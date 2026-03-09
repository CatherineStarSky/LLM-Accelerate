#!/usr/bin/env python
"""Performance test runner, supporting Latency and Throughput testing"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.data_loader import MovieLensDataLoader
from benchmark.metrics import evaluate_batch, evaluate_ranking_quality
from src.inference.reranker import LLMReranker


class BenchmarkRunner:
    """Performance test runner"""
    
    def __init__(self, config_path: str = "benchmark/configs.yaml"):
        """
        Initialize the test runner
        
        Args:
            config_path: configuration file path
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
    
    def _load_config(self) -> Dict:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _prepare_data(self, n_users: int, n_candidates: int) -> List[Dict]:
        """Prepare test data"""
        loader = MovieLensDataLoader(
            data_dir=self.config['dataset']['data_dir']
        )
        
        samples = loader.generate_test_samples(
            n_users=n_users,
            n_candidates=n_candidates,
            seed=self.config['dataset']['seed']
        )
        
        return samples
    
    def run_latency_test(
        self,
        reranker: LLMReranker,
        samples: List[Dict],
        n_repeats: int,
        warmup_runs: int = 2
    ) -> Dict:
        """
        Run Latency Test
        
        Args:
            reranker: Inference engine instance
            samples: test samples
            n_repeats: number of repetitions
            warmup_runs: number of warmup times
        
        Returns:
            Test result dictionary
        """
        print(f"  Running latency test (warmup={warmup_runs}, repeats={n_repeats})...")
        
        # Warmup
        for _ in range(warmup_runs):
            for sample in samples[:min(2, len(samples))]:
                reranker.rerank(sample['context'], sample['candidates'])
        
        # formal test
        latencies = []
        
        for repeat in range(n_repeats):
            for sample in samples:
                _, elapsed = reranker.rerank(
                    sample['context'],
                    sample['candidates'],
                    return_time=True
                )
                latencies.append(elapsed * 1000)  # Convert to ms
        
        # Calculate statistical indicators
        latencies = np.array(latencies)
        result = {
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'n_samples': len(latencies)
        }
        
        return result
    
    def run_throughput_test(
        self,
        reranker: LLMReranker,
        samples: List[Dict],
        duration_seconds: int
    ) -> Dict:
        """
        Run throughput tests
        
        Args:
            reranker: Inference engine instance
            samples: test samples
            duration_seconds: test duration (seconds)
        
        Returns:
            Test result dictionary
        """
        print(f"  Running throughput test (duration={duration_seconds}s)...")
        
        start_time = time.time()
        total_candidates = 0
        n_requests = 0
        
        sample_idx = 0
        while time.time() - start_time < duration_seconds:
            sample = samples[sample_idx % len(samples)]
            reranker.rerank(sample['context'], sample['candidates'])
            
            total_candidates += len(sample['candidates'])
            n_requests += 1
            sample_idx += 1
        
        elapsed = time.time() - start_time
        
        result = {
            'throughput_candidates_per_sec': total_candidates / elapsed,
            'throughput_requests_per_sec': n_requests / elapsed,
            'total_candidates': total_candidates,
            'total_requests': n_requests,
            'elapsed_seconds': elapsed
        }
        
        return result
    
    def run_quality_evaluation(
        self,
        reranker: LLMReranker,
        samples: List[Dict],
        k_list: List[int] = [10, 50],
        rating_threshold: float = 4.0
    ) -> Dict:
        """
        Evaluate ranking quality metrics (NDCG, MRR, Recall)
        
        Args:
            reranker: Inference engine instance
            samples: Test samples with candidates and ground truth
            k_list: List of k values for metrics (e.g., [10, 50])
            rating_threshold: Rating threshold for binary relevance
        
        Returns:
            Dictionary of quality metrics
        """
        all_scores = []
        
        # Get predictions for all samples
        for sample in samples:
            scores, _ = reranker.rerank(
                sample['context'],
                sample['candidates'],
                return_time=True
            )
            all_scores.append(scores)
        
        # Evaluate ranking quality
        quality_metrics = evaluate_batch(
            samples,
            all_scores,
            k_list=k_list,
            rating_threshold=rating_threshold
        )
        
        return quality_metrics
    
    def run_single_config(
        self,
        use_simd: bool,
        batch_size: int,
        simd_threads: int,
        n_users: int,
        n_candidates: int,
        mode: str = 'small'
    ) -> Dict:
        """
        Run tests for a single configuration
        
        Args:
            use_simd: whether to use SIMD
            batch_size: Python layer batch size
            simd_threads: Number of C++ layer SIMD threads
            n_users: number of users
            n_candidates: number of candidates
            mode: test mode
        
        Returns:
            Test results
        """
        print(f"\n=== Config: SIMD={use_simd}, BatchSize={batch_size}, "
              f"SimdThreads={simd_threads}, Users={n_users}, Candidates={n_candidates} ===")
        
        # Prepare data
        print("  Preparing data...")
        samples = self._prepare_data(n_users, n_candidates)
        
        # Create an inference engine
        print("  Loading model...")
        reranker = LLMReranker(
            use_simd=use_simd,
            batch_size=batch_size,
            simd_threads=simd_threads,
            verbose=False
        )
        
        # Get test parameters
        mode_config = self.config['test_modes'][mode]
        
        result = {
            'use_simd': use_simd,
            'batch_size': batch_size,
            'simd_threads': simd_threads,
            'n_users': n_users,
            'n_candidates': n_candidates,
            'mode': mode
        }
        
        # Latency test
        latency_result = self.run_latency_test(
            reranker,
            samples,
            n_repeats=mode_config['n_repeats'],
            warmup_runs=mode_config['warmup_runs']
        )
        result.update({f'latency_{k}': v for k, v in latency_result.items()})
        
        # Throughput test (if configured)
        if mode_config.get('duration_seconds'):
            throughput_result = self.run_throughput_test(
                reranker,
                samples,
                duration_seconds=mode_config['duration_seconds']
            )
            result.update({f'throughput_{k}': v for k, v in throughput_result.items()})
        
        # Ranking quality evaluation (NDCG, MRR, Recall)
        print("  Evaluating ranking quality...")
        quality_result = self.run_quality_evaluation(
            reranker,
            samples,
            k_list=mode_config.get('quality_k_list', [10, 50])
        )
        result.update({f'quality_{k}': v for k, v in quality_result.items()})
        
        self.results.append(result)
        
        print(f"  ✓ Latency P50: {result['latency_p50']:.2f}ms, "
              f"P95: {result['latency_p95']:.2f}ms")
        
        if 'throughput_throughput_candidates_per_sec' in result:
            print(f"  ✓ Throughput: {result['throughput_throughput_candidates_per_sec']:.2f} candidates/s")
        
        # Print quality metrics
        if 'quality_avg_ndcg@10' in result:
            print(f"  ✓ NDCG@10: {result['quality_avg_ndcg@10']:.4f}, "
                  f"MRR@10: {result['quality_avg_mrr@10']:.4f}, "
                  f"Recall@50: {result['quality_avg_recall@50']:.4f}")
        
        return result
    
    def run_grid_search(self, mode: str = 'small'):
        """
        Run a parametric grid search
        
        Args:
            mode: test mode (quick/small/full)
        """
        print(f"\n{'='*60}")
        print(f"Starting Grid Search - Mode: {mode}")
        print(f"{'='*60}")
        
        mode_config = self.config['test_modes'][mode]
        
        # Get a list of candidates
        if 'n_candidates_list' in mode_config:
            n_candidates_list = mode_config['n_candidates_list']
        else:
            n_candidates_list = [mode_config['n_candidates']]
        
        # Iterate through all configurations
        batch_sizes = self.config.get('param_grid', {}).get('batch_sizes', [1])
        simd_threads_list = self.config.get('param_grid', {}).get('simd_threads', [1])
        for use_simd in [False, True]:
            for n_candidates in n_candidates_list:
                for batch_size in batch_sizes:
                    for simd_threads in simd_threads_list:
                        try:
                            self.run_single_config(
                                use_simd=use_simd,
                                batch_size=batch_size,
                                simd_threads=simd_threads,
                                n_users=mode_config['n_users'],
                                n_candidates=n_candidates,
                                mode=mode
                            )
                        except Exception as e:
                            print(f"  ✗ Error: {e}")
                            continue
    
    def save_results(self, output_path: str = None):
        """Save test results to CSV"""
        if output_path is None:
            output_dir = self.config['output']['results_dir']
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, self.config['output']['csv_filename'])
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        
        return output_path
    
    def print_summary(self):
        """Print test summary"""
        if not self.results:
            print("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        print(f"Total configurations tested: {len(df)}")
        print(f"\nLatency (ms):")
        latency_cols = ['use_simd', 'n_candidates', 'latency_p50', 'latency_p95']
        available_latency_cols = [col for col in latency_cols if col in df.columns]
        print(df[available_latency_cols].to_string(index=False))
        
        # Print quality metrics if available
        quality_cols = [col for col in df.columns if col.startswith('quality_avg_')]
        if quality_cols:
            print(f"\nRanking Quality Metrics:")
            quality_display_cols = ['use_simd', 'n_candidates'] + quality_cols[:6]  # Show first 6 quality metrics
            available_quality_cols = [col for col in quality_display_cols if col in df.columns]
            print(df[available_quality_cols].to_string(index=False))
        
        # Calculate speedup
        baseline_df = df[df['use_simd'] == False]
        simd_df = df[df['use_simd'] == True]
        if len(baseline_df) > 0 and len(simd_df) > 0:
            baseline = baseline_df.iloc[0]
            simd = simd_df.iloc[0]
            speedup = baseline['latency_p50'] / simd['latency_p50']
            print(f"\nSpeedup (SIMD vs Baseline): {speedup:.2f}x")
            
            # Compare quality metrics
            if 'quality_avg_ndcg@10' in baseline and 'quality_avg_ndcg@10' in simd:
                ndcg_baseline = baseline['quality_avg_ndcg@10']
                ndcg_simd = simd['quality_avg_ndcg@10']
                ndcg_diff = ndcg_simd - ndcg_baseline
                print(f"NDCG@10 difference (SIMD - Baseline): {ndcg_diff:.4f}")
        elif len(simd_df) == 0:
            print(f"\nWarning: No successful SIMD runs found. All SIMD tests failed.")


def main():
    parser = argparse.ArgumentParser(description="LLM Re-ranking Benchmark Runner")
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'small', 'full'],
                        help='Test mode')
    parser.add_argument('--config', type=str, default='benchmark/configs.yaml',
                        help='Config file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(config_path=args.config)
    runner.run_grid_search(mode=args.mode)
    runner.print_summary()
    runner.save_results(output_path=args.output)


if __name__ == "__main__":
    main()

