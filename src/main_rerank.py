#!/usr/bin/env python
"""LLM Re-ranking Main Entry Program"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.data_loader import MovieLensDataLoader
from benchmark.metrics import evaluate_ranking_quality
from src.inference.reranker import LLMReranker
from benchmark.benchmark_runner import BenchmarkRunner


def run_single_inference(args):
    """Run single inference test"""
    print("=== Single Inference Mode ===\n")
    
    # Load data
    print("Loading test data...")
    loader = MovieLensDataLoader(data_dir=args.data_dir)
    samples = loader.generate_test_samples(
        n_users=1,
        n_candidates=args.n_candidates,
        seed=42
    )
    
    sample = samples[0]
    print(f"User ID: {sample['user_id']}")
    print(f"Context: {sample['context']}")
    print(f"Number of candidates: {len(sample['candidates'])}\n")
    
    # Create inference engine
    print("Loading model...")
    reranker = LLMReranker(
        use_simd=args.simd,
        batch_size=args.batch_size,
        simd_threads=args.simd_threads,
        verbose=True
    )
    
    # Run inference
    print("\nRunning inference...")
    scores, elapsed = reranker.rerank(
        sample['context'],
        sample['candidates'],
        return_time=True
    )
    
    # Display results
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Inference time: {elapsed:.3f}s ({elapsed*1000:.2f}ms)")
    print(f"\nTop 5 candidates (by score):")
    
    # Sort and display top 5
    scored_candidates = list(zip(sample['candidates'], scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    for i, (cand, score) in enumerate(scored_candidates[:5], 1):
        print(f"{i}. {cand['text']}")
        print(f"   Score: {score:.4f}, Rating: {cand['rating']}")
    
    # Calculate ranking quality metrics
    print(f"\n{'='*60}")
    print("Ranking Quality Metrics:")
    print(f"{'='*60}")
    
    # Determine k values based on number of candidates
    n_cands = len(sample['candidates'])
    k_list = []
    if n_cands >= 50:
        k_list = [10, 50]
    elif n_cands >= 20:
        k_list = [10, 20]
    elif n_cands >= 10:
        k_list = [5, 10]
    else:
        k_list = [min(5, n_cands)]
    
    metrics = evaluate_ranking_quality(
        candidates=sample['candidates'],
        scores=scores,
        k_list=k_list,
        rating_threshold=4.0
    )
    
    # Display metrics
    for k in k_list:
        print(f"\nMetrics @{k}:")
        if f'ndcg@{k}' in metrics:
            print(f"  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        if f'ndcg@{k}_continuous' in metrics:
            print(f"  NDCG@{k} (continuous): {metrics[f'ndcg@{k}_continuous']:.4f}")
        if f'mrr@{k}' in metrics:
            print(f"  MRR@{k}: {metrics[f'mrr@{k}']:.4f}")
        if f'recall@{k}' in metrics:
            print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    
    # Show ground truth count
    ground_truth_count = len(sample.get('ground_truth', []))
    print(f"\nGround truth: {ground_truth_count} relevant items (rating >= 4)")
    
    print(f"\n{'='*60}")


def run_benchmark(args):
    """Run performance benchmark"""
    print("=== Benchmark Mode ===\n")
    
    runner = BenchmarkRunner(config_path='benchmark/configs.yaml')
    runner.run_grid_search(mode=args.benchmark_mode)
    runner.print_summary()
    
    csv_path = runner.save_results()
    
    # Auto-generate report
    if args.generate_report:
        print("\nGenerating report...")
        from benchmark.report_generator import ReportGenerator
        
        report_path = csv_path.replace('.csv', '_report.md')
        generator = ReportGenerator(csv_path=csv_path, output_path=report_path)
        generator.generate()


def main():
    parser = argparse.ArgumentParser(
        description="LLM Re-ranking Baseline - Main Entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single inference (without SIMD)
  python src/main_rerank.py --mode single --n-candidates 50
  
  # Single inference (with SIMD)
  python src/main_rerank.py --mode single --simd --n-candidates 50
  
  # Quick benchmark test
  python src/main_rerank.py --mode benchmark --benchmark-mode quick
  
  # Complete benchmark test
  python src/main_rerank.py --mode benchmark --benchmark-mode full --generate-report
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='single',
        choices=['single', 'benchmark'],
        help='运行模式：single=单次推理，benchmark=性能测试'
    )
    
    parser.add_argument(
        '--simd',
        action='store_true',
        help='启用SIMD优化'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Python层batch大小（>=2时启用批量推理）'
    )
    
    parser.add_argument(
        '--simd-threads',
        type=int,
        default=1,
        help='C++层SIMD算子线程数'
    )
    
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=50,
        help='候选数量'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='ml-100k',
        help='MovieLens数据目录'
    )
    
    parser.add_argument(
        '--benchmark-mode',
        type=str,
        default='quick',
        choices=['quick', 'quick2x', 'medium', 'small', 'full'],
        help='Benchmark测试模式'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Benchmark完成后自动生成报告'
    )
    
    args = parser.parse_args()
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Execute the corresponding mode
    if args.mode == 'single':
        run_single_inference(args)
    elif args.mode == 'benchmark':
        run_benchmark(args)


if __name__ == "__main__":
    main()

