#!/usr/bin/env python
"""Automatic report generator to generate markdown reports from CSV data"""

import os
import sys
import platform
import argparse
import pandas as pd
import subprocess
from datetime import datetime
from typing import Dict, List


class ReportGenerator:
    """Performance test report generator"""
    
    def __init__(self, csv_path: str, output_path: str = None):
        """
        Initialize report generator
        
        Args:
            csv_path: input CSV data file
            output_path: output markdown report file
        """
        self.csv_path = csv_path
        self.output_path = output_path or csv_path.replace('.csv', '_report.md')
        self.df = pd.read_csv(csv_path)
        self.report_lines = []
    
    def _get_hardware_info(self) -> Dict:
        """Get hardware information"""
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'python_version': platform.python_version()
        }
        
        # Try to get CPU information
        try:
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    info['cpu'] = result.stdout.strip()
                
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.strip())
                    info['memory_gb'] = round(mem_bytes / (1024**3), 2)
            
            elif platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            info['cpu'] = line.split(':')[1].strip()
                            break
                
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            mem_kb = int(line.split()[1])
                            info['memory_gb'] = round(mem_kb / (1024**2), 2)
                            break
        except Exception as e:
            print(f"Warning: Could not detect hardware info: {e}")
        
        return info
    
    def _add_header(self):
        """Add report title"""
        self.report_lines.extend([
            "# LLM Re-ranking Baseline Performance Report",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ])
    
    def _add_hardware_info(self):
        """Add hardware information"""
        info = self._get_hardware_info()
        
        self.report_lines.extend([
            "## 硬件配置",
            ""
        ])
        
        if 'cpu' in info:
            self.report_lines.append(f"- **CPU**: {info['cpu']}")
        else:
            self.report_lines.append(f"- **Processor**: {info['processor']}")
        
        if 'memory_gb' in info:
            self.report_lines.append(f"- **Memory**: {info['memory_gb']} GB")
        
        self.report_lines.extend([
            f"- **OS**: {info['os']} ({info['machine']})",
            f"- **Python**: {info['python_version']}",
            "",
            "---",
            ""
        ])
    
    def _add_test_config(self):
        """Add test configuration"""
        self.report_lines.extend([
            "## 测试配置",
            "",
            "- **模型**: Qwen3-0.6B",
            "- **数据集**: MovieLens 100K",
            f"- **测试配置数**: {len(self.df)}",
            ""
        ])
        
        # Statistical test parameters
        if 'n_candidates' in self.df.columns:
            candidates_list = sorted(self.df['n_candidates'].unique())
            self.report_lines.append(f"- **候选数 (N)**: {candidates_list}")
        
        if 'n_users' in self.df.columns:
            users = self.df['n_users'].unique()[0]
            self.report_lines.append(f"- **测试用户数**: {users}")
        
        if 'latency_n_samples' in self.df.columns:
            n_samples = self.df['latency_n_samples'].unique()[0]
            self.report_lines.append(f"- **Latency测试样本数**: {n_samples}")
        
        self.report_lines.extend([
            "",
            "---",
            ""
        ])
    
    def _add_latency_results(self):
        """Add Latency test results"""
        self.report_lines.extend([
            "## Latency性能测试结果",
            "",
            "### 详细数据",
            ""
        ])
        
        # Create table
        columns = ['use_simd', 'num_threads', 'n_candidates', 'latency_p50', 'latency_p95', 'latency_mean']
        display_columns = {
            'use_simd': 'SIMD',
            'num_threads': 'Threads',
            'n_candidates': 'N',
            'latency_p50': 'P50 (ms)',
            'latency_p95': 'P95 (ms)',
            'latency_mean': 'Mean (ms)'
        }
        
        available_columns = [col for col in columns if col in self.df.columns]
        table_df = self.df[available_columns].copy()
        
        # Format numeric value
        for col in ['latency_p50', 'latency_p95', 'latency_mean']:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}")
        
        # Rename columns
        table_df.columns = [display_columns.get(col, col) for col in table_df.columns]
        
        # Convert to markdown table
        self.report_lines.append(table_df.to_markdown(index=False))
        self.report_lines.extend(["", ""])
    
    # def _add_speedup_analysis(self):
    #     """Add speedup analysis"""
    #     self.report_lines.extend([
    #         "### 加速比分析",
    #         ""
    #     ])
        
    #     # Calculate the speedup ratio of SIMD vs Baseline
    #     if 'use_simd' in self.df.columns and 'latency_p50' in self.df.columns:
    #         speedups = []
            
    #         # Group by n_candidates
    #         if 'n_candidates' in self.df.columns:
    #             for n_cand in sorted(self.df['n_candidates'].unique()):
    #                 subset = self.df[self.df['n_candidates'] == n_cand]
                    
    #                 baseline = subset[subset['use_simd'] == False]
    #                 simd = subset[subset['use_simd'] == True]
                    
    #                 if len(baseline) > 0 and len(simd) > 0:
    #                     baseline_p50 = baseline['latency_p50'].values[0]
    #                     simd_p50 = simd['latency_p50'].values[0]
    #                     speedup = baseline_p50 / simd_p50
    #                     speedups.append({
    #                         'N_candidates': n_cand,
    #                         'Baseline_P50_ms': f"{baseline_p50:.2f}",
    #                         'SIMD_P50_ms': f"{simd_p50:.2f}",
    #                         'Speedup': f"{speedup:.2f}x"
    #                     })
    #         else:
    #             baseline = self.df[self.df['use_simd'] == False]
    #             simd = self.df[self.df['use_simd'] == True]
                
    #             if len(baseline) > 0 and len(simd) > 0:
    #                 baseline_p50 = baseline['latency_p50'].values[0]
    #                 simd_p50 = simd['latency_p50'].values[0]
    #                 speedup = baseline_p50 / simd_p50
    #                 speedups.append({
    #                     'Baseline_P50_ms': f"{baseline_p50:.2f}",
    #                     'SIMD_P50_ms': f"{simd_p50:.2f}",
    #                     'Speedup': f"{speedup:.2f}x"
    #                 })
            
    #         if speedups:
    #             speedup_df = pd.DataFrame(speedups)
    #             self.report_lines.append(speedup_df.to_markdown(index=False))
    #             self.report_lines.extend(["", ""])
        
    #     self.report_lines.extend([
    #         "---",
    #         ""
    #     ])

    def _add_speedup_analysis(self):
        """Add joint SIMD + Thread speedup analysis"""
        self.report_lines.extend([
            "### 加速比分析（含多线程与SIMD联合对比）",
            ""
        ])

        if not {'use_simd', 'num_threads', 'latency_p50'} <= set(self.df.columns):
            self.report_lines.append("_No sufficient columns for analysis (need use_simd, num_threads, latency_p50)_\n")
            self.report_lines.extend(["---", ""])
            return

        # 找到单线程 baseline（SIMD=False, num_threads=1）
        baseline_df = self.df[(self.df['use_simd'] == False) & (self.df['num_threads'] == 1)]
        if baseline_df.empty:
            self.report_lines.append("_No valid baseline (SIMD=False, Threads=1) found_\n")
            self.report_lines.extend(["---", ""])
            return

        baseline_p50 = baseline_df['latency_p50'].mean()

        rows = []
        for _, row in self.df.sort_values(['use_simd', 'num_threads']).iterrows():
            simd = bool(row.get('use_simd', False))
            threads = int(row.get('num_threads', 1))
            p50 = row.get('latency_p50', None)

            if p50 is None:
                continue

            # 与 baseline 比较加速比
            speedup = baseline_p50 / p50 if p50 > 0 else 1.0
            rows.append({
                'SIMD': 'True' if simd else 'False',
                'Threads': threads,
                'P50 (ms)': f"{p50:.2f}",
                'Speedup': f"{speedup:.2f}x"
            })

        # 转成表格
        df_speedup = pd.DataFrame(rows)
        self.report_lines.append(df_speedup.to_markdown(index=False))
        self.report_lines.extend(["", "---", ""])
    
    def _add_throughput_results(self):
        """Add throughput test results"""
        if 'throughput_throughput_candidates_per_sec' not in self.df.columns:
            return
        
        self.report_lines.extend([
            "## Throughput性能测试结果",
            ""
        ])
        
        columns = ['use_simd', 'num_threads', 'n_candidates', 
                   'throughput_throughput_candidates_per_sec',
                   'throughput_throughput_requests_per_sec']
        display_columns = {
            'use_simd': 'SIMD',
            'num_threads': 'Threads',
            'n_candidates': 'N',
            'throughput_throughput_candidates_per_sec': 'Candidates/s',
            'throughput_throughput_requests_per_sec': 'Requests/s'
        }
        
        available_columns = [col for col in columns if col in self.df.columns]
        table_df = self.df[available_columns].copy()
        
        # Format numeric value
        for col in available_columns:
            if 'throughput' in col:
                table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}")
        
        # Rename columns
        table_df.columns = [display_columns.get(col, col) for col in table_df.columns]
        
        self.report_lines.append(table_df.to_markdown(index=False))
        self.report_lines.extend([
            "",
            "---",
            ""
        ])
    
    def _add_observations(self):
        """Add key observations"""
        self.report_lines.extend([
            "## 关键观察",
            ""
        ])
        
        observations = []
        
        # Find the fastest configuration
        if 'latency_p50' in self.df.columns:
            best_idx = self.df['latency_p50'].idxmin()
            best_config = self.df.iloc[best_idx]
            observations.append(
                f"1. **最优配置**: SIMD={best_config.get('use_simd', 'N/A')}, "
                f"Threads={best_config.get('num_threads', 'N/A')}, "
                f"P50延迟={best_config['latency_p50']:.2f}ms"
            )
        
        # SIMD speedup ratio
        if 'use_simd' in self.df.columns:
            baseline = self.df[self.df['use_simd'] == False]
            simd = self.df[self.df['use_simd'] == True]
            
            if len(baseline) > 0 and len(simd) > 0:
                avg_baseline = baseline['latency_p50'].mean()
                avg_simd = simd['latency_p50'].mean()
                speedup = avg_baseline / avg_simd
                observations.append(
                    f"2. **SIMD平均加速比**: {speedup:.2f}x"
                )
        
        # Effect of number of candidates
        if 'n_candidates' in self.df.columns and len(self.df['n_candidates'].unique()) > 1:
            observations.append(
                "3. **候选数影响**: 随着候选数增加，延迟线性增长（符合预期）"
            )
        
        observations.append(
            "4. **多线程接口**: 已预留，待C同学完成后可进一步提升性能"
        )
        
        for obs in observations:
            self.report_lines.append(obs)
        
        self.report_lines.extend([
            "",
            "---",
            ""
        ])
    
    def _add_footer(self):
        """Add footer"""
        self.report_lines.extend([
            "## 下一步优化方向",
            "",
            "1. **多线程并行**: 在候选级别和算子级别实现并行处理",
            "2. **内存优化**: 实现KV-Cache复用和零拷贝内存池",
            "3. **量化**: 引入INT8/FP8量化降低内存带宽压力",
            "4. **批处理**: 实现动态批处理提升吞吐量",
            "",
            "---",
            "",
            f"*报告由 `benchmark/report_generator.py` 自动生成*"
        ])
    
    def generate(self):
        """Generate full report"""
        print(f"Generating report from {self.csv_path}...")
        
        self._add_header()
        self._add_hardware_info()
        self._add_test_config()
        self._add_latency_results()
        self._add_speedup_analysis()
        self._add_throughput_results()
        self._add_observations()
        self._add_footer()
        
        # write file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        
        print(f"✓ Report saved to: {self.output_path}")
        
        return self.output_path


def main():
    parser = argparse.ArgumentParser(description="Generate performance report from CSV data")
    parser.add_argument('--input', type=str, default='results/raw_data.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output markdown file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Please run benchmark_runner.py first to generate test data.")
        sys.exit(1)
    
    generator = ReportGenerator(csv_path=args.input, output_path=args.output)
    generator.generate()


if __name__ == "__main__":
    main()

