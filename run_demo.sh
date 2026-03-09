#!/bin/bash
# LLM Re-ranking一键运行脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "LLM Re-ranking Baseline Demo"
echo "========================================"
echo ""

# 激活Python环境
# echo "Activating CEG5206 environment..."
# source ~/miniconda3/bin/activate CEG5206 || source /opt/miniconda3/bin/activate CEG5206 || echo "Warning: Could not activate conda environment"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查SIMD库是否存在
if [ ! -f "build/simd_ops.dylib" ]; then
    echo "Warning: SIMD library not found at build/simd_ops.dylib"
    echo "Please build it first with:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && cmake --build ."
    echo ""
    echo "Continuing with baseline-only test..."
    SIMD_FLAG=""
else
    echo "✓ SIMD library found"
    SIMD_FLAG="--simd"
fi

echo ""
echo "========================================"
echo "Step 1: Running single inference test (with NDCG metrics)"
echo "========================================"
python src/main_rerank.py --mode single --n-candidates 10

echo ""
echo "========================================"
echo "Step 2: Running quick benchmark (includes NDCG/MRR/Recall)"
echo "========================================"
python src/main_rerank.py --mode benchmark --benchmark-mode quick --generate-report

echo ""
echo "========================================"
echo "Demo completed!"
echo "========================================"
echo ""
echo "Results saved in: results/"
echo "- results/raw_data.csv"
echo "- results/raw_data_report.md"
echo ""
echo "To run full benchmark:"
echo "  python src/main_rerank.py --mode benchmark --benchmark-mode full --generate-report"

