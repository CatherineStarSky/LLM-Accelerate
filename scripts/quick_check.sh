#!/bin/bash
# 快速检查项目配置

echo "=========================================="
echo "LLM Re-ranking Quick Check"
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."

# 检查目录结构
echo "--- Directory Structure ---"
for dir in src benchmark scripts results ml-100k build; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ NOT found"
    fi
done

echo ""
echo "--- Key Files ---"
for file in src/main_rerank.py benchmark/data_loader.py run_demo.sh; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file NOT found"
    fi
done

echo ""
echo "--- Data Files ---"
for file in ml-100k/u.item ml-100k/u.data; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✓ $file ($size)"
    else
        echo "✗ $file NOT found"
    fi
done

echo ""
echo "--- SIMD Library ---"
if [ -f "build/simd_ops.dylib" ]; then
    size=$(ls -lh "build/simd_ops.dylib" | awk '{print $5}')
    echo "✓ build/simd_ops.dylib ($size)"
else
    echo "✗ build/simd_ops.dylib NOT found"
    echo "  (Optional) Build with: cd build && cmake .. && cmake --build ."
fi

echo ""
echo "=========================================="
echo "Setup looks good!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate conda environment:"
echo "   conda activate CEG5206"
echo ""
echo "2. Run demo:"
echo "   ./run_demo.sh"
echo ""
echo "   或手动运行:"
echo "   python src/main_rerank.py --mode single --n-candidates 10"

