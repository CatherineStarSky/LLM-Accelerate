# LLM Re-ranking 快速使用指南

## 环境准备

```bash
# 激活conda环境
conda activate CEG5206

# 安装依赖（首次使用）
pip install pandas pyyaml tabulate
```

## 编译SIMD库（可选）

### macOS/Linux:
```bash
cd build
export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake ..
cmake --build .
```

### Windows:
```cmd
cd build
set CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake ..
cmake --build . --config Release
```

---

## 基本命令

### 1. 单次推理测试

```bash
# Baseline（无SIMD，串行）
python src/main_rerank.py --mode single --n-candidates 10

# 使用SIMD优化
python src/main_rerank.py --mode single --simd --n-candidates 10

# 使用batch处理（batch_size=4）
python src/main_rerank.py --mode single --batch-size 4 --n-candidates 10

# 使用SIMD + C++多线程
python src/main_rerank.py --mode single --simd --simd-threads 8 --n-candidates 10

# 完整配置：SIMD + batch + C++线程
python src/main_rerank.py --mode single --simd --batch-size 4 --simd-threads 8 --n-candidates 10
```

### 2. 性能测试

```bash
# 快速测试（10个候选，约1分钟）
python src/main_rerank.py --mode benchmark --benchmark-mode quick

# 快速测试+生成报告
python src/main_rerank.py --mode benchmark --benchmark-mode quick --generate-report

# 20个候选测试（约2分钟）
python src/main_rerank.py --mode benchmark --benchmark-mode quick2x --generate-report

# 50个候选测试（约30分钟）
python src/main_rerank.py --mode benchmark --benchmark-mode medium --generate-report
```

### 3. 一键运行

```bash
./run_demo.sh
```

---

## 命令行参数

```
--mode {single,benchmark}
    single: 单次推理测试
    benchmark: 性能测试

--simd
    启用SIMD优化

--batch-size N
    Python层batch大小（默认1，>=2时启用批量推理）
    推荐值：2, 4, 8

--simd-threads N
    C++层SIMD算子线程数（默认1，仅在--simd时生效）
    推荐值：4, 8（CPU核心数）

--n-candidates N
    候选数量（默认50）

--benchmark-mode {quick,quick2x,medium,small,full}
    quick: 3用户×10候选×3次 (1分钟)
    quick2x: 3用户×20候选×3次 (2分钟)
    medium: 3用户×50候选×2次 (22分钟)
    small: 10用户×50候选×5次 (3小时)
    full: 50用户×多种候选×10次 (30+小时)

--generate-report
    测试后自动生成性能报告
```

---

## 查看结果

测试完成后，结果保存在：
- `results/raw_data.csv` - 原始性能数据
- `results/raw_data_report.md` - 自动生成的报告

---

## 常见问题

**Q: 测试很慢？**
- 使用CPU推理，每个候选需要4-5秒
- 建议用quick模式快速验证
- 如需完整测试，使用GPU或减少测试规模

**Q: SIMD没加速？**
- 小数据量时SIMD开销可能超过收益
- 需要更大规模数据才能看到效果
- M1芯片PyTorch已高度优化

**Q: 找不到模型？**
- 首次运行会自动下载Qwen3-0.6B（约1.2GB）
- 需要网络连接

---

## 给B/C同学的接口

### B同学（SIMD优化）

**测试SIMD**：
```bash
python src/main_rerank.py --mode benchmark --benchmark-mode quick --generate-report
```

**扩展位置**：
- `src/simd/` - SIMD内核实现
- `src/ops/simd_ops.cpp` - PyTorch算子注册
- `src/inference/simd_patch.py` - 模型patch逻辑

### C同学（多线程）

**测试多线程**：
```bash
python src/main_rerank.py --mode single --simd --threads 4
```

**实现位置**：
- `src/inference/reranker.py` 第140行有TODO注释
- Python层：使用ThreadPoolExecutor
- C++层：在simd_ops.cpp中添加OpenMP

---

## 项目结构

```
CEG5206_Proj_O1/
├── src/
│   ├── main_rerank.py          # 主入口
│   └── inference/
│       ├── reranker.py         # 推理引擎
│       └── simd_patch.py       # SIMD patch
├── benchmark/
│   ├── data_loader.py          # 数据加载
│   ├── benchmark_runner.py     # 性能测试
│   ├── report_generator.py     # 报告生成
│   └── configs.yaml            # 测试配置
├── results/                    # 测试结果
├── ml-100k/                    # MovieLens数据集
└── build/
    └── simd_ops.dylib          # SIMD库
```

---

**完整文档**: 查看 `README.md`

