# NDCG排序指标使用指南

## 概述

本项目已集成NDCG（Normalized Discounted Cumulative Gain）排序质量评估指标，用于评估重排序系统的效果。NDCG是信息检索和推荐系统中广泛使用的评估指标，能够衡量排序结果的质量。

## 功能特性

### 支持的指标

1. **NDCG@k** - 归一化折损累积增益
   - Binary版本：使用二值相关性（rating >= threshold为相关）
   - Continuous版本：使用连续评分作为相关性分数

2. **MRR@k** - 平均倒数排名
   - 衡量第一个相关结果出现的位置

3. **Recall@k** - 召回率
   - 衡量前k个结果中包含的相关项目比例

## 使用方法

### 1. 在基准测试中自动计算

运行基准测试时，NDCG等指标会自动计算：

```bash
python3 benchmark/benchmark_runner.py --mode small
```

输出示例：
```
  ✓ Latency P50: 125.50ms, P95: 180.30ms
  ✓ NDCG@10: 0.8234, MRR@10: 0.7567, Recall@50: 0.9123
```

### 2. 单独使用metrics模块

```python
from benchmark.metrics import evaluate_ranking_quality

# 准备数据
candidates = [
    {'rating': 5, 'text': 'Movie A'},
    {'rating': 4, 'text': 'Movie B'},
    {'rating': 1, 'text': 'Movie C'},
    # ...
]

# 获取预测分数
scores = [4.5, 3.8, 1.2, ...]  # 从reranker获取

# 计算指标
metrics = evaluate_ranking_quality(
    candidates=candidates,
    scores=scores,
    k_list=[10, 50],
    rating_threshold=4.0  # rating >= 4.0 视为相关
)

print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
print(f"MRR@10: {metrics['mrr@10']:.4f}")
print(f"Recall@50: {metrics['recall@50']:.4f}")
```

### 3. 批量评估

```python
from benchmark.metrics import evaluate_batch

# 多个查询样本
samples = [
    {'candidates': [...], 'context': '...'},
    {'candidates': [...], 'context': '...'},
    # ...
]

# 对应的预测分数
all_scores = [
    [4.5, 3.8, ...],  # 第一个查询的分数
    [4.2, 3.9, ...],  # 第二个查询的分数
    # ...
]

# 批量评估
batch_metrics = evaluate_batch(
    samples=samples,
    all_scores=all_scores,
    k_list=[10, 50],
    rating_threshold=4.0
)

print(f"Average NDCG@10: {batch_metrics['avg_ndcg@10']:.4f}")
print(f"Std NDCG@10: {batch_metrics['std_ndcg@10']:.4f}")
```

## 配置说明

在 `benchmark/configs.yaml` 中可以配置评估参数：

```yaml
test_modes:
  small:
    quality_k_list: [10, 50]  # 计算NDCG@10, MRR@10, Recall@50等
```

## 指标解释

### NDCG@k

- **范围**: 0.0 到 1.0（越高越好）
- **含义**: 归一化的折损累积增益，考虑位置权重
- **公式**: NDCG = DCG / IDCG
  - DCG: 预测排序的折损累积增益
  - IDCG: 理想排序的折损累积增益

### MRR@k

- **范围**: 0.0 到 1.0（越高越好）
- **含义**: 第一个相关结果出现位置的倒数
- **示例**: 如果第一个相关结果在第3位，MRR = 1/3 = 0.333

### Recall@k

- **范围**: 0.0 到 1.0（越高越好）
- **含义**: 前k个结果中包含的相关项目比例
- **公式**: Recall@k = (前k个中的相关数) / (总相关数)

## 测试

运行测试脚本验证metrics功能：

```bash
python3 benchmark/test_metrics.py
```

## 结果输出

基准测试结果会保存到CSV文件（`results/raw_data.csv`），包含以下质量指标列：

- `quality_avg_ndcg@10` - 平均NDCG@10
- `quality_std_ndcg@10` - NDCG@10标准差
- `quality_avg_ndcg@10_continuous` - 连续版本NDCG@10
- `quality_avg_mrr@10` - 平均MRR@10
- `quality_avg_recall@50` - 平均Recall@50
- 等等...

## 注意事项

1. **相关性阈值**: 默认使用 `rating >= 4.0` 作为相关标准，可在调用时修改
2. **k值选择**: 建议根据候选数量选择合理的k值（如候选50个，使用k=10, 50）
3. **性能影响**: 质量评估需要额外的推理时间，在性能测试中会自动包含

## 示例输出

```
=== Config: SIMD=True, BatchSize=4, SimdThreads=4, Users=10, Candidates=50 ===
  Preparing data...
  Loading model...
  Running latency test (warmup=2, repeats=5)...
  Evaluating ranking quality...
  ✓ Latency P50: 125.50ms, P95: 180.30ms
  ✓ NDCG@10: 0.8234, MRR@10: 0.7567, Recall@50: 0.9123
```

## 相关文件

- `benchmark/metrics.py` - 指标计算实现
- `benchmark/benchmark_runner.py` - 基准测试运行器（已集成质量评估）
- `benchmark/test_metrics.py` - 测试脚本
- `benchmark/configs.yaml` - 配置文件

