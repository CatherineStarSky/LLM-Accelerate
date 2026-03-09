# I. Description
Our goal is to implement a recommendation system re-ranking module based on LLM on CPU. Recommendation systems are usually divided into two stages: the first stage uses traditional recall methods to generate a candidate set, and the second stage uses a small LLM to perform relevance matching between the context and candidates, scoring and ranking, and returning the top K most relevant results. We need to implement this system in three parts: 
1) build a pure FP32, single-thread baseline; 
2) design a memory-optimized LLM, using KV-Cache reuse, paging/chunking, zero-copy pooled allocation, NUMA affinity, SoA/Block-SoA layout; 
3) implement SIMD vectorization and multithreading. 

We need to compare the memory layouts of AoS and SoA and Block-SoA, report latency, throughput, and speedup ratio, and analyze trade-offs together with ranking quality (NDCG, MRR). We will compare the impact of different batch sizes, different memory methods, number of threads, and SIMD width on latency.

Regarding the terms and symbols used in this project: KV-Cache reuse technology means running the model once first to cache the KV of the context; for subsequent candidate items, only compute the KV of newly added tokens and concatenate to the cache to reduce redundant computation; paging/chunking means splitting the cache into fixed-size pages or blocks, and fetching the corresponding number of pages each time when applying; zero-copy pooled allocation means pre-allocating a large memory block as a pool, making memory continuous and improving hit rate; NUMA means that in multi-socket CPUs, memory uses a NUMA architecture, each CPU accesses its own node faster and other nodes slower, so we bind a certain thread to a certain CPU core and place the allocated data in the NUMA node corresponding to that core; AoS (Array of Structures) means each element is a structure with mixed fields, SoA (Structure of Arrays) means putting data of the same field in continuous memory, which can improve cache hit rate.

Regarding the two indicators of ranking quality: NDCG refers to “the degree to which relevant content is ranked in front”; MRR refers to the mean reciprocal rank, measuring “how quickly the system can rank the first relevant result to the front.”

# II. Required Implementations
1. Baseline
- Select 0.5B–1.3B open-source Decoder-only LLM, such as Qwen3-0.6B, model can be loaded as FP32 or FP16→FP32 computation, caching and chunking off, single-thread, single-sample inference.
- For each candidate generate "prompt = [user context + candidate description]", score each independently, scoring head can be used: binary classification logits for relevant/irrelevant, or pairwise comparison scoring.
- Python+PyTorch or C++ front end + oneDNN/BLAS back end, ensure correctness, for example consistency with small batch mode results.
2. SIMD Vectorization
- For common preprocessing, such as ASCII range filtering, space collapsing, case/Unicode simplification path using AVX2/AVX-512 batch processing of 32–64 byte blocks; using mask load/compare/pack instructions to reduce branching.
- Kernel hotspots:
  - LayerNorm / RMSNorm: vectorize mean/variance/scale shift, reduce scalar fallback.
  - QKV linear layer input packing and GEMM micro-kernel alignment (aligned by 8/16/32), use FMA/LDR instructions to fill SIMD lanes.
  - Position encoding (RoPE) trigonometric precomputation + vectorized reuse.
3. Multithreading
- Parallelize at candidate granularity, e.g., N candidates parallel append, parallel attention; or at layer/tensor block granularity, by head, by seq-block.
- Batch scheduler:
  - Build dynamic batching, collect multiple requests within T millisecond window, merge shared prefixes for batch inference, maximize throughput under P95 latency target.
  - Work stealing and NUMA-aware task queues, cross-queue stealing keeps load balanced.
- Static scheduling (e.g., OpenMP "schedule(static)") keeps affinity for KV pages; explicit prefetch reduces L1/L2 miss.

# III. Dataset and Shapes
- Dataset: use MovieLens as recommendation candidate dataset for this topic.
- Candidate description length: title+introduction Lt ≤128 tokens, candidate number N ∈ {50, 100, 200, 500, 1000}.
- Model input length: shared prefix 200 ≤ Lp ≤300, candidate segment 64 ≤ Lc ≤128.

# IV. Experiment Grid
For each implementation (Baseline, memory-optimized direct inference, +SIMD, +multithreading, +quantization), set the following parameters:
- Batch size B ∈ {1, 2, 4, 8, 16}
- Candidate number N ∈ {50, 100, 200, 500, 1000}
- Thread number T ∈ {1, 2, 4, 8, 16}
- SIMD: {SSE4.2, AVX2, AVX-512}
- Layout: {AoS, SoA, Block-SoA (blocked by head×seq)}
Hardware configuration is as below:
- CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz   2.69 GHz
- Memory: 32.0 GB
- OS: Windows 11, version 24H2

V. Measurement Protocol
- Performance metrics include latency (ms/request), statistics P50/P95; throughput counted by number of candidate items.
- Use hardware counters to collect: cycles, instructions, L1/L2 miss, LLC miss, branch misprediction, and compare memory bandwidth usage and memory access instructions under different strategies.
- Use recommendation quality metrics NDCG@10, MRR@10, Recall@50; when using quantization/cache strategies, report relative quality loss.
- Warm up 3–5 times during experiment; record median of 10 test results.
- All time statistics must cover end-to-end process, including tokenization, packing, inference, ranking, etc.

VI. Toolchain & Requirements
- Optional languages: C/C++17 (core inference and memory management); Python (data preparation/evaluation scripts, optional).
- Build tools: support CMake or Makefile. Build tools are used for automating code compilation, linking, dependency management, etc. CMake has strong cross-platform compatibility and can generate build files for different systems; Makefile is widely used in Linux/macOS environments. Both can meet the automation build needs of the project.
- Thread handling: can use OpenMP, TBB or pthreads. Multithreading parallelism is key to improving computing efficiency.
- SIMD instruction sets: conditional compilation adaptation for x86 architecture or ARM architecture. SIMD (Single Instruction Multiple Data) instructions can perform the same operation on multiple data at once, greatly improving parallel computing efficiency; x86 and ARM are mainstream processor architectures, each with different SIMD instruction sets; conditional compilation allows the program to automatically adapt to the optimal instruction set on different hardware.
- BLAS/GEMM libraries: can use oneDNN, OpenBLAS or BLIS (focus on data packing/unpacking and memory alignment).
- Operating systems: compatible with Linux, macOS and Windows. Cross-OS compatibility can expand the applicable scenarios of the project and meet the needs of different user environments.
- Reproducibility requirements: provide one-click run scripts to ensure demonstration configurations with N=100/200 can run on laptops with 16GB–32GB memory.

# VII. Plots
- Relationship between latency/throughput and candidate number N (multi-curve comparison: different thread configurations, different KV strategies). Multi-curve comparison can intuitively show how performance changes with N under “thread count change” and “different KV storage strategies”, helping to find the optimal configuration.
- Speedup ratio compared with baseline version (break down to show: effect of memory optimization, SIMD instruction, thread parallelism, quantization gradually stacked).
- Thread scalability curve (under optimal configuration, QPS trend with thread number change). Thread scalability curve is used to observe “whether performance increases linearly as the number of threads increases”, to find performance bottlenecks (e.g., when too many threads cause resource contention and QPS growth slows), and determine the optimal thread number.
- SIMD width sensitivity analysis. Analyze performance differences of different instruction sets to clarify the performance improvement range of hardware instruction sets and guide optimization directions for specific hardware.
- Quality-performance frontier curve (horizontal axis NDCG@10 loss, vertical axis throughput improvement, draw Pareto frontier). Pareto frontier is the set of optimal balance points between “performance improvement” and “quality loss” (i.e., cannot further improve performance without losing quality), helping to select the maximum performance improvement plan under acceptable quality loss.
- Cache performance observation (scatter plot of correlation between L1/L2/LLC cache miss rate and latency). Correlation analysis can reveal the impact of “cache utilization efficiency” on latency (e.g., high miss rate usually corresponds to high latency), providing basis for memory layout optimization (such as improving data locality).

# VIII. Advanced (Optional)
1. Memory-Optimized Direct Inference
Reconstruct inference data path around memory and access patterns:
- Encode "[user context]" as shared prefix, generate reusable KV-Cache; for N candidates only append candidate segment tokens, reducing redundant computation and memory bandwidth.
- Fixed-size pages (e.g., 1–2MB), avoid large continuous allocations, facilitate concurrent recycling and NUMA migration.
- Layout and pooling:
  - For attention KV use SoA (Structure of Arrays)/Block-SoA layout, so that the same dimension such as head, seq is blocked and along continuous memory direction, facilitating vectorization and prefetch.
  - Custom arena/pool, hierarchical management (small objects: token/offset, large objects: KV pages/tensor blocks), zero-copy reuse.
- Candidates are sharded to different NUMA nodes; thread core binding + first-touch ensure locality.
- Change tokenizer to batch, SIMD-accelerated implementation, and feed tokenization results directly into the model with zero-copy views.
2. Quantization and Mixed Precision 
- W(INT8/INT4) + A(INT8/FP16), weight quantization, for example on out-proj, FFN weights, activation INT8 or keep FP16; introduce symmetric quantization and calibration.
- KV stored as FP8/INT8, decode to FP16 during computation; maintain Top-K ranking quality degradation not significant, e.g., NDCG@10 loss <1% as target.


