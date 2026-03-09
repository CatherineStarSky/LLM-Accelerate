#include <torch/library.h>
#include <ATen/ATen.h>
#include <cstring>
#include "simd/kernels.h"
#include "thread_scheduler.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace simd;

static void check_float_cpu_contig(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(t.dtype() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

/**********************
 * 1) RMSNorm
 **********************/
at::Tensor rmsnorm_torch(const at::Tensor& x, const at::Tensor& gamma, double eps) {
  check_float_cpu_contig(x, "x");
  check_float_cpu_contig(gamma, "gamma");

  auto y = at::empty_like(x);

  const auto& K = get_kernels();  // rmsnorm

  const int64_t dim   = x.size(-1);
  const int64_t outer = x.numel() / dim;

  NormParams p{static_cast<float>(eps), gamma.data_ptr<float>()};

  const float* xp = x.data_ptr<float>();
  float*       yp = y.data_ptr<float>();

  ThreadScheduler& sched = ThreadScheduler::instance();
  // Prefer OpenMP static scheduling; fallback handled in scheduler
  sched.parallel_for_1d(outer, [&](int64_t i){
    K.rmsnorm(xp + i*dim, yp + i*dim, (size_t)dim, p);
  });
  return y;
}

/**********************
 * 2) RoPE
 **********************/
std::tuple<at::Tensor, at::Tensor>
rope_torch(const at::Tensor& q,
           const at::Tensor& k,
           const at::Tensor& cos,
           const at::Tensor& sin) {
  check_float_cpu_contig(q,   "q");
  check_float_cpu_contig(k,   "k");
  check_float_cpu_contig(cos, "cos");
  check_float_cpu_contig(sin, "sin");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");

  const int64_t dim = q.size(-1);
  const int64_t rows = q.numel() / dim;

  TORCH_CHECK(cos.numel() >= dim, "cos length must be >= dim");
  TORCH_CHECK(sin.numel() >= dim, "sin length must be >= dim");

  auto q_out = q.contiguous();
  auto k_out = k.contiguous();

  const auto& K = get_kernels();  // rope_apply

  float* qp = q_out.data_ptr<float>();
  float* kp = k_out.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* sp = sin.data_ptr<float>();

  // 多线程并行处理多行
  ThreadScheduler& sched = ThreadScheduler::instance();
  sched.parallel_for_1d(rows, [&](int64_t r) {
    // 每个线程处理一行
    float* qr = qp + r * dim;
    float* kr = kp + r * dim;
    const float* cr = cp + r * dim;
    const float* sr = sp + r * dim;
    // 调用 SIMD 实现处理单行（rows=1）
    K.rope_apply(qr, kr, (size_t)dim, 1, cr, sr);
  });

  return std::make_tuple(q_out, k_out);
}

/**********************
 * 3) Pack QKV 
 **********************/
at::Tensor pack_qkv_torch(const at::Tensor& qkv) {
  check_float_cpu_contig(qkv, "qkv");
  TORCH_CHECK(qkv.dim() == 3, "qkv must be 3D: [rows, 3, dim]");

  const int64_t rows = qkv.size(0);
  const int64_t three = qkv.size(1);
  const int64_t dim = qkv.size(2);
  TORCH_CHECK(three == 3, "qkv.shape[1] must be 3 (q, k, v)");

  auto out = at::empty_like(qkv);

  const auto& K = get_kernels();  // There is pack_qkv in it

  const float* src = qkv.data_ptr<float>();
  float*       dst = out.data_ptr<float>();

  // pack_qkv function signature: (const float* x, float* x_packed, size_t in_dim, size_t K_block)
  // For [rows, 3, dim] format, we need to repack QKV
  // Optimization: Use SIMD functions to accelerate batch copying
  // Each Q/K/V block size is dim, so block_size = dim means copying the entire block at once
  const size_t block_size = (size_t)dim;
  
  // Safety check: ensure dim > 0
  if (dim <= 0) {
    return out;
  }

  // Use parallel processing for multiple rows, but handle Q/K/V separately for each row
  ThreadScheduler& sched = ThreadScheduler::instance();
  // Parallelize over rows, but process Q/K/V separately for each row
  sched.parallel_for_1d(rows, [&](int64_t r) {
    const float* row_src = src + r * 3 * dim;
    float* row_dst = dst + r * 3 * dim;
    
    // Process Q, K, V separately with SIMD-optimized copying
    // When K_block >= in_dim, pack_qkv uses SIMD for efficient batch copying
    for (int64_t which = 0; which < 3; ++which) {
      const float* qkv_src = row_src + which * dim;
      float* qkv_dst = row_dst + which * dim;
      
      // Use SIMD-optimized pack_qkv function
      // When block_size >= dim, function uses SIMD instructions for efficient batch copying
      // For small dimensions (< 4), use memcpy for safety
      if (dim < 4) {
        std::memcpy(qkv_dst, qkv_src, sizeof(float) * dim);
      } else {
        K.pack_qkv(qkv_src, qkv_dst, (size_t)dim, block_size);
      }
    }
  });

  return out;
}

/**********************
 * PyTorch
 **********************/
TORCH_LIBRARY(simd_opt, m) {
  m.def("rmsnorm(Tensor x, Tensor gamma, float eps=1e-5) -> Tensor");
  m.def("rope(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
  m.def("pack_qkv(Tensor qkv) -> Tensor");
  m.def("set_threads(int n) -> ()");
}

TORCH_LIBRARY_IMPL(simd_opt, CPU, m) {
  m.impl("rmsnorm",  rmsnorm_torch);
  m.impl("rope",     rope_torch);
  m.impl("pack_qkv", pack_qkv_torch);
  m.impl("set_threads", +[](int64_t n){ ThreadScheduler::instance().set_num_threads(static_cast<int>(n)); });
}