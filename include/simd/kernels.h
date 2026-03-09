#pragma once
#include <cstddef>

namespace simd {

struct NormParams {
  float eps;
  const float* gamma;
};

using rmsnorm_fn = void(*)(const float* x, float* y, size_t dim, const NormParams& p);


using rope_fn = void(*)(float* q, float* k,
                        size_t head_dim, size_t n_tokens,
                        const float* cos_table, const float* sin_table);


using packqkv_fn = void(*)(const float* x, float* x_packed, size_t in_dim, size_t K_block);

struct KernelVTable {
  rmsnorm_fn  rmsnorm;
  rope_fn     rope_apply;
  packqkv_fn  pack_qkv;
};

const KernelVTable& get_kernels();

bool has_avx2();
bool has_neon();

void rmsnorm_scalar(const float* x, float* y, size_t dim, const NormParams& p);
void rmsnorm_scalar_novec(const float* x, float* y, size_t dim, const NormParams& p);
void rope_scalar(float* q, float* k, size_t head_dim, size_t n_tokens,
                 const float* cos_tab, const float* sin_tab);
void packqkv_scalar(const float* x, float* x_packed, size_t in_dim, size_t K_block);
} // namespace simd