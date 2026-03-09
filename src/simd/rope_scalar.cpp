#include "simd/kernels.h"
#include <cstddef>


namespace simd {

static inline void apply_one_token(float* v, size_t head_dim,
                                   const float* cos_tab, const float* sin_tab) {
  const size_t pairs = head_dim / 2;
  const bool half = true; 
  for (size_t i = 0; i < pairs; ++i) {
    size_t e = 2 * i;
    size_t o = e + 1;
    float c = cos_tab[half ? i : e];
    float s = sin_tab[half ? i : e];
    float x0 = v[e], x1 = v[o];
    v[e] = x0 * c - x1 * s;
    v[o] = x0 * s + x1 * c;
  }
}

void rope_scalar(float* q, float* k, size_t head_dim, size_t n_tokens,
                 const float* cos_tab, const float* sin_tab) {
  for (size_t t = 0; t < n_tokens; ++t) {
    if (q) apply_one_token(q + t * head_dim, head_dim, cos_tab, sin_tab);
    if (k) apply_one_token(k + t * head_dim, head_dim, cos_tab, sin_tab);
  }
}

} // namespace simd