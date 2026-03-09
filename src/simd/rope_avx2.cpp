#include "simd/kernels.h"
#include <immintrin.h>
#include <cstddef>


namespace simd {

static inline void apply_block8(float* base, const float* c, const float* s) {
  // base: e0,o0,e1,o1,e2,o2,e3,o3
  __m256 vx = _mm256_loadu_ps(base); // x0,y0,x1,y1,x2,y2,x3,y3

  float x[8]; _mm256_storeu_ps(x, vx);

  for (int i = 0; i < 4; ++i) {
    float x0 = x[2*i], x1 = x[2*i+1];
    float cc = c[i], ss = s[i];
    x[2*i]   = x0 * cc - x1 * ss;
    x[2*i+1] = x0 * ss + x1 * cc;
  }
  _mm256_storeu_ps(base, _mm256_loadu_ps(x));
}

void rope_avx2(float* q, float* k, size_t head_dim, size_t n_tokens,
               const float* cos_tab, const float* sin_tab) {
  const size_t pairs = head_dim / 2;
  for (size_t t = 0; t < n_tokens; ++t) {
    for (int which = 0; which < 2; ++which) {
      float* v = (which == 0 ? q : k);
      if (!v) continue;
      float* row = v + t * head_dim;

      size_t i = 0;

      for (; i + 4 <= pairs; i += 4) {
        size_t off = 2 * i; 
        apply_block8(row + off, cos_tab + i, sin_tab + i);
      }
      for (; i < pairs; ++i) {
        size_t e = 2 * i, o = e + 1;
        float cc = cos_tab[i], ss = sin_tab[i];
        float x0 = row[e],   x1 = row[o];
        row[e]   = x0 * cc - x1 * ss;
        row[o]   = x0 * ss + x1 * cc;
      }
    }
  }
}

} // namespace simd