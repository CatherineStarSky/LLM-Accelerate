#include "simd/kernels.h"
#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace simd {


static inline float hsum256_ps(__m256 v) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);                // or _mm256_storeu_ps
    return tmp[0] + tmp[1] + tmp[2] + tmp[3]
         + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

void rmsnorm_avx2(const float* x, float* y, size_t dim, const NormParams& p) {
  const size_t V = 8;
  size_t i = 0;

  __m256 vsum = _mm256_setzero_ps();
  for (; i + V <= dim; i += V) {
    __m256 vx = _mm256_loadu_ps(x + i);
    vsum = _mm256_fmadd_ps(vx, vx, vsum);
  }
  float sum = hsum256_ps(vsum);
  for (; i < dim; ++i) sum += x[i] * x[i];

  float rms = 1.0f / std::sqrt(sum / float(dim) + p.eps);
  __m256 vrms = _mm256_set1_ps(rms);
  size_t j = 0;
  if (p.gamma) {
    for (; j + V <= dim; j += V) {
      __m256 vx = _mm256_loadu_ps(x + j);
      __m256 vg = _mm256_loadu_ps(p.gamma + j);
      __m256 vout = _mm256_mul_ps(_mm256_mul_ps(vx, vrms), vg);
      _mm256_storeu_ps(y + j, vout);
    }
    for (; j < dim; ++j) y[j] = x[j] * rms * p.gamma[j];
  } else {
    for (; j + V <= dim; j += V) {
      __m256 vx = _mm256_loadu_ps(x + j);
      __m256 vout = _mm256_mul_ps(vx, vrms);
      _mm256_storeu_ps(y + j, vout);
    }
    for (; j < dim; ++j) y[j] = x[j] * rms;
  }
}

} // namespace simd