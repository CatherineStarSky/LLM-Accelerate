#include "simd/kernels.h"
#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace simd {

void packqkv_avx2(const float* x, float* x_packed, size_t in_dim, size_t K_block) {
  size_t off = 0;
  const size_t V = 8;
  for (size_t k = 0; k < in_dim; k += K_block) {
    size_t len = std::min(K_block, in_dim - k);
    size_t i = 0;
    for (; i + V <= len; i += V) {
      __m256 vx = _mm256_loadu_ps(x + k + i);
      _mm256_storeu_ps(x_packed + off + i, vx);
    }
    for (; i < len; ++i) x_packed[off + i] = x[k + i];
    off += len;
  }
}

} // namespace simd