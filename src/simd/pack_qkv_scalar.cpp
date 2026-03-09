#include "simd/kernels.h"
#include <cstring>
#include <algorithm>

namespace simd {

//  [0..K_block), [K_block..2K_block), ...
void packqkv_scalar(const float* x, float* x_packed, size_t in_dim, size_t K_block) {
  size_t off = 0;
  for (size_t k = 0; k < in_dim; k += K_block) {
    size_t len = std::min(K_block, in_dim - k);
    std::memcpy(x_packed + off, x + k, sizeof(float) * len);
    off += len;
  }
}

} // namespace simd