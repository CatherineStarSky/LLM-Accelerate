#include "simd/kernels.h"
#include <cmath>
#if defined(__clang__)
#define NO_VEC _Pragma("clang loop vectorize(disable)") \
               _Pragma("clang loop interleave(disable)")
#else
#define NO_VEC
#endif

namespace simd {
__attribute__((noinline))
void rmsnorm_scalar_novec(const float* x, float* y, size_t dim, const NormParams& p) {
  double sum = 0.0;
  NO_VEC
  for (size_t i = 0; i < dim; ++i) sum += double(x[i]) * double(x[i]);
  float rms = 1.0f / std::sqrt(float(sum / double(dim)) + p.eps);
  if (p.gamma) {
    NO_VEC
    for (size_t i = 0; i < dim; ++i) y[i] = x[i] * rms * p.gamma[i];
  } else {
    NO_VEC
    for (size_t i = 0; i < dim; ++i) y[i] = x[i] * rms;
  }
}
} // namespace simd