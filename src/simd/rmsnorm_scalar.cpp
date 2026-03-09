#include "simd/kernels.h"
#include <cmath>

namespace simd {

void rmsnorm_scalar(const float* x, float* y, size_t dim, const NormParams& p) {
  double sum = 0.0;
  for (size_t i = 0; i < dim; ++i) sum += double(x[i]) * double(x[i]);
  float rms = 1.0f / std::sqrt(float(sum / double(dim)) + p.eps);
  if (p.gamma) {
    for (size_t i = 0; i < dim; ++i) y[i] = x[i] * rms * p.gamma[i];
  } else {
    for (size_t i = 0; i < dim; ++i) y[i] = x[i] * rms;
  }
}

} // namespace simd