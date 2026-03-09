#include "simd/kernels.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace simd;
using clockT = std::chrono::high_resolution_clock;


float bench_rmsnorm(const rmsnorm_fn fn,
                    const float* x, float* y,
                    size_t outer, size_t dim,
                    const NormParams& p, int iters) {
  auto t0 = clockT::now();
  for (int it = 0; it < iters; ++it) {
    for (size_t i = 0; i < outer; ++i) {
      fn(x + i*dim, y + i*dim, dim, p);
    }
  }
  auto t1 = clockT::now();
  std::chrono::duration<double, std::milli> ms = t1 - t0;
  return static_cast<float>(ms.count());
}

int main() {

  const size_t dim   = 1024;   
  const size_t outer = 2048;   
  const int    iters = 5;     


  std::mt19937 rng(123);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);

  std::vector<float> x(outer * dim), y(outer * dim), y2(outer * dim), gamma(dim);
  for (auto& v : x) v = ud(rng);
  for (size_t i = 0; i < dim; ++i) gamma[i] = 0.5f + 0.5f * ud(rng);

  NormParams p{1e-5f, gamma.data()};


  bool avx2 = has_avx2(); 
  bool neon =
  #if defined(SIMD_ARM64)
    true;
  #else
    false;
  #endif

  std::cout << "AVX2 enabled? " << (avx2 ? "yes" : "no")
            << " | NEON enabled? " << (neon ? "yes" : "no") << "\n";

  
  const auto& K = get_kernels();
  auto fn_opt    = K.rmsnorm;
  auto fn_scalar = rmsnorm_scalar;

  std::cout << "opt == scalar ? " << (fn_opt == fn_scalar ? "yes" : "no") << "\n";


  (void)bench_rmsnorm(fn_scalar, x.data(), y2.data(), outer, dim, p, 1);
  (void)bench_rmsnorm(fn_opt,    x.data(), y.data(),  outer, dim, p, 1);


  float t_scalar = bench_rmsnorm(fn_scalar, x.data(), y2.data(), outer, dim, p, iters);
  float t_opt    = bench_rmsnorm(fn_opt,    x.data(), y.data(),  outer, dim, p, iters);


  float maxdiff = 0.f;
  for (size_t i = 0; i < outer*dim; ++i)
    maxdiff = std::max(maxdiff, std::abs(y[i] - y2[i]));

  
  std::cout << "RMSNorm(y[0..3]): "
            << y[0] << ", " << y[1] << ", " << y[2] << ", " << y[3] << "\n";

  std::cout << "RMSNorm scalar: " << t_scalar << " ms\n";
  std::cout << "RMSNorm  opt  : " << t_opt    << " ms\n";
  std::cout << "Speedup = " << (t_scalar / t_opt) << "x,  max|diff|=" << maxdiff << "\n";

  extern void rmsnorm_scalar_novec(const float*, float*, size_t, const simd::NormParams&);


  float t_true_scalar =
      bench_rmsnorm(simd::rmsnorm_scalar_novec, x.data(), y2.data(), outer, dim, p, iters);
  std::cout << "RMSNorm true-scalar(no-vec): " << t_true_scalar << " ms\n";
  std::cout << "OK\n";
  return 0;
}