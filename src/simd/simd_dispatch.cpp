#include "simd/kernels.h"

#if defined(SIMD_ARM64)
  #include <arm_neon.h>
#endif

#if defined(SIMD_X86)
  #include <immintrin.h>
#endif

namespace simd {


void rmsnorm_scalar(const float*, float*, size_t, const NormParams&);
void rope_scalar(float*, float*, size_t, size_t, const float*, const float*);
void packqkv_scalar(const float*, float*, size_t, size_t);

#if defined(SIMD_X86)
void rmsnorm_avx2  (const float*, float*, size_t, const NormParams&);
void rope_avx2  (float*, float*, size_t, size_t, const float*, const float*);
void packqkv_avx2  (const float*, float*, size_t, size_t);
#endif

#if defined(SIMD_ARM64)
void rmsnorm_neon(const float*, float*, size_t, const NormParams&);
void rope_neon(float*, float*, size_t, size_t, const float*, const float*);
void packqkv_neon(const float*, float*, size_t, size_t);
#endif

static KernelVTable vt;
static bool inited = false;
static bool g_has_avx2 = false;

#if defined(SIMD_X86)
static bool cpu_has_avx2_runtime() {
  #if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
  #else
    return false;
  #endif
}
#endif

const KernelVTable& get_kernels() {
  if (!inited) {
#if defined(SIMD_X86)
    g_has_avx2 = cpu_has_avx2_runtime();
    if (g_has_avx2) {
      vt.rmsnorm    = rmsnorm_avx2;
      vt.rope_apply = rope_avx2;
      vt.pack_qkv   = packqkv_avx2;
    } else
#endif
    {
#if defined(SIMD_ARM64)
      vt.rmsnorm    = rmsnorm_neon;   
      vt.rope_apply = rope_neon;
      vt.pack_qkv   = packqkv_neon;
#else
      vt.rmsnorm    = rmsnorm_scalar;
      vt.rope_apply = rope_scalar;
      vt.pack_qkv   = packqkv_scalar;
#endif
    }
    inited = true;
  }
  return vt;
}

bool has_avx2() {
#if defined(SIMD_X86)
  get_kernels(); 
  return g_has_avx2;
#else
  return false;
#endif
}

bool has_neon() {
#if defined(SIMD_ARM64)
  return true;   
#else
  return false;
#endif
}

} // namespace simd