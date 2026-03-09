#include "simd/kernels.h"
#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace simd {


static inline float hsum_f32(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t low  = vget_low_f32(v);
    float32x2_t high = vget_high_f32(v);
    float32x2_t sum2 = vadd_f32(low, high);
    float32x2_t shuf = vrev64_f32(sum2);
    return vget_lane_f32(vpadd_f32(sum2, shuf), 0);
#endif
}

void rmsnorm_neon(const float* x, float* y, size_t dim, const NormParams& p) {
    const size_t V = 4; // NEON 128-bit = 4 x float
    size_t i = 0;

    float32x4_t vsum = vdupq_n_f32(0.f);
    for (; i + V <= dim; i += V) {
        float32x4_t vx = vld1q_f32(x + i);
        vsum = vmlaq_f32(vsum, vx, vx); // vsum += vx*vx
    }
    float sum = hsum_f32(vsum);
    for (; i < dim; ++i) sum += x[i] * x[i];

    float rms = 1.0f / std::sqrt(sum / float(dim) + p.eps);
    float32x4_t vrms = vdupq_n_f32(rms);

    size_t j = 0;
    if (p.gamma) {
        for (; j + V <= dim; j += V) {
            float32x4_t vx = vld1q_f32(x + j);
            float32x4_t vg = vld1q_f32(p.gamma + j);
            float32x4_t vout = vmulq_f32(vmulq_f32(vx, vrms), vg);
            vst1q_f32(y + j, vout);
        }
        for (; j < dim; ++j) y[j] = x[j] * rms * p.gamma[j];
    } else {
        for (; j + V <= dim; j += V) {
            float32x4_t vx = vld1q_f32(x + j);
            float32x4_t vout = vmulq_f32(vx, vrms);
            vst1q_f32(y + j, vout);
        }
        for (; j < dim; ++j) y[j] = x[j] * rms;
    }
}

} // namespace simd