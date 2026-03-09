// src/simd/rope_neon.cpp
#include <cmath>
#include "simd/kernels.h"

#if defined(__aarch64__)
  #include <arm_neon.h>
#endif

namespace simd {

// 说明：本内核按照vtable 的签名：
//   void (*rope_apply)(float* q, float* k, size_t rows, size_t dim,
//                      const float* cos, const float* sin);
// 约定：每一“行”是一个长度为 dim 的向量；cos/sin 也按行对齐（同一行同一个 s 的 cos/sin）。
// 这里在每一行上做 in-place 旋转：
//   [x_even', x_odd'] = [x_even * c - x_odd * s,  x_even * s + x_odd * c]

static inline void rope_row_scalar(float* __restrict__ x,
                                   const float* __restrict__ c,
                                   const float* __restrict__ s,
                                   size_t dim) {
  // 标量兜底：偶/奇位复数旋转
  for (size_t i = 0; i + 1 < dim; i += 2) {
    float xe = x[i + 0];
    float xo = x[i + 1];
    float cc = c[i + 0];  // cos 对应偶位
    float ss = s[i + 0];  // sin 对应偶位（通常 cos/sin 在偶/奇位是同一个值）
    float xe2 = xe * cc - xo * ss;
    float xo2 = xe * ss + xo * cc;
    x[i + 0] = xe2;
    x[i + 1] = xo2;
  }
}

#if defined(__aarch64__)
static inline void rope_row_neon_f32(float* __restrict__ x,
                                     const float* __restrict__ c,
                                     const float* __restrict__ s,
                                     size_t dim) {
  // NEON：一次处理 8 个元素（4 个偶 + 4 个奇），利用 vld2q/vst2q 做偶/奇拆/并
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    // x:  a0,b0,a1,b1,a2,b2,a3,b3 -> even=[a0..a3], odd=[b0..b3]
    float32x4x2_t xv = vld2q_f32(x + i);
    float32x4_t xe = xv.val[0];
    float32x4_t xo = xv.val[1];

    float32x4x2_t cv = vld2q_f32(c + i);
    float32x4x2_t sv = vld2q_f32(s + i);
    // 对于大多数实现，cos/sin 在偶/奇位取值相同，这里按内存布局也做偶/奇流
    float32x4_t ce = cv.val[0];
    float32x4_t se = sv.val[0];

    // 复数旋转
    // e' = e*c - o*s
    // o' = e*s + o*c
    float32x4_t xe2 = vmlsq_f32( vmulq_f32(xe, ce), xo, se );     // xe*ce - xo*se
    float32x4_t xo2 = vmlaq_f32( vmulq_f32(xe, se), xo, ce );     // xe*se + xo*ce

    float32x4x2_t out;
    out.val[0] = xe2;
    out.val[1] = xo2;
    vst2q_f32(x + i, out);
  }
  // 尾部
  for (; i + 1 < dim; i += 2) {
    float xe = x[i + 0];
    float xo = x[i + 1];
    float cc = c[i + 0];
    float ss = s[i + 0];
    float xe2 = xe * cc - xo * ss;
    float xo2 = xe * ss + xo * cc;
    x[i + 0] = xe2;
    x[i + 1] = xo2;
  }
}
#endif

void rope_neon(float* __restrict__ q,
               float* __restrict__ k,
               size_t rows,
               size_t dim,
               const float* __restrict__ cos_,
               const float* __restrict__ sin_) {
  // 逐行处理；每一行长度 = dim。cos/sin 也按行对齐（同一行使用同一段 cos/sin）
  for (size_t r = 0; r < rows; ++r) {
    float* qr = q + r * dim;
    float* kr = k + r * dim;
    const float* cr = cos_ + r * dim;
    const float* sr = sin_ + r * dim;

#if defined(__aarch64__)
    rope_row_neon_f32(qr, cr, sr, dim);
    rope_row_neon_f32(kr, cr, sr, dim);
#else
    // 非 NEON 平台做标量兜底（理论上 ARM64 才会编这个文件，这里只是更健壮）
    rope_row_scalar(qr, cr, sr, dim);
    rope_row_scalar(kr, cr, sr, dim);
#endif
  }
}

} // namespace simd