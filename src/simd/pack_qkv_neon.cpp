#include "simd/kernels.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace simd {

// NEON 优化的 QKV 打包函数
// 使用 128-bit NEON 寄存器，每次处理 4 个 float
// 注意：vld1q_f32 和 vst1q_f32 在 ARM64 上支持未对齐访问
void packqkv_neon(const float* x, float* x_packed, size_t in_dim, size_t K_block) {
  if (in_dim == 0) return;
  
  size_t off = 0;
  const size_t V = 4; // NEON 128-bit = 4 x float32
  
  for (size_t k = 0; k < in_dim; k += K_block) {
    size_t len = std::min(K_block, in_dim - k);
    if (len == 0) break;
    
    size_t i = 0;
    
    // NEON 向量化处理：每次处理 4 个 float
    // 在 ARM64 上，vld1q_f32 和 vst1q_f32 支持未对齐访问
    for (; i + V <= len; i += V) {
      float32x4_t vx = vld1q_f32(x + k + i);
      vst1q_f32(x_packed + off + i, vx);
    }
    
    // 处理剩余的元素（标量回退）
    for (; i < len; ++i) {
      x_packed[off + i] = x[k + i];
    }
    
    off += len;
  }
}

} // namespace simd

