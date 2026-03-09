# import os, torch

# lib = os.path.abspath("build/simd_ops.dylib")  # mac
# torch.ops.load_library(lib)

# x = torch.randn(2, 16, 1024)
# gamma = torch.ones(1024)
# y = torch.ops.simd_opt.rmsnorm(x, gamma, 1e-5)
# print(y.shape)

import os
import torch


torch.ops.load_library(os.path.abspath("build/simd_ops.dylib"))

# 1) RMSNorm
x = torch.randn(2, 16, 1024, dtype=torch.float32)
gamma = torch.ones(1024, dtype=torch.float32)
y1 = torch.ops.simd_opt.rmsnorm(x, gamma, 1e-5)
print("rmsnorm ok:", y1.shape)

# 2) RoPE
q = torch.randn(2, 16, 128, dtype=torch.float32)
k = torch.randn(2, 16, 128, dtype=torch.float32)
cos = torch.randn(128, dtype=torch.float32)
sin = torch.randn(128, dtype=torch.float32)
q2, k2 = torch.ops.simd_opt.rope(q, k, cos, sin)
print("rope ok:", q2.shape, k2.shape)

# 3) PackQKV
rows = 4
dim = 64
qkv = torch.randn(rows, 3, dim, dtype=torch.float32)
packed = torch.ops.simd_opt.pack_qkv(qkv)
print("pack_qkv ok:", packed.shape) 