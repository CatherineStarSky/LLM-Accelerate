# import os
# import torch

# ROOT = os.path.dirname(os.path.abspath(__file__))
# lib_path = os.path.join(ROOT, "build", "simd_ops.dylib") 
# torch.ops.load_library(lib_path)
# print("[simd] loaded:", lib_path)

# class SimdRMSNorm(torch.nn.Module):
#     def __init__(self, orig_norm: torch.nn.Module):
#         super().__init__()
#         self.weight = orig_norm.weight
#         eps = getattr(orig_norm, "variance_epsilon", None)
#         if eps is None:
#             eps = getattr(orig_norm, "eps", 1e-6)
#         self.variance_epsilon = float(eps)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         in_dtype = hidden_states.dtype
#         w_dtype = self.weight.dtype

#         hs = hidden_states
#         w = self.weight

#         if hs.dtype != torch.float32:
#             hs = hs.float()
#         if w.dtype != torch.float32:
#             w = w.float()

#         out = torch.ops.simd_opt.rmsnorm(hs, w, self.variance_epsilon)

#         if out.dtype != in_dtype:
#             out = out.to(in_dtype)
#         return out

# def replace_rmsnorm_with_simd(module: torch.nn.Module, verbose: bool = True):
#     for name, child in list(module.named_children()):
#         cls = child.__class__.__name__
#         if cls in ("Qwen2RMSNorm", "QwenRMSNorm", "RMSNorm", "LlamaRMSNorm"):
#             if verbose:
#                 print(f"[simd] patching {name}: {cls} -> SimdRMSNorm")
#             setattr(module, name, SimdRMSNorm(child))
#         else:
#             replace_rmsnorm_with_simd(child, verbose=verbose)


# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, AutoTokenizer

#     model_path = "Qwen/Qwen3-0.6B"  
#     tok = AutoTokenizer.from_pretrained(model_path)
#     device = "cpu"
#     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

#     print("[simd] model loaded")

#     replace_rmsnorm_with_simd(model, verbose=True)

#     print("[simd] all RMSNorm patched.")

#     inputs = tok("Please explain what SIMD is", return_tensors="pt")
#     with torch.no_grad():
#         out = model(**inputs)

#     print("[simd] forward ok, logits shape:", out.logits.shape)




import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(ROOT, "build", "simd_ops.dylib")
torch.ops.load_library(lib_path)
print("[simd] loaded:", lib_path)

############################################
# 2. RMSNorm 
############################################

class SimdRMSNorm(torch.nn.Module):
    def __init__(self, orig_norm: torch.nn.Module):
        super().__init__()
        self.weight = orig_norm.weight
        eps = getattr(orig_norm, "variance_epsilon", None)
        if eps is None:
            eps = getattr(orig_norm, "eps", 1e-6)
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        in_dtype = hidden_states.dtype
        w = self.weight
        x = hidden_states
        if x.dtype != torch.float32:
            x = x.float()
        if w.dtype != torch.float32:
            w = w.float()
        out = torch.ops.simd_opt.rmsnorm(x, w, self.variance_epsilon)
        if out.dtype != in_dtype:
            out = out.to(in_dtype)
        return out

def patch_rmsnorm(module: torch.nn.Module, verbose: bool = True):
    for name, child in list(module.named_children()):
        cls = child.__class__.__name__
        if cls in ("Qwen3RMSNorm", "QwenRMSNorm", "RMSNorm", "LlamaRMSNorm"):
            if verbose:
                print(f"[simd] patching {name}: {cls} -> SimdRMSNorm")
            setattr(module, name, SimdRMSNorm(child))
        else:
            patch_rmsnorm(child, verbose=verbose)

############################################
# 3. RoPE
############################################

import transformers.models.qwen3.modeling_qwen3 as q3_mod

_orig_apply_rope = q3_mod.apply_rotary_pos_emb  # Save it first and restore it if necessary

def apply_rotary_pos_emb_simd(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_dtype = q.dtype
    k_dtype = k.dtype

    B, Hq, S, D = q.shape
    B2, Hk, S2, D2 = k.shape
    assert B == B2 and S == S2 and D == D2, "q/k shape mismatch on non-head dims"

    need_shrink_back = False
    factor = 1
    if Hq != Hk:
        assert Hq % Hk == 0, "cannot broadcast k heads to q heads"
        factor = Hq // Hk
        k = k.repeat_interleave(factor, dim=1)  # [B, Hq, S, D]
        need_shrink_back = True

    q32 = q.contiguous().float()
    k32 = k.contiguous().float()
    cos32 = cos.contiguous().float()
    sin32 = sin.contiguous().float()

    q32_out, k32_out = torch.ops.simd_opt.rope(q32, k32, cos32, sin32)

    q_out = q32_out.to(q_dtype)
    k_out = k32_out.to(k_dtype)

    if need_shrink_back:
        k_out = k_out[:, ::factor, ...] 

    return q_out, k_out

q3_mod.apply_rotary_pos_emb = apply_rotary_pos_emb_simd
print("[simd] qwen3.apply_rotary_pos_emb patched -> simd_opt.rope")

############################################
# 4. attach pack_qkv hook
############################################

def attach_pack_hook_to_qwen3_attn(model: torch.nn.Module, verbose: bool = True):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention):

            def _hook(m, args, kwargs):
                if len(args) > 0:
                    hs = args[0]
                else:
                    hs = kwargs.get("hidden_states", None)

                if hs is None:
                    return 
                dim = hs.shape[-1]
                fake_qkv = torch.zeros((1, 3, dim), dtype=torch.float32, device=hs.device)

                _ = torch.ops.simd_opt.pack_qkv(fake_qkv)

                return
            module.register_forward_pre_hook(_hook, with_kwargs=True)

            if verbose:
                print(f"[simd] attach pack_qkv hook on {name}")

############################################
# 5. Qwen3-0.6B
############################################

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   
        device_map=None             
    ).to("cpu")

    print("[simd] model loaded")

    patch_rmsnorm(model, verbose=True)
    print("[simd] all RMSNorm patched.")

    attach_pack_hook_to_qwen3_attn(model, verbose=True)

    inputs = tok("SIMD on Qwen3 test.", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    print("[simd] forward ok, logits:", out.logits.shape)