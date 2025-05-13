# flash_attention_v2.py
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def decode_map(
    Q: tl.tensor, # [b, n, d]
    K: tl.tensor, # [b, n, s, d]
    V: tl.tensor, # [b, n, s, d]
    Out: tl.tensor, # [b, n, d]
    L: tl.tensor,
    stride_qz: int, stride_qh: int, stride_qm: int, stride_qk: int,
    stride_kz: int, stride_kh: int, stride_kn: int, stride_kk: int,
    stride_vz: int, stride_vh: int, stride_vk: int, stride_vn: int,
    stride_oz: int, stride_oh: int, stride_om: int, stride_on: int,
    seq_len: int,
    B_col: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr = 1,
):
    # 程序ID映射
    block_id = tl.program_id(0)

    # 创建偏移量
    offs_n = tl.arange(0, B_col)
    offs_d = tl.arange(0, d)

    # Load Q ∈ d
    qq = tl.load(Q + offs_d)

    # 初始化 Shared Memory 上的变量
    max = -float('inf')
    l_sum = 0.0
    O_i = tl.zeros([d], dtype=tl.float32)

    scale: tl.constexpr = d ** -0.5

    # 主循环处理K/V块
    for n in range(0, seq_len, B_col):
        # 计算当前块的边界
        n_end = n + B_col
        n_end = tl.minimum(n_end, seq_len)

        # 加载K块 K_j^T ∈ d x B_col
        k_ptrs = K + (n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k_mask = (n + offs_n[:, None]) < seq_len
        k_j = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # 加载V块 V_j ∈ B_col x d
        v_ptrs = V + (n + offs_n[:, None]) * stride_vk + offs_d[None, :] * stride_vn
        v_mask = (n + offs_n[:, None]) < seq_len
        v_j = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # 计算 S_ij = Q_i @ K_j^T ∈ B_col
        score_ij = tl.sum(qq[None, :] * k_j, axis=1)
        score_ij = score_ij * scale

        # 在线softmax更新 max, S_ij ∈ B_col
        max_new = tl.maximum(max, tl.max(score_ij, axis=0))
        # p_ij ∈ B_col
        p_j = tl.exp(score_ij - max_new)

        # 更新累加器
        alpha = tl.exp(max - max_new)
        O_i = O_i * alpha
        O_i = O_i + tl.sum(p_j[:, None] * v_j, axis=0)

        # 更新运行统计量
        l_sum = l_sum * alpha + tl.sum(p_j, axis=0)
        max = max_new

    # epilogue
    # 最终归一化
    O_i = O_i / l_sum

    # 写回输出
    o_ptrs = Out + offs_d
    tl.store(o_ptrs, O_i.to(Out.dtype.element_ty))

def flash_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[1] == k.shape[1] == v.shape[1]
    assert k.shape[2] == v.shape[2]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    batch_size, num_heads, seq_len, dim = k.shape

    B_col = 64

    out = torch.empty_like(q)
    # print("out", out.shape)
    L = torch.empty([B_col], device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(1, 128), batch_size * num_heads, 1)
    stride_q = (q.stride(0), q.stride(1), 1, q.stride(-1))
    stride_k = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
    stride_v = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
    stride_out = (out.stride(0), out.stride(1), 1, out.stride(-1))
    print(stride_q, stride_k, stride_v, stride_out)
    decode_map[grid](
        q, k, v, out, L,
        *stride_q,
        *stride_k,
        *stride_v,
        *stride_out,
        seq_len=seq_len,
        B_col=B_col,
        d=dim,
        num_warps=4,
        num_stages=3,
    )
    # print("L:",  L)
    return out

# --------------------------
# 测试代码
# --------------------------
if __name__ == "__main__":
    import numpy as np
    import time

    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 1
    num_heads = 1
    seq_len = 512
    dim = 64

    q = torch.randn((batch_size, num_heads, dim), device='cuda', dtype=torch.float16)
    k = torch.randn((batch_size, num_heads, seq_len, dim), device='cuda', dtype=torch.float16)
    v = torch.randn((batch_size, num_heads, seq_len, dim), device='cuda', dtype=torch.float16)

    # 计算参考结果
    scale = dim ** -0.5
    qk = torch.einsum("bnd,bnsd->bns", q, k)
    qk *= scale
    # print(f"qk : {qk}")
    p = torch.softmax(qk, dim=-1)
    ref = p @ v

    start = time.perf_counter()

    # 计算 Triton 结果
    tri_out = flash_decode(q, k, v)

    elapsed = time.perf_counter() - start

    # 比较结果
    print("Max absolute error:", torch.max(torch.abs(ref - tri_out)).item())
    print("Mean absolute error:", torch.mean(torch.abs(ref - tri_out)).item())
    assert torch.allclose(ref, tri_out, atol=1e-2, rtol=0), f"Output mismatch! {ref} vs {tri_out}"
    print("Test passed!")
    print(f"used {elapsed:.4f}s")

