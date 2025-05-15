# flash_attention_v2.py
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _fwd_kernel(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    Out: tl.tensor,
    L: tl.tensor,
    stride_qz: int, stride_qh: int, stride_qm: int, stride_qk: int,
    stride_kz: int, stride_kh: int, stride_kn: int, stride_kk: int,
    stride_vz: int, stride_vh: int, stride_vk: int, stride_vn: int,
    stride_oz: int, stride_oh: int, stride_om: int, stride_on: int,
    seq_len: int,
    BLOCK_row: tl.constexpr,
    BLOCK_col: tl.constexpr,
    d: tl.constexpr,
):
    # 程序ID映射
    id_row = tl.program_id(0)
    id_head = tl.program_id(1)
    id_batch = tl.program_id(2)

    Q_start = id_row * BLOCK_row

    # 创建偏移量
    offs_m = Q_start + tl.arange(0, BLOCK_row)
    offs_n = tl.arange(0, BLOCK_col)
    offs_d = tl.arange(0, d)

    # Load Qi ∈ B_row x d
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk) + \
        id_head * stride_qh + id_batch * stride_qz
    q_mask = (offs_m[:, None]) < seq_len
    q_i = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 初始化 Shared Memory 上的变量
    m_i = tl.full((BLOCK_row,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_row,), dtype=tl.float32)
    O_i = tl.zeros((BLOCK_row, d), dtype=tl.float32)

    scale: tl.constexpr = d ** -0.5

    # 主循环处理K/V块
    for start_col in range(0, seq_len, BLOCK_col):
        # 计算当前块的边界
        # end_col = start_col + BLOCK_col
        # end_col = tl.minimum(end_col, seq_len)

        offs_col = start_col + offs_n
        mask_col = offs_col[:, None] < seq_len

        # 加载K块 K_j^T ∈ d x B_col
        k_ptrs = K + (start_col + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk + \
            id_head * stride_kh + id_batch * stride_kz
        k_mask = mask_col
        k_j = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # 加载V块 V_j ∈ B_col x d
        v_ptrs = V + (start_col + offs_n[:, None]) * stride_vk + offs_d[None, :] * stride_vn + \
            id_head * stride_vh + id_batch * stride_vz
        v_mask = mask_col
        v_j = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # 计算 S_ij = Q_i @ K_j^T ∈ B_row x B_col
        score_ij = tl.dot(q_i, tl.trans(k_j))
        score_ij = score_ij * scale

        # 在线softmax更新 m_i ∈ B_row, s_ij ∈ B_row x B_col
        m_i_new = tl.maximum(m_i, tl.max(score_ij, axis=1))
        # p_ij ∈ B_row x B_col
        p_ij = tl.exp(score_ij - m_i_new[:, None])

        # 更新累加器
        alpha = tl.exp(m_i - m_i_new)

        O_i = O_i * alpha[:, None]
        O_i += tl.dot(p_ij.to(v_j.dtype), v_j)

        # 更新运行统计量
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)
        m_i = m_i_new

    # epilogue
    # 最终归一化
    O_i = O_i / l_i[:, None]

    # backward pass
    # l_ptrs = L + offs_m
    # tl.store(l_ptrs, m_i + tl.log(l_i))

    # 写回输出
    o_ptrs = Out + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_on) + \
        id_head * stride_oh + id_batch * stride_oz
    tl.store(o_ptrs, O_i.to(Out.dtype.element_ty), mask=q_mask)

def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.float16
    assert q.shape == k.shape == v.shape

    batch_size, num_heads, seq_len, dim = q.shape
    out = torch.empty_like(q)
    L = torch.empty((batch_size * num_heads, seq_len), device=q.device, dtype=torch.float32)

    B_row = 128
    B_col = 64

    grid = (triton.cdiv(seq_len, 128), num_heads, batch_size)
    stride_q = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    stride_k = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
    stride_v = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
    stride_out = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))
    print(stride_q, stride_k, stride_v, stride_out)

    _fwd_kernel[grid](
        q, k, v, out, L,
        *stride_q,
        *stride_k,
        *stride_v,
        *stride_out,
        seq_len=seq_len,
        BLOCK_row=B_row,
        BLOCK_col=B_col,
        d=dim,
        num_warps=4,
        num_stages=3,
    )
    return out

# --------------------------
# 测试代码
# --------------------------
if __name__ == "__main__":
    import numpy as np
    import time

    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    num_heads = 8
    seq_len = 512
    dim = 64

    q = torch.randn((batch_size, num_heads, seq_len, dim), device='cuda', dtype=torch.float16)
    k = torch.randn((batch_size, num_heads, seq_len, dim), device='cuda', dtype=torch.float16)
    v = torch.randn((batch_size, num_heads, seq_len, dim), device='cuda', dtype=torch.float16)

    # 计算参考结果
    scale = dim ** -0.5
    qk = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    ref = torch.softmax(qk, dim=-1) @ v

    start = time.perf_counter()

    # 计算 Triton 结果
    tri_out = flash_attention_v2(q, k, v)

    elapsed = time.perf_counter() - start

    # 比较结果
    print("Max absolute error:", torch.max(torch.abs(ref - tri_out)).item())
    print("Mean absolute error:", torch.mean(torch.abs(ref - tri_out)).item())
    assert torch.allclose(ref, tri_out, atol=1e-2, rtol=0)
    print("Test passed!")
    print(f"used {elapsed:.4f}s")

