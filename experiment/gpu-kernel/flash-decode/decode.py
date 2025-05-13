#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

@triton.jit
def decode_map(
    Q: tl.tensor, # [b, n, d]
    K: tl.tensor, # [b, n, s, d]
    V: tl.tensor, # [b, n, s, d]
    MidO: tl.tensor, # [b, n, n_split, d]
    L: tl.tensor,
    stride_qz: int, stride_qh: int, stride_qm: int, stride_qk: int,
    stride_kz: int, stride_kh: int, stride_kn: int, stride_kk: int,
    stride_vz: int, stride_vh: int, stride_vk: int, stride_vn: int,
    stride_Lb: int, stride_Ln: int, stride_Ld: int,
    stride_midOb: int, stride_midOn: int, stride_midOs: int, stride_midOd: int,
    seq_len: int,
    B_col: tl.constexpr,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # 程序ID映射
    split_id = tl.program_id(0)

    # 创建偏移量
    offs_n = tl.arange(0, B_col)
    offs_d = tl.arange(0, d)

    # Load Q ∈ d
    q = tl.load(Q + offs_d)

    # 初始化 Shared Memory 上的变量
    max = -float('inf')
    l_sum = 0.0
    O_i = tl.zeros([d], dtype=tl.float32)

    scale: tl.constexpr = d ** -0.5
    q *= scale # scale ops: d

    # 主循环处理K/V块
    split_size: tl.constexpr = tl.cdiv(seq_len, NUM_SPLITS)
    split_start = split_id * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)
    if split_start <= split_end:
        for n in range(split_start, split_end, B_col):
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
            score_ij = tl.sum(q[None, :] * k_j, axis=1)
            # score_ij = score_ij * scale # scale ops: split_size / B_col * B_col

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
        o_ptrs = MidO + split_id * stride_midOs + offs_d
        tl.store(o_ptrs, O_i)

        # 计算 logsumexp
        l_ptrs = L + split_id * stride_Ld
        tl.store(l_ptrs, max + tl.log(l_sum))


@triton.jit
def decode_reduce(
    MidO: tl.tensor, # [b, n, n_split, d]
    Logsumexp: tl.tensor, # [b, n, n_split]
    Out: tl.tensor, # [b, n, d]
    stride_midOb: int, stride_midOn: int, stride_midOs: int, stride_midOd: int,
    stride_Lb: int, stride_Ln: int, stride_Ld: int,
    stride_Ob: int, stride_On: int, stride_Od: int,
    seq_len: int,
    d: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    offs_d = tl.arange(0, d)
    offs_midO = offs_d

    split_size = tl.cdiv(seq_len, NUM_SPLITS)

    max = -float('inf')
    l_sum = 0.0
    O = tl.zeros([d], dtype=tl.float32)

    # Reduce Split
    for split_id in range(0, NUM_SPLITS):
        # 计算当前 split 的边界
        split_start = split_id * split_size
        split_end = tl.minimum(split_start + split_size, seq_len)

        if split_start <= split_end:
            midO_i = tl.load(MidO + split_id * stride_midOs + offs_midO)
            logsumexp_i = tl.load(Logsumexp + split_id * stride_Ld)

            # 更新 max
            max_new = tl.maximum(logsumexp_i, max)

            # 更新累加器
            alpha = tl.exp(max - max_new)
            O = O * alpha
            exp_logic = tl.exp(logsumexp_i - max_new)
            O = O + exp_logic * midO_i

            # 更新运行统计量
            l_sum = l_sum * alpha + exp_logic
            max = max_new

    # epilogue
    # 最终归一化
    O = O / l_sum

    # 写回输出
    o_ptrs = Out + offs_d
    tl.store(o_ptrs, O.to(Out.dtype.element_ty))


def flash_decode(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.dtype == K.dtype == V.dtype == torch.float16
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[1] == K.shape[1] == V.shape[1]
    assert K.shape[2] == V.shape[2]
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]

    batch_size, num_heads, seq_len, dim = K.shape

    B_col = 32
    NUM_SPLITS = 8

    Out = torch.empty_like(Q)
    # print("out", out.shape)
    MidO = torch.empty((batch_size, num_heads, NUM_SPLITS, dim), device=Q.device, dtype=torch.float32)
    Logsumexp = torch.empty((batch_size, num_heads, NUM_SPLITS), device=Q.device, dtype=torch.float32)

    grid = (NUM_SPLITS, batch_size, num_heads)
    stride_q = (Q.stride(0), Q.stride(1), 1, Q.stride(-1))
    stride_k = (K.stride(0), K.stride(1), K.stride(2), K.stride(3))
    stride_v = (V.stride(0), V.stride(1), V.stride(2), V.stride(3))
    stride_midO = (MidO.stride(0), MidO.stride(1), MidO.stride(2), MidO.stride(3))
    stride_L = (Logsumexp.stride(0), Logsumexp.stride(1), Logsumexp.stride(2))
    stride_O = (Out.stride(0), Out.stride(1), Out.stride(2))
    print(stride_q, stride_k, stride_v, stride_midO, stride_L, stride_O)
    decode_map[grid](
        Q, K, V, MidO, Logsumexp,
        *stride_q,
        *stride_k,
        *stride_v,
        *stride_L,
        *stride_midO,
        seq_len=seq_len,
        B_col=B_col,
        d=dim,
        num_warps=4,
        num_stages=3,
        NUM_SPLITS=NUM_SPLITS,
    )
    # print("MidO:", MidO)
    # print("L:", Logsumexp)
    decode_reduce[(batch_size, num_heads, 1)](
        MidO, Logsumexp, Out,
        *stride_midO,
        *stride_L,
        *stride_O,
        seq_len=seq_len,
        d=dim,
        NUM_SPLITS=NUM_SPLITS,
    )
    # print("O:",  Out)
    return Out

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
    seq_len = 1024
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

