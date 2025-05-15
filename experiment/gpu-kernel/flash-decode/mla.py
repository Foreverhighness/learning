#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl

@triton.jit
def mla_decode_split(
    Q_absorbed_nope: tl.tensor, # [b, n, dim_latent]
    C_KV: tl.tensor, # [b, s, dim_latent]
    O: tl.tensor, # [b, n, s, d]
    MidO: tl.tensor, # [b, n, n_split, dim_latent]
    L: tl.tensor, # [b, n, n_split]
    stride_qz: int, stride_qh: int, stride_qm: int, stride_qk: int,
    stride_c_kv_b: int, stride_kh: int, stride_c_kv_s: int, stride_c_kv_d: int,
    stride_vz: int, stride_vh: int, stride_vk: int, stride_vn: int,
    stride_midOb: int, stride_midOn: int, stride_midOs: int, stride_midOd: int,
    stride_Lb: int, stride_Ln: int, stride_Ls: int,
    seq_len: int,
    BLOCK_col: tl.constexpr,
    BLOCK_SIZE_head: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    DIM_LATENT: tl.constexpr,
    DIM_HEAD: tl.constexpr,
):
    # 程序ID映射
    id_split = tl.program_id(0)
    id_head_block = tl.program_id(1)
    id_batch = tl.program_id(2)

    # 创建偏移量
    offs_n = tl.arange(0, BLOCK_col)   # KV seq_len
    offs_h = tl.arange(0, BLOCK_SIZE_head)  # KV num_heads
    offs_dl = tl.arange(0, DIM_LATENT)

    offs_cur_head = id_head_block * BLOCK_SIZE_head + offs_h

    # Load Q_absorbed ∈ B_head x dim_latent
    # Q_absorbed = Q [n, d] @ (W^UK [dim_latent, d])^T
    q_absorbed_ptrs = Q_absorbed_nope + (offs_cur_head[:, None] * stride_qh + offs_dl[None, :] * stride_qk) + \
        id_head_block * BLOCK_SIZE_head + id_batch * stride_qz
    q_absorbed = tl.load(q_absorbed_ptrs)

    # 初始化 Shared Memory 上的变量
    max = tl.full([BLOCK_SIZE_head], float('-inf'), dtype=tl.float32)
    l_sum = tl.zeros([BLOCK_SIZE_head], dtype=tl.float32)
    O_i = tl.zeros([BLOCK_SIZE_head, DIM_LATENT], dtype=tl.float32)

    scale: tl.constexpr = DIM_HEAD ** -0.5
    q_absorbed *= scale # scale ops: B_head * dim_latent

    # 主循环处理K/V块
    split_size = tl.cdiv(seq_len, NUM_SPLITS)
    split_start = id_split * split_size
    split_end = tl.minimum(split_start + split_size, seq_len)
    if split_start <= split_end:
        for start_col in range(split_start, split_end, BLOCK_col):
            # 计算当前块的边界
            offs_col = start_col + offs_n
            mask_col = offs_col[:, None] < split_end

            # 加载 Compressed KV 块 C ∈ B_col x dim_latent
            # C = tokens [s, emb] @ W^DKV [emb, dim_latent]
            # KV ∈ n x B_col x d
            c_kv_ptrs = C_KV + (offs_col[:, None] * stride_c_kv_s + offs_dl[None, :] * stride_c_kv_d) + \
                id_batch * stride_c_kv_b
            c_kv_mask = mask_col
            c_kv = tl.load(c_kv_ptrs, mask=c_kv_mask, other=0.0)

            # 计算 S_ij = Q_absorbed @ C_j^T ∈ n x B_col
            score_ij = tl.dot(q_absorbed, tl.trans(c_kv).to(q_absorbed.dtype))

            # 在线softmax更新 max ∈ B_head, s_ij ∈ n x B_col
            max_new = tl.maximum(max, tl.max(score_ij, axis=1))
            # p_j ∈ n x B_col
            p_j = tl.exp(score_ij - max_new[:, None])

            # 更新累加器
            alpha = tl.exp(max - max_new)

            O_i = O_i * alpha[:, None]
            O_i = O_i + tl.dot(p_j.to(c_kv.dtype), c_kv)

            # 更新运行统计量
            l_sum = l_sum * alpha + tl.sum(p_j, axis=1)
            max = max_new

        # epilogue
        # 最终归一化
        O_i = O_i / l_sum[:, None]

        # 写回输出
        o_ptrs = MidO + (offs_cur_head[:, None] * stride_midOn + offs_dl[None, :] * stride_midOd) + \
            id_batch * stride_midOb + id_split * stride_midOs
        tl.store(o_ptrs, O_i)

        # 计算 logsumexp
        l_ptrs = L + (offs_cur_head * stride_Ln) + \
            id_batch * stride_Lb + id_split * stride_Ls
        tl.store(l_ptrs, max + tl.log(l_sum))


@triton.jit
def mla_decode_combine(
    MidO: tl.tensor, # [b, n, n_split, dim_latent]
    Logsumexp: tl.tensor, # [b, n, n_split]
    Out: tl.tensor, # [b, n, dim_latent]
    stride_midOb: int, stride_midOn: int, stride_midOs: int, stride_midOd: int,
    stride_Lb: int, stride_Ln: int, stride_Ls: int,
    stride_Ob: int, stride_On: int, stride_Od: int,
    seq_len: int,
    NUM_SPLITS: tl.constexpr,
    DIM_LATENT: tl.constexpr,
):
    id_batch = tl.program_id(0)
    id_head_block = tl.program_id(1)

    split_size = tl.cdiv(seq_len, NUM_SPLITS)

    offs_dl = tl.arange(0, DIM_LATENT)

    max = -float('inf')
    l_sum = 0.0
    O = tl.zeros([DIM_LATENT], dtype=tl.float32)

    # Reduce Split
    for id_split in range(0, NUM_SPLITS):
        # 计算当前 split 的边界
        start_split = id_split * split_size
        end_split = tl.minimum(start_split + split_size, seq_len)

        if start_split <= end_split:
            # load MidO ∈ dim_latent
            midO_ptrs = MidO + (offs_dl * stride_midOd) + \
                id_batch * stride_midOb + id_head_block * stride_midOn + id_split * stride_midOs
            midO_i = tl.load(midO_ptrs)

            # load logsumexp scalar
            logsumexp_ptr = Logsumexp + \
                id_batch * stride_Lb + id_head_block * stride_Ln + id_split * stride_Ls
            logsumexp_i = tl.load(logsumexp_ptr)

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
    o_ptrs = Out + (offs_dl * stride_Od) + \
        id_batch * stride_Ob + id_head_block * stride_On
    tl.store(o_ptrs, O.to(Out.dtype.element_ty))


def mla_decode(Q: torch.Tensor, C_KV: torch.Tensor, dim_head: tl.constexpr, num_heads: tl.constexpr) -> torch.Tensor:
    assert Q.dtype == C_KV.dtype == torch.float16

    batch_size, seq_len, dim_latent = C_KV.shape

    B_col = 16
    BLOCK_SIZE_HEAD = 32
    NUM_SPLITS = 8

    NUM_HEAD_BLOCKS = triton.cdiv(num_heads, BLOCK_SIZE_HEAD)

    Out = torch.empty_like(Q)
    # print("out", out.shape)
    MidO = torch.zeros((batch_size, num_heads, NUM_SPLITS, dim_latent), device=Q.device, dtype=torch.float32)
    Logsumexp = torch.zeros((batch_size, num_heads, NUM_SPLITS), device=Q.device, dtype=torch.float32)

    grid = (NUM_SPLITS, NUM_HEAD_BLOCKS, batch_size)
    stride_q = (Q.stride(0), Q.stride(1), 1, Q.stride(-1))
    stride_c_kv = (C_KV.stride(0), 1, C_KV.stride(-2), C_KV.stride(-1))
    stride_midO = (MidO.stride(0), MidO.stride(1), MidO.stride(2), MidO.stride(3))
    stride_L = (Logsumexp.stride(0), Logsumexp.stride(1), Logsumexp.stride(2))
    stride_O = (Out.stride(0), Out.stride(1), Out.stride(2))
    print(stride_q, stride_c_kv, stride_midO, stride_L, stride_O)

    test_out = torch.zeros((batch_size, num_heads, seq_len), device=Q.device, dtype=torch.float32)
    mla_decode_split[grid](
        Q, C_KV, test_out, MidO, Logsumexp,
        *stride_q,
        *stride_c_kv,
        *stride_midO,
        *stride_midO,
        *stride_L,
        seq_len=seq_len,
        BLOCK_col=B_col,
        BLOCK_SIZE_head=BLOCK_SIZE_HEAD,
        NUM_SPLITS=NUM_SPLITS,
        DIM_LATENT=dim_latent,
        DIM_HEAD=dim_head,
        num_warps=4,
        num_stages=1,
    )
    # print("MidO:", MidO)
    # print("L:", Logsumexp)
    mla_decode_combine[(batch_size, num_heads, 1)](
        MidO, Logsumexp, Out,
        *stride_midO,
        *stride_L,
        *stride_O,
        seq_len=seq_len,
        NUM_SPLITS=NUM_SPLITS,
        DIM_LATENT=dim_latent,
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

    embedding_size = 32

    dim_latent = 32

    # 创建测试数据
    batch_size = 2
    num_heads = 32
    seq_len = 512
    dim_head = 64

    # query token
    # [b, s, e]
    tokens = torch.randn((batch_size, seq_len, embedding_size), device='cuda', dtype=torch.float16)
    x = tokens[:, -1, :]

    # Q 投影矩阵
    W_Q_down = torch.randn((batch_size, embedding_size, dim_latent), device='cuda', dtype=torch.float16)
    W_Q_up = torch.randn((batch_size, num_heads, dim_latent, dim_head), device='cuda', dtype=torch.float16)

    # [b, n, e, d]
    W_Q = torch.einsum("bel,bnld->bned", W_Q_down, W_Q_up)

    # [b, n, e, l]
    W_KV_down = torch.randn((batch_size, embedding_size, dim_latent), device='cuda', dtype=torch.float16)

    # K/V 投影矩阵
    # [b, n, l, d]
    W_K_up = torch.randn((batch_size, num_heads, dim_latent, dim_head), device='cuda', dtype=torch.float16)
    W_V_up = torch.randn((batch_size, num_heads, dim_latent, dim_head), device='cuda', dtype=torch.float16)

    W_K = torch.einsum("bel,bnld->bned", W_KV_down, W_K_up)
    W_V = torch.einsum("bel,bnld->bned", W_KV_down, W_V_up)

    q = torch.einsum("be,bned->bnd", x, W_Q)
    k = torch.einsum("bse,bned->bnsd", tokens, W_K)

    q_absorbed = torch.einsum("bnd,bnld->bnl", q, W_K_up)
    compressed_kv = torch.einsum("bse,bel->bsl", tokens, W_KV_down)

    # 计算参考结果
    scale = dim_head ** -0.5
    qk = torch.einsum("bnd,bnsd->bns", q, k)
    qk *= scale
    # print(f"qk : {qk}")
    p = torch.softmax(qk, dim=-1)
    ref = p @ compressed_kv

    start = time.perf_counter()

    # 计算 Triton 结果
    tri_out = mla_decode(q_absorbed, compressed_kv, dim_head, num_heads)

    elapsed = time.perf_counter() - start

    # 比较结果
    print("Max absolute error:", torch.max(torch.abs(ref - tri_out)).item())
    print("Mean absolute error:", torch.mean(torch.abs(ref - tri_out)).item())
    assert torch.allclose(ref, tri_out, atol=1e-1, rtol=0), f"Output mismatch! {ref} vs {tri_out}"
    print("Test passed!")
    print(f"used {elapsed:.4f}s")

