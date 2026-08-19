# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 数据面到 Ascend 自定义算子的薄适配层。

本模块只固定 LIDU、KSC、SFA-Offload 和满块复制的 tensor ABI。请求阶段、
resident 行分配、DRAM block 预留和 attention 时序由上层 runtime 负责；
这里不隐式创建输出 tensor，也不执行 D2H 或 list-to-tensor 转换。
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from vllm_ascend.dsa_offload.contracts import (
    DSA_A5_ATTENTION_CAPACITY,
    DSA_A5_PACKED_KV_ROW_BYTES,
    DSA_SFA_COMPUTE_TOPK,
)
from vllm_ascend.utils import load_custom_op_library


class DSALightningIndexerOutputs(NamedTuple):
    """逐层复用的 caller-owned LIDU 输出缓冲。

    A3 BF16 路径把四个 buffer 解释为 topK token、resident slot、miss 数与
    tail；A5 C8 融合路径复用前三个承载 KSC 的 copy src/dst/count，attention
    slots 与 resident 长度另有固定地址输出，第四列在该路径不消费。
    """

    topk_index: torch.Tensor
    topk_slots: torch.Tensor
    miss_count: torch.Tensor
    tail_info: torch.Tensor


class DSAOffloadSelectionOutput(NamedTuple):
    """LIDU/KSC 完成后交给 SFA-Offload 的逐行 resident 视图。"""

    sparse_indices: torch.Tensor
    tail_info: torch.Tensor
    resident_seq_lengths: torch.Tensor | None = None


_REQUIRED_OPS = (
    "npu_lightning_indexer_decode_update_out",
    "npu_kvcache_scatter_copy",
    "npu_sparse_flash_attention_for_offload",
    "kv_cache_full_block_dump",
)

_REQUIRED_A5_C8_OPS = (
    "npu_dsa_a5_li_manage_nomtp_c8_out",
    "npu_dsa_a5_kvcache_scatter_copy_c8_out",
    "kv_cache_full_block_dump_c8",
)

_REQUIRED_A5_NATIVE_OPS = (
    "npu_quant_lightning_indexer",
    "npu_kv_quant_sparse_flash_attention",
)


def require_dsa_offload_ops(*, packed_c8: bool = False) -> None:
    """在 worker 初始化期确认当前 cache 模式所需设备算子均已部署。"""

    if packed_c8 and not load_custom_op_library():
        raise RuntimeError(
            "DSA A5 packed-C8 custom operator library failed to load; "
            "check the vllm_ascend_C installation and custom-op runtime libraries"
        )
    if packed_c8:
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("DSA A5 packed-C8 requires torch_npu native Quant-LI and QSFA operators") from exc
        missing_native = [op_name for op_name in _REQUIRED_A5_NATIVE_OPS if not hasattr(torch_npu, op_name)]
        if missing_native:
            raise RuntimeError(f"DSA A5 native operators are unavailable: {tuple(missing_native)}")
    required_ops = _REQUIRED_A5_C8_OPS if packed_c8 else _REQUIRED_OPS
    missing = [op_name for op_name in required_ops if not hasattr(torch.ops._C_ascend, op_name)]
    if missing:
        raise RuntimeError(f"DSA sparse offload custom operators are not installed: {tuple(missing)}")


def _packed_byte_view(cache: torch.Tensor, *, name: str) -> torch.Tensor:
    if cache.ndim != 4 or tuple(cache.shape[1:]) != (128, 1, DSA_A5_PACKED_KV_ROW_BYTES):
        raise ValueError(f"{name} must be [blocks,128,1,656], got {tuple(cache.shape)}")
    if not cache.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if cache.dtype == torch.int8:
        return cache
    if cache.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{name} must be float8_e4m3fn or int8, got {cache.dtype}")
    return cache.view(torch.int8)


def _normalize_a5_indexer_key_scale(
    key_dequant_scale: torch.Tensor,
) -> torch.Tensor:
    """将 A5 Indexer scale cache 统一为原生 Quant-LI/融合 LIDU 的三维 ABI。"""

    if key_dequant_scale.ndim == 4:
        if tuple(key_dequant_scale.shape[2:]) != (1, 1):
            raise ValueError(f"A5 Indexer scale cache must end in [1,1], got {tuple(key_dequant_scale.shape)}")
        return key_dequant_scale.squeeze(2)
    if key_dequant_scale.ndim == 3 and key_dequant_scale.shape[2] == 1:
        return key_dequant_scale
    raise ValueError(
        f"A5 Indexer scale cache must be [blocks,128,1] or [blocks,128,1,1], got {tuple(key_dequant_scale.shape)}"
    )


def quant_lightning_indexer_topk(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """调用 A5 原生 Quant-LI，只负责精确 score/top-2048。"""

    import torch_npu

    result = torch_npu.npu_quant_lightning_indexer(
        query=query,
        key=key,
        # A5 native Quant-LI accepts the strided suffix view produced by
        # wk_weights_proj. Only the A3 custom LIDU requires a compact copy.
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=_normalize_a5_indexer_key_scale(key_dequant_scale),
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=DSA_SFA_COMPUTE_TOPK,
        sparse_mode=3,
    )
    expected_numel = int(query.shape[0]) * DSA_SFA_COMPUTE_TOPK
    if isinstance(result, torch.Tensor):
        topk = result
    else:
        topk = next(
            (
                tensor
                for tensor in result
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int32 and tensor.numel() == expected_numel
            ),
            None,
        )
    if topk is None or topk.dtype != torch.int32 or topk.numel() != expected_numel:
        raise RuntimeError(
            "A5 Quant-LI returned an invalid top-k tensor: "
            f"expected_numel={expected_numel}, "
            f"actual={None if topk is None else (topk.dtype, topk.numel())}"
        )
    return topk.reshape(
        int(query.shape[0]),
        1,
        DSA_SFA_COMPUTE_TOPK,
    ).contiguous()


def a5_lightning_indexer_decode_update_c8(
    *,
    index_weights: torch.Tensor,
    query: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    index_key_dequant_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_lens: torch.Tensor,
    final_seq_lengths_kv: torch.Tensor,
    row_modes: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    attention_slots: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
) -> None:
    index_key_dequant_scale = _normalize_a5_indexer_key_scale(index_key_dequant_scale)
    torch.ops._C_ascend.npu_dsa_a5_li_manage_nomtp_c8_out(
        index_weights,
        query,
        query_dequant_scale,
        actual_seq_lengths_query,
        index_key_cache,
        index_key_dequant_scale,
        index_block_table,
        candidate_lens,
        final_seq_lengths_kv,
        row_modes,
        req_pool_entries,
        cache_slots,
        attention_slots,
        resident_seq_lengths,
        outputs.topk_index,
        outputs.topk_slots,
        outputs.miss_count,
    )


def a5_kvcache_scatter_copy_c8(
    *,
    resident_packed_cache: torch.Tensor,
    dram_packed_arena: torch.Tensor,
    resident_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        _packed_byte_view(resident_packed_cache, name="resident_packed_cache"),
        _packed_byte_view(dram_packed_arena, name="dram_packed_arena"),
        resident_block_table,
        dram_block_table,
        source_token_ids,
        destination_slots,
        copy_counts,
    )


def sparse_flash_attention_for_offload_c8(
    *,
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """以 KSC 生成的 resident slot plan 调用 A5 原生 QSFA。"""

    if sparse_indices.shape[-1] != DSA_A5_ATTENTION_CAPACITY:
        raise ValueError(
            f"A5 DSA attention slots must have width {DSA_A5_ATTENTION_CAPACITY}, got {tuple(sparse_indices.shape)}"
        )
    query = query.contiguous()
    kwargs = dict(
        query=query,
        key=packed_kv,
        value=packed_kv,
        sparse_indices=sparse_indices,
        scale_value=float(scale_value),
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=resident_seq_lengths,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=64,
    )
    # Match the native A5 SFA path. The _C_ascend adapter has a different
    # return/LSE contract and is used by the A3 implementation instead.
    import torch_npu

    result = torch_npu.npu_kv_quant_sparse_flash_attention(**kwargs)
    return result[0] if isinstance(result, tuple) else result


def dump_full_kv_cache_blocks_c8(
    *,
    resident_packed_cache: torch.Tensor,
    dram_packed_arena: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    if src_block_ids.numel() == 0:
        return
    torch.ops._C_ascend.kv_cache_full_block_dump_c8(
        _packed_byte_view(resident_packed_cache, name="resident_packed_cache"),
        _packed_byte_view(dram_packed_arena, name="dram_packed_arena"),
        src_block_ids,
        dst_block_ids,
    )


def _squeeze_cache_head_dim(
    cache: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if cache.ndim == 4 and cache.shape[2] == 1:
        return cache.squeeze(2)
    if cache.ndim == 3:
        return cache
    raise ValueError(f"{name} must be [blocks, block, 1, dim] or [blocks, block, dim], got {tuple(cache.shape)}")


def _normalize_lidu_weights_layout(weights: torch.Tensor) -> torch.Tensor:
    """将 v0.23 合并投影产生的 weights view 收敛到 LIDU 连续布局。

    v0.23 的 SFA 通过 ``wk_weights_proj`` 一次生成 key 与 weights，
    ``kw[:, head_dim:]`` 在 batch size 大于 1 时是带较大行 stride 的列后缀
    view。原生 lightning-indexer 能接收该 view，但 LIDU AscendC kernel 按
    紧凑 ``[B, N_idx]`` 布局寻址。这里只规范化这一个小输入；cache 和固定
    容量元数据仍必须由各自 owner 提供连续 tensor，避免在逐层热路径隐式
    复制大张量。
    """

    return weights.contiguous()


def lightning_indexer_decode_update(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    row_modes: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
) -> None:
    weights = _normalize_lidu_weights_layout(weights)
    torch.ops._C_ascend.npu_lightning_indexer_decode_update_out(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots,
        row_modes,
        actual_seq_lengths_key,
        block_table,
        outputs.topk_index,
        outputs.topk_slots,
        outputs.miss_count,
        outputs.tail_info,
    )


def kvcache_scatter_copy(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    resident_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    src_token_ids: torch.Tensor,
    dst_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    torch.ops._C_ascend.npu_kvcache_scatter_copy(
        _squeeze_cache_head_dim(
            resident_rope_cache,
            name="resident_rope_cache",
        ),
        _squeeze_cache_head_dim(
            resident_nope_cache,
            name="resident_nope_cache",
        ),
        _squeeze_cache_head_dim(
            dram_rope_arena,
            name="dram_rope_arena",
        ),
        _squeeze_cache_head_dim(
            dram_nope_arena,
            name="dram_nope_arena",
        ),
        resident_block_table,
        dram_block_table,
        src_token_ids,
        dst_slots,
        copy_counts,
    )


def sparse_flash_attention_for_offload(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    sparse_indices: torch.Tensor,
    tail_info: torch.Tensor,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_sparse_flash_attention_for_offload(
        query,
        key,
        key,
        sparse_indices,
        tail_info,
        float(scale_value),
        1,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        "TND",
        "PA_BSND",
        3,
    )


def dump_full_kv_cache_blocks(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    if src_block_ids.numel() == 0:
        return
    torch.ops._C_ascend.kv_cache_full_block_dump(
        _squeeze_cache_head_dim(
            resident_nope_cache,
            name="resident_nope_cache",
        ),
        _squeeze_cache_head_dim(
            resident_rope_cache,
            name="resident_rope_cache",
        ),
        _squeeze_cache_head_dim(
            dram_nope_arena,
            name="dram_nope_arena",
        ),
        _squeeze_cache_head_dim(
            dram_rope_arena,
            name="dram_rope_arena",
        ),
        src_block_ids,
        dst_block_ids,
    )
