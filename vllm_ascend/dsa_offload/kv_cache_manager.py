# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 两个 KV-cache plane 的类型化 manager 定义。

两个 single-type manager 复用 vLLM 的 full-attention 块生命周期，但由
DSA coordinator 注入彼此独立的 BlockPool。顶层
顶层仍使用 vLLM 原生 ``KVCacheManager``；Ascend patch 仅在其 coordinator
为 DSA 双平面实现时，把 ``allocate_slots`` 中的标量容量判断替换为
component-wise 判断。这样 scheduler 无论在 patch 前后导入 manager 类，
都不会产生一套平行的 manager 类型。

该文件不负责 dense -> resident 阶段转换。阶段转换会在后续 P3 生命周期
层建立在这两个稳定 manager identity 之上。
"""

from __future__ import annotations

from vllm.v1.core.kv_cache_manager import (
    KVCacheBlocks,
    KVCacheManager,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import Request


class DSAIndexerKVCacheManager(FullAttentionManager):
    """管理完整上下文 Indexer dense plane 的 block table。"""


class DSAResidentMLAKVCacheManager(FullAttentionManager):
    """管理 dense-prefill / sparse-decode 共用的 MLA resident plane。"""


def allocate_dsa_slots(
    manager: KVCacheManager,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
) -> KVCacheBlocks | None:
    """为 DSA 双物理池执行逐 component admission。

    vLLM v0.23 的 ``KVCacheCoordinator`` 已允许多个 cache group，但顶层
    ``KVCacheManager`` 仍把所有 group 的待分配块数求和后，与单个
    ``BlockPool`` 的空闲块数比较。Indexer/MLA 容量不同后，该标量比较会
    产生假阳性，因此这里只逐句复用原流程并替换两处容量判断。

    该函数由平台 patch 条件调用；非 DSA manager 不会进入这里。
    """

    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )

    coordinator = manager.coordinator
    if not isinstance(coordinator, DSAKVCacheCoordinator):
        raise RuntimeError(f"allocate_dsa_slots requires DSAKVCacheCoordinator, got {type(coordinator).__name__}")

    if num_new_tokens == 0 and num_external_computed_tokens == 0:
        raise ValueError("num_new_tokens must be greater than 0 when there are no external computed tokens")

    if new_computed_blocks is not None:
        new_computed_block_list = new_computed_blocks.blocks
    else:
        new_computed_block_list = manager.empty_kv_cache_blocks.blocks

    num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    total_computed_tokens = min(
        num_local_computed_tokens + num_external_computed_tokens,
        manager.max_model_len,
    )

    if full_sequence_must_fit:
        full_num_tokens = min(request.num_tokens, manager.max_model_len)
        full_requirements = coordinator.get_num_blocks_to_allocate_by_group(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )
        if not coordinator.can_allocate(
            full_requirements,
            reserved_blocks=reserved_blocks,
        ):
            return None

    num_tokens_main_model = total_computed_tokens + num_new_tokens
    num_tokens_need_slot = min(
        num_tokens_main_model + num_lookahead_tokens,
        manager.max_model_len,
    )

    coordinator.remove_skipped_blocks(
        request.request_id,
        total_computed_tokens,
    )

    requirements = coordinator.get_num_blocks_to_allocate_by_group(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=(num_local_computed_tokens + num_external_computed_tokens),
        num_tokens_main_model=num_tokens_main_model,
    )
    if not coordinator.can_allocate(
        requirements,
        reserved_blocks=reserved_blocks,
    ):
        return None

    if new_computed_block_list is not manager.empty_kv_cache_blocks.blocks or num_external_computed_tokens > 0:
        coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=new_computed_block_list,
            num_local_computed_tokens=num_local_computed_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
        )

    new_blocks = coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens_need_slot,
        num_tokens_main_model,
        num_encoder_tokens,
    )

    if not manager.enable_caching or delay_cache_blocks:
        return manager.create_kv_cache_blocks(new_blocks)

    num_tokens_to_cache = min(
        total_computed_tokens + num_new_tokens,
        request.num_tokens,
    )
    coordinator.cache_blocks(request, num_tokens_to_cache)
    return manager.create_kv_cache_blocks(new_blocks)
