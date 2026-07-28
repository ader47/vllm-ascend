# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 两个 KV-cache plane 的类型化 manager 定义。

两个 single-type manager 复用 vLLM 的 full-attention 块生命周期，但由
DSA coordinator 注入彼此独立的 BlockPool。顶层仍使用 vLLM 原生
``KVCacheManager``；Ascend patch 仅在其 coordinator
为 DSA 双平面实现时，把 ``allocate_slots`` 中的标量容量判断替换为
component-wise 判断和生命周期分配。这样 scheduler 无论在 patch 前后
导入 manager 类，都不会产生一套平行的 manager 类型。

请求阶段由 ``request_cache_layout`` 以 plan/commit 协议统一规划；本文件只把
规划结果翻译成两个 manager 的实际 block 分配、ENTER 收缩和 steady grow。
任何容量不足都必须发生在物理表修改前，不能让失败重试提前推进阶段。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_manager import (
    KVCacheBlocks,
    KVCacheManager,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlan,
    DSARequestCacheStage,
)

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )


class DSAIndexerKVCacheManager(FullAttentionManager):
    """管理完整上下文 Indexer dense plane 的 block table。"""


class DSAResidentMLAKVCacheManager(FullAttentionManager):
    """管理 dense-prefill / sparse-decode 共用的 MLA resident plane。"""


def _require_initial_runtime_contract(
    *,
    manager: KVCacheManager,
    new_computed_blocks: KVCacheBlocks | None,
    num_new_computed_tokens: int,
    num_lookahead_tokens: int,
    num_external_computed_tokens: int,
    delay_cache_blocks: bool,
    num_encoder_tokens: int,
    reserved_blocks: int,
) -> tuple[Sequence[KVCacheBlock], ...]:
    """把首版支持矩阵在分配边界再次收紧，避免静默走错布局。"""

    unsupported = {
        "num_new_computed_tokens": num_new_computed_tokens,
        "num_lookahead_tokens": num_lookahead_tokens,
        "num_external_computed_tokens": num_external_computed_tokens,
        "delay_cache_blocks": delay_cache_blocks,
        "num_encoder_tokens": num_encoder_tokens,
        "reserved_blocks": reserved_blocks,
    }
    active = {name: value for name, value in unsupported.items() if bool(value)}
    if active:
        raise RuntimeError(f"DSA request cache layout received unsupported allocation features: {active}")

    if new_computed_blocks is None:
        blocks = manager.empty_kv_cache_blocks.blocks
    else:
        blocks = new_computed_blocks.blocks
    if any(blocks):
        raise RuntimeError("DSA request cache layout does not yet support prefix-cache hits")
    return blocks


def _make_group_blocks(
    coordinator: DSAKVCacheCoordinator,
    *,
    indexer_blocks: list[KVCacheBlock],
    resident_blocks: list[KVCacheBlock],
) -> tuple[list[KVCacheBlock], ...]:
    blocks: list[list[KVCacheBlock]] = [[] for _ in range(coordinator.num_single_type_manager)]
    blocks[coordinator.group_ids.indexer] = indexer_blocks
    blocks[coordinator.group_ids.resident_mla] = resident_blocks
    return tuple(blocks)


def _allocate_dense_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    empty_blocks: tuple[Sequence[KVCacheBlock], ...],
) -> KVCacheBlocks | None:
    requirements = coordinator.get_num_blocks_to_allocate_by_group(
        request_id=request.request_id,
        num_tokens=plan.indexer_tokens_need_slot,
        new_computed_blocks=empty_blocks,
        num_encoder_tokens=0,
        total_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=plan.indexer_tokens_need_slot,
    )
    if not coordinator.can_allocate(requirements):
        return None

    new_blocks = coordinator.allocate_new_blocks(
        request.request_id,
        plan.indexer_tokens_need_slot,
        plan.indexer_tokens_need_slot,
    )
    coordinator.request_cache_layout.commit(plan)
    return manager.create_kv_cache_blocks(new_blocks)


def _release_resident_blocks_for_enter(
    coordinator: DSAKVCacheCoordinator,
    request_id: str,
    *,
    preserve_tail_block: bool,
) -> KVCacheBlock | None:
    resident_manager = coordinator.resident_manager
    req_blocks = resident_manager.req_to_blocks.get(request_id, [])
    preserved_tail = req_blocks[-1] if preserve_tail_block and req_blocks else None
    releasable = req_blocks[:-1] if preserved_tail is not None else req_blocks
    resident_manager.req_to_blocks[request_id] = []
    if releasable:
        resident_manager.block_pool.free_blocks(reversed(releasable))
    resident_manager.num_cached_block.pop(request_id, None)
    return preserved_tail


def _allocate_sparse_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
) -> KVCacheBlocks | None:
    indexer_manager = coordinator.indexer_manager
    resident_manager = coordinator.resident_manager
    request_id = request.request_id
    empty: tuple[KVCacheBlock, ...] = ()

    indexer_blocks_needed = indexer_manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=plan.indexer_tokens_need_slot,
        new_computed_blocks=empty,
        total_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=plan.indexer_tokens_need_slot,
    )
    if indexer_blocks_needed > indexer_manager.block_pool.get_num_free_blocks():
        return None

    resident_new_blocks_needed = 0
    preserve_tail = False
    releasable_resident_blocks = 0
    if plan.replace_resident_blocks:
        existing_resident_blocks = resident_manager.req_to_blocks.get(
            request_id,
            [],
        )
        preserve_tail = plan.preserve_resident_tail_block and bool(existing_resident_blocks)
        releasable_resident_blocks = len(existing_resident_blocks) - int(preserve_tail)
        target_resident_blocks = (
            plan.resident_valid_tokens + resident_manager.block_size - 1
        ) // resident_manager.block_size
        resident_new_blocks_needed = target_resident_blocks - int(preserve_tail)
        resident_available_after_release = (
            resident_manager.block_pool.get_num_free_blocks() + releasable_resident_blocks
        )
        if resident_new_blocks_needed > resident_available_after_release:
            return None
    else:
        expected_resident_blocks = (
            plan.resident_valid_tokens + resident_manager.block_size - 1
        ) // resident_manager.block_size
        actual_resident_blocks = len(resident_manager.req_to_blocks.get(request_id, ()))
        if actual_resident_blocks != expected_resident_blocks:
            raise RuntimeError(
                "DSA steady sparse resident block table changed "
                f"unexpectedly: req_id={request_id}, "
                f"actual={actual_resident_blocks}, "
                f"expected={expected_resident_blocks}"
            )

    new_indexer_blocks = indexer_manager.allocate_new_blocks(
        request_id,
        plan.indexer_tokens_need_slot,
        plan.indexer_tokens_need_slot,
    )
    new_resident_blocks: list[KVCacheBlock] = []
    if plan.replace_resident_blocks:
        preserved_tail = _release_resident_blocks_for_enter(
            coordinator,
            request_id,
            preserve_tail_block=preserve_tail,
        )
        resident_slots_to_allocate = (
            plan.sparse_budget_tokens if preserved_tail is not None else plan.resident_valid_tokens
        )
        new_resident_blocks = resident_manager.allocate_new_blocks(
            request_id,
            resident_slots_to_allocate,
            resident_slots_to_allocate,
        )
        if preserved_tail is not None:
            resident_manager.req_to_blocks[request_id].append(preserved_tail)
        if len(new_resident_blocks) != resident_new_blocks_needed:
            raise RuntimeError(
                "DSA ENTER resident allocation disagreed with its "
                f"capacity precheck: req_id={request_id}, "
                f"allocated={len(new_resident_blocks)}, "
                f"expected={resident_new_blocks_needed}"
            )

    coordinator.request_cache_layout.commit(plan)
    return manager.create_kv_cache_blocks(
        _make_group_blocks(
            coordinator,
            indexer_blocks=new_indexer_blocks,
            resident_blocks=new_resident_blocks,
        )
    )


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
    """为 DSA 双物理池执行请求生命周期感知的分配。

    DENSE/PREFILL 同时扩展两个 plane；ENTER 只执行一次 resident 块表
    收缩并保留不满尾块；SPARSE 仅扩展完整上下文 Indexer，resident 的
    budget+tail 物理表保持不变。所有阶段都先完成双 pool 容量预检，再
    修改 block table，最后 commit 请求账本。
    """

    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )

    coordinator = manager.coordinator
    if not isinstance(coordinator, DSAKVCacheCoordinator):
        raise RuntimeError(f"allocate_dsa_slots requires DSAKVCacheCoordinator, got {type(coordinator).__name__}")

    if num_new_tokens <= 0:
        raise ValueError("DSA num_new_tokens must be greater than 0")

    new_computed_block_list = _require_initial_runtime_contract(
        manager=manager,
        new_computed_blocks=new_computed_blocks,
        num_new_computed_tokens=num_new_computed_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
        delay_cache_blocks=delay_cache_blocks,
        num_encoder_tokens=num_encoder_tokens,
        reserved_blocks=reserved_blocks,
    )

    if full_sequence_must_fit:
        full_num_tokens = min(request.num_tokens, manager.max_model_len)
        full_requirements = coordinator.get_num_blocks_to_allocate_by_group(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )
        if not coordinator.can_allocate(full_requirements):
            return None

    plan = coordinator.request_cache_layout.plan(
        request,
        num_new_tokens=num_new_tokens,
        max_model_len=manager.max_model_len,
    )
    if plan.stage in (
        DSARequestCacheStage.PREFILL,
        DSARequestCacheStage.DENSE_DECODE,
    ):
        return _allocate_dense_plan(
            manager,
            coordinator,
            request,
            plan,
            new_computed_block_list,
        )
    return _allocate_sparse_plan(
        manager,
        coordinator,
        request,
        plan,
    )
