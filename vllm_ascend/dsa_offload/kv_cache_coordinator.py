# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA Indexer/MLA 双物理池 coordinator。

vLLM v0.23 的多 group coordinator 仍默认所有 manager 共用一个
``BlockPool``，这适用于“不同逻辑 cache 共享同一物理 block id 空间”的
hybrid cache。DSA 的 Indexer dense plane 与 MLA resident plane 容量不同，
block id 也分别索引各自 tensor，因此必须为每个 group 建立独立 BlockPool，
并在 admission 时逐 component 检查。

本模块只改变物理池所有权和容量判断，不复制 Scheduler 主循环，也不处理
dense -> resident 阶段转换。prefix cache、KV connector 和 speculative
decode 已由首版配置合同提前拒绝。
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.dsa_offload.kv_cache import (
    get_dsa_group_num_blocks,
    validate_dsa_kv_cache_config,
)


class DSABlockPoolView:
    """向 vLLM 顶层暴露两个物理 BlockPool 的只读聚合视图。

    single-type manager 不通过该对象分配，而是直接持有各自的真实 pool。
    聚合视图仅服务于 usage、reset、event 等 ``KVCacheManager`` 公共接口；
    admission 必须调用 ``DSAKVCacheCoordinator.can_allocate``，不能使用
    聚合 free-block 数做标量判断。
    """

    def __init__(self, block_pools: Sequence[BlockPool]) -> None:
        if not block_pools:
            raise ValueError("DSA requires at least one physical block pool")
        self.block_pools = tuple(block_pools)
        self.num_gpu_blocks = sum(pool.num_gpu_blocks for pool in self.block_pools)
        self.enable_caching = False

    def get_num_free_blocks(self) -> int:
        return sum(pool.get_num_free_blocks() for pool in self.block_pools)

    def get_usage(self) -> float:
        # 调度压力由最先耗尽的 component 决定，使用最大利用率比按总块数
        # 加权平均更能反映 DSA 的真实可调度容量。
        return max(pool.get_usage() for pool in self.block_pools)

    def reset_prefix_cache(self) -> bool:
        # ``all(generator)`` 会在首个 False 处短路，导致后续物理池没有被
        # reset。即使首版关闭 prefix cache，也必须保持公共接口的双池原子
        # 语义，便于异常恢复和后续扩展。
        results = [pool.reset_prefix_cache() for pool in self.block_pools]
        return all(results)

    def evict_blocks(self, block_ids: set[int]) -> None:
        # 两个物理池的 block id 均从 0 开始。首版禁用 KV connector 和
        # prefix cache；若未来开放按 block-id 驱逐，接口必须增加 group id，
        # 不能把同一组裸 block id 无差别广播给两个 pool。
        if block_ids:
            raise RuntimeError("DSA split KV-cache eviction requires group-qualified block ids")

    def take_events(self) -> list:
        events = []
        for pool in self.block_pools:
            events.extend(pool.take_events())
        return events


class DSAKVCacheCoordinator(KVCacheCoordinator):
    """为每个 DSA cache group 构造独立 BlockPool 和 manager。"""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        if enable_caching:
            raise ValueError("DSA split KV-cache does not yet support prefix caching")
        if use_eagle:
            raise ValueError("DSA split KV-cache does not yet support speculative decode")
        if enable_kv_cache_events:
            raise ValueError(
                "DSA split KV-cache does not yet support KV-cache events: "
                "the independent pools currently reuse block IDs"
            )
        if metrics_collector is not None:
            raise ValueError(
                "DSA split KV-cache does not yet support KV-cache metrics: "
                "the independent pools currently reuse block IDs"
            )
        if scheduler_block_size % hash_block_size != 0 or any(
            scheduler_block_size % group.kv_cache_spec.block_size != 0 for group in kv_cache_config.kv_cache_groups
        ):
            raise ValueError(
                "DSA scheduler block size must be divisible by the hash block size and every KV-cache group block size"
            )

        validate_dsa_kv_cache_config(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = False
        self.scheduler_block_size = scheduler_block_size
        self.retention_interval = None
        self.eagle_group_ids: set[int] = set()

        physical_pools = tuple(
            BlockPool(
                num_gpu_blocks=get_dsa_group_num_blocks(
                    kv_cache_config,
                    group,
                ),
                enable_caching=False,
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
            )
            for group in kv_cache_config.kv_cache_groups
        )
        self.physical_block_pools = physical_pools
        self.block_pool = DSABlockPoolView(physical_pools)

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                block_pool=physical_pools[group_id],
                enable_caching=False,
                kv_cache_group_id=group_id,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=scheduler_block_size,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_blocks_to_allocate_by_group(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> tuple[int, ...]:
        requirements: list[int] = []
        for group_id, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                requirement = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    (),
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                requirement = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[group_id],
                    total_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            requirements.append(requirement)
        return tuple(requirements)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # 保留 vLLM 公共接口的“总待分配块数”语义；真正 admission 使用
        # get_num_blocks_to_allocate_by_group + can_allocate。
        return sum(
            self.get_num_blocks_to_allocate_by_group(
                request_id=request_id,
                num_tokens=num_tokens,
                new_computed_blocks=new_computed_blocks,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        )

    def can_allocate(
        self,
        requirements: Sequence[int],
        *,
        reserved_blocks: int = 0,
    ) -> bool:
        if len(requirements) != len(self.physical_block_pools):
            raise ValueError(
                "DSA component requirement count does not match physical "
                f"pools: requirements={len(requirements)}, "
                f"pools={len(self.physical_block_pools)}"
            )
        if reserved_blocks:
            raise RuntimeError("DSA split KV-cache does not support connector block reservations")
        return all(
            required <= pool.get_num_free_blocks()
            for required, pool in zip(
                requirements,
                self.physical_block_pools,
                strict=True,
            )
        )

    def get_num_common_prefix_blocks(
        self,
        running_request_id: str,
    ) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        del block_hashes, max_cache_hit_length
        return (
            tuple([] for _ in range(self.num_single_type_manager)),
            0,
        )
