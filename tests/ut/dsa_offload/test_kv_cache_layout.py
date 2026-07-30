# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSAKVCacheCoordinator,
)
from vllm_ascend.dsa_offload.kv_cache_manager import allocate_dsa_slots
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlanner,
    DSARequestCacheStage,
)


@dataclass(frozen=True)
class _Block:
    block_id: int


class _Pool:
    def __init__(self, num_blocks: int, *, id_base: int) -> None:
        self._free = num_blocks
        self._next_id = id_base

    def get_num_free_blocks(self) -> int:
        return self._free

    def allocate(self, count: int) -> list[_Block]:
        if count > self._free:
            raise RuntimeError("test pool exhausted")
        blocks = [_Block(self._next_id + offset) for offset in range(count)]
        self._next_id += count
        self._free -= count
        return blocks

    def free_blocks(self, blocks) -> None:
        self._free += len(list(blocks))


class _SingleTypeManager:
    block_size = 128

    def __init__(self, pool: _Pool) -> None:
        self.block_pool = pool
        self.req_to_blocks = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        required = (num_tokens + self.block_size - 1) // self.block_size
        return max(required - len(self.req_to_blocks[request_id]), 0)

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[_Block]:
        del num_tokens_main_model
        count = self.get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            (),
            0,
            num_tokens,
        )
        blocks = self.block_pool.allocate(count)
        self.req_to_blocks[request_id].extend(blocks)
        return blocks

    def free(self, request_id: str) -> None:
        blocks = self.req_to_blocks.pop(request_id, [])
        self.block_pool.free_blocks(reversed(blocks))
        self.num_cached_block.pop(request_id, None)


class _TopManager:
    def __init__(
        self,
        coordinator: DSAKVCacheCoordinator,
        max_model_len: int = 16384,
    ) -> None:
        self.coordinator = coordinator
        self.max_model_len = max_model_len
        self.empty_kv_cache_blocks = SimpleNamespace(blocks=((), ()))

    @staticmethod
    def create_kv_cache_blocks(blocks):
        return blocks


@dataclass
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_tokens: int


def _make_manager(
    *,
    indexer_blocks: int = 128,
    resident_blocks: int = 64,
) -> tuple[_TopManager, DSAKVCacheCoordinator]:
    indexer_pool = _Pool(indexer_blocks, id_base=1000)
    resident_pool = _Pool(resident_blocks, id_base=2000)
    indexer_manager = _SingleTypeManager(indexer_pool)
    resident_manager = _SingleTypeManager(resident_pool)

    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.max_model_len = 16384
    coordinator.num_single_type_manager = 2
    coordinator.group_ids = SimpleNamespace(indexer=0, resident_mla=1)
    coordinator.indexer_manager = indexer_manager
    coordinator.resident_manager = resident_manager
    coordinator.single_type_managers = (
        indexer_manager,
        resident_manager,
    )
    coordinator.physical_block_pools = (
        indexer_pool,
        resident_pool,
    )
    coordinator.request_cache_layout = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=2048,
        prompt_budget_thresholds=(),
        resident_budget_tokens=(2048,),
    )
    return _TopManager(coordinator), coordinator


def test_dense_enter_sparse_keeps_indexer_full_and_resident_fixed() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 3000, 0, 0, 3000)

    allocate_dsa_slots(manager, request, 3000)  # type: ignore[arg-type]
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 24
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 24

    request.num_computed_tokens = 3000
    coordinator.release_prefill_resident_blocks(
        "req",
        preserve_tail_block=True,
    )
    preserved_tail = coordinator.resident_manager.req_to_blocks["req"][0]

    request.num_output_tokens = 1
    request.num_tokens = 3001
    allocate_dsa_slots(manager, request, 1)  # type: ignore[arg-type]
    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 24
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 17
    assert coordinator.resident_manager.req_to_blocks["req"][-1] == preserved_tail

    resident_ids = tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"])
    request.num_computed_tokens = 3072
    request.num_output_tokens = 73
    request.num_tokens = 3073
    allocate_dsa_slots(manager, request, 1)  # type: ignore[arg-type]
    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.SPARSE_DECODE
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 25
    assert tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"]) == resident_ids


def test_chunked_prefill_grows_both_dense_planes_incrementally() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 3000, 0, 0, 3000)

    for computed, chunk_size, expected_blocks in (
        (0, 1024, 8),
        (1024, 1024, 16),
        (2048, 952, 24),
    ):
        request.num_computed_tokens = computed
        allocated = allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            chunk_size,
        )
        assert allocated is not None
        assert (
            len(coordinator.indexer_manager.req_to_blocks["req"])
            == expected_blocks
        )
        assert (
            len(coordinator.resident_manager.req_to_blocks["req"])
            == expected_blocks
        )
        state = coordinator.get_request_cache_state("req")
        assert state is not None
        assert state.stage == DSARequestCacheStage.PREFILL

    request.num_computed_tokens = request.num_prompt_tokens
    assert coordinator.request_cache_layout.should_release_resident_after_prefill(
        request
    )


def test_enter_capacity_failure_keeps_stage_and_resident_table() -> None:
    manager, coordinator = _make_manager(
        indexer_blocks=16,
        resident_blocks=32,
    )
    request = _Request("req", 2048, 0, 0, 2048)
    allocate_dsa_slots(manager, request, 2048)  # type: ignore[arg-type]
    resident_ids = tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"])

    request.num_computed_tokens = 2048
    request.num_output_tokens = 1
    request.num_tokens = 2049
    assert (
        allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            1,
        )
        is None
    )

    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.PREFILL
    assert tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"]) == resident_ids


def test_block_aligned_prefill_enter_allocates_a_new_tail_block() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 2048, 0, 0, 2048)
    allocate_dsa_slots(manager, request, 2048)  # type: ignore[arg-type]

    request.num_computed_tokens = 2048
    assert coordinator.release_prefill_resident_blocks(
        "req",
        preserve_tail_block=False,
    )
    assert not coordinator.resident_manager.req_to_blocks["req"]

    request.num_output_tokens = 1
    request.num_tokens = 2049
    allocated = allocate_dsa_slots(  # type: ignore[arg-type]
        manager,
        request,
        1,
    )

    assert allocated is not None
    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert state.resident_valid_tokens == 2049
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 17


def test_dense_component_failure_does_not_allocate_or_commit() -> None:
    manager, coordinator = _make_manager(
        indexer_blocks=32,
        resident_blocks=1,
    )
    request = _Request("req", 256, 0, 0, 256)

    assert (
        allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            256,
        )
        is None
    )
    assert coordinator.request_cache_layout.get_state("req") is None
    assert not coordinator.indexer_manager.req_to_blocks["req"]
    assert not coordinator.resident_manager.req_to_blocks["req"]


def test_free_clears_both_planes_and_request_cache_state() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 512, 0, 0, 512)
    allocate_dsa_slots(manager, request, 512)  # type: ignore[arg-type]

    coordinator.free("req")

    assert "req" not in coordinator.indexer_manager.req_to_blocks
    assert "req" not in coordinator.resident_manager.req_to_blocks
    assert coordinator.request_cache_layout.get_state("req") is None
