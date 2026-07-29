# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm_ascend.dsa_offload.dram_store import (
    DSAHotDRAMStore,
    calculate_dram_usable_blocks,
)


def _cpu_arena(
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(
        (capacity, *block_shape),
        dtype=dtype,
        device=device,
    )


def _make_store() -> DSAHotDRAMStore:
    return DSAHotDRAMStore(
        usable_blocks=8,
        storage_rows=3,
        max_logical_blocks=6,
        device=torch.device("cpu"),
        arena_factory=_cpu_arena,
    )


def test_dram_capacity_uses_fractional_multiplier_with_ceil() -> None:
    assert calculate_dram_usable_blocks(7, 1.5) == 11


def test_layer_arenas_follow_resident_cache_block_shapes() -> None:
    store = _make_store()
    resident_nope = torch.empty(4, 128, 1, 512)
    resident_rope = torch.empty(4, 128, 1, 64)

    store.add_layer(
        layer_id=0,
        resident_nope_cache=resident_nope,
        resident_rope_cache=resident_rope,
    )
    arenas = store.get_layer_arenas(0)

    assert arenas.nope.shape == (9, 128, 1, 512)
    assert arenas.rope.shape == (9, 128, 1, 64)
    assert torch.count_nonzero(arenas.nope[0]).item() == 0
    assert torch.count_nonzero(arenas.rope[0]).item() == 0


def test_reservation_is_idempotent_and_release_reclaims_whole_row() -> None:
    store = _make_store()
    pool_rows = np.array([0, 0, 1], dtype=np.intp)
    logical_blocks = np.array([0, 2, 0], dtype=np.intp)

    first = store.reserve_blocks(
        pool_indices=pool_rows,
        logical_block_indices=logical_blocks,
    )
    assert first.new_mask.tolist() == [True, True, True]
    assert np.all(first.physical_block_ids > 0)
    assert store.num_free_blocks == 5

    second = store.reserve_blocks(
        pool_indices=pool_rows,
        logical_block_indices=logical_blocks,
    )
    assert second.new_mask.tolist() == [False, False, False]
    assert second.physical_block_ids.tolist() == (
        first.physical_block_ids.tolist()
    )

    active = np.empty((2, 6), dtype=np.int32)
    store.gather_rows(
        pool_indices=np.array([1, 0], dtype=np.intp),
        output=active,
    )
    assert active[0, 0] == first.physical_block_ids[2]
    assert active[1, 0] == first.physical_block_ids[0]
    assert active[1, 2] == first.physical_block_ids[1]

    store.release_pool_index(0)
    assert store.num_free_blocks == 7
    assert np.count_nonzero(store.logical_block_table[0]) == 0
    assert store.logical_block_table[1, 0] != 0
