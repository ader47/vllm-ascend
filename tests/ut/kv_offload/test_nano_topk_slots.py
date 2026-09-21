# SPDX-License-Identifier: Apache-2.0
"""Unit tests for nano top-k slot binding and tail geometry."""

import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.nano_topk_slots import (
    NanoTopkSlotAllocator,
    nano_pool_capacity,
    nano_tail_geometry,
)


def test_nano_pool_capacity_includes_padding_rows():
    assert nano_pool_capacity(8) == 10


def test_nano_tail_geometry_keeps_last_full_block_for_aligned_prefix():
    assert nano_tail_geometry(10240, 128) == (128, 79)
    assert nano_tail_geometry(4096, 128) == (128, 31)
    assert nano_tail_geometry(128, 128) == (128, 0)
    assert nano_tail_geometry(0, 128) == (0, 0)


def test_nano_tail_geometry_keeps_incomplete_last_block():
    assert nano_tail_geometry(10367, 128) == (127, 80)
    assert nano_tail_geometry(129, 128) == (1, 1)


@pytest.mark.parametrize("kv_tokens", [1, 127, 128, 129, 4096, 4097, 4224])
def test_nano_tail_geometry_covers_first_decode_after_last_token_rewind(kv_tokens):
    tail_tokens, tail_block = nano_tail_geometry(kv_tokens, 128)
    previous_len = kv_tokens - 1
    prefix = previous_len // 128 * 128
    assert tail_block * 128 == prefix
    assert tail_tokens >= previous_len - prefix
    assert tail_tokens <= 128


def test_nano_slot_allocator_reuses_and_releases():
    allocator = NanoTopkSlotAllocator(2)
    first = allocator.bind("req-a")
    second = allocator.bind("req-b")
    assert {first, second} == {0, 1}
    assert allocator.bind("req-a") == first
    allocator.release("req-a")
    assert allocator.get("req-a") is None
    reused = allocator.bind("req-c")
    assert reused == first


def test_nano_slot_allocator_exhausts_capacity():
    allocator = NanoTopkSlotAllocator(1)
    allocator.bind("req-a")
    with pytest.raises(RuntimeError, match="exhausted"):
        allocator.bind("req-b")
