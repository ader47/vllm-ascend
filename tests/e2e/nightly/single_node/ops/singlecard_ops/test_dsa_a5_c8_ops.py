# SPDX-License-Identifier: Apache-2.0
"""A5 packed-C8 DSA operator integration checks."""

import pytest
import torch
import torch_npu

from vllm_ascend.dsa_offload.contracts import DSA_LIDU_CACHE_ROW_ALIGNMENT
from vllm_ascend.dsa_offload.ops import require_dsa_offload_ops
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
)

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="packed-C8 DSA operators require Ascend A5",
)

_BLOCK_SIZE = 128
_PACKED_ROW_BYTES = 656
_TOPK = 2048
_COPY_CAPACITY = 16384
_ATTENTION_CAPACITY = _TOPK + _BLOCK_SIZE


def _cache_row_width(max_model_len: int) -> int:
    raw_width = max_model_len + 1
    alignment = DSA_LIDU_CACHE_ROW_ALIGNMENT
    return (raw_width + alignment - 1) // alignment * alignment


def _assert_exact_int_tensor(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = torch.nonzero(actual != expected, as_tuple=False)
    first_index = tuple(int(value) for value in mismatch[0].tolist())
    pytest.fail(
        "integer tensor mismatch: "
        f"first_index={first_index}, "
        f"actual={int(actual[first_index])}, "
        f"expected={int(expected[first_index])}, "
        f"mismatch_count={int(mismatch.shape[0])}"
    )


@pytest.fixture(scope="module", autouse=True)
def _load_a5_dsa_custom_ops() -> None:
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("packed-C8 DSA operators require Ascend A5")
    try:
        require_dsa_offload_ops(packed_c8=True)
    except RuntimeError as error:
        pytest.fail(str(error))


def _swapped_arena(shape: tuple[int, ...]) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        pytest.skip("torch_npu swapped-memory API is unavailable")
    return torch_npu.empty_with_swapped_memory(
        shape,
        dtype=torch.int8,
        device=torch.device(f"npu:{torch.npu.current_device()}"),
    )


def _read_swapped_arena(tensor: torch.Tensor) -> torch.Tensor:
    staging = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    staging.fill_(1).mul_(tensor)
    torch.npu.synchronize()
    return staging.cpu()


def _write_swapped_arena(
    destination: torch.Tensor,
    source: torch.Tensor,
) -> None:
    destination.zero_()
    staging = source.to(destination.device)
    destination.add_(staging)
    torch.npu.synchronize()


def test_packed_c8_full_block_dump_writes_only_active_rows() -> None:
    source = (
        torch.randint(
            -128,
            128,
            (2, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
            dtype=torch.int16,
        )
        .to(torch.int8)
        .to("npu")
    )
    destination = _swapped_arena((3, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES))
    destination.zero_()
    src_block_ids = torch.tensor([1, 999], dtype=torch.int32, device="npu")
    dst_block_ids = torch.tensor([2, -1], dtype=torch.int32, device="npu")

    torch.ops._C_ascend.kv_cache_full_block_dump_c8(
        source,
        destination,
        src_block_ids,
        dst_block_ids,
    )
    torch.npu.synchronize()

    actual = _read_swapped_arena(destination)
    expected_source = source[1].cpu()
    assert torch.equal(actual[2], expected_source)
    assert torch.count_nonzero(actual[:2]).item() == 0


@pytest.mark.parametrize(
    ("topk_start", "topk_rotation"),
    [
        (0, 0),
        (0, 1),
        (0, _TOPK - 1),
        (1, 0),
    ],
)
def test_packed_c8_li_manager_first_fill_then_steady(
    topk_start: int,
    topk_rotation: int,
) -> None:
    batch = 3
    max_model_len = 8192
    cache_row_width = _cache_row_width(max_model_len)
    budget = 6144
    sparse_actual_len = 6273
    expected_topk = torch.roll(
        torch.arange(
            topk_start,
            topk_start + _TOPK,
            dtype=torch.int32,
        ),
        shifts=-topk_rotation,
    )
    expected_topk_slots = torch.arange(_TOPK, dtype=torch.int32)
    topk = expected_topk.to("npu")
    topk = topk.view(1, 1, _TOPK).repeat(batch, 1, 1)
    req_pool_entries = torch.tensor(
        [3, 0, 2],
        dtype=torch.int32,
        device="npu",
    )
    cache_slots = torch.full(
        (4, cache_row_width),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[2, -1] = -budget
    row_modes = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
    actual_lengths = torch.tensor(
        [2048, 4096, sparse_actual_len],
        dtype=torch.int32,
        device="npu",
    )
    source_ids = torch.full(
        (batch, 1, _COPY_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    destination_slots = torch.full_like(source_ids, -77)
    miss_counts = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    tail_info = torch.full(
        (batch, 2),
        -77,
        dtype=torch.int32,
        device="npu",
    )

    def _launch() -> None:
        torch.ops._C_ascend.npu_dsa_a5_li_manage_c8_out(
            topk,
            req_pool_entries,
            cache_slots,
            row_modes,
            actual_lengths,
            source_ids,
            destination_slots,
            miss_counts,
            tail_info,
        )
        torch.npu.synchronize()

    _launch()
    first_sources = source_ids.cpu()
    first_slots = destination_slots.cpu()
    first_counts = miss_counts.cpu()
    first_tail = tail_info.cpu()
    pool = cache_slots.cpu()

    assert torch.all(first_sources[0] == -1)
    assert torch.all(first_slots[0] == -1)
    assert first_counts.tolist() == [0, 0, budget]
    assert first_tail.tolist() == [[-1, 0], [-1, 0], [budget, 1]]
    # First-fill must publish the exact topK first, followed by a unique set
    # of resident filler tokens. The filler order is an implementation detail;
    # its token->slot mapping and slot uniqueness are the actual contract.
    first_fill_count = sparse_actual_len - 1
    sparse_sources = first_sources[2, 0]
    sparse_slots = first_slots[2, 0]
    _assert_exact_int_tensor(
        sparse_sources[:_TOPK],
        expected_topk,
    )
    _assert_exact_int_tensor(
        sparse_slots[:_TOPK],
        expected_topk_slots,
    )
    valid_sources = sparse_sources[:first_fill_count]
    assert torch.all((valid_sources >= 0) & (valid_sources < first_fill_count))
    unique_count = torch.unique(valid_sources).numel()
    if unique_count != first_fill_count:
        token_counts = torch.bincount(
            valid_sources.to(torch.int64),
            minlength=first_fill_count,
        )
        missing = torch.nonzero(token_counts == 0, as_tuple=False).flatten()
        duplicated = torch.nonzero(token_counts > 1, as_tuple=False).flatten()
        duplicate_details = []
        for token in duplicated[:8].tolist():
            positions = torch.nonzero(
                valid_sources == token,
                as_tuple=False,
            ).flatten()
            duplicate_details.append(
                (token, int(token_counts[token]), positions[:8].tolist())
            )
        pool_tokens_at_topk = pool[2, :-1][expected_topk.to(torch.int64)]
        pool_mismatch_ranks = torch.nonzero(
            pool_tokens_at_topk != expected_topk_slots,
            as_tuple=False,
        ).flatten()
        pool_topk_mismatch = int(pool_mismatch_ranks.numel())
        pool_mismatch_details = [
            (
                int(rank),
                int(expected_topk[rank]),
                int(pool_tokens_at_topk[rank]),
                int(expected_topk_slots[rank]),
            )
            for rank in pool_mismatch_ranks[:8].tolist()
        ]
        pytest.fail(
            "first-fill source IDs are not unique: "
            f"topk_start={topk_start}, "
            f"topk_rotation={topk_rotation}, "
            f"unique={unique_count}/{first_fill_count}, "
            f"missing={missing[:16].tolist()}, "
            f"duplicates={duplicate_details}, "
            f"tail={valid_sources[-16:].tolist()}, "
            f"pool_topk_mismatch={pool_topk_mismatch}, "
            f"pool_mismatch_details={pool_mismatch_details}, "
            f"pool_topk_tail={pool_tokens_at_topk[-16:].tolist()}"
        )
    _assert_exact_int_tensor(
        sparse_slots[:first_fill_count],
        torch.arange(first_fill_count, dtype=torch.int32),
    )
    assert torch.all(sparse_sources[first_fill_count:] == -1)
    assert torch.all(sparse_slots[first_fill_count:] == -1)

    pool_tokens = pool[2, :-1]
    cached_sources = valid_sources[:budget].to(torch.int64)
    _assert_exact_int_tensor(
        pool_tokens[cached_sources],
        torch.arange(budget, dtype=torch.int32),
    )
    resident_slots = pool_tokens[pool_tokens >= 0]
    assert resident_slots.numel() == budget
    _assert_exact_int_tensor(
        torch.sort(resident_slots).values,
        torch.arange(budget, dtype=torch.int32),
    )
    assert int(pool[2, -1]) == budget
    _assert_exact_int_tensor(first_sources[1, 0, :_TOPK], expected_topk)
    _assert_exact_int_tensor(first_slots[1, 0, :_TOPK], expected_topk)
    _launch()
    assert miss_counts.cpu().tolist() == [0, 0, 0]
    assert tail_info.cpu().tolist() == [[-1, 0], [-1, 0], [budget, 1]]
    second_sources = source_ids.cpu()[2, 0]
    second_slots = destination_slots.cpu()[2, 0]
    _assert_exact_int_tensor(second_sources[:_TOPK], expected_topk)
    _assert_exact_int_tensor(
        second_slots[:_TOPK],
        pool_tokens[expected_topk.to(torch.int64)],
    )
    assert torch.all(second_sources[_TOPK:] == -1)
    assert torch.all(second_slots[_TOPK:] == -1)


def test_packed_c8_ksc_copies_one_opaque_row() -> None:
    hbm = torch.zeros(
        (17, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        dtype=torch.int8,
        device="npu",
    )
    dram_cpu = torch.zeros(
        (16, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        dtype=torch.int8,
    )
    expected = torch.arange(_PACKED_ROW_BYTES, dtype=torch.int16).remainder(256).sub(128).to(torch.int8)
    dram_cpu[1, 1, 0] = expected
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm_table = torch.arange(17, dtype=torch.int32, device="npu").view(1, -1)
    dram_table = torch.arange(16, dtype=torch.int32, device="npu").view(1, -1)
    source_ids = torch.full(
        (1, 1, _COPY_CAPACITY),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    destination_slots = torch.full_like(source_ids, -1)
    source_ids[0, 0, 0] = 129
    destination_slots[0, 0, 0] = 3
    attention_slots = torch.full(
        (1, 1, _ATTENTION_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    resident_seq_lengths = torch.full(
        (1,),
        -77,
        dtype=torch.int32,
        device="npu",
    )

    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        hbm_table,
        dram_table,
        source_ids,
        destination_slots,
        torch.tensor([1], dtype=torch.int32, device="npu"),
        torch.tensor([2048], dtype=torch.int32, device="npu"),
        torch.tensor([2048], dtype=torch.int32, device="npu"),
        torch.tensor([2049], dtype=torch.int32, device="npu"),
        attention_slots,
        resident_seq_lengths,
    )
    torch.npu.synchronize()

    assert torch.equal(hbm[0, 3, 0].cpu(), expected)
    slots = attention_slots.cpu()[0, 0]
    assert int(slots[0]) == 3
    assert torch.all(slots[1:_TOPK] == -1)
    assert int(slots[_TOPK]) == 2048
    assert torch.all(slots[_TOPK + 1 :] == -1)
    assert resident_seq_lengths.cpu().tolist() == [2049]


def test_packed_c8_ksc_dense_rows_are_metadata_only() -> None:
    actual_lengths = (1024, 2048, 2176, 2177, 4096, 6144)
    batch = len(actual_lengths)
    max_blocks = max(actual_lengths) // _BLOCK_SIZE
    hbm = torch.zeros(
        (max_blocks, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        dtype=torch.int8,
        device="npu",
    )
    hbm_before = hbm.clone()
    dram = _swapped_arena((1, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES))
    dram.zero_()
    hbm_table = torch.arange(
        max_blocks,
        dtype=torch.int32,
        device="npu",
    ).repeat(batch, 1)
    dram_table = torch.zeros(
        (batch, max_blocks),
        dtype=torch.int32,
        device="npu",
    )

    source_ids = torch.full(
        (batch, 1, _COPY_CAPACITY),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    destination_slots = torch.full_like(source_ids, -1)
    selected = torch.arange(_TOPK, dtype=torch.int32, device="npu")
    source_ids[:, 0, :_TOPK] = selected
    destination_slots[:, 0, :_TOPK] = selected
    copy_counts = torch.zeros(batch, dtype=torch.int32, device="npu")
    cache_tokens = torch.zeros(batch, dtype=torch.int32, device="npu")
    candidate_lens = torch.tensor(
        actual_lengths,
        dtype=torch.int32,
        device="npu",
    )
    actual_seq_lengths_kv = candidate_lens.clone()
    attention_slots = torch.full(
        (batch, 1, _ATTENTION_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    resident_seq_lengths = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )

    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        hbm_table,
        dram_table,
        source_ids,
        destination_slots,
        copy_counts,
        cache_tokens,
        candidate_lens,
        actual_seq_lengths_kv,
        attention_slots,
        resident_seq_lengths,
    )
    torch.npu.synchronize()

    assert torch.equal(hbm, hbm_before)
    assert resident_seq_lengths.cpu().tolist() == list(actual_lengths)
    slots_cpu = attention_slots.cpu()
    for row, actual_len in enumerate(actual_lengths):
        selected_count = min(actual_len, _TOPK)
        assert torch.equal(
            slots_cpu[row, 0, :selected_count],
            torch.arange(selected_count, dtype=torch.int32),
        )
        assert torch.all(slots_cpu[row, 0, selected_count:] == -1)
