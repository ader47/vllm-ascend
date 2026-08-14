# SPDX-License-Identifier: Apache-2.0
"""A5 packed-C8 DSA operator integration checks."""

import pytest
import torch
import torch_npu

from vllm_ascend.dsa_offload.contracts import DSA_LIDU_CACHE_ROW_ALIGNMENT
from vllm_ascend.dsa_offload.ops import (
    quant_lightning_indexer_topk,
    require_dsa_offload_ops,
)
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


def _make_fused_lidu_inputs(
    *,
    candidate_len: int,
    final_len: int,
    row_mode: int,
    batch: int = 1,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    heads = 32
    blocks = (candidate_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    weights_storage = torch.randn(
        (batch, 128 + heads),
        dtype=torch.bfloat16,
        device="npu",
    )
    return {
        "weights": weights_storage[:, 128:],
        "query": torch.randn(
            (batch, heads, 128),
            dtype=torch.bfloat16,
            device="npu",
        ).to(torch.float8_e4m3fn),
        # Keep every row/head distinct so the batch=2 DENSE cases also catch
        # accidental reuse of the strided weights offset for this compact
        # scale tensor.
        "query_scale": torch.linspace(
            0.5,
            1.5,
            steps=batch * heads,
            dtype=torch.float32,
            device="npu",
        ).view(batch, heads),
        "query_ends": torch.arange(
            1,
            batch + 1,
            dtype=torch.int32,
            device="npu",
        ),
        "key": torch.randn((blocks, 128, 1, 128), dtype=torch.bfloat16, device="npu").to(torch.float8_e4m3fn),
        "key_scale": torch.ones((blocks, 128, 1), dtype=torch.float32, device="npu"),
        "block_table": torch.arange(
            blocks,
            dtype=torch.int32,
            device="npu",
        )
        .view(1, -1)
        .repeat(batch, 1),
        "candidate_lens": torch.full(
            (batch,),
            candidate_len,
            dtype=torch.int32,
            device="npu",
        ),
        "final_lens": torch.full(
            (batch,),
            final_len,
            dtype=torch.int32,
            device="npu",
        ),
        "row_modes": torch.full(
            (batch,),
            row_mode,
            dtype=torch.int32,
            device="npu",
        ),
        "req_entries": (
            torch.arange(batch, dtype=torch.int32, device="npu")
            if row_mode == 2
            else torch.full(
                (batch,),
                -1,
                dtype=torch.int32,
                device="npu",
            )
        ),
    }


def _allocate_fused_lidu_outputs(
    batch: int = 1,
) -> tuple[torch.Tensor, ...]:
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
    copy_src_ids = torch.full(
        (batch, 1, _COPY_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    copy_dst_slots = torch.full_like(copy_src_ids, -77)
    copy_counts = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    return (
        attention_slots,
        resident_seq_lengths,
        copy_src_ids,
        copy_dst_slots,
        copy_counts,
    )


def _launch_fused_lidu(
    inputs: dict[str, torch.Tensor],
    cache_slots: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_li_manage_nomtp_c8_out(
        inputs["weights"],
        inputs["query"],
        inputs["query_scale"],
        inputs["query_ends"],
        inputs["key"],
        inputs["key_scale"],
        inputs["block_table"],
        inputs["candidate_lens"],
        inputs["final_lens"],
        inputs["row_modes"],
        inputs["req_entries"],
        cache_slots,
        *outputs,
    )


def _native_topk(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return quant_lightning_indexer_topk(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_scale"],
        key_dequant_scale=inputs["key_scale"],
        actual_seq_lengths_query=inputs["query_ends"],
        candidate_lens=inputs["candidate_lens"],
        block_table=inputs["block_table"],
    )


@pytest.mark.parametrize("dense_len", [1, 2048, 2049, 2176, 2177, 6144])
def test_packed_c8_fused_lidu_dense_matches_framework_semantics(
    dense_len: int,
) -> None:
    batch = 2
    inputs = _make_fused_lidu_inputs(
        candidate_len=dense_len,
        final_len=dense_len,
        row_mode=1,
        batch=batch,
    )
    assert inputs["weights"].stride() == (160, 1)
    assert not inputs["weights"].is_contiguous()
    expected_slots = (
        torch.arange(dense_len, dtype=torch.int32).view(1, 1, -1).repeat(batch, 1, 1)
        if dense_len <= _TOPK
        else _native_topk(inputs).cpu()
    )
    cache_slots = torch.full(
        (batch + 1, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    before = cache_slots.clone()
    outputs = _allocate_fused_lidu_outputs(batch)

    _launch_fused_lidu(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    selected_count = min(dense_len, _TOPK)
    _assert_exact_int_tensor(attention[:, :, :selected_count], expected_slots)
    assert torch.all(attention[:, :, selected_count:] == -1)
    assert resident_lengths.tolist() == [dense_len] * batch
    assert counts.tolist() == [0] * batch
    assert torch.equal(cache_slots, before)


@pytest.mark.parametrize("budget", [6144, 10240, 12288])
def test_packed_c8_fused_lidu_sparse_first_fill_then_steady(
    budget: int,
) -> None:
    candidate_len = budget + _BLOCK_SIZE
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate_len,
        final_len=candidate_len + 1,
        row_mode=2,
        seed=19,
    )
    expected_topk = _native_topk(inputs).cpu()[0, 0]
    cache_slots = torch.full(
        (2, _cache_row_width(16384)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    outputs = _allocate_fused_lidu_outputs()

    _launch_fused_lidu(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    pool = cache_slots.cpu()[0]
    attention, resident_lengths, source_ids, destination_slots, counts = (tensor.cpu() for tensor in outputs)
    assert counts.tolist() == [budget]
    assert resident_lengths.tolist() == [budget + 1]
    assert int(pool[-1]) == budget
    valid_sources = source_ids[0, 0, :budget]
    valid_destinations = destination_slots[0, 0, :budget]
    assert torch.unique(valid_sources).numel() == budget
    _assert_exact_int_tensor(
        valid_destinations,
        torch.arange(budget, dtype=torch.int32),
    )
    _assert_exact_int_tensor(
        pool[valid_sources.to(torch.int64)],
        valid_destinations,
    )
    _assert_exact_int_tensor(
        attention[0, 0, :_TOPK],
        pool[expected_topk.to(torch.int64)],
    )
    assert int(attention[0, 0, _TOPK]) == budget
    assert torch.all(attention[0, 0, _TOPK + 1 :] == -1)

    repeat_outputs = _allocate_fused_lidu_outputs()
    pool_before_repeat = cache_slots.clone()
    _launch_fused_lidu(inputs, cache_slots, repeat_outputs)
    torch.npu.synchronize()
    assert repeat_outputs[-1].cpu().tolist() == [0]
    assert torch.equal(cache_slots, pool_before_repeat)


def test_packed_c8_fused_lidu_mixed_dense_sparse_and_pad_rows() -> None:
    batch = 3
    dense_len = 2177
    sparse_candidate_len = 6272
    sparse_final_len = 6273
    inputs = _make_fused_lidu_inputs(
        candidate_len=sparse_candidate_len,
        final_len=sparse_final_len,
        row_mode=2,
        batch=batch,
        seed=23,
    )
    inputs["candidate_lens"].copy_(
        torch.tensor(
            [dense_len, sparse_candidate_len, _TOPK],
            dtype=torch.int32,
            device="npu",
        )
    )
    inputs["final_lens"].copy_(
        torch.tensor(
            [dense_len, sparse_final_len, 1],
            dtype=torch.int32,
            device="npu",
        )
    )
    inputs["row_modes"].copy_(torch.tensor([1, 2, 0], dtype=torch.int32, device="npu"))
    inputs["req_entries"].copy_(torch.tensor([-1, 0, -1], dtype=torch.int32, device="npu"))
    expected_topk = _native_topk(inputs).cpu()
    cache_slots = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -6144
    outputs = _allocate_fused_lidu_outputs(batch)

    _launch_fused_lidu(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    _assert_exact_int_tensor(
        attention[0, :, :_TOPK],
        expected_topk[0],
    )
    assert resident_lengths.tolist() == [dense_len, 6145, 0]
    assert counts.tolist() == [0, 6144, 0]
    assert torch.all(attention[2] == -1)


def test_packed_c8_fused_lidu_graph_replay_changes_pool_state() -> None:
    inputs = _make_fused_lidu_inputs(
        candidate_len=6272,
        final_len=6273,
        row_mode=2,
        seed=31,
    )
    initial_pool = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    initial_pool[:, -1].zero_()
    initial_pool[0, -1] = -6144

    expected_pool = initial_pool.clone()
    expected_outputs = _allocate_fused_lidu_outputs()
    _launch_fused_lidu(inputs, expected_pool, expected_outputs)
    torch.npu.synchronize()
    expected_pool_cpu = expected_pool.cpu()
    expected_outputs_cpu = tuple(tensor.cpu() for tensor in expected_outputs)

    graph_pool_state = expected_pool.clone()
    graph_outputs = _allocate_fused_lidu_outputs()
    stable_pointers = (
        graph_pool_state.data_ptr(),
        *(tensor.data_ptr() for tensor in graph_outputs),
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        _launch_fused_lidu(inputs, graph_pool_state, graph_outputs)
    # A5 graph capture records the launch but does not guarantee one eager
    # execution. Replay explicitly before validating the resident steady state.
    graph.replay()
    torch.npu.synchronize()
    assert graph_outputs[-1].cpu().tolist() == [0]

    graph_pool_state.copy_(initial_pool)
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(graph_pool_state.cpu(), expected_pool_cpu)
    for actual, expected in zip(graph_outputs, expected_outputs_cpu):
        assert torch.equal(actual.cpu(), expected)
    assert stable_pointers == (
        graph_pool_state.data_ptr(),
        *(tensor.data_ptr() for tensor in graph_outputs),
    )


def test_packed_c8_fused_lidu_copy_plan_is_consumed_by_ksc() -> None:
    budget = 6144
    candidate_len = 6272
    blocks = candidate_len // _BLOCK_SIZE
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate_len,
        final_len=candidate_len + 1,
        row_mode=2,
        seed=37,
    )
    cache_slots = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    outputs = _allocate_fused_lidu_outputs()
    _launch_fused_lidu(inputs, cache_slots, outputs)

    dram_cpu = torch.randint(
        -128,
        128,
        (blocks, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        dtype=torch.int16,
    ).to(torch.int8)
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm = torch.zeros_like(dram_cpu, device="npu")
    block_table = torch.arange(
        blocks,
        dtype=torch.int32,
        device="npu",
    ).view(1, -1)
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        block_table,
        block_table,
        outputs[2],
        outputs[3],
        outputs[4],
    )
    torch.npu.synchronize()

    source_ids = outputs[2][0, 0, :budget].cpu().to(torch.int64)
    destination_slots = outputs[3][0, 0, :budget].cpu().to(torch.int64)
    assert outputs[4].cpu().tolist() == [budget]
    actual_rows = hbm.cpu().view(-1, _PACKED_ROW_BYTES)[destination_slots]
    expected_rows = dram_cpu.view(-1, _PACKED_ROW_BYTES)[source_ids]
    assert torch.equal(actual_rows, expected_rows)


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
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        hbm_table,
        dram_table,
        source_ids,
        destination_slots,
        torch.tensor([1], dtype=torch.int32, device="npu"),
    )
    torch.npu.synchronize()

    assert torch.equal(hbm[0, 3, 0].cpu(), expected)
