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
        require_dsa_offload_ops(packed_c8=True, mtp=True)
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


def _make_unique_packed_rows(num_rows: int) -> torch.Tensor:
    """Encode the complete source token ID into every opaque packed row."""

    base = torch.arange(_PACKED_ROW_BYTES, dtype=torch.int16).remainder(251).sub(125).to(torch.int8)
    rows = base.repeat(num_rows, 1)
    token_ids = torch.arange(num_rows, dtype=torch.int32)
    rows[:, 0] = token_ids.to(torch.int8)
    rows[:, 1] = torch.div(token_ids, 256, rounding_mode="floor").to(torch.int8)
    rows[:, 2] = torch.div(token_ids, 65536, rounding_mode="floor").to(torch.int8)
    return rows


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
    heads: int = 32,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
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
    total_query_rows: int | None = None,
) -> tuple[torch.Tensor, ...]:
    if total_query_rows is None:
        total_query_rows = batch
    attention_slots = torch.full(
        (total_query_rows, 1, _ATTENTION_CAPACITY),
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


def _launch_mtp_lim(
    inputs: dict[str, torch.Tensor],
    cache_slots: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_li_manage_c8_out(
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


def _launch_packed_ksc(
    hbm: torch.Tensor,
    dram: torch.Tensor,
    block_table: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        block_table,
        block_table,
        outputs[2],
        outputs[3],
        outputs[4],
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


def test_packed_c8_mtp_lim_dense_short_rows_are_causal() -> None:
    batch = 2
    queries_per_request = 4
    total_query_rows = batch * queries_per_request
    final_len = 16
    inputs = _make_fused_lidu_inputs(
        candidate_len=final_len,
        final_len=final_len,
        row_mode=1,
        batch=total_query_rows,
        seed=29,
    )
    inputs["query_ends"] = torch.tensor(
        [4, 8],
        dtype=torch.int32,
        device="npu",
    )
    for name in (
        "block_table",
        "candidate_lens",
        "final_lens",
        "row_modes",
        "req_entries",
    ):
        inputs[name] = inputs[name][:batch]
    cache_slots = torch.full(
        (batch + 1, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    before = cache_slots.clone()
    outputs = _allocate_fused_lidu_outputs(
        batch,
        total_query_rows,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (
        tensor.cpu() for tensor in outputs
    )
    for query_row in range(total_query_rows):
        local_query = query_row % queries_per_request
        visible_len = final_len - (
            queries_per_request - 1 - local_query
        )
        _assert_exact_int_tensor(
            attention[query_row, 0, :visible_len],
            torch.arange(visible_len, dtype=torch.int32),
        )
        assert torch.all(
            attention[query_row, 0, visible_len:] == -1
        )
    assert resident_lengths.tolist() == [final_len] * batch
    assert counts.tolist() == [0] * batch
    assert torch.equal(cache_slots, before)


def test_packed_c8_mtp_lim_dense_long_rows_are_causal() -> None:
    batch = 2
    queries_per_request = 4
    total_query_rows = batch * queries_per_request
    final_len = 2180
    inputs = _make_fused_lidu_inputs(
        candidate_len=final_len,
        final_len=final_len,
        row_mode=1,
        batch=total_query_rows,
        seed=31,
    )
    # Request-level tensors keep B rows while query/weight/scale keep T rows.
    inputs["query_ends"] = torch.tensor(
        [4, 8],
        dtype=torch.int32,
        device="npu",
    )
    inputs["block_table"] = inputs["block_table"][:batch]
    inputs["candidate_lens"] = inputs["candidate_lens"][:batch]
    inputs["final_lens"] = inputs["final_lens"][:batch]
    inputs["row_modes"] = inputs["row_modes"][:batch]
    inputs["req_entries"] = inputs["req_entries"][:batch]
    # This is the exact native target-model call shape: compact TND Q rows,
    # per-request final KV lengths, and sparse_mode=3 for causal visibility.
    expected_topk = _native_topk(inputs).cpu().reshape(
        total_query_rows,
        _TOPK,
    )
    cache_slots = torch.full(
        (batch + 1, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    before = cache_slots.clone()
    outputs = _allocate_fused_lidu_outputs(
        batch,
        total_query_rows,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, raw_topk, error_metadata, counts = (
        tensor.cpu() for tensor in outputs
    )
    if any(count < 0 for count in counts.tolist()):
        diagnostics = []
        for request, count in enumerate(counts.tolist()):
            if count >= 0:
                continue
            raw = raw_topk[request, 0, :_TOPK]
            query_row = int(error_metadata[request, 0, 0])
            visible_len = int(error_metadata[request, 0, 1])
            invalid = torch.nonzero(
                (raw < 0) | (raw >= visible_len),
                as_tuple=False,
            ).view(-1)
            diagnostics.append(
                {
                    "request": request,
                    "query_row": query_row,
                    "visible_len": visible_len,
                    "invalid_count": int(invalid.numel()),
                    "first_invalid_indices": invalid[:16].tolist(),
                    "raw_prefix": raw[:16].tolist(),
                    "raw_suffix": raw[-16:].tolist(),
                    "valid_unique": int(
                        torch.unique(
                            raw[(raw >= 0) & (raw < visible_len)]
                        ).numel()
                    ),
                }
            )
        pytest.fail(f"MTP LIM dense selection error: {diagnostics}")
    for query_row in range(total_query_rows):
        actual_row = attention[query_row, 0, :_TOPK]
        expected_row = expected_topk[query_row]
        if not torch.equal(actual_row, expected_row):
            mismatch = torch.nonzero(
                actual_row != expected_row,
                as_tuple=False,
            ).view(-1)
            invalid = torch.nonzero(
                (actual_row < 0)
                | (actual_row >= final_len),
                as_tuple=False,
            ).view(-1)
            pytest.fail(
                "MTP LIM dense top-k mismatch: "
                f"query_row={query_row}, "
                f"mismatch_count={int(mismatch.numel())}, "
                f"first_mismatch_indices={mismatch[:16].tolist()}, "
                f"actual_at_mismatch={actual_row[mismatch[:16]].tolist()}, "
                f"expected_at_mismatch={expected_row[mismatch[:16]].tolist()}, "
                f"invalid_count={int(invalid.numel())}, "
                f"invalid_indices={invalid[:16].tolist()}, "
                "same_multiset="
                f"{torch.equal(torch.sort(actual_row).values, torch.sort(expected_row).values)}"
            )
        assert torch.all(
            attention[query_row, 0, _TOPK:] == -1
        )
    assert resident_lengths.tolist() == [final_len] * batch
    assert counts.tolist() == [0] * batch
    assert torch.equal(cache_slots, before)


def test_packed_c8_mtp_lim_uses_per_query_candidates_and_dual_tails() -> None:
    budget = 8192
    source_len = 8192
    final_len = 8321
    queries_per_request = 4
    inputs = _make_fused_lidu_inputs(
        candidate_len=final_len - 1,
        final_len=final_len,
        row_mode=2,
        batch=queries_per_request,
        seed=47,
    )
    inputs["query_ends"] = torch.tensor(
        [queries_per_request],
        dtype=torch.int32,
        device="npu",
    )
    inputs["block_table"] = inputs["block_table"][:1]
    inputs["candidate_lens"] = torch.tensor(
        [source_len],
        dtype=torch.int32,
        device="npu",
    )
    inputs["final_lens"] = torch.tensor(
        [final_len],
        dtype=torch.int32,
        device="npu",
    )
    inputs["row_modes"] = torch.tensor(
        [2],
        dtype=torch.int32,
        device="npu",
    )
    inputs["req_entries"] = torch.tensor(
        [0],
        dtype=torch.int32,
        device="npu",
    )

    expected_topk_rows: list[torch.Tensor] = []
    candidate_ends: list[int] = []
    for query_row in range(queries_per_request):
        visible_len = final_len - (queries_per_request - 1 - query_row)
        candidate_end = (
            (visible_len - 1) // _BLOCK_SIZE * _BLOCK_SIZE
        )
        candidate_ends.append(candidate_end)
        row_inputs = {
            "query": inputs["query"][query_row : query_row + 1],
            "key": inputs["key"],
            "weights": inputs["weights"][query_row : query_row + 1],
            "query_scale": inputs["query_scale"][
                query_row : query_row + 1
            ],
            "key_scale": inputs["key_scale"],
            "query_ends": torch.tensor(
                [1],
                dtype=torch.int32,
                device="npu",
            ),
            "candidate_lens": torch.tensor(
                [candidate_end],
                dtype=torch.int32,
                device="npu",
            ),
            "block_table": inputs["block_table"],
        }
        expected_topk_rows.append(
            _native_topk(row_inputs).cpu().view(-1)
        )

    cache_slots = torch.full(
        (2, _cache_row_width(final_len)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    outputs = _allocate_fused_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (
        tensor.cpu() for tensor in outputs
    )
    cache_row = cache_slots[0].cpu()
    for query_row, expected_tokens in enumerate(expected_topk_rows):
        expected_slots = torch.where(
            expected_tokens < source_len,
            cache_row[expected_tokens.to(torch.long)],
            budget + expected_tokens.remainder(2 * _BLOCK_SIZE),
        ).to(torch.int32)
        _assert_exact_int_tensor(
            attention[query_row, 0, :_TOPK],
            expected_slots,
        )

        visible_len = final_len - (queries_per_request - 1 - query_row)
        tail_start = candidate_ends[query_row]
        tail_tokens = torch.arange(
            tail_start,
            visible_len,
            dtype=torch.int32,
        )
        expected_tail_slots = (
            budget + tail_tokens.remainder(2 * _BLOCK_SIZE)
        )
        tail_len = int(tail_tokens.numel())
        _assert_exact_int_tensor(
            attention[
                query_row,
                0,
                _TOPK : _TOPK + tail_len,
            ],
            expected_tail_slots,
        )
        assert torch.all(
            attention[query_row, 0, _TOPK + tail_len :] == -1
        )

    assert resident_lengths.tolist() == [budget + 2 * _BLOCK_SIZE]
    assert counts.tolist() == [budget]


@pytest.mark.parametrize("heads", [32, 64])
def test_packed_c8_mtp_lim_steady_fast_path_updates_union_and_dual_tails(
    heads: int,
) -> None:
    budget = 8192
    candidate_len = 8320
    final_len = candidate_len + 4
    queries_per_request = 4

    def make_request(seed: int) -> dict[str, torch.Tensor]:
        inputs = _make_fused_lidu_inputs(
            candidate_len=candidate_len,
            final_len=final_len,
            row_mode=2,
            batch=queries_per_request,
            seed=seed,
            heads=heads,
        )
        inputs["query_ends"] = torch.tensor(
            [queries_per_request],
            dtype=torch.int32,
            device="npu",
        )
        for name in (
            "block_table",
            "candidate_lens",
            "final_lens",
            "row_modes",
            "req_entries",
        ):
            inputs[name] = inputs[name][:1]
        return inputs

    cache_slots = torch.full(
        (2, _cache_row_width(final_len)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    # The final pool element is metadata, deliberately leaving a logical
    # token capacity that is not a 128-token QuantLI score-row stride.
    assert (int(cache_slots.shape[1]) - 1) % _BLOCK_SIZE != 0

    # Establish the persistent request row through the correctness path. The
    # positive metadata left by this call is part of the fast-path contract.
    cold_outputs = _allocate_fused_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )
    _launch_mtp_lim(make_request(seed=53), cache_slots, cold_outputs)
    torch.npu.synchronize()
    assert int(cache_slots[0, -1].cpu()) == budget

    steady_inputs = make_request(seed=59)
    expected_topk_rows = []
    for route in range(queries_per_request):
        row_inputs = {
            **steady_inputs,
            "query": steady_inputs["query"][route : route + 1],
            "weights": steady_inputs["weights"][route : route + 1],
            "query_scale": steady_inputs["query_scale"][route : route + 1],
            # candidate_lens is a durable prefix before every speculative
            # query. A batched sparse_mode=3 reference would instead treat
            # the queries as part of the key sequence and causally hide the
            # final 3/2/1 prefix tokens from its early rows.
            "query_ends": torch.tensor(
                [1],
                dtype=torch.int32,
                device="npu",
            ),
        }
        expected_topk_rows.append(_native_topk(row_inputs).cpu().view(-1))
    expected_topk = torch.stack(expected_topk_rows)
    pool_before = cache_slots[0].cpu().clone()
    expected_misses = torch.unique(
        expected_topk[pool_before[expected_topk.to(torch.long)] < 0]
    )
    assert expected_misses.numel() > 0

    outputs = _allocate_fused_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )
    _launch_mtp_lim(steady_inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, source_ids, destination_slots, counts = (
        tensor.cpu() for tensor in outputs
    )
    pool_after = cache_slots[0].cpu()
    copy_count = int(counts[0])
    assert copy_count == int(expected_misses.numel())
    actual_sources = source_ids[0, 0, :copy_count]
    actual_destinations = destination_slots[0, 0, :copy_count]
    _assert_exact_int_tensor(
        torch.sort(actual_sources).values,
        torch.sort(expected_misses).values,
    )
    assert torch.unique(actual_destinations).numel() == copy_count
    assert torch.all((actual_destinations >= 0) & (actual_destinations < budget))
    _assert_exact_int_tensor(
        pool_after[actual_sources.to(torch.long)],
        actual_destinations,
    )

    inverse_before = torch.full((budget,), -1, dtype=torch.int32)
    mapped_sources = torch.nonzero(
        pool_before[:candidate_len] >= 0,
        as_tuple=False,
    ).view(-1)
    inverse_before[pool_before[mapped_sources].to(torch.long)] = (
        mapped_sources.to(torch.int32)
    )
    victim_sources = inverse_before[actual_destinations.to(torch.long)]
    assert torch.all(victim_sources >= 0)
    assert torch.all(pool_after[victim_sources.to(torch.long)] == -1)

    for route in range(queries_per_request):
        expected_slots = pool_after[expected_topk[route].to(torch.long)]
        actual_slots = attention[route, 0, :_TOPK]
        # The optimized path deliberately publishes the miss destinations
        # before the compacted hit slots. Sparse attention consumes an
        # unordered resident-slot set, so preserving native TopK score order
        # would only add another 2048-entry reorder/round trip. Keep the test
        # strict about membership, multiplicity and cardinality instead.
        _assert_exact_int_tensor(
            torch.sort(actual_slots).values,
            torch.sort(expected_slots).values,
        )
        assert torch.unique(actual_slots).numel() == _TOPK
        visible_len = final_len - (queries_per_request - 1 - route)
        tail_tokens = torch.arange(
            candidate_len,
            visible_len,
            dtype=torch.int32,
        )
        expected_tail = budget + tail_tokens.remainder(2 * _BLOCK_SIZE)
        tail_len = int(tail_tokens.numel())
        _assert_exact_int_tensor(
            attention[route, 0, _TOPK : _TOPK + tail_len],
            expected_tail,
        )
        assert torch.all(attention[route, 0, _TOPK + tail_len :] == -1)

    assert resident_lengths.tolist() == [budget + 2 * _BLOCK_SIZE]
    assert int(pool_after[-1]) == budget


@pytest.mark.parametrize("heads", [32, 64])
def test_packed_c8_mtp_lim_steady_fast_path_has_exact_monotonic_topk(
    heads: int,
) -> None:
    budget = 8192
    candidate_len = budget
    final_len = candidate_len + 4
    queries_per_request = 4
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate_len,
        final_len=final_len,
        row_mode=2,
        batch=queries_per_request,
        seed=67,
        heads=heads,
    )
    inputs["query_ends"] = torch.tensor(
        [queries_per_request],
        dtype=torch.int32,
        device="npu",
    )
    for name in (
        "block_table",
        "candidate_lens",
        "final_lens",
        "row_modes",
        "req_entries",
    ):
        inputs[name] = inputs[name][:1]

    # QK=128 and the head reduction is heads*128, both powers of two.
    # Scaling consecutive positive BF16 values by that factor therefore
    # produces 8192 unique, exactly ordered final BF16 scores without ties.
    inputs["query"] = torch.ones(
        tuple(inputs["query"].shape),
        dtype=torch.bfloat16,
        device="npu",
    ).to(torch.float8_e4m3fn)
    inputs["key"] = torch.ones(
        tuple(inputs["key"].shape),
        dtype=torch.bfloat16,
        device="npu",
    ).to(torch.float8_e4m3fn)
    inputs["weights"].fill_(1)
    inputs["query_scale"].fill_(1)
    score_bits = torch.arange(
        0x2000,
        0x2000 + candidate_len,
        dtype=torch.int32,
    ).to(torch.int16)
    desired_scores = score_bits.view(torch.bfloat16).to(torch.float32)
    key_scales = desired_scores / float(heads * 128)
    inputs["key_scale"].view(-1).copy_(key_scales.to(device="npu"))

    cache_slots = torch.full(
        (2, _cache_row_width(final_len)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, :candidate_len] = torch.arange(
        candidate_len,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[0, -1] = budget
    before = cache_slots.clone()
    outputs = _allocate_fused_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (
        tensor.cpu() for tensor in outputs
    )
    expected_topk = torch.arange(
        candidate_len - _TOPK,
        candidate_len,
        dtype=torch.int32,
    )
    for route in range(queries_per_request):
        actual_topk = attention[route, 0, :_TOPK]
        sorted_actual = torch.sort(actual_topk).values
        if not torch.equal(sorted_actual, expected_topk):
            missing = expected_topk[~torch.isin(expected_topk, actual_topk)]
            extra = actual_topk[~torch.isin(actual_topk, expected_topk)]
            pytest.fail(
                "MTP LIM monotonic TopK mismatch: "
                f"heads={heads}, route={route}, "
                f"unique={int(torch.unique(actual_topk).numel())}, "
                f"actual_min={int(sorted_actual[0])}, "
                f"actual_max={int(sorted_actual[-1])}, "
                f"missing_count={int(missing.numel())}, "
                f"extra_count={int(extra.numel())}, "
                f"missing_prefix={missing[:16].tolist()}, "
                f"extra_prefix={extra[:16].tolist()}"
            )
        visible_len = final_len - (queries_per_request - 1 - route)
        tail_tokens = torch.arange(
            candidate_len,
            visible_len,
            dtype=torch.int32,
        )
        expected_tail = budget + tail_tokens.remainder(2 * _BLOCK_SIZE)
        tail_len = int(tail_tokens.numel())
        _assert_exact_int_tensor(
            attention[route, 0, _TOPK : _TOPK + tail_len],
            expected_tail,
        )
        assert torch.all(attention[route, 0, _TOPK + tail_len :] == -1)

    assert counts.tolist() == [0]
    assert resident_lengths.tolist() == [budget + 2 * _BLOCK_SIZE]
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


def test_packed_c8_full_shared_next_full_graph_chain() -> None:
    budget = 6144
    candidate_len = budget + _BLOCK_SIZE
    blocks = candidate_len // _BLOCK_SIZE
    full_inputs = (
        _make_fused_lidu_inputs(
            candidate_len=candidate_len,
            final_len=candidate_len + 1,
            row_mode=2,
            seed=41,
        ),
        _make_fused_lidu_inputs(
            candidate_len=candidate_len,
            final_len=candidate_len + 1,
            row_mode=2,
            seed=43,
        ),
    )
    initial_selection_states = torch.full(
        (2, 2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    initial_selection_states[:, :, -1].zero_()
    initial_selection_states[:, 0, -1] = -budget

    expected_states = initial_selection_states.clone()
    expected_outputs = (
        _allocate_fused_lidu_outputs(),
        _allocate_fused_lidu_outputs(),
    )
    for state_id in range(2):
        _launch_fused_lidu(
            full_inputs[state_id],
            expected_states[state_id],
            expected_outputs[state_id],
        )
    torch.npu.synchronize()
    assert [int(outputs[4].cpu()[0]) for outputs in expected_outputs] == [
        budget,
        budget,
    ]
    assert not torch.equal(
        expected_outputs[0][2][:, :, :budget],
        expected_outputs[1][2][:, :, :budget],
    )

    block_table = torch.arange(
        blocks,
        dtype=torch.int32,
        device="npu",
    ).view(1, -1)
    dram_cpu_layers = []
    dram_layers = []
    hbm_layers = []
    for layer_id in range(3):
        dram_cpu = _make_unique_packed_rows(candidate_len)
        dram_cpu[:, 3] = layer_id + 1
        dram_cpu = dram_cpu.view(
            blocks,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        )
        dram = _swapped_arena(tuple(dram_cpu.shape))
        _write_swapped_arena(dram, dram_cpu)
        dram_cpu_layers.append(dram_cpu)
        dram_layers.append(dram)
        hbm_layers.append(torch.zeros_like(dram_cpu, device="npu"))

    graph_states = initial_selection_states.clone()
    shared_scratch = _allocate_fused_lidu_outputs()
    shared_attention_snapshot = torch.empty_like(shared_scratch[0])
    shared_length_snapshot = torch.empty_like(shared_scratch[1])
    stable_pointers = (
        graph_states.data_ptr(),
        *(tensor.data_ptr() for tensor in shared_scratch),
        shared_attention_snapshot.data_ptr(),
        shared_length_snapshot.data_ptr(),
        *(tensor.data_ptr() for tensor in hbm_layers),
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        # full 0 produces one complete selection plan. shared 1 consumes the
        # same scratch but copies from its own DRAM arena into its own HBM.
        _launch_fused_lidu(
            full_inputs[0],
            graph_states[0],
            shared_scratch,
        )
        _launch_packed_ksc(
            hbm_layers[0],
            dram_layers[0],
            block_table,
            shared_scratch,
        )
        _launch_packed_ksc(
            hbm_layers[1],
            dram_layers[1],
            block_table,
            shared_scratch,
        )
        # Snapshot the attention metadata consumed by the shared SFA path so
        # the following full layer cannot hide an early scratch overwrite.
        shared_attention_snapshot.copy_(shared_scratch[0])
        shared_length_snapshot.copy_(shared_scratch[1])
        # The next full layer owns a different compact selection state and may
        # overwrite the shared scratch only after the follower consumed it.
        _launch_fused_lidu(
            full_inputs[1],
            graph_states[1],
            shared_scratch,
        )
        _launch_packed_ksc(
            hbm_layers[2],
            dram_layers[2],
            block_table,
            shared_scratch,
        )

    graph_states.copy_(initial_selection_states)
    for hbm in hbm_layers:
        hbm.zero_()
    graph.replay()
    torch.npu.synchronize()

    assert torch.equal(graph_states.cpu(), expected_states.cpu())
    assert torch.equal(
        shared_attention_snapshot.cpu(),
        expected_outputs[0][0].cpu(),
    )
    assert torch.equal(
        shared_length_snapshot.cpu(),
        expected_outputs[0][1].cpu(),
    )
    for actual, expected in zip(shared_scratch, expected_outputs[1]):
        assert torch.equal(actual.cpu(), expected.cpu())
    assert stable_pointers == (
        graph_states.data_ptr(),
        *(tensor.data_ptr() for tensor in shared_scratch),
        shared_attention_snapshot.data_ptr(),
        shared_length_snapshot.data_ptr(),
        *(tensor.data_ptr() for tensor in hbm_layers),
    )

    # Layers 0 and 1 consume full 0's plan; layer 2 consumes full 2's plan.
    source_plans = (expected_outputs[0], expected_outputs[0], expected_outputs[1])
    probe_indices = (0, budget // 2, budget - 1)
    for layer_id, outputs in enumerate(source_plans):
        source_ids = outputs[2][0, 0, :budget].cpu().to(torch.int64)
        destination_slots = outputs[3][0, 0, :budget].cpu().to(torch.int64)
        hbm_rows = hbm_layers[layer_id].cpu().view(-1, _PACKED_ROW_BYTES)
        dram_rows = dram_cpu_layers[layer_id].view(-1, _PACKED_ROW_BYTES)
        for probe_index in probe_indices:
            assert torch.equal(
                hbm_rows[destination_slots[probe_index]],
                dram_rows[source_ids[probe_index]],
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

    dram_cpu = _make_unique_packed_rows(blocks * _BLOCK_SIZE).view(
        blocks,
        _BLOCK_SIZE,
        1,
        _PACKED_ROW_BYTES,
    )
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm = torch.zeros_like(dram_cpu, device="npu")
    block_table = torch.arange(
        blocks,
        dtype=torch.int32,
        device="npu",
    ).view(1, -1)
    _launch_packed_ksc(hbm, dram, block_table, outputs)
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
