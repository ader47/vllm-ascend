# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

import vllm_ascend.dsa_offload.runtime as runtime_module
from vllm_ascend.dsa_offload.dram_store import DSAHotDRAMStore
from vllm_ascend.dsa_offload.input_batch import DSAInputBatchCacheLayout
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool
from vllm_ascend.dsa_offload.runtime import (
    DSALayerOffloadContext,
    DSAOffloadRuntime,
)
from vllm_ascend.dsa_offload.scheduler_output import (
    DSARequestCacheLayoutProjection,
    DSAResidentBlockTableReplacement,
)


class _ResidentBlockTable:
    def __init__(self) -> None:
        self._rows = np.array([[10, 11, 12, 0]], dtype=np.int32)
        self.num_blocks_per_row = np.array([3], dtype=np.int32)

    def get_numpy_array(self) -> np.ndarray:
        return self._rows


def _make_runtime(
    max_num_reqs: int = 1,
    *,
    packed_c8: bool = False,
) -> tuple[
    DSAResidentTokenPool,
    DSAOffloadRuntime,
    DSAHotDRAMStore,
]:
    resident_pool = DSAResidentTokenPool(
        max_num_reqs=max_num_reqs,
        num_layers=2,
        max_model_len=512,
        max_resident_budget_tokens=256,
        device=torch.device("cpu"),
    )
    runtime = DSAOffloadRuntime(
        max_num_reqs=max_num_reqs,
        max_num_tokens=512,
        num_layers=2,
        max_model_len=512,
        block_size=128,
        resident_token_pool=resident_pool,
        device=torch.device("cpu"),
        pin_memory=False,
        packed_c8=packed_c8,
    )
    store = DSAHotDRAMStore(
        usable_blocks=8,
        storage_rows=resident_pool.storage_rows,
        max_logical_blocks=runtime.max_logical_blocks,
        device=torch.device("cpu"),
        arena_factory=lambda shape, dtype, capacity, device: torch.zeros(
            (capacity, *shape),
            dtype=dtype,
            device=device,
        ),
    )
    runtime.bind_dram_store(store)
    return resident_pool, runtime, store


def test_lidu_scratch_is_shared_but_cache_slots_remain_per_layer() -> None:
    resident_pool, runtime, _ = _make_runtime()

    first = runtime.get_lidu_outputs(num_reqs=1)
    second = runtime.get_lidu_outputs(num_reqs=1)

    assert first is second
    assert first.topk_index.data_ptr() == runtime._lidu_topk_index.data_ptr()
    assert first.topk_slots.data_ptr() == runtime._lidu_topk_slots.data_ptr()
    assert resident_pool.get_cache_slots(0).data_ptr() != resident_pool.get_cache_slots(1).data_ptr()


def test_a5_selection_scratch_is_allocated_only_for_packed_c8() -> None:
    _, bf16_runtime, _ = _make_runtime()
    _, c8_runtime, _ = _make_runtime(packed_c8=True)

    assert bf16_runtime._a5_attention_slots is None
    assert bf16_runtime._a5_resident_seq_lengths is None
    assert c8_runtime._a5_attention_slots is not None
    assert c8_runtime._a5_attention_slots.shape == (1, 1, 2176)
    assert c8_runtime._a5_resident_seq_lengths is not None
    assert c8_runtime._a5_resident_seq_lengths.shape == (1,)


def test_candidate_lengths_keep_dense_history_and_exclude_sparse_tail() -> None:
    resident_pool, runtime, _ = _make_runtime(max_num_reqs=2)
    state = DSAInputBatchCacheLayout(
        max_num_reqs=2,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    runtime.active_num_reqs = 2
    runtime._tokens_after_schedule[:2] = [4096, 6273]
    state.row_modes_cpu[:2] = [1, 2]

    runtime._refresh_candidate_lens(state)

    assert state.candidate_lens_cpu[:2].tolist() == [4096, 6272]


def test_sparse_candidate_lengths_preserve_exact_boundary_tail_block() -> None:
    resident_pool, runtime, _ = _make_runtime(max_num_reqs=3)
    state = DSAInputBatchCacheLayout(
        max_num_reqs=3,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    runtime.active_num_reqs = 3
    runtime._tokens_after_schedule[:3] = [6145, 6272, 6273]
    state.row_modes_cpu[:3] = [2, 2, 2]

    runtime._refresh_candidate_lens(state)

    assert state.candidate_lens_cpu[:3].tolist() == [6144, 6144, 6272]


def test_a5_selection_chain_reuses_preallocated_outputs(monkeypatch) -> None:
    resident_pool, runtime, store = _make_runtime(
        max_num_reqs=2,
        packed_c8=True,
    )
    resident_cache = torch.empty(4, 128, 1, 656, dtype=torch.int8)
    store.add_packed_layer(
        layer_id=0,
        resident_packed_cache=resident_cache,
    )
    indexer_cache = (
        torch.empty(8, 128, 1, 128, dtype=torch.int8),
        torch.empty(8, 128, 1, 1, dtype=torch.float32),
    )
    captured: dict[str, torch.Tensor] = {}

    def _fake_fused_lidu(*, outputs, attention_slots, resident_seq_lengths, **kwargs) -> None:
        captured["candidate_lens"] = kwargs["candidate_lens"]
        outputs.topk_index.fill_(7)
        outputs.topk_slots.fill_(9)
        outputs.miss_count.copy_(torch.tensor([0, 3], dtype=torch.int32))
        attention_slots.fill_(11)
        resident_seq_lengths.copy_(torch.tensor([4096, 4224], dtype=torch.int32))
        captured["copy_src_ids"] = outputs.topk_index
        captured["copy_dst_slots"] = outputs.topk_slots
        captured["attention_slots"] = attention_slots
        captured["resident_seq_lengths"] = resident_seq_lengths

    def _fake_scatter(*, source_token_ids, destination_slots, copy_counts, **kwargs) -> None:
        captured["scatter_source_token_ids"] = source_token_ids
        captured["scatter_destination_slots"] = destination_slots
        captured["scatter_copy_counts"] = copy_counts

    monkeypatch.setattr(runtime_module, "a5_lightning_indexer_decode_update_c8", _fake_fused_lidu)
    monkeypatch.setattr(
        runtime_module,
        "a5_kvcache_scatter_copy_c8",
        _fake_scatter,
    )

    context = DSALayerOffloadContext(
        layer_id=0,
        indexer_cache=indexer_cache,
        runtime=runtime,
        packed_c8=True,
    )
    candidate_lens = torch.tensor([4096, 6272], dtype=torch.int32)
    selection = context.execute_decode_selection(
        query=torch.empty(2, 32, 128),
        weights=torch.empty(2, 32),
        row_modes=torch.tensor([1, 2], dtype=torch.int32),
        resident_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([4096, 6273], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([1, 2], dtype=torch.int32),
        indexer_block_table=torch.zeros(2, 64, dtype=torch.int32),
        resident_cache=(resident_cache,),
        resident_block_table=torch.zeros(2, 64, dtype=torch.int32),
        dram_block_table=torch.zeros(2, 64, dtype=torch.int32),
        candidate_lens=candidate_lens,
        query_dequant_scale=torch.ones(2, 32),
        query_shape=(2, 32, 128),
    )

    assert captured["candidate_lens"].data_ptr() == candidate_lens.data_ptr()
    assert captured["copy_src_ids"].data_ptr() == (runtime._lidu_topk_index.data_ptr())
    assert captured["scatter_source_token_ids"].data_ptr() == (runtime._lidu_topk_index.data_ptr())
    assert captured["scatter_destination_slots"].data_ptr() == (runtime._lidu_topk_slots.data_ptr())
    assert captured["scatter_copy_counts"].data_ptr() == (runtime._lidu_miss_count.data_ptr())
    assert runtime._a5_attention_slots is not None
    assert selection.sparse_indices.data_ptr() == (runtime._a5_attention_slots.data_ptr())
    assert selection.sparse_indices.tolist() == [[[11] * 2176]] * 2
    assert runtime._a5_resident_seq_lengths is not None
    assert selection.resident_seq_lengths is not None
    assert selection.resident_seq_lengths.data_ptr() == (runtime._a5_resident_seq_lengths.data_ptr())
    assert selection.resident_seq_lengths.tolist() == [4096, 4224]


def test_dump_plan_is_compact_and_idempotent() -> None:
    resident_pool, runtime, store = _make_runtime()
    state = DSAInputBatchCacheLayout(
        max_num_reqs=1,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        block_table=[_ResidentBlockTable()],
    )
    state.refresh(
        input_batch=input_batch,
        projection=DSARequestCacheLayoutProjection(
            request_ids=("req-0",),
            stages=(int(DSARequestCacheStage.PREFILL),),
            target_resident_budget_tokens=(256,),
            sparse_budget_tokens=(0,),
            resident_valid_tokens=(-1,),
            resident_block_table_replacements=(),
        ),
    )
    positions = torch.arange(257, dtype=torch.int64)
    req_indices = torch.zeros(257, dtype=torch.int64)
    scheduled = np.array([257], dtype=np.int32)

    returned_positions = runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=scheduled,
        req_indices=req_indices,
        positions=positions,
        num_reqs=1,
        num_tokens=257,
        resident_group_id=0,
    )

    assert returned_positions.data_ptr() == positions.data_ptr()
    assert runtime.dump_job_count == 2
    assert runtime.dump_src_block_ids.np[:2].tolist() == [10, 11]
    assert store.logical_block_table[0, :2].tolist() != [0, 0]

    runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=scheduled,
        req_indices=req_indices,
        positions=positions,
        num_reqs=1,
        num_tokens=257,
        resident_group_id=0,
    )
    assert runtime.dump_job_count == 0


def test_consecutive_prefill_chunks_dump_only_newly_completed_blocks() -> None:
    resident_pool, runtime, store = _make_runtime()
    state = DSAInputBatchCacheLayout(
        max_num_reqs=1,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        block_table=[_ResidentBlockTable()],
    )
    state.refresh(
        input_batch=input_batch,
        projection=DSARequestCacheLayoutProjection(
            request_ids=("req-0",),
            stages=(int(DSARequestCacheStage.PREFILL),),
            target_resident_budget_tokens=(256,),
            sparse_budget_tokens=(0,),
            resident_valid_tokens=(-1,),
            resident_block_table_replacements=(),
        ),
    )

    runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=np.array([257], dtype=np.int32),
        req_indices=torch.zeros(257, dtype=torch.int64),
        positions=torch.arange(257, dtype=torch.int64),
        num_reqs=1,
        num_tokens=257,
        resident_group_id=0,
    )
    assert runtime.dump_job_count == 2
    first_two_dram_blocks = store.logical_block_table[0, :2].copy()

    input_batch.num_computed_tokens_cpu[0] = 257
    runtime.prepare_forward(
        input_batch=input_batch,
        state=state,
        num_scheduled_tokens=np.array([128], dtype=np.int32),
        req_indices=torch.zeros(128, dtype=torch.int64),
        positions=torch.arange(257, 385, dtype=torch.int64),
        num_reqs=1,
        num_tokens=128,
        resident_group_id=0,
    )

    assert runtime.dump_job_count == 1
    assert runtime.dump_src_block_ids.np[0] == 12
    assert store.logical_block_table[0, :2].tolist() == (first_two_dram_blocks.tolist())
    assert store.logical_block_table[0, 2] != 0


def test_enter_rejects_missing_dram_source_blocks() -> None:
    resident_pool, runtime, _ = _make_runtime()
    state = DSAInputBatchCacheLayout(
        max_num_reqs=1,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_pool,
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        num_computed_tokens_cpu=np.array([256], dtype=np.int32),
        block_table=[_ResidentBlockTable()],
    )
    state.refresh(
        input_batch=input_batch,
        projection=DSARequestCacheLayoutProjection(
            request_ids=("req-0",),
            stages=(int(DSARequestCacheStage.ENTER_SPARSE_DECODE),),
            target_resident_budget_tokens=(256,),
            sparse_budget_tokens=(256,),
            resident_valid_tokens=(257,),
            resident_block_table_replacements=(
                DSAResidentBlockTableReplacement(
                    request_id="req-0",
                    block_ids=(10, 11, 12),
                ),
            ),
        ),
    )

    try:
        runtime.prepare_forward(
            input_batch=input_batch,
            state=state,
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            req_indices=torch.zeros(1, dtype=torch.int64),
            positions=torch.tensor([256], dtype=torch.int64),
            num_reqs=1,
            num_tokens=1,
            resident_group_id=0,
        )
    except RuntimeError as error:
        assert "incomplete DRAM block table" in str(error)
        assert "first_missing_logical_block=0" in str(error)
    else:
        raise AssertionError("ENTER must reject a null DRAM source mapping")


def test_graph_execution_view_pads_dump_jobs_with_noop_rows() -> None:
    _, runtime, _ = _make_runtime(max_num_reqs=4)
    runtime.active_num_reqs = 2
    runtime.dump_job_count = 1
    runtime.dump_src_block_ids.np[0] = 7
    runtime.dump_dst_block_ids.np[0] = 9

    execution_rows = runtime.prepare_execution_view(
        active_num_reqs=2,
        graph_row_count=4,
    )

    assert execution_rows == 4
    assert runtime.execution_num_reqs == 4
    assert runtime.dump_launch_count == 4
    assert runtime.dump_src_block_ids.gpu[:4].tolist() == [7, 0, 0, 0]
    assert runtime.dump_dst_block_ids.gpu[:4].tolist() == [9, -1, -1, -1]


def test_eager_execution_view_keeps_compact_dump_jobs() -> None:
    _, runtime, _ = _make_runtime(max_num_reqs=4)
    runtime.active_num_reqs = 2
    runtime.dump_job_count = 1
    runtime.dump_src_block_ids.np[0] = 7
    runtime.dump_dst_block_ids.np[0] = 9

    execution_rows = runtime.prepare_execution_view(
        active_num_reqs=2,
        graph_row_count=None,
    )

    assert execution_rows == 2
    assert runtime.execution_num_reqs == 2
    assert runtime.dump_launch_count == 1
    assert runtime.dump_src_block_ids.gpu[:1].tolist() == [7]
    assert runtime.dump_dst_block_ids.gpu[:1].tolist() == [9]


def test_graph_capture_runtime_can_be_reused_for_multiple_sizes() -> None:
    _, runtime, _ = _make_runtime(max_num_reqs=4)

    for row_count in (4, 2, 1):
        runtime.prepare_graph_capture(row_count=row_count)

        assert runtime.graph_capture_row_count == row_count
        assert runtime.active_num_reqs == row_count
        assert runtime.execution_num_reqs == row_count
        assert runtime.dump_launch_count == row_count
        assert runtime.active_dram_block_table.gpu[:row_count].eq(0).all()
        assert runtime.dump_dst_block_ids.gpu[:row_count].eq(-1).all()

        runtime.restore_after_graph_capture()

        assert runtime.graph_capture_row_count == 0
        assert runtime.active_num_reqs == 0
        assert runtime.execution_num_reqs == 0
        assert runtime.dump_job_count == 0
        assert runtime.dump_launch_count == 0
