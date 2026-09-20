from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("memfabric_hybrid")

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (  # noqa: E402
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.nano_topk_slots import (  # noqa: E402
    NanoTopkSlotAllocator,
    nano_pool_capacity,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    FSA_EXTERNAL_PLAN_READY_MARKER,
    FSA_PAIRED_SELECTION_COPY_MARKER,
    FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
    SparseKVOffloadManager,
    _nano_finalized_block_slots,
)


def _make_sparse_kv_ops():
    return SimpleNamespace(
        sparse_kv_warmup_lru_resident_threads=MagicMock(return_value=8),
        sparse_kv_lru_resident_compact_with_plan_stable_rows=MagicMock(),
        sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows=MagicMock(),
    )


def _make_plan_manager():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.tp_rank = 0
    manager.topk = 4
    manager.topk_buffer_size = 8
    manager.max_model_len = 64
    manager.max_num_topk_rows = 4
    manager.lru_workspace_threads = 2
    manager.layer_name_to_offload_id = {
        "layer.0": 0,
        "layer.1": 1,
        "layer.2": 2,
        "layer.3": 3,
        "layer.4": 4,
    }
    manager.lru_req_ids_cpu = torch.empty(4, dtype=torch.int64)
    manager.lru_topk_indices_cpu = torch.empty((4, 4), dtype=torch.int32)
    manager.lru_stable_prefix_lens_cpu = torch.empty(4, dtype=torch.int32)
    manager.lru_visible_seq_lens_cpu = torch.empty(4, dtype=torch.int32)
    manager.lru_physical_row_workspace = torch.arange(12, dtype=torch.int32)
    manager.fused_plan_metadata_npu = torch.zeros(5, dtype=torch.int32)
    manager.fused_plan_status_npu = manager.fused_plan_metadata_npu[:1]
    manager.fused_plan_current_linear_slots_npu = manager.fused_plan_metadata_npu[1:]
    manager.current_kv_by_layer = {}
    manager.fused_overlap_membership_map = None
    manager.fused_overlap_membership_map_rows = 0
    manager.fused_overlap_plan_owner_layer_id = None
    manager.fused_overlap_plan_topk = None
    manager.fused_overlap_plan_num_tokens = 0
    manager.fused_overlap_plan_membership_map = None
    pointer_names = (
        "lru_req_ids_ptr",
        "lru_topk_indices_ptr",
        "lru_stable_prefix_lens_ptr",
        "lru_visible_seq_lens_ptr",
        "lru_current_slots_ptr",
        "lru_token_mark_workspace_ptr",
        "lru_token_pos_workspace_ptr",
        "lru_slot_workspace_ptr",
        "lru_miss_position_workspace_ptr",
        "lru_epochs_ptr",
        "lru_physical_row_workspace_ptr",
    )
    for pointer, name in enumerate(pointer_names, start=101):
        setattr(manager, name, pointer)
    manager.lru_last_req_ids_ptrs = [201 + layer for layer in range(5)]
    manager.lru_slot_to_token_ptrs = [211 + layer for layer in range(5)]
    manager.lru_slots_ptrs = [221 + layer for layer in range(5)]
    manager.lru_miss_count_ptrs = [231 + layer for layer in range(5)]
    manager.lru_miss_tokens_ptrs = [241 + layer for layer in range(5)]
    manager.lru_miss_slots_ptrs = [251 + layer for layer in range(5)]
    manager.tp_group = MagicMock()
    return manager


def test_external_lru_planner_thread_warmup_runs_on_tp0():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.tp_rank = 0
    manager.lru_workspace_threads = 8
    sparse_kv_ops = _make_sparse_kv_ops()

    with patch.object(manager_module, "_sparse_kv_ops", return_value=sparse_kv_ops):
        assert manager._warmup_external_lru_planner_threads() == 8

    sparse_kv_ops.sparse_kv_warmup_lru_resident_threads.assert_called_once_with(8)


@pytest.mark.parametrize(
    ("use_fused_overlap", "tp_rank"),
    [(False, 0), (True, 1)],
)
def test_external_lru_planner_thread_warmup_is_gated(
    use_fused_overlap,
    tp_rank,
):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = use_fused_overlap
    manager.tp_rank = tp_rank
    manager.lru_workspace_threads = 8
    sparse_kv_ops = _make_sparse_kv_ops()

    with patch.object(manager_module, "_sparse_kv_ops", return_value=sparse_kv_ops):
        assert manager._warmup_external_lru_planner_threads() == 0

    sparse_kv_ops.sparse_kv_warmup_lru_resident_threads.assert_not_called()


def test_mapped_membership_allocation_initializes_external_plan_control():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.topk = 2048
    manager.tp_rank = 0
    manager.tp_group = MagicMock()
    real_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_zeros(*args, **kwargs)

    with (
        patch.object(manager_module.torch, "zeros", side_effect=cpu_zeros),
        patch.object(
            manager_module.offload,
            "empty",
            side_effect=lambda shape, dtype, pin_memory: torch.empty(shape, dtype=dtype),
        ),
    ):
        membership = manager.allocate_fused_overlap_membership_map(3)
        membership_again = manager.allocate_fused_overlap_membership_map(3)

    assert membership.shape == (3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT)
    assert membership_again.data_ptr() == membership.data_ptr()
    assert membership.dtype == torch.int16
    control = membership[
        :,
        FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
        + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
    ]
    assert control[:, 0].tolist() == [-1, -1, -1]
    assert control[:, 1].tolist() == [FSA_EXTERNAL_PLAN_READY_MARKER] * 3
    assert control[:, 2].tolist() == [2048] * 3
    assert control[:, 3].tolist() == [14336] * 3
    assert control[:, 7].tolist() == [FSA_PAIRED_SELECTION_COPY_MARKER] * 3
    manager.tp_group.broadcast.assert_called_once()
    manager.tp_group.barrier.assert_called_once_with()


def test_external_lru_plan_is_reused_by_three_skip_layers_and_replanned_at_owner():
    manager = _make_plan_manager()
    sparse_kv_ops = _make_sparse_kv_ops()
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    topk = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    req_ids = torch.tensor([101], dtype=torch.int64)
    stable_prefix_lens = torch.tensor([10], dtype=torch.int32)
    visible_seq_lens = torch.tensor([11], dtype=torch.int32)

    common_args = dict(
        num_tokens=1,
        topk_indices_npu=topk,
        req_ids_npu=req_ids,
        stable_prefix_lens_npu=stable_prefix_lens,
        visible_seq_lens_npu=visible_seq_lens,
        selection_membership_map=membership,
        capturing=False,
    )
    with patch.object(manager_module, "_sparse_kv_ops", return_value=sparse_kv_ops):
        assert manager.prepare_fused_overlap_external_plan(layer_name="layer.0", skip_topk=False, **common_args)
        for layer_id in range(1, 4):
            assert manager.prepare_fused_overlap_external_plan(
                layer_name=f"layer.{layer_id}", skip_topk=True, **common_args
            )

        planner = sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
        assert planner.call_count == 1
        manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)
        assert manager.fused_overlap_plan_owner_layer_id == 0

        assert manager.prepare_fused_overlap_external_plan(layer_name="layer.4", skip_topk=False, **common_args)
        assert planner.call_count == 2
        assert planner.call_args_list[-1].args[4] == manager.lru_slot_to_token_ptrs[4]
        assert manager.fused_overlap_plan_owner_layer_id == 4


def test_eager_external_plan_copies_inputs_and_preserves_cpp_argument_order():
    manager = _make_plan_manager()
    sparse_kv_ops = _make_sparse_kv_ops()
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    topk = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int32)
    req_ids = torch.tensor([101, 202], dtype=torch.int64)
    stable_prefix_lens = torch.tensor([10, 20], dtype=torch.int32)
    visible_seq_lens = torch.tensor([11, 21], dtype=torch.int32)

    with patch.object(manager_module, "_sparse_kv_ops", return_value=sparse_kv_ops):
        assert manager.prepare_fused_overlap_external_plan(
            layer_name="layer.0",
            num_tokens=2,
            topk_indices_npu=topk,
            req_ids_npu=req_ids,
            stable_prefix_lens_npu=stable_prefix_lens,
            visible_seq_lens_npu=visible_seq_lens,
            selection_membership_map=membership,
            capturing=False,
        )

    torch.testing.assert_close(manager.lru_topk_indices_cpu[:2], topk)
    torch.testing.assert_close(manager.lru_req_ids_cpu[:2], req_ids)
    torch.testing.assert_close(manager.lru_stable_prefix_lens_cpu[:2], stable_prefix_lens)
    torch.testing.assert_close(manager.lru_visible_seq_lens_cpu[:2], visible_seq_lens)
    plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - manager.topk
    planner = sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
    planner.assert_called_once_with(
        manager.lru_req_ids_ptr,
        manager.lru_last_req_ids_ptrs[0],
        manager.lru_topk_indices_ptr,
        manager.lru_stable_prefix_lens_ptr,
        manager.lru_slot_to_token_ptrs[0],
        manager.lru_slots_ptrs[0],
        manager.lru_current_slots_ptr,
        manager.lru_miss_count_ptrs[0],
        manager.lru_miss_tokens_ptrs[0],
        manager.lru_miss_slots_ptrs[0],
        manager.lru_token_mark_workspace_ptr,
        manager.lru_token_pos_workspace_ptr,
        manager.lru_slot_workspace_ptr,
        manager.lru_miss_position_workspace_ptr,
        manager.lru_epochs_ptr,
        manager.lru_physical_row_workspace_ptr,
        manager.max_num_topk_rows,
        membership[:, plan_start:].data_ptr(),
        membership.stride(0),
        2,
        manager.topk,
        manager.topk_buffer_size,
        manager.max_model_len,
        manager.lru_workspace_threads,
        manager.lru_workspace_threads,
        manager.lru_visible_seq_lens_ptr,
    )
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)


def test_capture_external_plan_and_current_kv_use_separate_side_streams():
    manager = _make_plan_manager()
    sparse_kv_ops = _make_sparse_kv_ops()
    manager.current_kv_save_stream = MagicMock()
    manager.fused_plan_stream = MagicMock()
    current_stream = MagicMock()
    input_event = object()
    current_stream.record_event.return_value = input_event
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    topk = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    req_ids = torch.tensor([101], dtype=torch.int64)
    stable_prefix_lens = torch.tensor([10], dtype=torch.int32)
    visible_seq_lens = torch.tensor([11], dtype=torch.int32)
    manager._offload_new_kv_on_current_stream = MagicMock()

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
        patch.object(manager_module, "_sparse_kv_ops", return_value=sparse_kv_ops),
    ):
        manager.offload_new_kv(
            layer_name="layer.0",
            slot_mapping=torch.tensor([0]),
            k_cache_cpu=torch.empty(1),
            v_cache_cpu=torch.empty(1),
            k_cache_npu=None,
            v_cache_npu=None,
            k=torch.empty(1),
            v=torch.empty(1),
            capturing=True,
        )
        assert manager.prepare_fused_overlap_external_plan(
            layer_name="layer.0",
            num_tokens=1,
            topk_indices_npu=topk,
            req_ids_npu=req_ids,
            stable_prefix_lens_npu=stable_prefix_lens,
            visible_seq_lens_npu=visible_seq_lens,
            selection_membership_map=membership,
            capturing=True,
        )
        manager.inject_current_kv_into_selection = MagicMock()
        manager.wait_for_current_kv_writeback(capturing=True)

    assert current_stream.record_event.call_count == 2
    manager.current_kv_save_stream.wait_event.assert_called_once_with(input_event)
    manager.fused_plan_stream.wait_event.assert_called_once_with(input_event)
    sparse_kv_ops.sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows.assert_called_once()
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)
    current_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)
    assert manager.current_kv_by_layer[0][0].numel() == 1
    assert manager.current_kv_by_layer[0][1].numel() == 1


def test_capture_without_fused_overlap_stays_on_current_stream():
    manager = _make_plan_manager()
    manager.use_fused_overlap = False
    manager.current_kv_save_stream = MagicMock()
    manager.fused_plan_stream = MagicMock()
    current_stream = MagicMock()
    manager._offload_new_kv_on_current_stream = MagicMock()

    with patch.object(
        manager_module.torch_npu.npu,
        "current_stream",
        return_value=current_stream,
    ):
        manager.offload_new_kv(
            layer_name="layer.0",
            slot_mapping=torch.tensor([0]),
            k_cache_cpu=torch.empty(1),
            v_cache_cpu=torch.empty(1),
            k_cache_npu=None,
            v_cache_npu=None,
            k=torch.empty(1),
            v=torch.empty(1),
            capturing=True,
        )
        manager.wait_for_current_kv_writeback(capturing=True)

    manager._offload_new_kv_on_current_stream.assert_called_once()
    current_stream.record_event.assert_not_called()
    current_stream.wait_stream.assert_not_called()
    manager.current_kv_save_stream.wait_event.assert_not_called()
    manager.fused_plan_stream.wait_event.assert_not_called()


def test_nano_d2h_flushes_only_completed_full_blocks():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 8
    manager.token_size_bytes_v = 4
    manager.topk_buffers_k = [torch.zeros((512, 2), dtype=torch.float32)]
    manager.topk_buffers_v = [torch.zeros((512, 1), dtype=torch.float32)]
    manager.k_caches_cpu = [torch.zeros((512, 2), dtype=torch.float32)]
    manager.v_caches_cpu = [torch.zeros((512, 1), dtype=torch.float32)]
    manager.nano_d2h_src_ptrs_npu = torch.empty(4, dtype=torch.int64)
    manager.nano_d2h_dst_ptrs_npu = torch.empty(4, dtype=torch.int64)
    manager.nano_d2h_lengths_npu = torch.empty(4, dtype=torch.int32)
    manager.nano_d2h_size_npu = torch.empty(1, dtype=torch.int32)
    sparse_copy = MagicMock(return_value=0)

    with patch.object(
        manager_module,
        "offload",
        SimpleNamespace(sparse_copy=sparse_copy),
        create=True,
    ):
        manager._offload_nano_full_blocks_on_current_stream(
            layer_id=0,
            source_slots=torch.tensor([128, 256], dtype=torch.int64),
            destination_slots=torch.tensor([256, 384], dtype=torch.int64),
            active=torch.tensor([True, False]),
        )

    k_npu = manager.topk_buffers_k[0]
    v_npu = manager.topk_buffers_v[0]
    k_cpu = manager.k_caches_cpu[0]
    v_cpu = manager.v_caches_cpu[0]
    assert manager.nano_d2h_src_ptrs_npu[:2].tolist() == [
        k_npu.data_ptr() + 128 * manager.token_size_bytes_k,
        k_npu.data_ptr() + 256 * manager.token_size_bytes_k,
    ]
    assert manager.nano_d2h_src_ptrs_npu[2:4].tolist() == [
        v_npu.data_ptr() + 128 * manager.token_size_bytes_v,
        v_npu.data_ptr() + 256 * manager.token_size_bytes_v,
    ]
    assert manager.nano_d2h_dst_ptrs_npu[:2].tolist() == [
        k_cpu.data_ptr() + 256 * manager.token_size_bytes_k,
        k_cpu.data_ptr() + 384 * manager.token_size_bytes_k,
    ]
    assert manager.nano_d2h_dst_ptrs_npu[2:4].tolist() == [
        v_cpu.data_ptr() + 256 * manager.token_size_bytes_v,
        v_cpu.data_ptr() + 384 * manager.token_size_bytes_v,
    ]
    assert manager.nano_d2h_lengths_npu[:4].tolist() == [1024, 0, 512, 0]
    assert manager.nano_d2h_size_npu.item() == 4
    sparse_copy.assert_called_once()


def test_nano_finalize_uses_accepted_mtp_rows_at_block_boundary():
    source_slots, destination_slots, active, committed_lens = _nano_finalized_block_slots(
        seq_lens=torch.full((5,), 136, dtype=torch.int32),
        query_ends=torch.tensor([16, 32, 48, 64, 80], dtype=torch.int32),
        rejected_rows=torch.tensor([16, 11, 8, 6, 0], dtype=torch.int32),
        pools=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        active=torch.ones(5, dtype=torch.bool),
        block_table=torch.tensor([[10], [11], [12], [13], [14]], dtype=torch.int32),
        stride_blocks=34,
        hot_tokens=4096,
    )

    assert committed_lens.tolist() == [120, 125, 128, 130, 136]
    assert active.tolist() == [False, False, True, True, True]
    assert source_slots[2:].tolist() == [12800, 17152, 21504]
    assert destination_slots[2:].tolist() == [1536, 1664, 1792]


def test_nano_finalize_does_not_advance_on_invalid_rejection_count():
    _, _, active, committed_lens = _nano_finalized_block_slots(
        seq_lens=torch.tensor([136], dtype=torch.int32),
        query_ends=torch.tensor([16], dtype=torch.int32),
        rejected_rows=torch.tensor([17], dtype=torch.int32),
        pools=torch.tensor([0], dtype=torch.int32),
        active=torch.tensor([True]),
        block_table=torch.tensor([[10]], dtype=torch.int32),
        stride_blocks=34,
        hot_tokens=4096,
    )

    assert committed_lens.tolist() == [120]
    assert active.tolist() == [False]


def test_nano_finalize_truncates_residency_but_does_not_invent_mtp_writes():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_nano = True
    manager.max_num_reqs = 2
    manager.nano_pool_capacity = nano_pool_capacity(manager.max_num_reqs)
    manager.nano_plan_capacity = 2
    manager.topk_buffer_size = 4096
    manager.nano_rejected_rows_npu = torch.zeros(2, dtype=torch.int32)
    manager.nano_confirmed_generations = torch.full((2,), -1, dtype=torch.int64)
    manager.nano_confirmed_src_slots = torch.zeros(2, dtype=torch.int64)
    manager.nano_confirmed_dst_slots = torch.zeros(2, dtype=torch.int64)
    manager.nano_confirmed_active = torch.zeros(2, dtype=torch.bool)
    generations = torch.full((4,), -1, dtype=torch.int64)
    resident_lens = torch.zeros(4, dtype=torch.int32)
    generations[:2] = torch.tensor([7, 9])
    resident_lens[:2] = 136
    draft_generations = torch.tensor([7, -1, -1, -1], dtype=torch.int64)
    draft_lens = torch.tensor([120, 0, 0, 0], dtype=torch.int32)
    manager.nano_tail_states = {
        "layer.0": (generations, resident_lens),
        "draft": (draft_generations, draft_lens),
    }
    metadata = SimpleNamespace(
        nano_enabled=True,
        nano_seq_lens=torch.tensor([136, 136], dtype=torch.int32),
        nano_query_ends=torch.tensor([16, 32], dtype=torch.int32),
        nano_pool_entries=torch.tensor([0, 1], dtype=torch.int32),
        nano_generations=torch.tensor([7, 9], dtype=torch.int64),
        nano_active=torch.tensor([True, True]),
        nano_source_block_table=torch.tensor([[10], [11]], dtype=torch.int32),
        nano_flush_src_slots=torch.empty(2, dtype=torch.int64),
        nano_flush_dst_slots=torch.empty(2, dtype=torch.int64),
        nano_flush_active=torch.empty(2, dtype=torch.bool),
    )

    manager.finalize_nano_full_blocks(
        metadata,
        torch.tensor([11, 8], dtype=torch.int32),
    )

    assert resident_lens[:2].tolist() == [125, 128]
    assert generations[:2].tolist() == [7, 9]
    assert draft_generations[:2].tolist() == [7, -1]
    assert draft_lens[:2].tolist() == [120, 0]
    assert manager.nano_confirmed_generations.tolist() == [7, 9]
    assert manager.nano_confirmed_active.tolist() == [False, True]
    assert manager.nano_confirmed_src_slots.tolist() == [4224, 8448]
    assert manager.nano_confirmed_dst_slots.tolist() == [1280, 1408]


def test_nano_mtp_gathers_confirmed_plan_by_pool_and_generation():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.max_num_reqs = 3
    manager.nano_pool_capacity = nano_pool_capacity(manager.max_num_reqs)
    manager.nano_plan_capacity = 3
    manager.nano_confirmed_generations = torch.tensor([10, 20, 30], dtype=torch.int64)
    manager.nano_confirmed_src_slots = torch.tensor([100, 200, 300], dtype=torch.int64)
    manager.nano_confirmed_dst_slots = torch.tensor([1000, 2000, 3000], dtype=torch.int64)
    manager.nano_confirmed_active = torch.tensor([True, False, True])

    source, destination, active = manager.gather_nano_confirmed_plan(
        pools=torch.tensor([2, 0, 1], dtype=torch.int32),
        generations=torch.tensor([30, 10, 99], dtype=torch.int64),
        active=torch.tensor([True, True, True]),
    )

    assert source.tolist() == [300, 100, 200]
    assert destination.tolist() == [3000, 1000, 2000]
    assert active.tolist() == [True, True, False]


@pytest.mark.parametrize("max_real_reqs", [1, 2])
def test_nano_confirmation_dummy_slot_cannot_alias_real_pool(max_real_reqs):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_nano = True
    manager.max_num_reqs = max_real_reqs
    manager.nano_pool_capacity = nano_pool_capacity(manager.max_num_reqs)
    manager.nano_plan_capacity = 8
    manager.topk_buffer_size = 4096
    manager.nano_rejected_rows_npu = torch.zeros(2, dtype=torch.int32)
    manager.nano_confirmed_generations = torch.full((8,), -1, dtype=torch.int64)
    manager.nano_confirmed_src_slots = torch.zeros(8, dtype=torch.int64)
    manager.nano_confirmed_dst_slots = torch.zeros(8, dtype=torch.int64)
    manager.nano_confirmed_active = torch.zeros(8, dtype=torch.bool)
    manager.nano_tail_states = {}
    metadata = SimpleNamespace(
        nano_enabled=True,
        nano_seq_lens=torch.tensor([128, 1], dtype=torch.int32),
        nano_query_ends=torch.tensor([1, 2], dtype=torch.int32),
        nano_pool_entries=torch.tensor([max_real_reqs - 1, 4], dtype=torch.int32),
        nano_generations=torch.tensor([7, -1], dtype=torch.int64),
        nano_active=torch.tensor([True, False]),
        nano_source_block_table=torch.tensor([[3], [-1]], dtype=torch.int32),
        nano_flush_src_slots=torch.empty(2, dtype=torch.int64),
        nano_flush_dst_slots=torch.empty(2, dtype=torch.int64),
        nano_flush_active=torch.empty(2, dtype=torch.bool),
    )

    manager.finalize_nano_full_blocks(metadata, None)

    assert manager.nano_confirmed_generations[max_real_reqs - 1].item() == 7
    assert manager.nano_confirmed_active[max_real_reqs - 1].item()
    assert manager.nano_confirmed_generations[4].item() == -1


def _make_nano_confirmation_manager(max_num_reqs):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_nano = True
    manager.max_num_reqs = max_num_reqs
    manager.nano_pool_capacity = nano_pool_capacity(max_num_reqs)
    manager.nano_plan_capacity = 2 * manager.nano_pool_capacity
    manager.topk_buffer_size = 4096
    manager.nano_rejected_rows_npu = torch.zeros(manager.nano_pool_capacity, dtype=torch.int32)
    manager.nano_confirmed_generations = torch.full((manager.nano_plan_capacity,), -1, dtype=torch.int64)
    manager.nano_confirmed_src_slots = torch.zeros(manager.nano_plan_capacity, dtype=torch.int64)
    manager.nano_confirmed_dst_slots = torch.zeros_like(manager.nano_confirmed_src_slots)
    manager.nano_confirmed_active = torch.zeros(manager.nano_plan_capacity, dtype=torch.bool)
    manager.nano_tail_states = {}
    return manager


@pytest.mark.parametrize("max_num_reqs", [2, 4])
@pytest.mark.parametrize("rejected_rows", [0, 2, 3, 4])
def test_nano_mtp_confirmation_survives_pd_slot_turnover(max_num_reqs, rejected_rows):
    manager = _make_nano_confirmation_manager(max_num_reqs)
    allocator = NanoTopkSlotAllocator(manager.nano_pool_capacity)
    # Finished requests return their slots to the queue's tail. The next two
    # real requests therefore use the extra slots, without exceeding concurrency.
    for index in range(max_num_reqs):
        request_id = f"finished-{index}"
        allocator.bind(request_id)
        allocator.release(request_id)
    pools = [allocator.bind("request-a"), allocator.bind("request-b")]
    assert pools == [max_num_reqs, max_num_reqs + 1]

    dummy_pool = manager.nano_pool_capacity
    metadata = SimpleNamespace(
        nano_enabled=True,
        nano_seq_lens=torch.tensor([130, 130, 1], dtype=torch.int32),
        nano_query_ends=torch.tensor([4, 8, 9], dtype=torch.int32),
        nano_pool_entries=torch.tensor([*pools, dummy_pool], dtype=torch.int32),
        nano_generations=torch.tensor([7, 9, -1], dtype=torch.int64),
        nano_active=torch.tensor([True, True, False]),
        nano_source_block_table=torch.tensor([[10], [11], [-1]], dtype=torch.int32),
    )
    manager.finalize_nano_full_blocks(
        metadata,
        torch.tensor([rejected_rows, rejected_rows], dtype=torch.int32),
    )

    expected_active = rejected_rows <= 2  # Previous length 126; the boundary is 128.
    assert manager.nano_confirmed_generations[pools].tolist() == [7, 9]
    assert manager.nano_confirmed_active[pools].tolist() == [expected_active, expected_active]
    assert manager.nano_confirmed_generations[dummy_pool].item() == -1
    assert not manager.nano_confirmed_active[dummy_pool].item()

    # MTP can use a different row order. Gather must follow slot ownership,
    # not the target batch row or the maximum number of concurrent requests.
    order = torch.tensor([1, 0, 2])
    source, destination, active = manager.gather_nano_confirmed_plan(
        metadata.nano_pool_entries[order],
        metadata.nano_generations[order],
        metadata.nano_active[order],
    )
    assert active.tolist() == [expected_active, expected_active, False]
    if expected_active:
        stride = manager.topk_buffer_size + 2 * manager_module.NANO_TAIL_BLOCK_SIZE
        assert source[:2].tolist() == [pool * stride + manager.topk_buffer_size for pool in reversed(pools)]
        assert destination[:2].tolist() == [11 * 128, 10 * 128]

    # Reusing a slot for a new generation must not inherit the previous plan.
    _, _, stale_active = manager.gather_nano_confirmed_plan(
        metadata.nano_pool_entries[:2],
        metadata.nano_generations[:2] + 1,
        metadata.nano_active[:2],
    )
    assert stale_active.tolist() == [False, False]


@pytest.mark.parametrize("pool", [-1, 4, 7])
def test_nano_confirmation_excludes_slots_outside_real_pool(pool):
    manager = _make_nano_confirmation_manager(max_num_reqs=2)
    metadata = SimpleNamespace(
        nano_enabled=True,
        nano_seq_lens=torch.tensor([128], dtype=torch.int32),
        nano_query_ends=torch.tensor([1], dtype=torch.int32),
        nano_pool_entries=torch.tensor([pool], dtype=torch.int32),
        nano_generations=torch.tensor([7], dtype=torch.int64),
        nano_active=torch.tensor([True]),
        nano_source_block_table=torch.tensor([[10]], dtype=torch.int32),
    )
    manager.finalize_nano_full_blocks(metadata, None)
    assert manager.nano_confirmed_generations.eq(-1).all()
    assert not manager.nano_confirmed_active.any()

    # Even a populated entry in the dummy arena must fail the gather bound.
    manager.nano_confirmed_generations.fill_(7)
    manager.nano_confirmed_active.fill_(True)
    _, _, active = manager.gather_nano_confirmed_plan(
        metadata.nano_pool_entries,
        metadata.nano_generations,
        metadata.nano_active,
    )
    assert active.tolist() == [False]


def test_nano_shared_plan_rejects_different_layer_tail_layouts():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.tp_rank = 0
    manager.topk_buffers_k = [
        torch.empty((2, 128, 1, 4), dtype=torch.bfloat16),
        torch.empty((3, 128, 1, 4), dtype=torch.bfloat16),
    ]
    manager.topk_buffers_v = [
        torch.empty((2, 128, 1, 2), dtype=torch.bfloat16),
        torch.empty((3, 128, 1, 2), dtype=torch.bfloat16),
    ]
    manager.k_caches_cpu = [
        torch.empty((4, 128, 1, 4), dtype=torch.bfloat16),
        torch.empty((4, 128, 1, 4), dtype=torch.bfloat16),
    ]
    manager.v_caches_cpu = [
        torch.empty((4, 128, 1, 2), dtype=torch.bfloat16),
        torch.empty((4, 128, 1, 2), dtype=torch.bfloat16),
    ]

    with pytest.raises(ValueError, match="identical device tail layout"):
        manager._validate_nano_shared_block_layout()


def test_nano_d2h_runs_on_side_stream_without_a_per_layer_join():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_nano = True
    manager.tp_rank = 0
    manager.max_num_reqs = 1
    manager.num_layers = 2
    manager._get_offload_layer_id = MagicMock(return_value=1)
    manager.nano_d2h_stream = MagicMock()
    manager.nano_d2h_inputs_ready = [MagicMock(), MagicMock()]
    manager._offload_nano_full_blocks_on_current_stream = MagicMock()
    compute_stream = MagicMock()
    inputs_ready = manager.nano_d2h_inputs_ready[1]
    source_slots = torch.tensor([128, 0, 0], dtype=torch.int64)
    destination_slots = torch.tensor([256, 0, 0], dtype=torch.int64)
    active = torch.tensor([True, False, False])

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=compute_stream),
        patch.object(manager_module.torch_npu.npu, "stream", return_value=nullcontext()),
    ):
        manager.offload_nano_full_blocks("layer.1", source_slots, destination_slots, active)

    manager.nano_d2h_stream.wait_event.assert_called_once_with(inputs_ready)
    manager._offload_nano_full_blocks_on_current_stream.assert_called_once_with(
        1,
        source_slots,
        destination_slots,
        active,
    )
    inputs_ready.record.assert_called_once_with(compute_stream)
    compute_stream.wait_event.assert_not_called()


def test_nano_d2h_join_records_tail_before_wait_without_host_sync():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_nano = True
    manager.tp_rank = 0
    manager.num_layers = 2
    manager.nano_d2h_stream = MagicMock()
    manager.nano_d2h_done = MagicMock()
    compute_stream = MagicMock()
    calls = []
    manager.nano_d2h_stream.wait_stream.side_effect = lambda _: calls.append("fork")
    manager.nano_d2h_done.record.side_effect = lambda _: calls.append("record")
    compute_stream.wait_event.side_effect = lambda _: calls.append("wait")

    with patch.object(manager_module.torch_npu.npu, "current_stream", return_value=compute_stream):
        manager.join_nano_d2h()

    assert calls == ["fork", "record", "wait"]
    compute_stream.wait_event.assert_called_once_with(manager.nano_d2h_done)
    manager.nano_d2h_done.synchronize.assert_not_called()


def test_nano_iteration_orders_tp_consumers_after_d2h_join():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_nano = True
    manager.nano_completion_token = torch.zeros(1, dtype=torch.int32)
    calls = []
    manager.join_nano_d2h = lambda: calls.append("join")
    group = SimpleNamespace(all_reduce=lambda _: calls.append("tp"))
    with (
        patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=2),
        patch.object(manager_module, "get_tp_group", return_value=group),
    ):
        manager.wait_for_nano_host_copies()
    assert calls == ["join", "tp"]


def test_current_kv_injection_uses_planner_linear_slots_and_sentinel():
    manager = _make_plan_manager()
    manager.topk_buffer_size = 4
    manager.current_kv_by_layer[0] = (
        torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
        torch.tensor([[30.0], [40.0]]),
    )
    manager.fused_plan_current_linear_slots_npu[:2].copy_(torch.tensor([6, 3]))
    selection_kv = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    selection_rope = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    def scatter(destination, indices, updates):
        destination.index_copy_(0, indices.reshape(-1).to(torch.int64), updates)

    with patch.object(
        manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=scatter, create=True
    ) as scatter_op:
        manager.inject_current_kv_into_selection(
            layer_name="layer.0",
            num_tokens=2,
            selection_kv_cache=selection_kv,
            selection_k_rope=selection_rope,
            capturing=False,
        )

    torch.testing.assert_close(selection_kv[6], torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(selection_rope[6], torch.tensor([30.0]))
    torch.testing.assert_close(selection_kv[3], torch.tensor([20.0, 21.0]))
    torch.testing.assert_close(selection_rope[3], torch.tensor([40.0]))
    torch.testing.assert_close(selection_kv[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(selection_rope[0], torch.tensor([0.0]))
    assert scatter_op.call_count == 2
