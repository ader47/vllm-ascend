"""Regression tests for SFA KV-offload attention metadata."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.sfa_kv_offload import (  # noqa: E402
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
    _nano_candidate_block_slots,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    FSA_EXTERNAL_PLAN_READY_MARKER,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
)


def _make_boundary_decode_metadata():
    return SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=1,
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        is_prefilling=torch.tensor([True]),
        req_ids_tensor=torch.tensor([7]),
        token_to_req=torch.tensor([0]),
    )


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (None, False),
        (SimpleNamespace(is_kv_consumer=False, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=False), True),
    ],
)
def test_pd_decode_consumer_is_derived_from_kv_role(kv_transfer_config, expected):
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    with patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None):
        builder = AscendSFAKVOffloadMetadataBuilder(
            kv_cache_spec=None,
            layer_names=[],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.is_pd_decode_consumer is expected


@pytest.mark.parametrize(
    ("is_pd_decode_consumer", "expected_decodes", "expected_prefills"),
    [
        (True, 1, 0),
        (False, 0, 1),
    ],
)
def test_boundary_token_classification_depends_on_pd_decode_role(
    is_pd_decode_consumer,
    expected_decodes,
    expected_prefills,
):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = False
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = is_pd_decode_consumer
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, _make_boundary_decode_metadata())

    assert metadata.num_decodes == expected_decodes
    assert metadata.num_prefills == expected_prefills
    assert metadata.num_decode_tokens == expected_decodes
    assert metadata.req_ids_tensor.tolist() == [7]
    assert metadata.token_to_req.tolist() == [0]
    assert AscendSFAKVOffloadImpl._is_decode_only(metadata) is is_pd_decode_consumer


def test_pd_decode_consumer_still_rejects_long_prefill_classification():
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = False
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = True
    metadata = SimpleNamespace()
    common_metadata = _make_boundary_decode_metadata()
    common_metadata.max_query_len = 2
    common_metadata.num_actual_tokens = 2
    common_metadata.query_start_loc_cpu = torch.tensor([0, 2])

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, common_metadata)

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 1
    assert metadata.num_decode_tokens == 0


@pytest.mark.parametrize(("width", "prefills", "eligible"), [(8, 0, True), (1, 1, True), (1, 0, False)])
def test_nano_rejects_unsafe_fallback(width, prefills, eligible):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = builder.is_pd_decode_consumer = True
    builder.decode_threshold = 7
    common = SimpleNamespace(
        nano_eligible=eligible,
        offload_dummy=False,
        max_query_len=width,
        req_ids_tensor=None,
        token_to_req=None,
    )
    with (
        patch("vllm_ascend.attention.sfa_kv_offload.split_decodes_and_prefills", return_value=(1, prefills, width, 0)),
        pytest.raises(ValueError, match="discard unoffloaded partial tails"),
    ):
        builder._populate_offload_metadata(SimpleNamespace(), common)


def test_nano_candidate_metadata_selects_only_scheduled_boundary_crossings():
    source_slots, destination_slots, active = _nano_candidate_block_slots(
        seq_lens=torch.tensor([128, 129, 256], dtype=torch.int32),
        widths=torch.tensor([1, 1, 3], dtype=torch.int32),
        pools=torch.tensor([2, 3, 4], dtype=torch.int32),
        active=torch.tensor([True, True, True]),
        block_table=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
        stride_blocks=34,
        hot_tokens=4096,
    )

    torch.testing.assert_close(active, torch.tensor([True, False, True]))
    assert source_slots.tolist() == [12800, 17152, 21632]
    assert destination_slots[[0, 2]].tolist() == [128, 768]


@pytest.mark.parametrize("base", [0, 128, 8192])
def test_nano_draft_metadata_keeps_full_block_in_tail_and_uses_actual_positions(base):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = True
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            max_model_len=16384,
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        ),
    )
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=8192)),
    ):
        builder._init_nano_metadata_buffers(config, torch.device("cpu"))
    cm = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([base + 130], dtype=torch.int32),
        req_topk_buffer_slots=torch.tensor([1], dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor([11], dtype=torch.int64),
        block_table_tensor=torch.arange(128, dtype=torch.int32).view(1, -1),
        positions=torch.arange(base + 126, base + 130),
        slot_mapping=torch.arange(base + 126, base + 130),
        req_ids_tensor=None,
        token_to_req=None,
        nano_eligible=True,
        nano_draft=True,
        offload_dummy=False,
        max_query_len=4,
        num_reqs=1,
        num_input_tokens=4,
    )
    manager = MagicMock()
    manager.gather_nano_confirmed_plan.return_value = (torch.tensor([base]), torch.tensor([base]), torch.tensor([True]))
    with (
        patch("vllm_ascend.attention.sfa_kv_offload.split_decodes_and_prefills", return_value=(1, 0, 4, 0)),
        patch("vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager", return_value=manager),
    ):
        first = builder._populate_offload_metadata(SimpleNamespace(), cm)
        assert first.nano_prefix_lens.tolist() == [base]
        assert first.nano_flush_active.tolist() == [True]
        saved_table = first.nano_hbm_block_table.clone()
        # Rejection kept only two of step 0's four rows. The actual next
        # position is 128, although the optimistic seq_len says 131.
        cm.query_start_loc = cm.query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
        cm.positions = torch.tensor([base + 128])
        cm.seq_lens = torch.tensor([base + 131], dtype=torch.int32)
        cm.max_query_len = cm.num_input_tokens = 1
        for step in (1, 2):
            md = builder._populate_offload_metadata(SimpleNamespace(), cm, draft_index=step)
            assert md.nano_seq_lens.tolist() == [base + 128 + step]
            assert md.nano_prefix_lens.tolist() == [base]
            assert md.nano_logical_lens.tolist() == [min(base, 8192) + 128 + step]
            assert not md.nano_flush_active[0]
            assert md.nano_device_slots.tolist() == [8448 + 8192 + (base + 127 + step) % 256]
            torch.testing.assert_close(md.nano_hbm_block_table, saved_table)
            cm.positions += 1
        assert first.nano_seq_lens.tolist() == [base + 130]
        assert first.nano_flush_active.tolist() == [True]


def test_nano_partial_tail_stays_resident_until_state_is_discontinuous():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.nano_tail_generation = torch.full((4,), -1, dtype=torch.int64)
    impl.nano_tail_seq_len = torch.zeros(4, dtype=torch.int32)
    impl.nano_tail_start = torch.zeros(4, dtype=torch.int32)
    impl.nano_restore_active = torch.empty(2, dtype=torch.bool)
    metadata = SimpleNamespace(
        nano_pool_entries=torch.tensor([0], dtype=torch.int32),
        nano_query_ends=torch.tensor([1], dtype=torch.int32),
        nano_seq_lens=torch.tensor([101], dtype=torch.int32),
        nano_prefix_lens=torch.tensor([0], dtype=torch.int32),
        nano_generations=torch.tensor([7], dtype=torch.int64),
        nano_active=torch.tensor([True]),
    )

    impl._prepare_nano_tail_state(metadata)
    assert impl.nano_restore_active[0]

    # Each layer records its own write before the next draft step runs.
    impl._update_nano_tail_state(metadata)
    metadata.nano_seq_lens[0] = 102
    impl._prepare_nano_tail_state(metadata)
    assert not impl.nano_restore_active[0]

    metadata.nano_seq_lens[0] = 500
    impl._prepare_nano_tail_state(metadata)
    assert impl.nano_restore_active[0]


def test_nano_each_draft_write_advances_residency_without_moving_tail_base():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.nano_tail_generation = torch.tensor([7, -1], dtype=torch.int64)
    impl.nano_tail_start = torch.zeros(2, dtype=torch.int32)
    impl.nano_tail_seq_len = torch.tensor([126, 0], dtype=torch.int32)
    impl.nano_restore_active = torch.empty(1, dtype=torch.bool)
    md = SimpleNamespace(
        nano_pool_entries=torch.tensor([0], dtype=torch.int32),
        nano_query_ends=torch.tensor([1], dtype=torch.int32),
        nano_seq_lens=torch.tensor([127], dtype=torch.int32),
        nano_prefix_lens=torch.tensor([0], dtype=torch.int32),
        nano_generations=torch.tensor([7], dtype=torch.int64),
        nano_active=torch.tensor([True]),
    )
    for end in (127, 128, 129):
        md.nano_seq_lens.fill_(end)
        impl._prepare_nano_tail_state(md)
        assert not impl.nano_restore_active[0]
        impl._update_nano_tail_state(md)
        assert impl.nano_tail_seq_len[0] == end
        assert impl.nano_tail_start[0] == 0
    # An end watermark alone is not enough if the older page was recycled.
    impl.nano_tail_start[0] = 128
    impl._prepare_nano_tail_state(md)
    assert impl.nano_restore_active[0]
    md.nano_active.fill_(False)
    md.nano_seq_lens.fill_(0)
    impl._update_nano_tail_state(md)
    assert impl.nano_tail_seq_len[0] == 129


def test_nano_short_history_uses_identity_misses_only_when_cold():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.sfa_sparse_topk = 2048
    impl.nano_dense_slots = torch.arange(2048, dtype=torch.int32)
    impl.nano_miss_src = torch.full((4, 2048), -1, dtype=torch.int32)
    impl.nano_miss_dst = impl.nano_miss_src.clone()
    impl.nano_misses = torch.tensor([0, 0, 7, 0], dtype=torch.int32)
    impl.nano_states = torch.tensor([-2, -1, -2, -3], dtype=torch.int32)
    md = SimpleNamespace(
        nano_pool_entries=torch.arange(4),
        nano_prefix_lens=torch.tensor([128, 256, 4096, 128], dtype=torch.int32),
        nano_active=torch.tensor([True, True, True, False]),
    )
    impl._prepare_nano_dense_prefix(md)
    assert impl.nano_misses.tolist() == [128, 0, 7, 0]
    torch.testing.assert_close(impl.nano_miss_src[0, :128], torch.arange(128, dtype=torch.int32))
    torch.testing.assert_close(impl.nano_miss_dst[0, :128], impl.nano_miss_src[0, :128])
    assert (impl.nano_miss_src[2:] == -1).all()


@pytest.mark.parametrize("history", [0, 128, 2048])
def test_nano_reused_topk_keeps_full_tail_visible_without_host_misses(history):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.nano_indexer_owner = impl
    impl.skip_topk = impl.has_indexer = True
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.scale = 4, 2, 1.0
    impl.sfa_sparse_topk = 2048
    impl.nano_reuse_topk_misses = impl.nano_reuse_misses = torch.zeros(2, dtype=torch.int32)
    impl.nano_topk_src = impl.nano_topk_dst = torch.zeros((2, 1, 2048), dtype=torch.int32)
    impl.nano_miss_src = impl.nano_miss_dst = torch.zeros((2, 2048), dtype=torch.int32)
    md = SimpleNamespace(
        num_decode_tokens=2,
        nano_pool_entries=torch.tensor([0, 1]),
        nano_cache_tokens=torch.tensor([history, 2048], dtype=torch.int32),
        nano_logical_lens=torch.tensor([history + 129, 2048], dtype=torch.int32),
        nano_query_ends=torch.tensor([1, 2], dtype=torch.int32),
        nano_token_active=torch.tensor([True, False]),
        nano_hbm_block_table=torch.arange(36, dtype=torch.int32).view(2, -1),
        nano_source_block_table=torch.arange(36, dtype=torch.int32).view(2, -1),
    )
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _: 0,
        topk_buffers_k=[torch.zeros((18, 128, 1, 4))],
        topk_buffers_v=[torch.zeros((18, 128, 1, 2))],
        k_caches_cpu=[torch.zeros((18, 128, 4))],
        v_caches_cpu=[torch.zeros((18, 128, 2))],
    )
    with patch.object(torch.ops._C_ascend, "npu_fused_copy_sfa_mtp", create=True) as kernel:
        kernel.side_effect = lambda *args: args[-1].fill_(1)
        output = impl._nano_attention(torch.zeros((2, 8, 4)), torch.zeros((2, 8, 2)), None, md, manager, "mtp")
    args = kernel.call_args.args
    assert args[3].tolist() == [history + 129, 2048]  # dummy has no logical tail
    assert args[4].tolist() == [history if history >= 2048 else 0, 2048]
    assert args[7].tolist() == args[10].tolist() == [0, 0]
    assert (output[0] == 1).all() and (output[1] == 0).all()


@pytest.mark.parametrize(
    ("draft_step", "host_visibility", "pd_tail_ready"),
    [(-1, False, False), (0, False, False), (1, False, False), (2, False, False), (-1, True, False), (0, False, True)],
)
def test_nano_decode_writes_kv_directly_to_topk_tail(draft_step, host_visibility, pd_tail_ready):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn"
    impl._current_layer_name = None
    impl.block_size = 2
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl._prepare_nano_tail_state = MagicMock()
    impl._nano_restore_tail = MagicMock()
    impl._update_nano_tail_state = MagicMock()
    impl.nano_requires_host_visibility = host_visibility
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(4),
        variance_epsilon=1e-5,
    )

    topk_k = torch.zeros((1, 6, 1, 4))
    topk_v = torch.zeros((1, 6, 1, 2))
    manager = SimpleNamespace(
        tp_rank=0,
        use_nano=True,
        topk_buffers_k=[topk_k],
        topk_buffers_v=[topk_v],
        k_caches_cpu=[torch.zeros((8, 1, 1, 4))],
        v_caches_cpu=[torch.zeros((8, 1, 1, 2))],
        _get_offload_layer_id=lambda _: 0,
        offload_nano_full_blocks=MagicMock(),
        wait_for_nano_host_copies=MagicMock(),
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_prefills=0,
        num_decodes=2,
        nano_enabled=True,
        nano_draft_step=draft_step,
        nano_skip_tail_restore=pd_tail_ready,
        nano_device_slots=torch.tensor([4, 5], dtype=torch.int64),
        nano_token_active=torch.tensor([True, False]),
        nano_flush_src_slots=torch.tensor([4, 0], dtype=torch.int64),
        nano_flush_dst_slots=torch.tensor([128, 0], dtype=torch.int64),
        nano_flush_active=torch.tensor([True, False]),
    )
    fused_kv = MagicMock()

    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.torch_npu.npu_kv_rmsnorm_rope_cache",
            fused_kv,
        ),
    ):
        result = impl.exec_kv(
            torch.zeros((2, 6)),
            torch.zeros((2, 1, 1, 2)),
            torch.zeros((2, 1, 1, 2)),
            (),
            torch.tensor([3, 7], dtype=torch.int64),
            metadata,
        )

    assert result == (None, None)
    fused_args = fused_kv.call_args.args
    torch.testing.assert_close(fused_args[4], metadata.nano_device_slots)
    assert fused_args[5].data_ptr() == topk_v.data_ptr()
    assert fused_args[6].data_ptr() == topk_k.data_ptr()
    impl._update_nano_tail_state.assert_called_once_with(metadata)
    if draft_step <= 0:
        if pd_tail_ready:
            impl._nano_restore_tail.assert_not_called()
        else:
            impl._nano_restore_tail.assert_called_once_with(metadata, manager, impl.layer_name)
        manager.offload_nano_full_blocks.assert_called_once_with(
            layer_name=impl.layer_name,
            source_slots=metadata.nano_flush_src_slots,
            destination_slots=metadata.nano_flush_dst_slots,
            active=metadata.nano_flush_active,
        )
    else:
        impl._nano_restore_tail.assert_not_called()
        manager.offload_nano_full_blocks.assert_not_called()
    assert manager.wait_for_nano_host_copies.call_count == int(host_visibility)


def test_non_nano_decode_keeps_immediate_offload_path():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn"
    impl._current_layer_name = None
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(4),
        variance_epsilon=1e-5,
    )
    k_nope = torch.ones((2, 1, 1, 4))
    k_pe = torch.ones((2, 1, 1, 2))
    manager = SimpleNamespace(
        tp_rank=0,
        use_nano=False,
        k_caches_cpu=[torch.zeros((8, 1, 1, 4))],
        v_caches_cpu=[torch.zeros((8, 1, 1, 2))],
        _get_offload_layer_id=lambda _: 0,
        offload_new_kv=MagicMock(),
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_prefills=0,
        num_decodes=2,
        nano_enabled=False,
    )

    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch.object(impl, "_compute_kv_only", return_value=(k_nope, k_pe)),
        patch.object(impl, "_in_graph_runtime", return_value=False),
    ):
        result = impl.exec_kv(
            torch.zeros((2, 6)),
            torch.zeros((2, 1, 1, 2)),
            torch.zeros((2, 1, 1, 2)),
            (),
            torch.tensor([3, 7], dtype=torch.int64),
            metadata,
        )

    assert result[0] is k_pe
    assert result[1] is k_nope
    offload_kwargs = manager.offload_new_kv.call_args.kwargs
    assert offload_kwargs["k"] is k_nope
    assert offload_kwargs["v"] is k_pe
    assert "source_slot_mapping" not in offload_kwargs


def _make_fused_overlap_impl() -> AscendSFAKVOffloadImpl:
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.block_size = 4
    impl.local_num_heads = 2
    impl.lru_resident_capacity = 8
    impl.sfa_sparse_topk = 4
    impl.max_num_topk_rows = 4
    impl.scale = 0.5
    impl.selection_kv_block_table = None
    impl.selection_kv_block_status = None
    impl.selection_membership_map = None
    impl.fused_overlap_last_req_ids = None
    impl._fused_overlap_selection_capacity = None
    impl._fused_overlap_decode_logged = True
    impl.skip_topk = False
    return impl


def test_mtp_rewrite_invalidates_membership_slot_map():
    impl = _make_fused_overlap_impl()
    topk_count = 4
    selection_status = torch.full((4, 1, 8), -1, dtype=torch.int32)
    selection_status[:, 0, :topk_count] = torch.tensor([8, 9, 10, 11])
    selection_status[:, 0, topk_count] = topk_count
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    control = membership[
        :, FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + 8
    ]
    control[:, 0] = 0x5A4D
    last_req_ids = torch.full((4,), 7, dtype=torch.int64)
    metadata = SimpleNamespace(
        token_to_req=torch.zeros(4, dtype=torch.int32),
        req_ids_tensor=torch.tensor([7], dtype=torch.int64),
    )

    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
        return_value=SimpleNamespace(capturing=False),
    ):
        impl._invalidate_fused_overlap_selection_rows(
            selection_status,
            membership,
            last_req_ids,
            metadata,
            num_tokens=4,
            num_reqs=1,
            topk_count=topk_count,
            seq_lens=torch.tensor([12], dtype=torch.int32),
            cum_query_lens=torch.tensor([4], dtype=torch.int32),
        )

    assert bool((selection_status[:, 0, :topk_count] == -1).all())
    assert bool((control[:, 0] == -1).all())


def test_fused_overlap_external_plan_passes_raw_topk_and_full_selection_state():
    """
    Test that the fused overlap external plan receives the raw topk indices and the full selection state,
    and that the fused op receives the correct inputs.
    """
    impl = _make_fused_overlap_impl()
    ql_nope = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    q_pe = torch.ones((2, 2, 1), dtype=torch.float32)
    topk = torch.tensor([[0, -1, 2, 5], [3, 7, 1, -1]], dtype=torch.int64)
    full_kv_cpu = torch.zeros((3, 4, 1, 3), dtype=torch.float32)
    full_rope_cpu = torch.zeros((3, 4, 1, 1), dtype=torch.float32)
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    call_order = []
    fused_inputs = {}
    plan_inputs = {}

    def allocate_membership(row_capacity):
        membership = torch.full(
            (row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
            -1,
            dtype=torch.int16,
        )
        control = membership[
            :,
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + 8,
        ]
        control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = 4
        control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - 4
        return membership

    def prepare_plan(**kwargs):
        call_order.append("plan")
        plan_inputs.update(kwargs)
        return True

    def inject_current(**kwargs):
        call_order.append("inject")
        assert kwargs["layer_name"] == "model.layers.0.self_attn"
        assert kwargs["num_tokens"] == 2
        assert kwargs["selection_kv_cache"].shape == (8, 4, 3)
        assert kwargs["selection_k_rope"].shape == (8, 4, 1)
        assert kwargs["capturing"] is False

    def wait_for_writeback(capturing):
        call_order.append("wait_writeback")
        assert capturing is False

    def fused_op(**kwargs):
        call_order.append("fused")
        fused_inputs.update(kwargs)
        return kwargs["query"] + 10

    manager = SimpleNamespace(
        topk_buffers_k=[torch.zeros((4, 8, 1, 3), dtype=torch.float32)],
        topk_buffers_v=[torch.zeros((4, 8, 1, 1), dtype=torch.float32)],
        _get_offload_layer_id=lambda _: 0,
        get_fused_overlap_cpu_kv_inputs=lambda _: (full_kv_cpu, full_rope_cpu),
        allocate_fused_overlap_membership_map=allocate_membership,
        prepare_fused_overlap_external_plan=prepare_plan,
        inject_current_kv_into_selection=inject_current,
        wait_for_current_kv_writeback=wait_for_writeback,
    )

    with (
        patch.object(impl, "_require_custom_op", return_value=fused_op),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
    ):
        output = impl._execute_fused_overlap_offload_decode(
            ql_nope,
            q_pe,
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            "model.layers.0.self_attn",
        )

    torch.testing.assert_close(output, ql_nope + 10)
    assert call_order == [
        "plan",
        "inject",
        "fused",
        "wait_writeback",
    ]
    assert fused_inputs["selection_kv_cache"].shape == (8, 4, 3)
    assert fused_inputs["selection_k_rope"].shape == (8, 4, 1)
    assert fused_inputs["selection_kv_block_table"].shape == (4, 2)
    assert fused_inputs["selection_kv_block_status"].shape == (4, 1, 8)
    assert fused_inputs["selection_membership_map"].shape == (4, 16400)
    assert fused_inputs["selection_membership_map"].dtype == torch.int16
    torch.testing.assert_close(
        fused_inputs["query"],
        torch.cat([ql_nope, q_pe], dim=-1),
    )
    assert plan_inputs["selection_membership_map"].data_ptr() == fused_inputs["selection_membership_map"].data_ptr()
    expected_topk = torch.tensor(
        [[[0, -1, 2, 5]], [[3, 7, 1, -1]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        fused_inputs["selection_topk_indices"],
        expected_topk,
    )
    torch.testing.assert_close(
        plan_inputs["topk_indices_npu"],
        expected_topk.squeeze(1),
    )
    torch.testing.assert_close(
        plan_inputs["req_ids_npu"],
        torch.tensor([101, 202], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan_inputs["stable_prefix_lens_npu"],
        torch.tensor([3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan_inputs["visible_seq_lens_npu"],
        torch.tensor([4, 4], dtype=torch.int32),
    )
    assert plan_inputs["capturing"] is False


def test_fused_overlap_common_inputs_are_reused_only_within_one_forward():
    impl = _make_fused_overlap_impl()
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    topk = torch.tensor(
        [[[0, 1, 2, 3]], [[3, 2, 1, 0]]],
        dtype=torch.int32,
    )
    query_lens = torch.tensor([1, 2], dtype=torch.int32)
    kv_lens = torch.tensor([4, 5], dtype=torch.int32)
    first_forward = SimpleNamespace(capturing=False)
    second_forward = SimpleNamespace(capturing=False)
    current_forward = [first_forward]

    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
        side_effect=lambda: current_forward[0],
    ):
        first = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )
        reused = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )
        kv_lens.add_(1)
        current_forward[0] = second_forward
        refreshed = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )

    assert reused is first
    assert refreshed is not first
    torch.testing.assert_close(
        first.seq_len_thresholds.reshape(-1),
        torch.tensor([4, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        refreshed.seq_len_thresholds.reshape(-1),
        torch.tensor([5, 6], dtype=torch.int32),
    )
