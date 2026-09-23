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
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    FSA_EXTERNAL_PLAN_READY_MARKER,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
)

MODULE = "vllm_ascend.attention.sfa_kv_offload"


def _make_nano_builder(max_num_seqs: int = 2):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = True
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=16,
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            max_model_len=16384,
            hf_text_config=SimpleNamespace(
                kv_lora_rank=4,
                qk_rope_head_dim=2,
            ),
        ),
    )
    with patch(
        MODULE + ".get_ascend_config",
        return_value=SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=8192),
        ),
    ):
        builder._init_nano_metadata_buffers(config, torch.device("cpu"))
    return builder


def _populate_nano_metadata(builder, common, draft_index: int = 0):
    with patch(
        MODULE + ".split_decodes_and_prefills",
        return_value=(common.num_reqs, 0, common.num_input_tokens, 0),
    ):
        return builder._populate_offload_metadata(
            SimpleNamespace(),
            common,
            draft_index=draft_index,
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
    with (
        patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None),
        patch(
            MODULE + ".get_ascend_config",
            return_value=SimpleNamespace(
                sparse_kv_offload_config=SimpleNamespace(use_nano=False),
            ),
        ),
    ):
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


@pytest.mark.parametrize(
    ("width", "prefills", "eligible"),
    [(8, 0, True), (1, 1, True), (1, 0, False)],
)
def test_nano_rejects_unsafe_fallback(width, prefills, eligible):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = True
    builder.is_pd_decode_consumer = True
    builder.decode_threshold = 7
    common = SimpleNamespace(
        nano_eligible=eligible,
        offload_dummy=False,
        max_query_len=width,
        req_ids_tensor=None,
        token_to_req=None,
    )

    with (
        patch(
            MODULE + ".split_decodes_and_prefills",
            return_value=(1, prefills, width, 0),
        ),
        pytest.raises(ValueError, match="discard resident partial tails"),
    ):
        builder._populate_offload_metadata(SimpleNamespace(), common)


def test_nano_full_pool_generation_and_padding_are_isolated():
    builder = _make_nano_builder(max_num_seqs=2)
    count = builder.nano_pool_capacity
    common = SimpleNamespace(
        query_start_loc=torch.arange(count + 1, dtype=torch.int32),
        query_start_loc_cpu=torch.arange(count + 1, dtype=torch.int32),
        seq_lens=torch.tensor([129, 999, 999, 999], dtype=torch.int32),
        # Slot 3 is the second extra legal pool row. The other rows model an
        # out-of-range slot, a negative slot, and ordinary graph padding.
        req_topk_buffer_slots=torch.tensor([3, 4, -1, 1], dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor([7, 8, 9, -1], dtype=torch.int64),
        req_topk_buffer_stable_prefixes=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        block_table_tensor=torch.arange(count * 128, dtype=torch.int32).view(count, -1),
        positions=torch.arange(count, dtype=torch.int64),
        slot_mapping=torch.arange(count, dtype=torch.int64),
        req_ids_tensor=None,
        token_to_req=None,
        nano_eligible=True,
        offload_dummy=False,
        max_query_len=1,
        num_reqs=count,
        num_input_tokens=count,
    )

    metadata = _populate_nano_metadata(builder, common)

    assert builder.nano_pool_capacity == 4
    assert metadata.nano_active.tolist() == [True, False, False, False]
    assert metadata.nano_pool_entries.tolist() == [3, 5, 6, 7]
    private_start = builder.nano_pool_capacity * builder.nano_stride_blocks * 128
    assert metadata.nano_device_slots[0] < private_start
    assert bool((metadata.nano_device_slots[1:] >= private_start).all())

    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.nano_states = torch.empty(count, dtype=torch.int32)
    impl.nano_last_generation = torch.full((count * 2,), -1, dtype=torch.int64)
    impl.nano_last_prefix = torch.zeros(count * 2, dtype=torch.int32)
    impl.nano_last_cache = torch.zeros(count * 2, dtype=torch.int32)
    impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states.tolist() == [-2, -3, -3, -3]
    impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states.tolist() == [-1, -3, -3, -3]
    metadata.nano_generations[0] += 1
    impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states.tolist() == [-2, -3, -3, -3]


def test_nano_later_draft_keeps_step_zero_prefix_and_full_tail():
    builder = _make_nano_builder()
    base = 8192
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([base + 130], dtype=torch.int32),
        req_topk_buffer_slots=torch.tensor([1], dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor([11], dtype=torch.int64),
        req_topk_buffer_stable_prefixes=torch.tensor([base], dtype=torch.int32),
        block_table_tensor=torch.arange(128, dtype=torch.int32).view(1, -1),
        positions=torch.arange(base + 126, base + 130, dtype=torch.int64),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        req_ids_tensor=None,
        token_to_req=None,
        nano_eligible=True,
        offload_dummy=False,
        max_query_len=4,
        num_reqs=1,
        num_input_tokens=4,
    )
    step_zero = _populate_nano_metadata(builder, common)
    assert step_zero.nano_prefix_lens.tolist() == [base]

    # Rejection leaves the next MTP row at position base + 128. Recomputing
    # floor((S-Q)/128) here would advance the prefix and hide the full page
    # that was not part of step zero's sparse TopK selection.
    common.query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    common.query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
    common.seq_lens = torch.tensor([base + 131], dtype=torch.int32)
    common.positions = torch.tensor([base + 128], dtype=torch.int64)
    common.slot_mapping = torch.tensor([base + 128], dtype=torch.int64)
    # Even if a later metadata input observes a newer published prefix, all
    # proposal steps must retain step 0's sparse-history boundary.
    common.req_topk_buffer_stable_prefixes.fill_(base + 128)
    common.max_query_len = 1
    common.num_input_tokens = 1
    later = _populate_nano_metadata(builder, common, draft_index=1)

    assert later.nano_seq_lens.tolist() == [base + 129]
    assert later.nano_prefix_lens.tolist() == [base]
    assert later.nano_cache_tokens.tolist() == [base]
    assert later.nano_logical_lens.tolist() == [base + 129]


def test_nano_prefix_advances_only_after_published_completion():
    builder = _make_nano_builder()
    base = 8192
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([base + 129], dtype=torch.int32),
        req_topk_buffer_slots=torch.tensor([1], dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor([11], dtype=torch.int64),
        req_topk_buffer_stable_prefixes=torch.tensor([base], dtype=torch.int32),
        block_table_tensor=torch.arange(128, dtype=torch.int32).view(1, -1),
        positions=torch.tensor([base + 128], dtype=torch.int64),
        slot_mapping=torch.tensor([base + 128], dtype=torch.int64),
        req_ids_tensor=None,
        token_to_req=None,
        nano_eligible=True,
        offload_dummy=False,
        max_query_len=1,
        num_reqs=1,
        num_input_tokens=1,
    )

    before_completion = _populate_nano_metadata(builder, common)
    assert before_completion.nano_prefix_lens.tolist() == [base]
    assert before_completion.nano_logical_lens.tolist() == [base + 129]
    assert not hasattr(before_completion, "nano_copy_src_offsets")

    common.req_topk_buffer_stable_prefixes.fill_(base + 128)
    after_completion = _populate_nano_metadata(builder, common)
    assert after_completion.nano_prefix_lens.tolist() == [base + 128]
    assert after_completion.nano_logical_lens.tolist() == [base + 1]


def test_nano_decode_uses_fused_resident_tail_write_without_per_token_d2h():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    k_nope = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    k_pe = torch.tensor([[[[5.0]]], [[[6.0]]]])
    impl._is_decode_only = lambda _metadata: True
    impl._compute_kv_only = MagicMock()
    impl._offload_layer_name = lambda: "layer"
    impl._cpu_cache_pair = MagicMock()
    impl.num_kv_heads = 1
    impl.kv_lora_rank = 2
    impl.qk_rope_head_dim = 1
    impl.block_size = 2
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(2),
        variance_epsilon=1e-5,
    )
    manager = MagicMock()
    manager._get_offload_layer_id.return_value = 0
    resident_k = torch.zeros((2, 4, 1, 2))
    resident_v = torch.zeros((2, 4, 1, 1))
    manager.topk_buffers_k = [resident_k]
    manager.topk_buffers_v = [resident_v]
    metadata = SimpleNamespace(
        nano_enabled=True,
        nano_device_slots=torch.tensor([2, 5]),
    )

    with (
        patch(MODULE + ".get_sparse_kv_offload_manager", return_value=manager),
        patch(
            MODULE + ".torch_npu.npu_kv_rmsnorm_rope_cache",
            return_value=(resident_v.view(-1, 2, 1, 1), resident_k.view(-1, 2, 1, 2), k_pe, k_nope),
        ) as fused_write,
    ):
        result = impl.exec_kv(
            torch.zeros((2, 3)),
            torch.zeros((2, 1)),
            torch.zeros((2, 1)),
            (),
            torch.tensor([10, 11]),
            metadata,
        )

    assert result[0] is k_pe
    assert result[1] is k_nope
    fused_write.assert_called_once()
    args = fused_write.call_args.args
    assert args[5].shape == (4, 2, 1, 1)
    assert args[6].shape == (4, 2, 1, 2)
    assert args[5].data_ptr() == resident_v.data_ptr()
    assert args[6].data_ptr() == resident_k.data_ptr()
    torch.testing.assert_close(args[4], metadata.nano_device_slots)
    assert fused_write.call_args.kwargs["is_output_kv"] is True
    impl._compute_kv_only.assert_not_called()
    impl._cpu_cache_pair.assert_not_called()
    manager.offload_new_kv.assert_not_called()


def test_non_nano_decode_keeps_existing_per_token_offload():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    k_nope = torch.ones((1, 1, 1, 2))
    k_pe = torch.ones((1, 1, 1, 1))
    impl._is_decode_only = lambda _metadata: True
    impl._compute_kv_only = lambda *_args: (k_nope, k_pe)
    impl._offload_layer_name = lambda: "layer"
    impl._cpu_cache_pair = MagicMock(return_value=(torch.empty(0), torch.empty(0)))
    impl._in_graph_runtime = lambda: False
    manager = MagicMock()
    slots = torch.tensor([10])

    with patch(MODULE + ".get_sparse_kv_offload_manager", return_value=manager):
        impl.exec_kv(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            (),
            slots,
            SimpleNamespace(nano_enabled=False),
        )

    manager.offload_new_kv.assert_called_once()
    assert manager.offload_new_kv.call_args.kwargs["slot_mapping"] is slots


def test_nano_reused_topk_passes_resident_tail_to_attention():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.nano_indexer_owner = impl
    impl.skip_topk = True
    impl.has_indexer = True
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 2
    impl.scale = 1.0
    impl.sfa_sparse_topk = 2048
    impl.nano_reuse_topk_misses = torch.zeros(1, dtype=torch.int32)
    impl.nano_reuse_misses = torch.zeros(1, dtype=torch.int32)
    impl.nano_topk_src = torch.zeros((1, 1, 2048), dtype=torch.int32)
    impl.nano_topk_dst = torch.zeros_like(impl.nano_topk_src)
    impl.nano_miss_src = torch.zeros((1, 2048), dtype=torch.int32)
    impl.nano_miss_dst = torch.zeros_like(impl.nano_miss_src)
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        nano_pool_entries=torch.tensor([0], dtype=torch.int32),
        nano_cache_tokens=torch.tensor([2048], dtype=torch.int32),
        nano_logical_lens=torch.tensor([2177], dtype=torch.int32),
        nano_query_ends=torch.tensor([1], dtype=torch.int32),
        nano_token_active=torch.tensor([True]),
        nano_hbm_block_table=torch.arange(18, dtype=torch.int32).view(1, -1),
        nano_source_block_table=torch.arange(18, dtype=torch.int32).view(1, -1),
    )
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _: 0,
        topk_buffers_k=[torch.zeros((18, 128, 1, 4))],
        topk_buffers_v=[torch.zeros((18, 128, 1, 2))],
        k_caches_cpu=[torch.zeros((18, 128, 4))],
        v_caches_cpu=[torch.zeros((18, 128, 2))],
    )
    captured = {}

    def copy_sfa(*args):
        captured["logical_lens"] = args[3].clone()
        captured["cache_tokens"] = args[4].clone()
        args[-1].fill_(1)

    with patch.object(
        torch.ops._C_ascend,
        "npu_fused_copy_sfa_mtp",
        side_effect=copy_sfa,
        create=True,
    ):
        output = impl._nano_attention(
            torch.zeros((1, 8, 4)),
            torch.zeros((1, 8, 2)),
            None,
            metadata,
            manager,
            "mtp",
        )

    assert captured["cache_tokens"].tolist() == [2048]
    assert captured["logical_lens"].tolist() == [2177]
    assert bool((output == 1).all())


def test_nano_short_history_uses_dense_identity_rows_only_when_cold():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.sfa_sparse_topk = 2048
    impl.nano_dense_slots = torch.arange(2048, dtype=torch.int32)
    impl.nano_miss_src = torch.full((4, 2048), -1, dtype=torch.int32)
    impl.nano_miss_dst = impl.nano_miss_src.clone()
    impl.nano_misses = torch.tensor([0, 0, 7, 0], dtype=torch.int32)
    impl.nano_states = torch.tensor([-2, -1, -2, -3], dtype=torch.int32)
    metadata = SimpleNamespace(
        nano_pool_entries=torch.arange(4),
        nano_prefix_lens=torch.tensor([128, 256, 4096, 128], dtype=torch.int32),
        nano_active=torch.tensor([True, True, True, False]),
    )

    impl._prepare_nano_dense_prefix(metadata)

    assert impl.nano_misses.tolist() == [128, 0, 7, 0]
    torch.testing.assert_close(
        impl.nano_miss_src[0, :128],
        torch.arange(128, dtype=torch.int32),
    )
    torch.testing.assert_close(
        impl.nano_miss_dst[0, :128],
        impl.nano_miss_src[0, :128],
    )
    assert bool((impl.nano_miss_src[2:] == -1).all())


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
