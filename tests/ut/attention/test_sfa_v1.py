import os
import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm.distributed.parallel_state import GroupCoordinator
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(), AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):
    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            cum_query_lens_cpu=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)
        assert torch.equal(metadata.cum_query_lens_cpu, common_attn_metadata.query_start_loc_cpu[1:11])

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
        self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly


class TestAscendSFAIndexerChunking(TestBase):
    def setUp(self):
        super().setUp()
        self.original_chunk_size = os.environ.get("VLLM_ASCEND_SFA_INDEXER_CHUNK_SIZE")

    def tearDown(self):
        if self.original_chunk_size is None:
            os.environ.pop("VLLM_ASCEND_SFA_INDEXER_CHUNK_SIZE", None)
        else:
            os.environ["VLLM_ASCEND_SFA_INDEXER_CHUNK_SIZE"] = self.original_chunk_size
        _EXTRA_CTX.capturing = False

    def test_sparse_c8_indexer_chunks_by_request_boundary(self):
        os.environ["VLLM_ASCEND_SFA_INDEXER_CHUNK_SIZE"] = "7"
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.sfa_indexer_chunk_size = 7

        calls = []

        def fake_indexer(**kwargs):
            calls.append(kwargs)
            return torch.full((kwargs["query"].shape[0], 2), len(calls), dtype=torch.int32)

        impl._call_sparse_c8_indexer = fake_indexer

        result = impl._execute_sparse_c8_indexer(
            query=torch.arange(10, dtype=torch.int8).view(10, 1, 1),
            key=torch.empty(1),
            weights=torch.arange(30, dtype=torch.float16).view(10, 3),
            query_dequant_scale=torch.ones(10, 1),
            key_dequant_scale=torch.ones(1),
            actual_seq_lengths_query=torch.tensor([3, 7, 10], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([30, 70, 100], dtype=torch.int32),
            block_table=torch.arange(12).view(3, 4),
            cum_query_lens_cpu=torch.tensor([3, 7, 10], dtype=torch.int32),
            num_actual_tokens=10,
        )

        assert result.shape == (10, 2)
        assert len(calls) == 2
        assert calls[0]["query"].shape[0] == 7
        assert calls[1]["query"].shape[0] == 3
        assert torch.equal(calls[0]["actual_seq_lengths_query"], torch.tensor([3, 7], dtype=torch.int32))
        assert torch.equal(calls[1]["actual_seq_lengths_query"], torch.tensor([3], dtype=torch.int32))
        assert torch.equal(calls[1]["actual_seq_lengths_key"], torch.tensor([100], dtype=torch.int32))
        assert torch.equal(calls[1]["block_table"], torch.arange(8, 12).view(1, 4))

    def test_sparse_c8_indexer_chunking_can_be_disabled(self):
        os.environ["VLLM_ASCEND_SFA_INDEXER_CHUNK_SIZE"] = "0"
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.sfa_indexer_chunk_size = 0

        calls = []

        def fake_indexer(**kwargs):
            calls.append(kwargs)
            return torch.empty(kwargs["query"].shape[0], 2)

        impl._call_sparse_c8_indexer = fake_indexer
        impl._execute_sparse_c8_indexer(
            query=torch.empty(10, 1, 1),
            key=torch.empty(1),
            weights=torch.empty(10, 3),
            query_dequant_scale=torch.empty(10, 1),
            key_dequant_scale=torch.empty(1),
            actual_seq_lengths_query=torch.tensor([10], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([10], dtype=torch.int32),
            block_table=torch.empty(1, 4),
            cum_query_lens_cpu=torch.tensor([10], dtype=torch.int32),
            num_actual_tokens=10,
        )

        assert len(calls) == 1
        assert calls[0]["query"].shape[0] == 10

    def test_sparse_c8_indexer_keeps_original_call_for_padded_query(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.sfa_indexer_chunk_size = 7

        calls = []

        def fake_indexer(**kwargs):
            calls.append(kwargs)
            return torch.empty(kwargs["query"].shape[0], 2)

        impl._call_sparse_c8_indexer = fake_indexer
        impl._execute_sparse_c8_indexer(
            query=torch.empty(12, 1, 1),
            key=torch.empty(1),
            weights=torch.empty(12, 3),
            query_dequant_scale=torch.empty(12, 1),
            key_dequant_scale=torch.empty(1),
            actual_seq_lengths_query=torch.tensor([3, 7, 10, 10], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([30, 70, 100, 0], dtype=torch.int32),
            block_table=torch.empty(4, 4),
            cum_query_lens_cpu=torch.tensor([3, 7, 10, 10], dtype=torch.int32),
            num_actual_tokens=10,
        )

        assert len(calls) == 1
        assert calls[0]["query"].shape[0] == 12

    def test_sparse_c8_indexer_keeps_original_call_when_capturing(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.sfa_indexer_chunk_size = 7
        _EXTRA_CTX.capturing = True

        calls = []

        def fake_indexer(**kwargs):
            calls.append(kwargs)
            return torch.empty(kwargs["query"].shape[0], 2)

        impl._call_sparse_c8_indexer = fake_indexer
        impl._execute_sparse_c8_indexer(
            query=torch.empty(10, 1, 1),
            key=torch.empty(1),
            weights=torch.empty(10, 3),
            query_dequant_scale=torch.empty(10, 1),
            key_dequant_scale=torch.empty(1),
            actual_seq_lengths_query=torch.tensor([3, 7, 10], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([30, 70, 100], dtype=torch.int32),
            block_table=torch.empty(3, 4),
            cum_query_lens_cpu=torch.tensor([3, 7, 10], dtype=torch.int32),
            num_actual_tokens=10,
        )

        assert len(calls) == 1
        assert calls[0]["query"].shape[0] == 10

    def test_sparse_c8_indexer_keeps_original_call_for_fia_padded_query(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_dsa_cp = False
        impl.sfa_indexer_chunk_size = 7

        calls = []

        def fake_indexer(**kwargs):
            calls.append(kwargs)
            return torch.empty(kwargs["query"].shape[0], 2)

        impl._call_sparse_c8_indexer = fake_indexer
        impl._execute_sparse_c8_indexer(
            query=torch.empty(12, 1, 1),
            key=torch.empty(1),
            weights=torch.empty(12, 3),
            query_dequant_scale=torch.empty(12, 1),
            key_dequant_scale=torch.empty(1),
            actual_seq_lengths_query=torch.tensor([3, 7, 10, 12], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([30, 70, 100, 0], dtype=torch.int32),
            block_table=torch.empty(4, 4),
            cum_query_lens_cpu=torch.tensor([3, 7, 10, 12], dtype=torch.int32),
            num_actual_tokens=10,
        )

        assert len(calls) == 1
        assert calls[0]["query"].shape[0] == 12
