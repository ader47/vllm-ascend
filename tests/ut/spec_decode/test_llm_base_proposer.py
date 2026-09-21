#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
    _draft_embed_accepts_mm,
)

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


@pytest.mark.parametrize("draft_steps,num_tokens", [(1, 4), (3, 8), (3, 10), (6, 16)])
def test_nano_dummy_capture_matches_runtime_step_layout(draft_steps, num_tokens):
    query_width = draft_steps + 1
    ends = [0, query_width, 2 * query_width]
    if num_tokens > ends[-1]:
        ends.append(num_tokens)
    num_reqs = len(ends) - 1
    capacity = 16

    def buffer(data):
        cpu = data.clone()
        gpu = data.clone()
        return SimpleNamespace(cpu=cpu, gpu=gpu, copy_to_gpu=lambda: gpu.copy_(cpu))

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=capacity),
        speculative_config=SimpleNamespace(num_speculative_tokens=draft_steps),
        model_config=SimpleNamespace(
            max_model_len=1024,
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        ),
    )
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = builder.is_pd_decode_consumer = True
    builder.decode_threshold = query_width
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=2048)),
    ):
        builder._init_nano_metadata_buffers(config, torch.device("cpu"))

    group = SimpleNamespace(layer_names=["mtp"], get_metadata_builder=lambda: builder)
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = config
    proposer.num_speculative_tokens = draft_steps
    proposer.use_cuda_graph = True
    proposer.use_compress = proposer.uses_mrope = proposer.has_gdn = proposer.supports_mm_inputs = False
    proposer.method = "mtp"
    proposer.dcp_size = proposer.extra_slots_per_request = 1
    proposer.kv_cache_gid = 0
    proposer.block_size = 128
    proposer.max_model_len = 1024
    proposer.sliding_window = None
    proposer.draft_attn_groups = [group]
    proposer.attn_layer_names = group.layer_names
    proposer.arange = torch.arange(capacity + 1, dtype=torch.int32)
    proposer.token_arange_np = proposer.arange.numpy()
    proposer.positions = torch.full((capacity,), 17, dtype=torch.int64)
    proposer._get_positions = lambda count: proposer.positions[:count]
    proposer.query_start_loc = buffer(torch.tensor(ends, dtype=torch.int32))
    proposer.slot_mapping_group = torch.empty((draft_steps, capacity), dtype=torch.int32)
    proposer.seq_lens_group = torch.empty((draft_steps, capacity), dtype=torch.int32)
    proposer.query_start_loc_group = torch.empty((draft_steps, capacity + 1), dtype=torch.int32)
    proposer.token_indices_to_sample = torch.zeros(capacity, dtype=torch.int64)
    proposer._runnable = MagicMock()
    block_table = torch.arange(num_reqs * 8, dtype=torch.int32).view(num_reqs, 8)
    proposer.runner = SimpleNamespace(
        sparse_kv_offload_enabled=True,
        sparse_kv_offload_config=SimpleNamespace(use_nano=True),
        _sync_metadata_across_dp=lambda n, **kwargs: (n, None, None),
        synchronize_input_prep=nullcontext,
        _prepare_nano_request_slots=MagicMock(),
        attn_groups=[group],
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.zeros(num_reqs, dtype=torch.int32),
            block_table=[
                SimpleNamespace(
                    get_device_tensor=lambda: block_table,
                    slot_mapping=buffer(torch.arange(capacity, dtype=torch.int32)),
                )
            ],
        ),
        query_start_loc=buffer(torch.tensor(ends, dtype=torch.int32)),
        optimistic_seq_lens_cpu=torch.full((num_reqs,), 128, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), 128, dtype=torch.int32),
        positions=torch.full((capacity,), 23, dtype=torch.int64),
        actual_seq_lengths_q=ends[1:],
        attn_state=None,
        decode_token_per_req=query_width,
        group_len=buffer(torch.zeros(num_reqs, dtype=torch.int32)),
        group_key_idx=buffer(torch.zeros(num_reqs, dtype=torch.int32)),
        group_key_cache_idx=buffer(torch.zeros(num_reqs, dtype=torch.int32)),
        _offload_pool_slots=buffer(torch.arange(num_reqs, dtype=torch.int32) + 4),
        _offload_pool_generations=buffer(torch.full((num_reqs,), -1, dtype=torch.int64)),
        _offload_req_ids_tensor=None,
        _offload_token_to_req=None,
        dynamic_eplb=False,
    )
    module = "vllm_ascend.spec_decode.llm_base_proposer"
    with (
        patch(module + ".AscendCommonAttentionMetadata", side_effect=lambda **kw: SimpleNamespace(**kw)),
        patch(module + ".prepare_sparse_kv_offload_mtp_dummy_metadata", return_value=(None, None)),
        patch(module + ".set_ascend_forward_context", return_value=nullcontext()),
        patch(module + ".get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)),
        patch(module + "._EXTRA_CTX", SimpleNamespace(capturing=True)),
        patch.object(AscendSFAMetadataBuilder, "_build", side_effect=lambda *a, **kw: SimpleNamespace()),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.split_decodes_and_prefills",
            side_effect=lambda cm, **kw: (cm.num_reqs, 0, cm.num_input_tokens, 0),
        ),
    ):
        proposer.dummy_run(num_tokens, num_reqs=num_reqs, aclgraph_runtime_mode=CUDAGraphMode.FULL)
        captured = [entry["mtp"] for entry in proposer._runnable.call_args.kwargs["multi_steps_attn_metadata"]]
        assert len(captured) == draft_steps
        per_step_fields = (
            "nano_device_slots",
            "nano_query_ends",
            "nano_hbm_block_table",
            "nano_token_active",
        )
        for name in per_step_fields:
            assert len({getattr(metadata, name).data_ptr() for metadata in captured}) == draft_steps
        assert captured[0].nano_query_ends.tolist() == ends[1:]
        for step, metadata in enumerate(captured):
            assert metadata.nano_device_slots.numel() == num_tokens
            assert not metadata.nano_active.any()
            assert not metadata.nano_token_active.any()
            if step:
                assert metadata.nano_query_ends.tolist() == list(range(1, num_reqs + 1))
                assert metadata.num_decode_tokens == num_reqs

        runtime_common = SimpleNamespace(
            num_reqs=num_reqs,
            num_input_tokens=num_tokens,
            max_query_len=query_width,
            query_start_loc=proposer.query_start_loc_group[0][: num_reqs + 1],
            query_start_loc_cpu=torch.tensor(ends, dtype=torch.int32),
            seq_lens=torch.full((num_reqs,), 128, dtype=torch.int32),
            seq_lens_cpu=torch.full((num_reqs,), 128, dtype=torch.int32),
            _seq_lens_cpu=None,
            num_computed_tokens_cpu=None,
            block_table_tensor=block_table,
            positions=proposer.positions.clone(),
            slot_mapping=proposer.slot_mapping_group[0],
            req_topk_buffer_slots=torch.arange(num_reqs, dtype=torch.int32),
            req_topk_buffer_generations=torch.tensor([11, 12] + [-1] * (num_reqs - 2)),
            nano_eligible=True,
            offload_dummy=False,
            req_ids_tensor=None,
            token_to_req=None,
        )
        runtime = [builder.build(0, runtime_common)]
        runtime_positions = torch.tensor([127, 127], dtype=torch.int64)
        for step in range(1, draft_steps):
            runtime_common, metadata = proposer.attn_update_stack_num_spec_norm(
                step,
                runtime_common,
                batch_size=2,
                input_batch_size=num_tokens,
                used_update_positions=runtime_positions,
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
                attn_group=group,
            )
            runtime.append(metadata)
            assert metadata.nano_seq_lens[:2].tolist() == [128 + step, 128 + step]

        assert len(runtime) == draft_steps
        for capture_metadata, runtime_metadata in zip(captured, runtime):
            assert capture_metadata.num_decode_tokens == runtime_metadata.num_decode_tokens
            for name in per_step_fields:
                assert getattr(capture_metadata, name).shape == getattr(runtime_metadata, name).shape
                assert getattr(capture_metadata, name).data_ptr() == getattr(runtime_metadata, name).data_ptr()

    assert (proposer.positions == 17).all()
    assert (proposer.runner.positions == 23).all()


class TestMultimodalImageTokenIndex:
    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Step3p7ForConditionalGeneration",
            "Gemma4ForConditionalGeneration",
            "Gemma4UnifiedForConditionalGeneration",
            "Glm5NextForConditionalGeneration",
        ],
    )
    def test_models_using_image_token_id(self, model_name: str):
        config = SimpleNamespace(image_token_id=123, image_token_index=456)

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(model_name, config)

        assert image_token_index == 123

    def test_pixtral_uses_vision_config_image_token_id(self):
        config = SimpleNamespace(
            image_token_id=123,
            image_token_index=456,
            vision_config=SimpleNamespace(image_token_id=789),
        )

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
            "PixtralForConditionalGeneration", config
        )

        assert image_token_index == 789

    @pytest.mark.parametrize(
        "model_name",
        [
            "KimiK25ForConditionalGeneration",
            "KimiK3ForConditionalGeneration",
            "AscendKimiK3ForConditionalGeneration",
        ],
    )
    def test_kimi_uses_media_placeholder_token_id(self, model_name: str):
        config = SimpleNamespace(
            image_token_id=123,
            image_token_index=456,
            media_placeholder_token_id=789,
        )

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(model_name, config)

        assert image_token_index == 789

    def test_default_uses_image_token_index(self):
        config = SimpleNamespace(image_token_id=123, image_token_index=456)

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
            "OtherForConditionalGeneration", config
        )

        assert image_token_index == 456

    def test_model_with_multiple_image_sentinels_needs_no_single_index(self):
        config = SimpleNamespace()

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
            "AscendDeepseekV4ForConditionalGeneration", config
        )

        assert image_token_index is None


class TestMtpSharesTheTargetLmHead:
    """``_maybe_share_lm_head`` for the MTP branch.

    Weight equality only distinguishes the two heads when the draft loaded one.
    GLM-5.3 and friends ship no ``shared_head.head``, so its buffer holds
    whatever the allocation left behind; comparing that against the target head
    says "different" and would leave the draft predicting from garbage.
    """

    @staticmethod
    def _build(draft_head_weight, has_own_lm_head=None):
        target = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(4, 2)))
        mtp_layer = SimpleNamespace(shared_head=SimpleNamespace(head=SimpleNamespace(weight=draft_head_weight)))

        draft = MagicMock()
        draft.model.layers = {"78": mtp_layer}
        if has_own_lm_head is None:
            del draft.has_own_lm_head
        else:
            draft.has_own_lm_head = has_own_lm_head

        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.model = draft
        proposer.use_cuda_graph = False
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
        return proposer, target, mtp_layer

    def test_shares_when_the_checkpoint_has_no_mtp_head(self):
        proposer, target, mtp_layer = self._build(torch.randn(4, 2), has_own_lm_head=False)
        proposer._maybe_share_lm_head(target)
        assert mtp_layer.shared_head.head is target.lm_head

    def test_keeps_a_head_the_draft_owns(self):
        proposer, target, mtp_layer = self._build(torch.randn(4, 2), has_own_lm_head=True)
        own_head = mtp_layer.shared_head.head
        proposer._maybe_share_lm_head(target)
        assert mtp_layer.shared_head.head is own_head

    def test_still_deduplicates_an_identical_head(self):
        # DeepSeek ships shared_head.head equal to lm_head; sharing it saves a
        # copy of the vocabulary projection.
        proposer, target, mtp_layer = self._build(torch.ones(4, 2), has_own_lm_head=True)
        proposer._maybe_share_lm_head(target)
        assert mtp_layer.shared_head.head is target.lm_head


def test_load_model_reads_validated_draft_window_size():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace(additional_config={"draft_window_size": 64})
    proposer.maybe_eager_context = nullcontext()
    draft_model = MagicMock()
    proposer._get_model = MagicMock(return_value=draft_model)
    proposer.method = "eagle3"
    proposer.num_speculative_tokens = 4
    proposer.runner = SimpleNamespace(max_num_reqs=8)
    proposer.device = "cpu"
    proposer.parallel_drafting = False
    proposer.supports_mm_inputs = False
    proposer._maybe_share_embeddings = MagicMock()
    proposer._maybe_share_topk_indices = MagicMock()
    proposer._maybe_share_lm_head = MagicMock()

    draft_layer = MagicMock()
    draft_layer.get_kv_cache_spec.return_value = object()
    draft_layer.get_attn_backend.return_value.get_supported_kernel_block_sizes.return_value = [16]

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_pp_group") as mock_pp_group,
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.get_layers_from_vllm_config",
            side_effect=[{}, {"draft": draft_layer}, {}, {"draft": draft_layer}],
        ),
        patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_ascend_config,
        patch("vllm_ascend.spec_decode.llm_base_proposer.SlidingWindowAdapter") as mock_adapter,
        patch("vllm_ascend.spec_decode.llm_base_proposer.supports_multimodal", return_value=False),
    ):
        mock_pp_group.return_value.is_last_rank = True
        mock_get_ascend_config.return_value.draft_window_size = 4096

        proposer.load_model(MagicMock())

    assert proposer.draft_window_size == 4096
    mock_adapter.assert_called_once_with(4096, 16, 8, 4, "cpu")


def test_draft_vllm_config_only_propagates_draft_runner_type():
    draft_model_config = SimpleNamespace(
        runner_type="draft",
        architecture="draft-architecture",
        num_experts=0,
    )
    base_model_config = SimpleNamespace(
        runner_type="generate",
        architecture="target-architecture",
        num_experts=256,
    )
    base_vllm_config = SimpleNamespace(model_config=base_model_config)
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
    )

    with (
        patch(
            "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer._create_draft_vllm_config",
            return_value=base_vllm_config,
        ),
    ):
        draft_vllm_config = proposer._create_draft_vllm_config()

    assert draft_vllm_config is not base_vllm_config
    assert draft_vllm_config.model_config is not base_model_config
    assert draft_vllm_config.model_config is not draft_model_config
    assert draft_vllm_config.model_config.runner_type == "draft"
    assert draft_vllm_config.model_config.architecture == "target-architecture"
    assert draft_vllm_config.model_config.num_experts == 256
    assert base_model_config.runner_type == "generate"


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()


class TestMtpTargetLmHeadLookup:
    """Where the MTP branch of ``_maybe_share_lm_head`` finds the target head.

    Multimodal wrappers such as ``Glm5NextForConditionalGeneration`` own no
    ``lm_head``; it lives on the nested language model, so reading it off the
    wrapper raises ``AttributeError``.
    """

    @staticmethod
    def _share(target) -> SimpleNamespace:
        """Return the resulting draft head after sharing runs."""
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
        proposer.use_cuda_graph = False
        layers = {"0": SimpleNamespace(shared_head=SimpleNamespace(head=SimpleNamespace(weight=torch.zeros(4, 3))))}
        # The checkpoint ships no MTP head, so the draft head is shared
        # whenever the target head is reachable at all.
        proposer.model = SimpleNamespace(model=SimpleNamespace(layers=layers), has_own_lm_head=False)

        proposer._maybe_share_lm_head(target)

        return layers["0"].shared_head.head

    def test_resolves_lm_head_through_get_language_model(self):
        target_lm_head = SimpleNamespace(weight=torch.ones(4, 3))
        target = SimpleNamespace(get_language_model=lambda: SimpleNamespace(lm_head=target_lm_head))

        assert self._share(target) is target_lm_head

    def test_resolves_lm_head_through_the_language_model_attribute(self):
        target_lm_head = SimpleNamespace(weight=torch.ones(4, 3))
        target = SimpleNamespace(language_model=SimpleNamespace(lm_head=target_lm_head))

        assert self._share(target) is target_lm_head

    def test_unreachable_lm_head_keeps_the_draft_head(self):
        draft_head = self._share(SimpleNamespace())

        assert torch.equal(draft_head.weight, torch.zeros(4, 3))


class TestDraftEmbedMmSupport:
    """Text-only MTP heads such as Glm5NextMTP expose ``embed_input_ids``
    without multimodal parameters, so forwarding a multimodal target model's
    multimodal kwargs to them raises ``TypeError``.
    """

    def test_head_taking_multimodal_embeddings_accepts_mm(self):
        def embed_input_ids(input_ids, multimodal_embeddings=None):
            return input_ids

        assert _draft_embed_accepts_mm(embed_input_ids) is True

    def test_text_only_head_does_not_accept_mm(self):
        def embed_input_ids(input_ids):
            return input_ids

        assert _draft_embed_accepts_mm(embed_input_ids) is False

    @pytest.mark.parametrize("error", [TypeError("C-bound callable"), ValueError("no signature found")])
    def test_uninspectable_callable_is_treated_as_text_only(self, error: Exception):
        """Falling back to text-only cannot raise, whereas assuming multimodal
        support and forwarding the kwargs to a head that rejects them would.
        """

        def embed_input_ids(input_ids, multimodal_embeddings=None):
            return input_ids

        with patch("vllm_ascend.spec_decode.llm_base_proposer._inspect") as fake_inspect:
            fake_inspect.signature.side_effect = error

            assert _draft_embed_accepts_mm(embed_input_ids) is False
