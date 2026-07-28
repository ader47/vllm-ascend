# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.dsa_offload.model_support import (
    get_dsa_offload_model_capabilities,
    require_dsa_offload_model_support,
)


def _model_config(
    architecture: str,
    *,
    use_mla: bool = True,
    index_topk: int = 2048,
    compress_ratios=None,
):
    hf_text_config = SimpleNamespace(
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    hf_config = SimpleNamespace()
    if compress_ratios is not None:
        hf_text_config.compress_ratios = compress_ratios
        hf_config.compress_ratios = compress_ratios
    return SimpleNamespace(
        architecture=architecture,
        use_mla=use_mla,
        hf_text_config=hf_text_config,
        hf_config=hf_config,
    )


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"],
)
def test_model_support_is_capability_based(architecture: str) -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config(architecture))

    assert capabilities.supported
    assert capabilities.architecture == architecture


def test_unknown_architecture_with_required_capabilities_is_supported() -> None:
    capabilities = get_dsa_offload_model_capabilities(_model_config("FutureMlaIndexerForCausalLM"))

    assert capabilities.supported


@pytest.mark.parametrize(
    ("model_config", "missing"),
    [
        (_model_config("NoMla", use_mla=False), "MLA attention"),
        (_model_config("WrongTopK", index_topk=1024), "index_topk=2048"),
        (
            _model_config("CompressedDSA", compress_ratios=[2]),
            "sparse indexer",
        ),
    ],
)
def test_missing_capability_is_reported(model_config, missing: str) -> None:
    capabilities = get_dsa_offload_model_capabilities(model_config)

    assert not capabilities.supported
    assert any(missing in item for item in capabilities.missing_requirements)
    with pytest.raises(ValueError, match=missing):
        require_dsa_offload_model_support(model_config)
