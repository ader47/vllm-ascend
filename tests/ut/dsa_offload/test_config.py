# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.dsa_offload.config import (
    DSAOffloadConfig,
    DSAOffloadTraceConfig,
)


def _vllm_config(
    *,
    architecture: str = "GlmMoeDsaForCausalLM",
    async_scheduling: bool | None = False,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    enforce_eager: bool = True,
    block_size: int = 128,
    speculative_config=None,
    kv_transfer_config=None,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    kv_cache_metrics: bool = False,
    enable_kv_cache_events: bool = False,
):
    hf_text_config = SimpleNamespace(
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            use_mla=True,
            enforce_eager=enforce_eager,
            hf_text_config=hf_text_config,
            hf_config=hf_text_config,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            async_scheduling=async_scheduling,
            enable_chunked_prefill=enable_chunked_prefill,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        observability_config=SimpleNamespace(
            kv_cache_metrics=kv_cache_metrics,
        ),
        kv_events_config=SimpleNamespace(
            enable_kv_cache_events=enable_kv_cache_events,
        ),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
    )


def test_disabled_config_uses_typed_defaults() -> None:
    config = DSAOffloadConfig.from_dict(None)

    assert config.enabled is False
    assert config.split_indexer_cache is False
    assert config.resident_budget_tokens == (6144, 10240, 12288)
    assert config.max_resident_budget_tokens == 12288


def test_enabled_config_keeps_fractional_dram_multiplier() -> None:
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "hot_cpu_block_multiple": 1.5,
        },
        vllm_config=_vllm_config(),
    )

    assert config.enabled is True
    assert config.split_indexer_cache is True
    assert config.hot_cpu_block_multiple == 1.5
    assert config.model_capabilities is not None
    assert config.model_capabilities.architecture == "GlmMoeDsaForCausalLM"


def test_enabled_config_rejects_non_split_layout() -> None:
    with pytest.raises(ValueError, match="always requires a split"):
        DSAOffloadConfig.from_dict(
            {
                "enabled": True,
                "split_indexer_cache": False,
            }
        )


@pytest.mark.parametrize(
    ("raw_config", "message"),
    [
        ({"unknown": 1}, "Unknown dsa_sparse_config keys"),
        (
            {
                "prompt_budget_thresholds": [32768, 65536],
                "resident_budget_tokens": [6144, 10240],
            },
            "exactly one more",
        ),
        (
            {"resident_budget_tokens": [6144, 8192, 12288]},
            "not supported",
        ),
        ({"max_active_reqs": 0}, "must be positive"),
        ({"hot_cpu_block_multiple": float("nan")}, "positive and finite"),
        ({"hot_cpu_block_multiple": float("inf")}, "positive and finite"),
    ],
)
def test_invalid_static_config_is_rejected(
    raw_config,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DSAOffloadConfig.from_dict(raw_config)


@pytest.mark.parametrize(
    ("config_updates", "message"),
    [
        ({"async_scheduling": None}, "async_scheduling=False"),
        ({"async_scheduling": True}, "async_scheduling=False"),
        ({"enable_chunked_prefill": True}, "chunked prefill"),
        ({"enable_prefix_caching": True}, "prefix caching"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV transfer connectors"),
        (
            {"decode_context_parallel_size": 2},
            "context parallelism",
        ),
        (
            {"prefill_context_parallel_size": 2},
            "context parallelism",
        ),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"kv_cache_metrics": True}, "KV-cache metrics"),
        ({"enable_kv_cache_events": True}, "KV-cache events"),
    ],
)
def test_initial_runtime_support_matrix_is_enforced(
    config_updates,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(**config_updates),
        )


def test_graph_config_rejects_enforce_eager() -> None:
    with pytest.raises(ValueError, match="enforce_eager=False"):
        DSAOffloadConfig.from_dict(
            {
                "enabled": True,
                "enable_row_mode_decode_graph": True,
            },
            vllm_config=_vllm_config(enforce_eager=True),
        )


def test_block_contract_is_checked_after_backend_refresh() -> None:
    vllm_config = _vllm_config(block_size=16)
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=vllm_config,
    )

    with pytest.raises(ValueError, match="block_size=128"):
        config.validate_finalized_cache_contract(vllm_config)

    vllm_config.cache_config.block_size = 128
    config.validate_finalized_cache_contract(vllm_config)


def test_trace_config_is_parsed_once_into_immutable_filters() -> None:
    trace = DSAOffloadTraceConfig.from_value(
        {
            "enabled": True,
            "points": ["first_sample"],
            "ranks": [0, 2],
        }
    )

    assert trace.enabled is True
    assert trace.points == frozenset({"first_sample"})
    assert trace.ranks == frozenset({0, 2})
