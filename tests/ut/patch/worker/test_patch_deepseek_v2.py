# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _resolve_skip_topk,
    _should_skip_indexer_init,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_declared_shared_topology_overrides_legacy_full_policy():
    config = _config(
        indexer_types=["full", "full", "shared"],
        index_topk_freq=1,
    )

    assert _resolve_skip_topk(config, "model.layers.2.self_attn")


def test_declared_full_topology_overrides_legacy_skip_policy():
    config = _config(
        indexer_types=["full", "full", "full"],
        index_topk_pattern=["F", "F", "S"],
    )

    assert not _resolve_skip_topk(config, "model.layers.2.self_attn")


def test_mtp_layer_computes_topk_on_first_iteration():
    config = _config(
        indexer_types=["full"] * 80 + ["shared"],
        index_topk_pattern=["F"] * 80 + ["S"],
    )

    assert not _resolve_skip_topk(config, "model.layers.80.self_attn")
    assert not _should_skip_indexer_init(
        config,
        "model.layers.80.self_attn",
        skip_topk=True,
    )
