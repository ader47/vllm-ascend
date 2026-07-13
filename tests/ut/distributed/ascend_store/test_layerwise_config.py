"""Unit tests for layerwise KV cache reuse config parsing."""

from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    _DEFAULT_MAX_PREFETCH_LAYERS,
    build_sfa_layerwise_cache_plan,
    build_sfa_layerwise_cache_plan_from_group,
    get_gva_layerwise_config,
    get_layer_load_start_block,
    get_layerwise_config,
    get_layerwise_independent_layers,
    get_layerwise_kv_cache_num_tensors,
    get_layerwise_kv_cache_reuse_layers,
    get_layerwise_num_prefetch_layers,
    get_layerwise_num_shared_buffers,
    get_layerwise_storage_indices,
)


def _sfa_name(layer_id: int, suffix: str) -> str:
    return f"model.layers.{layer_id}.self_attn.{suffix}"


class TestSFALayerwiseCachePlan:
    def test_real_indexers_follow_main_layer_storage_schedule(self):
        specs = {
            **{
                _sfa_name(layer_id, "attn"): SimpleNamespace(page_size_bytes=100)
                for layer_id in range(6)
            },
            **{
                _sfa_name(layer_id, "indexer.k_cache"): SimpleNamespace(
                    page_size_bytes=20
                )
                for layer_id in (0, 2, 4)
            },
        }

        plan = build_sfa_layerwise_cache_plan(
            specs,
            {"layerwise_num_shared_buffers": 2},
        )

        assert plan is not None
        assert len(plan.entries) == 6
        assert [
            index
            for index, entry in enumerate(plan.entries)
            if entry.indexer_layer_name is not None
        ] == [0, 2, 4]
        assert plan.main_pools == (
            (_sfa_name(0, "attn"),),
            (_sfa_name(5, "attn"),),
            (_sfa_name(1, "attn"), _sfa_name(3, "attn")),
            (_sfa_name(2, "attn"), _sfa_name(4, "attn")),
        )
        assert plan.indexer_pools == (
            (_sfa_name(0, "indexer.k_cache"),),
            (
                _sfa_name(2, "indexer.k_cache"),
                _sfa_name(4, "indexer.k_cache"),
            ),
        )
        assert plan.logical_page_bytes == 660
        assert plan.physical_page_bytes == 440
        assert plan.host_page_bytes == 120

    def test_non_sfa_layout_is_not_claimed(self):
        specs = {"model.layers.0.self_attn.attn": SimpleNamespace(page_size_bytes=100)}
        assert build_sfa_layerwise_cache_plan(specs) is None

    def test_collapsed_scheduler_group_uses_preserved_specs(self):
        specs = {
            _sfa_name(0, "attn"): SimpleNamespace(page_size_bytes=100),
            _sfa_name(0, "indexer.k_cache"): SimpleNamespace(page_size_bytes=20),
        }
        group = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(page_size_bytes=20),
            _ascend_sfa_layerwise_cache_specs=specs,
        )

        plan = build_sfa_layerwise_cache_plan_from_group(group)

        assert plan is not None
        assert len(plan.entries) == 1
        assert plan.host_page_bytes == 120


class TestNumSharedBuffers:
    def test_default_equals_num_layers(self):
        assert get_layerwise_num_shared_buffers(27, None) == 27
        assert get_layerwise_num_shared_buffers(27, {}) == 27

    def test_from_extra_config(self):
        cfg = {"layerwise_num_shared_buffers": 6}
        assert get_layerwise_num_shared_buffers(27, cfg) == 6

    def test_string_value_parsed(self):
        cfg = {"layerwise_num_shared_buffers": "6"}
        assert get_layerwise_num_shared_buffers(27, cfg) == 6

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            get_layerwise_num_shared_buffers(27, {"layerwise_num_shared_buffers": True})

    def test_less_than_one_rejected(self):
        with pytest.raises(ValueError):
            get_layerwise_num_shared_buffers(27, {"layerwise_num_shared_buffers": 0})

    def test_default_rejects_zero_layers(self):
        with pytest.raises(ValueError):
            get_layerwise_num_shared_buffers(0, None)
        with pytest.raises(ValueError):
            get_layerwise_num_shared_buffers(0, {})


class TestNumPrefetchLayers:
    def test_default_equals_num_shared_buffers(self):
        assert get_layerwise_num_prefetch_layers(6, None) == 6
        assert get_layerwise_num_prefetch_layers(6, {}) == 6

    def test_default_capped_for_large_shared_buffers(self):
        assert get_layerwise_num_prefetch_layers(100, None) == _DEFAULT_MAX_PREFETCH_LAYERS

    def test_explicit_overrides_default(self):
        assert get_layerwise_num_prefetch_layers(6, {"layerwise_prefetch_layers": 3}) == 3

    def test_less_than_one_rejected(self):
        with pytest.raises(ValueError):
            get_layerwise_num_prefetch_layers(6, {"layerwise_prefetch_layers": 0})


class TestIndependentLayers:
    def test_default_first_and_last(self):
        assert get_layerwise_independent_layers(27, None) == [0, 26]

    def test_all(self):
        result = get_layerwise_independent_layers(5, {"layerwise_independent_layers": "all"})
        assert result == [0, 1, 2, 3, 4]

    def test_comma_separated(self):
        result = get_layerwise_independent_layers(27, {"layerwise_independent_layers": "3,5,10"})
        assert result == [3, 5, 10]

    def test_dedup_and_sort(self):
        result = get_layerwise_independent_layers(27, {"layerwise_independent_layers": "10,3,3"})
        assert result == [3, 10]

    def test_negative_normalizes_to_last(self):
        result = get_layerwise_independent_layers(27, {"layerwise_independent_layers": "-1"})
        assert result == [26]

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            get_layerwise_independent_layers(27, {"layerwise_independent_layers": "27"})

    def test_list_input(self):
        result = get_layerwise_independent_layers(10, {"layerwise_independent_layers": [1, 4]})
        assert result == [1, 4]


class TestLayerwiseConfig:
    def test_no_reuse_when_all_independent(self):
        cfg = get_layerwise_config(27, {"layerwise_independent_layers": "all"})
        assert cfg.has_layer_reuse is False
        assert cfg.prefetch_layer_map == {}

    def test_reuse_when_reused_exceeds_buffers(self):
        # default independent [0, 26] → 25 reused layers, 6 buffers → reuse.
        cfg = get_layerwise_config(27, {"layerwise_num_shared_buffers": 6})
        assert cfg.has_layer_reuse is True
        # prefetch: reused[6]=7 maps to reused[0]=1, reused[7]=8 to reused[1]=2.
        assert cfg.prefetch_layer_map[7] == 1
        assert cfg.prefetch_layer_map[8] == 2

    def test_reuse_layers_none_when_no_reuse(self):
        assert get_layerwise_kv_cache_reuse_layers(27, {"layerwise_independent_layers": "all"}) is None

    def test_reuse_layers_value_when_reuse(self):
        assert get_layerwise_kv_cache_reuse_layers(27, {"layerwise_num_shared_buffers": 6}) == 6

    def test_mtp_layer_can_enable_reuse(self):
        config = {"layerwise_num_shared_buffers": 2}

        assert get_layerwise_config(4, config).has_layer_reuse is False
        assert get_layerwise_config(5, config).has_layer_reuse is True


class TestGVALayerwiseConfig:
    def test_reads_ascend_store_child_config(self):
        ascend_store_config = {
            "backend": "memcache",
            "use_layerwise": True,
            "layerwise_num_shared_buffers": "2",
        }
        config = SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "layerwise_num_shared_buffers": "99",
                "connectors": [
                    {
                        "kv_connector": "SFAPDCpuOffloadConnector",
                        "kv_connector_extra_config": {"use_layerwise": True},
                    },
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_connector_extra_config": ascend_store_config,
                    },
                ],
            },
        )

        assert get_gva_layerwise_config(config) is ascend_store_config

    @pytest.mark.parametrize(
        "extra_config",
        [
            {"backend": "mooncake", "use_layerwise": True},
            {"backend": "memcache", "use_layerwise": False},
        ],
    )
    def test_rejects_unsupported_layerwise_path(self, extra_config):
        config = SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config=extra_config,
        )

        assert get_gva_layerwise_config(config) is None


class TestNumTensors:
    def test_none_when_no_reuse(self):
        assert get_layerwise_kv_cache_num_tensors(27, {"layerwise_independent_layers": "all"}) is None

    def test_independent_plus_shared_buffers(self):
        # default independent [0, 26] (2 layers) + 6 shared buffers = 8.
        assert get_layerwise_kv_cache_num_tensors(27, {"layerwise_num_shared_buffers": 6}) == 8

    def test_no_independent_equals_shared_buffers(self):
        # no independent layers -> only the shared buffers.
        assert (
            get_layerwise_kv_cache_num_tensors(
                27, {"layerwise_num_shared_buffers": 6, "layerwise_independent_layers": ""}
            )
            == 6
        )


class TestStorageIndices:
    def test_full_coverage_no_duplicates(self):
        indices = get_layerwise_storage_indices(27, {"layerwise_num_shared_buffers": 6})
        flat = [layer for slot in indices for layer in slot]
        assert sorted(flat) == list(range(27))
        assert len(flat) == len(set(flat))

    def test_independent_layers_own_slots(self):
        indices = get_layerwise_storage_indices(27, {"layerwise_num_shared_buffers": 6})
        # first two slots are the independent layers [0] and [26].
        assert indices[0] == [0]
        assert indices[1] == [26]

    def test_reused_round_robin_grouping(self):
        indices = get_layerwise_storage_indices(27, {"layerwise_num_shared_buffers": 6})
        # independent [0], [26] then 6 reused slots over reused [1..25].
        reused_slots = indices[2:]
        assert len(reused_slots) == 6
        # slot 0 holds reused_layers[0::6] = [1, 7, 13, 19, 25].
        assert reused_slots[0] == [1, 7, 13, 19, 25]

    def test_no_reuse_one_slot_per_layer(self):
        indices = get_layerwise_storage_indices(5, {"layerwise_independent_layers": "all"})
        assert indices == [[0], [1], [2], [3], [4]]


class TestGetLayerLoadStartBlock:
    # default independent layers are the first and last: [0, 26]
    INDEPENDENT = [0, 26]
    BLOCK_SIZE = 16

    def test_independent_skips_hbm_cached_blocks(self):
        # layer 0 independent, HBM has 32 tokens (2 blocks) -> start at block 2
        assert get_layer_load_start_block(0, self.INDEPENDENT, 32, self.BLOCK_SIZE, True) == 2

    def test_independent_no_hbm_hit_starts_at_zero(self):
        assert get_layer_load_start_block(0, self.INDEPENDENT, 0, self.BLOCK_SIZE, True) == 0

    def test_shared_layer_always_starts_at_zero(self):
        # layer 5 shared: HBM hits are unreliable (time-multiplexed) -> always 0
        assert get_layer_load_start_block(5, self.INDEPENDENT, 32, self.BLOCK_SIZE, True) == 0
        assert get_layer_load_start_block(5, self.INDEPENDENT, 0, self.BLOCK_SIZE, True) == 0

    def test_no_reuse_all_layers_skip_hbm(self):
        # without layer reuse, even a shared layer skips HBM-cached blocks
        assert get_layer_load_start_block(5, self.INDEPENDENT, 32, self.BLOCK_SIZE, False) == 2
