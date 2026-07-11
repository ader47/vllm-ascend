"""Unit tests for layerwise KV cache reuse config parsing."""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_config import (
    _DEFAULT_MAX_PREFETCH_LAYERS,
    get_layer_load_start_block,
    get_layerwise_config,
    get_layerwise_independent_layers,
    get_layerwise_kv_cache_num_tensors,
    get_layerwise_kv_cache_reuse_layers,
    get_layerwise_layout_class_grouping,
    get_layerwise_num_prefetch_layers,
    get_layerwise_num_shared_buffers,
    get_layerwise_storage_indices,
)


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


class TestLayoutClassGrouping:
    """Layout-class-aware reuse: layers of different kv_cache tuple-layouts
    (e.g. main MLA 2-tuple vs sparse indexer 3-tuple) must never share a
    buffer. Reproduces the P-node reshape crash from mixing classes."""

    @staticmethod
    def _no_mixed_slots(storage_indices, layer_class_list):
        for slot in storage_indices:
            classes = {layer_class_list[i] for i in slot}
            assert len(classes) == 1, f"mixed-layout slot {slot}: " \
                f"{[(i, layer_class_list[i]) for i in slot]}"

    def test_single_class_matches_global_grouping(self):
        extra = {"layerwise_num_shared_buffers": "2"}
        classes = ["M"] * 8
        si, pm, nt = get_layerwise_layout_class_grouping(8, classes, extra)
        # independent [0,7] + 2 shared slots for reused [1..6] -> 4 buffers
        assert nt == 4
        self._no_mixed_slots(si, classes)
        # matches the old global storage_indices
        assert sorted(si) == sorted(get_layerwise_storage_indices(8, extra))
        # prefetch within class (trivially, single class)
        for layer, mate in pm.items():
            assert classes[layer] == classes[mate]

    def test_grouped_classes_never_mix(self):
        # [M,M,M,M, I,I,I,I] grouped, nsb=2 -- the crash scenario
        extra = {"layerwise_num_shared_buffers": "2"}
        classes = ["M"] * 4 + ["I"] * 4
        si, pm, nt = get_layerwise_layout_class_grouping(8, classes, extra)
        self._no_mixed_slots(si, classes)
        # independent [0,7]; M reused [1,2] -> 2 slots; I reused [3,4,5,6] -> 2 slots
        # num_tensors = 2 (independent) + 2 (M) + 2 (I) = 6
        assert nt == 6
        for layer, mate in pm.items():
            assert classes[layer] == classes[mate], f"cross-class mate {layer}->{mate}"

    def test_interleaved_classes_never_mix(self):
        # [M,I,M,I,...] interleaved, nsb=2
        extra = {"layerwise_num_shared_buffers": "2"}
        classes = ["M", "I"] * 6  # 12 layers
        si, pm, nt = get_layerwise_layout_class_grouping(12, classes, extra)
        self._no_mixed_slots(si, classes)
        for layer, mate in pm.items():
            assert classes[layer] == classes[mate]

    def test_irregular_indexer_pattern_like_glm(self):
        # Mirrors the diagnosed GLM case: irregular idx=True/False, 12 layers,
        # nsb=2. Old global grouping would mix; layout-class grouping must not.
        extra = {"layerwise_num_shared_buffers": "2"}
        classes = ["T", "T", "T", "F", "F", "F", "T", "F", "F", "F", "T", "F"]
        si, pm, nt = get_layerwise_layout_class_grouping(12, classes, extra)
        self._no_mixed_slots(si, classes)
        for layer, mate in pm.items():
            assert classes[layer] == classes[mate]
        # num_tensors == number of buffers actually produced
        assert nt == len(si)

    def test_small_class_no_reuse_within_it(self):
        # A class with <= num_shared_buffers members reuses nothing.
        extra = {"layerwise_num_shared_buffers": "4"}
        classes = ["M"] * 3 + ["I"] * 10  # M has 3 < nsb=4 -> no M reuse
        si, pm, nt = get_layerwise_layout_class_grouping(13, classes, extra)
        self._no_mixed_slots(si, classes)
        # M layers (0,1,2) -- 0 independent, 1,2 reused but only 2 < nsb=4 -> each own buffer
        for layer, mate in pm.items():
            assert classes[layer] == classes[mate]

