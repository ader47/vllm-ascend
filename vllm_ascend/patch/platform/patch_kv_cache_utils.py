# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import copy
import math
from collections import defaultdict

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    DIRECT_SFA_HOST_CACHE_BYTES,
    is_direct_sfa_host_offload,
)
from vllm_ascend.utils import vllm_version_is

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_generate_scheduler_kv_cache_config = (
    vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config
)
_orig_get_kv_cache_config_from_groups = (
    vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
)
_orig_max_memory_usage_bytes_from_groups = (
    vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups
)

_SFA_LAYERWISE_SCHEDULER_SPECS_ATTR = "_ascend_sfa_layerwise_cache_specs"


def _get_direct_sfa_indexer_specs(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> dict[str, AscendSFAIndexerCacheSpec] | None:
    """Return physical indexer specs for the direct host-only main-KV mode."""
    if not is_direct_sfa_host_offload(vllm_config):
        return None
    if len(kv_cache_groups) != 1 or not isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        raise ValueError(
            "Direct SFA host offload requires one UniformType KV cache group; "
            f"got {len(kv_cache_groups)} groups."
        )

    layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    main_specs = {
        name: spec
        for name, spec in layer_specs.items()
        if isinstance(spec, AscendMLAAttentionSpec)
    }
    indexer_specs = {
        name: spec
        for name, spec in layer_specs.items()
        if isinstance(spec, AscendSFAIndexerCacheSpec)
    }
    unsupported = {
        name: type(spec).__name__
        for name, spec in layer_specs.items()
        if name not in main_specs and name not in indexer_specs
    }
    if not main_specs or not indexer_specs or unsupported:
        raise ValueError(
            "Direct SFA host offload expects main MLA specs and real SFA "
            "indexer specs only; "
            f"main={len(main_specs)}, indexer={len(indexer_specs)}, "
            f"unsupported={unsupported}."
        )
    return indexer_specs


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Allocate paged HBM only for real indexer owners in direct offload.

    Main SFA specs remain in the logical Uniform group so the scheduler still
    assigns their token blocks. Their physical K/V bytes live in the connector
    host cache and use fixed per-layer staging tensors during model execution.
    """
    indexer_specs = _get_direct_sfa_indexer_specs(
        vllm_config, kv_cache_groups
    )
    if indexer_specs is None:
        return _orig_get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

    physical_page_bytes = sum(
        spec.page_size_bytes for spec in indexer_specs.values()
    )
    if physical_page_bytes <= 0:
        raise ValueError("Direct SFA host offload has no physical indexer pages.")
    layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    host_page_bytes = sum(
        spec.page_size_bytes
        for spec in layer_specs.values()
        if isinstance(spec, AscendMLAAttentionSpec)
    )
    if host_page_bytes <= 0:
        raise ValueError("Direct SFA host offload has no logical main KV pages.")
    max_hbm_blocks = available_memory // physical_page_bytes
    max_host_blocks = DIRECT_SFA_HOST_CACHE_BYTES // host_page_bytes
    num_blocks = min(max_hbm_blocks, max_host_blocks)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    if num_blocks > max_hbm_blocks or num_blocks > max_host_blocks:
        raise ValueError(
            "Direct SFA host-offload block override exceeds physical "
            "capacity: "
            f"requested={num_blocks}, hbm_limit={max_hbm_blocks}, "
            f"host_limit={max_host_blocks}."
        )
    required_blocks = cdiv(
        vllm_config.model_config.max_model_len,
        kv_cache_groups[0].kv_cache_spec.block_size,
    )
    if num_blocks < required_blocks:
        raise ValueError(
            "Direct SFA host offload cannot hold one max-length request: "
            f"num_blocks={num_blocks}, required_blocks={required_blocks}, "
            f"max_model_len={vllm_config.model_config.max_model_len}."
        )
    kv_cache_tensors = [
        KVCacheTensor(
            size=spec.page_size_bytes * num_blocks,
            shared_by=[layer_name],
        )
        for layer_name, spec in indexer_specs.items()
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _ascend_max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Exclude host-only main KV from the startup maximum-length check."""
    indexer_specs = _get_direct_sfa_indexer_specs(
        vllm_config, kv_cache_groups
    )
    if indexer_specs is None:
        return _orig_max_memory_usage_bytes_from_groups(
            vllm_config, kv_cache_groups
        )
    return sum(
        spec.max_memory_usage_bytes(vllm_config)
        for spec in indexer_specs.values()
    )


def _ascend_generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """Retain split-SFA layer specs after vLLM collapses scheduler specs.

    vLLM replaces a UniformType group with one arbitrary representative spec
    before constructing the scheduler. That is sufficient for block-table
    management, but AscendStore also needs every main/indexer page size to
    allocate its fixed GVA transfer pages. Attach the original mapping as
    Ascend-only metadata instead of changing the representative manager spec.
    """

    scheduler_config = _orig_generate_scheduler_kv_cache_config(kv_cache_configs)
    worker_config = kv_cache_configs[0]
    for scheduler_group, worker_group in zip(
        scheduler_config.kv_cache_groups,
        worker_config.kv_cache_groups,
    ):
        worker_spec = worker_group.kv_cache_spec
        if not isinstance(worker_spec, UniformTypeKVCacheSpecs):
            continue
        layer_specs = worker_spec.kv_cache_specs
        if not any(
            layer_name.endswith(".self_attn.indexer.k_cache")
            for layer_name in layer_specs
        ):
            continue
        setattr(
            scheduler_group,
            _SFA_LAYERWISE_SCHEDULER_SPECS_ATTR,
            copy.deepcopy(layer_specs),
        )
    return scheduler_config


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = (
    _ascend_get_kv_cache_config_from_groups
)
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = (
    _ascend_max_memory_usage_bytes_from_groups
)
vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config = (
    _ascend_generate_scheduler_kv_cache_config
)
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# vllm v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
if vllm_version_is("0.23.0"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
else:
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.engine.core.generate_scheduler_kv_cache_config = (
    _ascend_generate_scheduler_kv_cache_config
)
