from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping

_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_EXTRA_CONFIG_KEY_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS = "layerwise_independent_layers"

# Prefetch auto-defaults to num_shared_buffers (the max useful overlap -- a reused
# load gates on its mate num_shared_buffers layers back). Capped so a very large
# num_shared_buffers doesn't burst-submit too many loads at layer 0.
_DEFAULT_MAX_PREFETCH_LAYERS = 8

_SFA_MAIN_SUFFIX = ".self_attn.attn"
_SFA_INDEXER_SUFFIX = ".self_attn.indexer.k_cache"
_SFA_LAYERWISE_SCHEDULER_SPECS_ATTR = "_ascend_sfa_layerwise_cache_specs"


@dataclass(frozen=True)
class SFALayerwiseCacheEntry:
    """Cache owners transferred by one transformer-layer callback."""

    main_layer_name: str
    indexer_layer_name: str | None


@dataclass(frozen=True)
class SFALayerwiseCachePlan:
    """Shared interpretation of split SFA caches for layerwise offload.

    The scheduler still exposes one UniformType KV cache group, but runtime
    callbacks are emitted only for main attention layers.  ``entries`` bridges
    those views, while the pool lists describe independent main and indexer
    scratch tensors that follow the same transformer-layer reuse schedule.
    """

    entries: tuple[SFALayerwiseCacheEntry, ...]
    main_pools: tuple[tuple[str, ...], ...]
    indexer_pools: tuple[tuple[str, ...], ...]
    logical_page_bytes: int
    physical_page_bytes: int
    host_page_bytes: int


def _sfa_layer_id(layer_name: str) -> int:
    marker = ".layers."
    if marker not in layer_name:
        raise ValueError(f"Cannot extract an SFA layer id from {layer_name!r}")
    layer_id = layer_name.split(marker, 1)[1].split(".", 1)[0]
    try:
        return int(layer_id)
    except ValueError as err:
        raise ValueError(f"Invalid SFA layer id in {layer_name!r}") from err


def _partition_pool_by_page_size(
    layer_names: list[str],
    layer_specs: Mapping[str, Any],
) -> list[tuple[str, ...]]:
    """Keep differently sized cache owners out of the same physical tensor."""

    owners_by_page_size: dict[int, list[str]] = {}
    for layer_name in layer_names:
        page_size = int(layer_specs[layer_name].page_size_bytes)
        owners_by_page_size.setdefault(page_size, []).append(layer_name)
    return [tuple(owners) for owners in owners_by_page_size.values()]


def build_sfa_layerwise_cache_plan(
    layer_specs: Mapping[str, Any],
    extra_config: dict[str, Any] | None = None,
) -> SFALayerwiseCachePlan | None:
    """Describe split SFA caches without turning indexers into callbacks.

    ``AscendSFAIndexerCacheSpec`` and the main MLA specs deliberately remain in
    one UniformType cache group.  vLLM therefore allocates one tensor per cache
    owner, while AscendStore receives one callback per main transformer layer.
    This plan pairs an optional real indexer owner with that callback and maps
    both cache families onto the main layer's reuse schedule.

    Returning ``None`` keeps non-SFA and legacy combined-cache layouts on their
    existing paths.
    """

    indexer_names = [name for name in layer_specs if name.endswith(_SFA_INDEXER_SUFFIX)]
    if not indexer_names:
        return None

    main_names = [name for name in layer_specs if name.endswith(_SFA_MAIN_SUFFIX)]
    recognized_names = set(main_names) | set(indexer_names)
    if not main_names or recognized_names != set(layer_specs):
        return None

    main_by_layer_id: dict[int, str] = {}
    for main_name in main_names:
        layer_id = _sfa_layer_id(main_name)
        if layer_id in main_by_layer_id:
            raise ValueError(f"Multiple SFA main cache owners for layer {layer_id}")
        main_by_layer_id[layer_id] = main_name

    indexer_by_layer_id: dict[int, str] = {}
    for indexer_name in indexer_names:
        layer_id = _sfa_layer_id(indexer_name)
        if layer_id not in main_by_layer_id:
            raise ValueError(
                f"SFA indexer cache {indexer_name!r} has no main attention cache owner"
            )
        if layer_id in indexer_by_layer_id:
            raise ValueError(f"Multiple SFA indexer cache owners for layer {layer_id}")
        indexer_by_layer_id[layer_id] = indexer_name

    ordered_layer_ids = sorted(main_by_layer_id)
    entries = tuple(
        SFALayerwiseCacheEntry(
            main_layer_name=main_by_layer_id[layer_id],
            indexer_layer_name=indexer_by_layer_id.get(layer_id),
        )
        for layer_id in ordered_layer_ids
    )

    storage_indices = get_layerwise_storage_indices(len(entries), extra_config)
    main_pools: list[tuple[str, ...]] = []
    indexer_pools: list[tuple[str, ...]] = []
    for slot in storage_indices:
        slot_main_names = [entries[index].main_layer_name for index in slot]
        main_pools.extend(_partition_pool_by_page_size(slot_main_names, layer_specs))

        slot_indexer_names: list[str] = []
        for index in slot:
            indexer_name = entries[index].indexer_layer_name
            if indexer_name is not None:
                slot_indexer_names.append(indexer_name)
        indexer_pools.extend(
            _partition_pool_by_page_size(slot_indexer_names, layer_specs)
        )

    logical_page_bytes = sum(int(spec.page_size_bytes) for spec in layer_specs.values())
    physical_page_bytes = sum(
        int(layer_specs[pool[0]].page_size_bytes)
        for pool in (*main_pools, *indexer_pools)
    )
    host_page_bytes = max(
        int(layer_specs[entry.main_layer_name].page_size_bytes)
        + (
            int(layer_specs[entry.indexer_layer_name].page_size_bytes)
            if entry.indexer_layer_name is not None
            else 0
        )
        for entry in entries
    )

    return SFALayerwiseCachePlan(
        entries=entries,
        main_pools=tuple(main_pools),
        indexer_pools=tuple(indexer_pools),
        logical_page_bytes=logical_page_bytes,
        physical_page_bytes=physical_page_bytes,
        host_page_bytes=host_page_bytes,
    )


def build_sfa_layerwise_cache_plan_from_group(
    kv_cache_group: Any,
    extra_config: dict[str, Any] | None = None,
) -> SFALayerwiseCachePlan | None:
    """Build a split-SFA plan from worker or collapsed scheduler metadata."""

    group_spec = getattr(kv_cache_group, "kv_cache_spec", None)
    layer_specs = getattr(group_spec, "kv_cache_specs", None)
    if layer_specs is None:
        # Upstream scheduler configuration keeps one representative spec.
        # patch_kv_cache_utils preserves the full split-SFA mapping here so
        # GVA page sizing stays identical on scheduler and worker processes.
        layer_specs = getattr(
            kv_cache_group,
            _SFA_LAYERWISE_SCHEDULER_SPECS_ATTR,
            None,
        )
    if layer_specs is None:
        return None
    return build_sfa_layerwise_cache_plan(layer_specs, extra_config)


@dataclass(frozen=True)
class LayerwiseConfig:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int | None]
    has_layer_reuse: bool


def get_gva_layerwise_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the config for the supported GVA layerwise KV pool path.

    Layer reuse is implemented by AscendStore's memcache layerwise path.  In a
    MultiConnector setup the relevant settings belong to the AscendStore child,
    not to the MultiConnector or any sibling connector.
    """

    if kv_transfer_config is None:
        return None

    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    root_extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if connector_name in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
        connector_configs = [
            {
                "kv_connector": connector_name,
                "kv_connector_extra_config": root_extra_config,
            }
        ]
    elif connector_name == "MultiConnector":
        connector_configs = root_extra_config.get("connectors", [])
    else:
        return None

    for connector_config in connector_configs:
        if not isinstance(connector_config, dict):
            continue
        if connector_config.get("kv_connector") not in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
            continue
        extra_config = connector_config.get("kv_connector_extra_config") or {}
        backend = str(extra_config.get("backend", "mooncake")).lower()
        if backend == "memcache" and extra_config.get("use_layerwise", False):
            return extra_config
    return None


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def get_layerwise_num_shared_buffers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int:
    value = extra_config.get(_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS) if extra_config else None
    if value is None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1 to default num_shared_buffers")
        return num_layers
    num_shared_buffers = _parse_int_config(value, _EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS)
    if num_shared_buffers < 1:
        raise ValueError(f"{_EXTRA_CONFIG_KEY_NUM_SHARED_BUFFERS} must be at least 1")
    return num_shared_buffers


def get_layerwise_num_prefetch_layers(
    num_shared_buffers: int,
    extra_config: dict[str, Any] | None = None,
) -> int:
    value = extra_config.get(_EXTRA_CONFIG_KEY_PREFETCH_LAYERS) if extra_config else None
    if value is None:
        # Default = num_shared_buffers: overlap is capped at nsb-1 by the gate, so
        # this is the sweet spot. Clip pathological nsb to bound the layer-0 burst.
        return min(num_shared_buffers, _DEFAULT_MAX_PREFETCH_LAYERS)
    num_prefetch_layers = _parse_int_config(value, _EXTRA_CONFIG_KEY_PREFETCH_LAYERS)
    if num_prefetch_layers < 1:
        raise ValueError(f"{_EXTRA_CONFIG_KEY_PREFETCH_LAYERS} must be at least 1")
    return num_prefetch_layers


def _parse_layer_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(index.strip()) for index in value.split(",")]
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, Iterable):
        return [_parse_int_config(index, _EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS) for index in value]
    raise TypeError(
        f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} must be a comma-separated string or an iterable of integers"
    )


def get_layerwise_independent_layers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> list[int]:
    value = extra_config.get(_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS) if extra_config else None
    if value is None:
        layer_indices = [0, num_layers - 1]
    elif isinstance(value, str) and value.strip().lower() == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = _parse_layer_indices(value)

    normalized_indices = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            layer_index += num_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(
                f"{_EXTRA_CONFIG_KEY_INDEPENDENT_LAYERS} contains "
                f"out-of-range layer index {layer_index}; valid range is "
                f"[0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)

    return sorted(normalized_indices)


def get_layerwise_config(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseConfig:
    num_shared_buffers = get_layerwise_num_shared_buffers(num_layers, extra_config)
    num_prefetch_layers = get_layerwise_num_prefetch_layers(num_shared_buffers, extra_config)
    independent_layers = get_layerwise_independent_layers(num_layers, extra_config)
    independent_layer_indices = set(independent_layers)
    reused_layers = [i for i in range(num_layers) if i not in independent_layer_indices]
    has_layer_reuse = len(reused_layers) > num_shared_buffers

    prefetch_layer_map: dict[int, int | None] = {}
    if has_layer_reuse:
        for next_index in range(num_shared_buffers, len(reused_layers)):
            prefetch_layer_map[reused_layers[next_index]] = reused_layers[next_index - num_shared_buffers]

    return LayerwiseConfig(
        num_shared_buffers=num_shared_buffers,
        num_prefetch_layers=num_prefetch_layers,
        independent_layers=independent_layers,
        prefetch_layer_map=prefetch_layer_map,
        has_layer_reuse=has_layer_reuse,
    )


def get_layerwise_kv_cache_reuse_layers(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int | None:
    layerwise_config = get_layerwise_config(num_layers, extra_config)
    if not layerwise_config.has_layer_reuse:
        return None
    return layerwise_config.num_shared_buffers


def get_layerwise_kv_cache_num_tensors(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> int | None:
    """Number of distinct KV cache buffers after layer reuse.

    Returns ``None`` when layer reuse is disabled (one buffer per layer).
    Otherwise returns the count of merged tensors the model runner will
    allocate: each independent layer keeps its own buffer, and the reused
    layers share ``num_shared_buffers`` buffers. The worker uses this to size
    the memory-inflation factor so total allocation stays within budget.
    """
    config = get_layerwise_config(num_layers, extra_config)
    if not config.has_layer_reuse:
        return None
    return len(config.independent_layers) + config.num_shared_buffers


def get_layerwise_storage_indices(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Group layer indices into shared storage slots for layer reuse.

    Each inner list holds the layer indices that time-multiplex one shared
    buffer. Independent layers each occupy their own slot; the reused layers
    are distributed across ``num_shared_buffers`` slots round-robin so that
    ``reused_layers[k]`` shares a buffer with
    ``reused_layers[k + num_shared_buffers]`` — matching the prefetch map
    computed in :func:`get_layerwise_config`.
    """
    config = get_layerwise_config(num_layers, extra_config)
    independent_set = set(config.independent_layers)
    reused_layers = [layer for layer in range(num_layers) if layer not in independent_set]
    storage_indices: list[list[int]] = [[layer] for layer in config.independent_layers]
    for slot in range(config.num_shared_buffers):
        members = list(range(slot, len(reused_layers), config.num_shared_buffers))
        if members:
            storage_indices.append([reused_layers[m] for m in members])
    return storage_indices


def get_layer_load_start_block(
    layer_id: int,
    independent_layers: list[int],
    vllm_cached_tokens: int,
    block_size: int,
    has_layer_reuse: bool,
) -> int:
    """First pool block to load for ``layer_id``.

    Without layer reuse every layer behaves the same and starts right after
    the HBM-cached blocks. With layer reuse, independent layers still skip
    HBM-cached blocks (their dedicated buffer keeps that KV valid), while
    shared (time-multiplexed) layers must reload every block from block 0
    because HBM hits are not reliable.
    """
    if not has_layer_reuse or layer_id in independent_layers:
        return vllm_cached_tokens // block_size
    return 0
