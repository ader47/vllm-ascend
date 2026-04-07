from __future__ import annotations

import importlib
import math
import threading
from collections.abc import Generator

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
    PoolKey
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    record_failed_blocks,
)

backend_map = {
    "mooncake": {
        "name": "MooncakeBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
    },
    "memcache": {
        "name": "MemcacheBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend",
    },
    "yuanrong": {
        "name": "YuanrongBackend",
        "path": "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend",
    },
}


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwize: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(model_config, "hf_text_config", None)
        hf_config = getattr(model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        self.use_sparse = hasattr(model_config.hf_text_config, "index_topk")
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        self._invalid_block_ids: set[int] = set()
        self._invalid_block_ids_lock = threading.Lock()
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
        self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
        self.use_mamba = self._uses_mamba_kv_cache(self.use_hybrid, kv_cache_config)
        self.original_block_size = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        cp_scale = self.pcp_size * self.dcp_size
        self.grouped_block_size = [block_size * cp_scale for block_size in self.original_block_size]
        requested_hash_block_size = vllm_config.cache_config.hash_block_size
        if not isinstance(requested_hash_block_size, int):
            requested_hash_block_size = None
        self.hash_block_size = (
            requested_hash_block_size if requested_hash_block_size is not None else min(self.original_block_size)
        ) * cp_scale
        for group_block_size in self.grouped_block_size:
            assert group_block_size % self.hash_block_size == 0, "block_size must be divisible by hash_block_size"
        self.block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*self.grouped_block_size)
        self.num_kv_cache_groups = len(self.grouped_block_size)
        self.kv_cache_group_families = self._infer_group_families()
        self.group_uses_align_state = self._infer_group_uses_align_state()
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        if self.use_layerwise and self.num_kv_cache_groups > 1:
            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")

        logger.info(
            "use_hybrid: %s, use_mamba: %s, num_kv_cache_groups: %s, hash_block_size: %s, lcm_block_size: %s",
            self.use_hybrid,
            self.use_mamba,
            self.num_kv_cache_groups,
            self.hash_block_size,
            self.lcm_block_size,
        )
        self.current_layer = 0
        self.num_layers = model_config.get_num_layers(parallel_config)

        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()

        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
            self.head_or_tp_rank = self.tp_rank // self.put_step
        else:
            self.head_or_tp_rank = self.tp_rank
            self.put_step = 1

        partitions = None
        if self.kv_role == "kv_consumer" and self.consumer_is_to_put:
            num_hidden_layers = model_config.hf_text_config.num_hidden_layers
            partition_list_str = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "prefill_pp_layer_partition", None
            )
            prefill_pp_size = int(vllm_config.kv_transfer_config.kv_connector_extra_config.get("prefill_pp_size", 1))

            if partition_list_str is not None:
                try:
                    partitions = [int(layer) for layer in partition_list_str.split(",")]
                except ValueError as err:
                    raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
                if len(partitions) != prefill_pp_size:
                    raise ValueError(f"{len(partitions)=} does not match {prefill_pp_size=}.")
                if sum(partitions) != num_hidden_layers:
                    raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
            else:
                layers_per_partition = num_hidden_layers // prefill_pp_size
                partitions = [layers_per_partition for _ in range(prefill_pp_size)]

                if remaining_layers := num_hidden_layers % prefill_pp_size:
                    for i in range(2, remaining_layers + 2):
                        partitions[-i] += 1

        self.metadata: list[KeyMetadata] = []
        for group_id in range(self.num_kv_cache_groups):
            # the mamba kv_heads is not same with the full attention, can't share the cache data
            group_tp_rank = self.tp_rank if self.group_uses_align_state[group_id] else self.head_or_tp_rank
            self.metadata.append(
                KeyMetadata(
                    model_config.model.rstrip("/").split("/")[-1],
                    group_tp_rank,
                    self.pcp_rank,
                    self.dcp_rank,
                    self.pp_rank,
                    group_id,
                )
            )

        self.token_database = ChunkedTokenDatabase(
            self.metadata, self.grouped_block_size, partitions, self.use_hybrid, self.hash_block_size
        )

        backend = backend_map.get(self.backend.lower())
        assert backend is not None
        backend_path = backend.get("path")
        backend_name = backend.get("name")
        assert backend_path is not None and backend_name is not None
        backend_module = importlib.import_module(backend_path)
        real_backend = getattr(backend_module, backend_name)

        backend_kwargs = {}
        if self.backend.lower() in {"mooncake", "memcache"}:
            # DSV4 exposes compress_ratios; only use lazy store init for this
            # compressed-model path.
            backend_kwargs["lazy_init"] = self.use_compress
        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config,
            **backend_kwargs,
        )
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = False
        if kv_event_config and kv_event_config.enable_kv_cache_events:
            self.enable_kv_events = True

        self.kv_send_thread: KVTransferThread | None = None
        self.kv_recv_thread: KVTransferThread | None = None

        self.finished_store_req: set[str] = set()

        self.layer_load_tasks = [[] for i in range(self.num_layers)]
        self.layer_save_tasks = [[] for i in range(self.num_layers)]
        self.layer_load_finished_events = None
        self.layer_save_finished_events = None
        # req_id, layer_id, block info
        self._request_addr_tracker: dict[str, dict[int, dict]] = {}

        import os
        self.num_reuse_layers = 3   # TODO 当作参数，配置方法？
        # self.num_reuse_layers = 30   # TODO 当作参数，配置方法？
        self.layer_next_map = {i:i+self.num_reuse_layers for i in range(self.num_layers - self.num_reuse_layers)}
        self.independent_layers = []    # TODO 不是必要的
        self.offload_start_ids = [i for i in range(self.num_reuse_layers)]
        self.layers_need_to_save = [i for i in range(self.num_layers) if i not in self.independent_layers]
        self.sync_save_events = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache_tuple = self._as_cache_tuple(first_kv_cache_tuple)
        first_kv_cache = first_kv_cache_tuple[0]

        self.num_blocks = (
            self.kv_cache_config.num_blocks if self.kv_cache_config is not None else first_kv_cache.shape[0]
        )
        logger.info("num_blocks: %s", self.num_blocks)
        self.group_kv_caches_base_addr: dict[int, list[int]] = {}
        self.group_block_len: dict[int, list[int]] = {}
        self.group_block_stride: dict[int, list[int]] = {}
        self.kv_caches = kv_caches
        self.group_kv_cache_families: dict[int, str] = {
            group_id: self._get_group_family(self.kv_cache_group_families, group_id)
            for group_id in range(self.num_kv_cache_groups)
        }
        self.group_num_layers: dict[int, int] = {}

        logger.info(
            "Registering KV_Caches. use_mla: %s, use_sparse: %s, shape %s",
            self.use_mla,
            self.use_sparse,
            first_kv_cache.shape,
        )

        registered_regions: dict[int, tuple[int, int]] = {}
        for cache_or_caches in kv_caches.values():
            for cache in self._as_cache_tuple(cache_or_caches):
                base_addr = cache.data_ptr()
                _, _, region_len, _ = self._get_cache_block_metadata(cache)
                if not isinstance(region_len, int):
                    region_len = 0
                storage_key = self._get_storage_key(cache)
                start = base_addr
                end = base_addr + region_len
                if storage_key in registered_regions:
                    old_start, old_end = registered_regions[storage_key]
                    registered_regions[storage_key] = (min(old_start, start), max(old_end, end))
                else:
                    registered_regions[storage_key] = (start, end)

        ptrs = [start for start, _ in registered_regions.values()]
        lengths = [end - start for start, end in registered_regions.values()]

        if self.kv_cache_config is not None and self.use_hybrid:
            for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                self._infer_cache_group_metadata(group_id, group_spec.layer_names)
        else:
            self._infer_cache_group_metadata(0, list(kv_caches.keys()))

        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_group_buffers(
            self.group_kv_caches_base_addr,
            self.group_block_len,
            self.group_block_stride,
            cache_role="kv",
            group_cache_families=self.group_kv_cache_families,
            group_num_layers=self.group_num_layers,
        )

        if self.use_layerwise:
            self.get_event = threading.Event()
            self.layer_load_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.layer_save_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.sync_save_events = [torch.npu.Event() for i in range(self.num_layers)]
            if self.kv_role in ["kv_producer", "kv_both"]:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    ready_event_sending,
                    self.num_layers,
                    self.layer_save_finished_events,
                    self.sync_save_events,
                    self.enable_kv_events,
                    self.layer_transfer_finished_events
                )
                self.kv_send_thread.start()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store,
                self.token_database,
                self.grouped_block_size,
                self.tp_rank,
                self.dcp_size,
                ready_event,
                self.get_event,
                self.layer_load_finished_events,
                self.layer_save_finished_events
            )
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    self.put_step,
                    self.kv_role,
                    ready_event_sending,
                    self.group_uses_align_state,
                    self.enable_kv_events,
                )
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store,
                    self.token_database,
                    self.grouped_block_size,
                    self.tp_rank,
                    self.dcp_size,
                    ready_event,
                    self._invalid_block_ids,
                    self._invalid_block_ids_lock,
                )
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        if len(metadata.requests) == 0:
            return
        self.current_layer = 0
        for request in metadata.requests:
            if self.use_layerwise:
                self.process_layer_data(request)
            else:
                load_spec = request.load_spec
                if load_spec is None or not load_spec.can_load:  # load =0
                    continue
                token_len = request.token_len_chunk
                if (load_spec.kvpool_cached_tokens % self.block_size != 0) and (
                    load_spec.kvpool_cached_tokens == token_len - 1
                ):
                    token_len = request.load_spec.kvpool_cached_tokens + 1
                else:
                    token_len = request.load_spec.kvpool_cached_tokens
                request.load_spec.token_len = token_len
                if self.load_async:
                    self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                        request,
                    )
                else:
                    addr_list = []
                    size_list = []
                    key_list = []
                    block_id_list: list[int] = []
                    for group_id in load_group_ids:
                        block_ids = request.block_ids_by_group[group_id]
                        group_block_size = self.grouped_block_size[group_id]
                        mask_num = request.load_spec.vllm_cached_tokens // group_block_size * group_block_size
                        skip_null = (
                            group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
                        )
                        for start, end, key, _ in self.token_database.process_tokens_with_block_ids(
                            token_len,
                            request.block_hashes,
                            block_ids,
                            mask_num,
                            kv_cache_group_id=group_id,
                            skip_null_blocks=skip_null,
                        ):
                            addr, size, block_id = self.token_database.prepare_value(
                                start,
                                end,
                                block_ids,
                                kv_cache_group_id=group_id,
                            )
                            key_list.append(key.to_string())
                            addr_list.append(addr)
                            size_list.append(size)
                            block_id_list.append(block_id)
                    if not key_list:
                        continue
                    key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
                    addr_list_c = (
                        addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
                    )
                    size_list_c = (
                        size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
                    )
                    self.m_store.get(key_list_c, addr_list_c, size_list_c)
        # TODO 这里的请求释放可能有问题
        # logger.info(f">>>>>>>>>>>> metadata.requests {len(metadata.requests)} metadata.unfinished_request_ids {metadata.unfinished_request_ids}")
        if self.use_layerwise and metadata.unfinished_request_ids:
            for layer_id in self.offload_start_ids:
                layer_load_task = self.layer_load_tasks[layer_id]
                self.kv_recv_thread.add_request((None, layer_load_task, layer_id))

    def process_layer_data(self, request: ReqMeta) -> None:
        if request.block_keys_by_layer is not None and request.starts is not None and request.ends is not None:
            starts = request.starts
            ends = request.ends
            keys_by_layer = request.block_keys_by_layer
        else:
            token_len = request.token_len_chunk
            starts, ends, keys = [], [], []
            for start, end, key in self.token_database.process_tokens(
                    token_len, request.block_hashes, req_id=f"{request.req_id}"):
                keys_multi_layer = key.split_layers(self.num_layers)
                starts.append(start)
                ends.append(end)
                keys.append([k.to_string() for k in keys_multi_layer])
            if keys:
                keys_by_layer = [list(row) for row in zip(*keys)]
            else:
                keys_by_layer = []

        for layer_id, keys_multi_chunk in enumerate(keys_by_layer):
            if layer_id in self.independent_layers:
                continue

            if request.req_id not in self._request_addr_tracker:
                self._request_addr_tracker[request.req_id] = {}
            if layer_id not in self._request_addr_tracker[request.req_id]:
                self._request_addr_tracker[request.req_id][layer_id] = {
                    'processed_count': 0,
                    'addr_list': [],
                    'size_list': [],
                    'gvas_list': [],
                }

            layer_tracker = self._request_addr_tracker[request.req_id][layer_id]
            processed_count = layer_tracker['processed_count']
            new_block_count = len(keys_multi_chunk) - processed_count

            can_save = request.can_save
            if can_save is not None and can_save:
                if new_block_count > 0:
                    new_keys = keys_multi_chunk[processed_count:]
                    new_starts = starts[processed_count:]
                    new_ends = ends[processed_count:]

                    for idx, key in enumerate(new_keys):
                        addr, size = self.token_database.prepare_value_layer(
                            new_starts[idx], new_ends[idx], request.block_ids, layer_id)
                        layer_tracker['addr_list'].extend(addr)
                        layer_tracker['size_list'].extend(size)
                        gva = request.key_gva_mapping[key] if request.key_gva_mapping else None
                        if gva is not None:
                            layer_tracker['gvas_list'].extend([gva, gva + size[0]])

                    layer_tracker['processed_count'] = len(keys_multi_chunk)

                req_meta = LasyerMultiBlockReqMeta(
                    request.req_id, keys_multi_chunk, starts, ends,
                    request.block_ids, layer_id, request.is_last_chunk
                )
                req_meta.key_gva_mapping = request.key_gva_mapping
                req_meta.addr_list = layer_tracker['addr_list'].copy()
                req_meta.size_list = layer_tracker['size_list'].copy()
                req_meta.gvas_list = layer_tracker['gvas_list'].copy()
                self.layer_save_tasks[layer_id].append(req_meta)

            load_spec = request.load_spec
            if load_spec is not None and load_spec.can_load:
                token_len = load_spec.kvpool_cached_tokens
                num_saved_blocks = token_len // self.block_size
                if token_len % self.block_size == 0:
                    load_keys = keys_multi_chunk[:num_saved_blocks]
                    load_starts = starts[:num_saved_blocks]
                    load_ends = ends[:num_saved_blocks]
                else:
                    last_block_key = (
                        f"{self.metadata.model_name}"
                        f"@pcp{self.metadata.pcp_rank}@dcp{self.metadata.dcp_rank}"
                        f"@head_or_tp_rank:{self.metadata.head_or_tp_rank}"
                        f"@{request.req_id}_lastblock@{layer_id}"
                    )
                    load_keys = keys_multi_chunk[:num_saved_blocks] + [last_block_key]
                    load_starts = starts
                    load_ends = ends

                load_tracker_key = f"{request.req_id}_load"
                if load_tracker_key not in self._request_addr_tracker:
                    self._request_addr_tracker[load_tracker_key] = {}
                if layer_id not in self._request_addr_tracker[load_tracker_key]:
                    self._request_addr_tracker[load_tracker_key][layer_id] = {
                        'processed_count': 0,
                        'addr_list': [],
                        'size_list': [],
                        'gvas_list': [],
                    }

                load_tracker = self._request_addr_tracker[load_tracker_key][layer_id]
                load_processed_count = load_tracker['processed_count']
                new_load_count = len(load_keys) - load_processed_count

                if new_load_count > 0:
                    new_load_keys = load_keys[load_processed_count:]
                    new_load_starts = load_starts[load_processed_count:] if load_processed_count < len(load_starts) else load_starts
                    new_load_ends = load_ends[load_processed_count:] if load_processed_count < len(load_ends) else load_ends

                    for idx, key in enumerate(new_load_keys):
                        start_idx = min(idx, len(new_load_starts) - 1) if new_load_starts else 0
                        end_idx = min(idx, len(new_load_ends) - 1) if new_load_ends else 0
                        addr, size = self.token_database.prepare_value_layer(
                            new_load_starts[start_idx] if new_load_starts else 0,
                            new_load_ends[end_idx] if new_load_ends else self.block_size,
                            request.block_ids, layer_id)
                        load_tracker['addr_list'].extend(addr)
                        load_tracker['size_list'].extend(size)
                        gva = request.key_gva_mapping[key] if request.key_gva_mapping else None
                        if gva is not None:
                            load_tracker['gvas_list'].extend([gva, gva + size[0]])

                    load_tracker['processed_count'] = len(load_keys)

                req_meta = LasyerMultiBlockReqMeta(
                    request.req_id, load_keys, load_starts, load_ends,
                    request.block_ids, layer_id
                )
                req_meta.key_gva_mapping = request.key_gva_mapping
                req_meta.addr_list = load_tracker['addr_list'].copy()
                req_meta.size_list = load_tracker['size_list'].copy()
                req_meta.gvas_list = load_tracker['gvas_list'].copy()
                self.layer_load_tasks[layer_id].append(req_meta)

    def wait_for_layer_load(self) -> None:
        is_finish = self.layer_load_finished_events[self.current_layer].wait(timeout=5)  #try---cache
        if not is_finish:
            logger.info(f"Layerwise {self.current_layer} load failed")
        self.layer_load_finished_events[self.current_layer].clear()

    def save_kv_layer(self, connector_metadata: AscendConnectorMetadata) -> None:
        # skip independent layers
        if len(self.layer_save_tasks[self.current_layer]) == 0:
            self.current_layer = self.current_layer + 1
            return
        # Wait for KV cache saving to complete on the final layer that requires offloading.
        if self.current_layer != self.layers_need_to_save[-1]:
            # add current layer save task
            self.sync_save_events[self.current_layer].record()
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
            # add load task, in both prefill and decode stages
            # 1. wait for save, and clear save event
            # 2. start load, for prefill layer_load_tasks is None, skip load in the recv thread.
            # 3. set layer_load_finished_events (both prefill & decode)
            if self.current_layer < self.num_layers - self.num_reuse_layers:
                self.kv_recv_thread.add_request(
                    (self.current_layer, self.layer_load_tasks[self.layer_next_map[self.current_layer]], self.layer_next_map[self.current_layer]))
        else:
            self.sync_save_events[self.current_layer].record()
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
            is_finish = self.layer_save_finished_events[self.current_layer].wait(timeout=5)  # try---cache
            if not is_finish:
                logger.info(f"Layerwise {self.current_layer} save failed")
            self.layer_save_finished_events[self.current_layer].clear()
            # Clear save events for tail layers—no downstream layers exist to reset them.
            for i in range(self.num_reuse_layers - 1, 0, -1):
                self.layer_save_finished_events[self.current_layer - i].clear()

        self.current_layer = self.current_layer + 1

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):
        current_event = None
        has_save_request = False
        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue
            current_event = torch.npu.Event()
            current_event.record()
            break

        for request in connector_metadata.requests:
            can_save = request.can_save
            if can_save is None or not can_save:
                continue

            request.skip_null_blocks_by_group = self.group_uses_align_state
            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id
            )
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request,
            )
            has_save_request = True

        if has_save_request:
            # vLLM expects wait_for_save() to make stores visible before the
            # request is reported as finished. Without this barrier a following
            # identical prompt can lookup before Mooncake put() has completed.
            self.kv_send_thread.request_queue.join()  # type: ignore[union-attr]

    def get_finished(self, finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]:
        done_sending = (
            # TODO 这里需要优化，可能需要使用 self.get_and_clear_finished_requests
            self.get_and_clear_finished_requests(
                finished_req_ids,
                meta,  # type: ignore[union-attr]
            )
            if self.kv_role in ["kv_producer", "kv_both"] or self.consumer_is_to_put
            else set()
        )

        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.load_async
            else set()
        )

        logger.debug(
            "Number of completed KV cache send requests: %d, receive requests: %d, tp_rank:%d",
            len(done_sending),
            len(done_recving),
            self.tp_rank,
        )
        return done_sending, done_recving

    def get_and_clear_finished_requests(self, finished_req_ids, meta: AscendConnectorMetadata) -> set[str]:
        finished_sending = set()
        for req_id in meta.preempted_req_ids:
            self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                req_id
            )
        for req_id in self.kv_send_thread.stored_requests.copy(  # type: ignore[union-attr]
        ):
            if (
                self.kv_send_thread.stored_requests[  # type: ignore[union-attr]
                    req_id
                ]
                == 0
                and req_id in self.finished_store_req
            ):
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )
                if req_id in self._request_addr_tracker:
                    del self._request_addr_tracker[req_id]
                load_tracker_key = f"{req_id}_load"
                if load_tracker_key in self._request_addr_tracker:
                    del self._request_addr_tracker[load_tracker_key]

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(  # type: ignore[union-attr]
                req_id
            )
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id
                )
                if req_id in self._request_addr_tracker:
                    del self._request_addr_tracker[req_id]
                load_tracker_key = f"{req_id}_load"
                if load_tracker_key in self._request_addr_tracker:
                    del self._request_addr_tracker[load_tracker_key]
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            kv_cache_group_ids = self._get_lookup_gate_group_ids(kv_cache_group_ids)
            for group_id in kv_cache_group_ids:
                end = 0
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    hits.append(0)
                    continue

                res = self.m_store.exists(keys)  # type: ignore[assignment]

                if use_layerwise:
                    res = self.check_all_layers_exists(res, self.num_layers)
                if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                    hit_end = 0
                    for index in range(len(ends) - 1, -1, -1):
                        if (
                            res[index] == 1  # type: ignore[index]
                            and ends[index] % self.cache_transfer_granularity == 0
                        ):
                            hit_end = ends[index]
                            break
                else:
                    hit_end = end
                    for index, value in enumerate(res):  # type: ignore[arg-type]
                        if value != 1:
                            hit_end = 0
                            for hit_index in range(index, 0, -1):
                                if starts[hit_index] % self.cache_transfer_granularity == 0:
                                    hit_end = starts[hit_index]
                                    break
                            break
                hits.append(hit_end)
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return start
        return end

    def lookup_scheduler(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        use_layerwise: bool = False,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            hits = []
            kv_cache_group_ids = kv_cache_group_ids or [0]
            kv_cache_group_ids = self._get_lookup_gate_group_ids(kv_cache_group_ids)
            for group_id in kv_cache_group_ids:
                keys = []
                starts = []
                ends = []
                for start, end, key in self.token_database.process_tokens(
                    token_len,
                    block_hashes,
                    kv_cache_group_id=group_id,
                ):
                    if use_layerwise:
                        keys_multi_layer = key.split_layers(self.num_layers)
                        for item in keys_multi_layer:
                            keys.append(item.to_string())
                    else:
                        keys.append(key.to_string())
                    starts.append(start)
                    ends.append(end)

                if not keys:
                    return 0

                multi_tp_keys = keys[:]
                group_tp_size = self.get_group_tp_size(group_id)
                for i in range(1, group_tp_size):
                    for item in keys:
                        new_str = item.replace(  # type: ignore[attr-defined]
                            "@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1
                        )
                        multi_tp_keys.append(new_str)

                pp_base_keys = multi_tp_keys.copy()
                for i in range(1, self.pp_size):
                    for item in pp_base_keys:
                        new_str = item.replace(  # type: ignore[attr-defined]
                            "@pp_rank:0", f"@pp_rank:{i}", 1
                        )
                        multi_tp_keys.append(new_str)

                res = self.m_store.exists(multi_tp_keys)  # type: ignore[assignment]
                num_block = len(keys)
                if use_layerwise:
                    res = self.check_all_layers_exists(res, self.num_layers)
                    num_block = len(keys) // self.num_layers
                multi_tp_values = [
                    res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                    for i in range(group_tp_size * self.pp_size)
                ]
                logger.debug(
                    "KV pool lookup request token_len=%d group=%d keys=%d multi_tp_keys=%d "
                    "exists_count=%d/%d exists_sample=%s sample_keys=%s",
                    token_len,
                    group_id,
                    len(keys),
                    len(multi_tp_keys),
                    sum(1 for value in res if value == 1),  # type: ignore[union-attr]
                    len(res),
                    list(res[: min(12, len(res))]),  # type: ignore[index]
                    multi_tp_keys[:3],
                )
                if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                    # mamba group with align mode will skip some null block, we must loop it in reverse order
                    for i in range(num_block - 1, -1, -1):
                        if (
                            all(values[i] == 1 for values in multi_tp_values)
                            and ends[i] % self.cache_transfer_granularity == 0
                        ):
                            hits.append(ends[i])
                            break
                    else:
                        return 0
                else:
                    index = self.find_max_hit_index(multi_tp_values, num_block)
                    if index == -1:
                        return 0
                    else:
                        for hit_index in range(index, -1, -1):
                            if ends[hit_index] % self.cache_transfer_granularity == 0:
                                hits.append(ends[hit_index])
                                break
                        else:
                            return 0
                logger.debug(
                    "KV pool scheduler lookup group=%d keys=%d hit=%d token_len=%d",
                    group_id,
                    len(keys),
                    hits[-1],
                    token_len,
                )
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return start
        return end

    def check_all_layers_exists(self, res: list[int], num_layers: int) -> list[int]:
        total_chunks = len(res) // num_layers
        result = []

        for chunk_idx in range(total_chunks):
            start = chunk_idx * num_layers
            end = start + num_layers
            chunk = res[start:end]
            result.append(1 if all(x == 1 for x in chunk) else 0)

        return result

    def find_max_hit_index(self, arr, num_blocks: int):
        for i in range(num_blocks):
            if any(row[i] != 1 for row in arr):
                return i - 1
        else:
            # if arr is not empty, all hits, else no hits
            return len(arr[0]) - 1 if arr else -1

    def get_kv_events(self) -> list[BlockStored]:
        if self.enable_kv_events and self.kv_send_thread is not None:
            # collect store kv events form sending thread
            events = self.kv_send_thread.get_kv_events()
            return events
        return []

    def build_connector_worker_meta(self) -> AscendStoreKVConnectorWorkerMetadata | None:
        if self.use_mamba and isinstance(self.kv_send_thread, KVCacheStoreSendingThread):
            if ce := self.kv_send_thread.get_completed_events():
                return AscendStoreKVConnectorWorkerMetadata(ce)
        return None
