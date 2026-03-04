import math
import threading
from typing import Dict, Generator, Optional, Type

import torch
from vllm.config import VllmConfig
from vllm.distributed import (get_decode_context_model_parallel_rank,
                              get_decode_context_model_parallel_world_size,
                              get_pcp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kvpool.backend.backend import Backend
from vllm_ascend.distributed.kvpool.backend.memcache_backend import \
    MemcacheBackend
from vllm_ascend.distributed.kvpool.backend.mooncake_backend import \
    MooncakeBackend
from vllm_ascend.distributed.kvpool.config_data import (
    AscendConnectorMetadata, ChunkedTokenDatabase, KeyMetadata,
    LasyerMultiBlockReqMeta, ReqMeta, PoolKey)
from vllm_ascend.distributed.kvpool.kv_transfer import (
    KVCacheStoreLayerRecvingThread, KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread, KVCacheStoreSendingThread, KVTransferThread)
from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

import copy, time
import torch.distributed as dist
import collections

backend_map: Dict[str, Type[Backend]] = {
    "mooncake": MooncakeBackend,
    "memcache": MemcacheBackend,
}


class KVPoolWorker:
    # The main class for the cache engine.

    def __init__(
            self,
            vllm_config: VllmConfig,
            use_layerwize: bool,
    ):
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.use_mla = False
        if (hasattr(model_config, "use_mla")
                and isinstance(model_config.use_mla, bool)
                and model_config.use_mla):
            self.use_mla = True
        self.use_layerwise = use_layerwize
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel_config.pipeline_parallel_size
        self.pp_rank = (parallel_config.rank // self.tp_size) % self.pp_size

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group(
        ).rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "load_async", False)
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False)
        self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "backend", "mooncake")
        self.block_size = vllm_config.cache_config.block_size

        if self.pcp_size > 1:
            self.block_size *= self.pcp_size
        if self.dcp_size > 1:
            self.block_size *= self.dcp_size
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

        self.metadata = KeyMetadata(
            model_config.model.split('/')[-1],
            self.head_or_tp_rank,
            self.pcp_rank,
            self.dcp_rank,
            self.pp_rank,
        )

        partitions = None
        if self.kv_role == "kv_consumer" and self.consumer_is_to_put:
            num_hidden_layers = model_config.hf_text_config.num_hidden_layers
            partition_list_str = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                "prefill_pp_layer_partition", None)
            prefill_pp_size = int(
                vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                    "prefill_pp_size", 1))

            if partition_list_str is not None:
                try:
                    partitions = [
                        int(layer) for layer in partition_list_str.split(",")
                    ]
                except ValueError as err:
                    raise ValueError("Invalid partition string: {}".format(
                        partition_list_str)) from err
                if len(partitions) != prefill_pp_size:
                    raise ValueError(
                        f"{len(partitions)=} does not match {prefill_pp_size=}."
                    )
                if sum(partitions) != num_hidden_layers:
                    raise ValueError(
                        f"{sum(partitions)=} does not match {num_hidden_layers=}."
                    )
            else:
                layers_per_partition = num_hidden_layers // prefill_pp_size
                partitions = [
                    layers_per_partition for _ in range(prefill_pp_size)
                ]

                if remaining_layers := num_hidden_layers % prefill_pp_size:
                    for i in range(2, remaining_layers + 2):
                        partitions[-i] += 1

        self.token_database = ChunkedTokenDatabase(self.metadata,
                                                   self.block_size,
                                                   self.use_mla, partitions)

        real_backend = backend_map.get(self.backend.lower())
        self.m_store = real_backend(  # type: ignore[misc]
            parallel_config)

        self.kv_send_thread: Optional[KVTransferThread] = None
        self.kv_recv_thread: Optional[KVTransferThread] = None

        self.layer_load_tasks = [[] for i in range(self.num_layers)]
        self.layer_save_tasks = [[] for i in range(self.num_layers)]

        self.finished_store_req: set[str] = set()

        self.seq_last_block_id = {}
        self.num_reuse_layers = 20  # TODO 当作参数，配置方法？
        self.layer_next_map = {i: i + self.num_reuse_layers for i in range(11, 31)}
        self.independent_layers = [i for i in range(11)] + [i for i in range(51, 61)]  # TODO 不是必要的
        self.offload_start_ids = [i for i in range(11, 31)]
        self.layers_need_to_save = [i for i in range(self.num_layers) if i not in self.independent_layers]

        # TODO 记录每个请求当前decode数量，可以使用这个decode标识每个last block,之前的block 是否有必要删除？
        self.seq_last_block_id = {}
        self.num_reuse_layers = 3  # TODO 当作参数，配置方法？
        # self.num_reuse_layers = 30   # TODO 当作参数，配置方法？
        self.layer_next_map = {i: i + self.num_reuse_layers for i in range(self.num_layers - self.num_reuse_layers)}
        self.independent_layers = []  # TODO 不是必要的
        self.offload_start_ids = [i for i in range(self.num_reuse_layers)]
        self.layers_need_to_save = [i for i in range(self.num_layers) if i not in self.independent_layers]

        self.sync_save_events = [torch.npu.Event() for i in range(self.num_layers)]
        # 添加缓存字典，用于缓存process_layer_data的计算结果
        # 缓存键格式：(req_id, tuple(block_hashes), token_len_chunk, can_save, load_spec_cache_key, is_last_chunk)
        # 缓存值：(keys, starts, ends)
        self.process_layer_cache = {}
        # 缓存清理阈值，当缓存大小超过此值时，清理最早的缓存项
        self.cache_max_size = 1000
        self.cache_lru = collections.OrderedDict()  # 用于实现LRU缓存，使用OrderedDict提高性能
        # 请求历史状态追踪
        # key: req_id, value: (prev_block_hashes, prev_token_len, prev_keys, prev_starts, prev_ends, prev_unfull_key_split)
        self.request_history = {}

        self.request_history = collections.defaultdict(tuple)

        # Add cache index for efficient cleanup by req_id
        self.cache_by_req_id = collections.defaultdict(set)  # Maps req_id to set of cache keys

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        _, first_kv_cache_tuple = next(iter(kv_caches.items()))
        first_kv_cache = first_kv_cache_tuple[0]

        # TODO(tms): Find a more robust way to detect and handle MLA
        if self.use_mla:
            # MLA case.[num_block, block_size, 1, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 3  # [block_size, latent_dim]
            block_shape_norm = first_kv_cache_tuple[0].shape[-block_rank:]
            block_shape_pe = first_kv_cache_tuple[1].shape[-block_rank:]
            self.block_len = [
                first_kv_cache[0].element_size() * math.prod(block_shape_norm),
                first_kv_cache[1].element_size() * math.prod(block_shape_pe)
            ]
            logger.info(
                "num_blocks: %s, block_shape_norm: %s, block_shape_pe: %s",
                self.num_blocks, block_shape_norm, block_shape_pe)
        else:
            # [num_block, block_size, num_head, hidden_dim]
            self.num_blocks = first_kv_cache.shape[0]
            kv_elem_size = first_kv_cache.element_size()
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            self.block_len = [kv_elem_size * math.prod(block_shape)]
            logger.info("num_blocks: %s, block_shape: %s", self.num_blocks,
                        block_shape)

        logger.info("Registering KV_Caches. use_mla: %s, shape %s",
                    self.use_mla, first_kv_cache.shape)

        self.kv_caches = kv_caches
        self.kv_caches_base_addr = []
        ptrs = []
        lengths = []
        # for cache_or_caches in kv_caches.values():
        for layer_id, cache_or_caches in enumerate(kv_caches.values()):
            # Normalize to always be a list of caches
            if self.use_mla:
                for i, cache in enumerate(cache_or_caches, 0):
                    base_addr = cache.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)
                    region_len = self.num_blocks * self.block_len[i % 2]
                    # avoid overlapped memory register
                    if layer_id in self.offload_start_ids:
                        ptrs.append(base_addr)
                        lengths.append(region_len)
            else:
                cache_list = [cache_or_caches
                              ] if self.use_mla else cache_or_caches
                for cache in cache_list:
                    base_addr = cache.data_ptr()
                    self.kv_caches_base_addr.append(base_addr)
                    region_len = self.num_blocks * self.block_len[0]
                    # avoid overlapped memory register
                    if layer_id in self.offload_start_ids:
                        ptrs.append(base_addr)
                        lengths.append(region_len)
        self.m_store.register_buffer(ptrs, lengths)
        self.token_database.set_kv_caches_base_addr(self.kv_caches_base_addr)
        logger.info(f"=================> block len {self.block_len}")
        self.token_database.set_block_len(self.block_len)

        if self.use_layerwise:
            self.layer_load_finished_events = [threading.Event() for i in range(self.num_layers)]
            self.layer_save_finished_events = [threading.Event() for i in range(self.num_layers)]
            if self.kv_role in ['kv_producer', 'kv_both']:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreLayerSendingThread(
                    self.m_store, self.token_database, self.block_size,
                    self.tp_rank, self.dcp_size, self.put_step,
                    ready_event_sending, self.num_layers, self.layer_save_finished_events, self.sync_save_events)
                self.kv_send_thread.start()
                ready_event_sending.wait()
            ready_event = threading.Event()
            self.kv_recv_thread = KVCacheStoreLayerRecvingThread(
                self.m_store, self.token_database, self.block_size,
                self.tp_rank, self.dcp_size, ready_event, self.layer_load_finished_events,
                self.layer_save_finished_events)
            self.kv_recv_thread.start()
            ready_event.wait()
        else:
            if self.kv_role in ['kv_producer', 'kv_both'
                                ] or self.consumer_is_to_put:
                ready_event_sending = threading.Event()
                self.kv_send_thread = KVCacheStoreSendingThread(
                    self.m_store, self.token_database, self.block_size,
                    self.tp_rank, self.dcp_size, self.put_step, self.kv_role,
                    ready_event_sending)
                self.kv_send_thread.start()
            if self.load_async:
                ready_event = threading.Event()
                self.kv_recv_thread = KVCacheStoreRecvingThread(
                    self.m_store, self.token_database, self.block_size,
                    self.tp_rank, self.dcp_size, ready_event)
                self.kv_recv_thread.start()
                ready_event.wait()

    def start_load_kv(self, metadata: AscendConnectorMetadata):
        self.current_layer = 0
        self.layerwise_retrievers = []
        for request in metadata.requests:
            if self.use_layerwise:
                self.seq_last_block_id[request.req_id] = self.seq_last_block_id.get(request.req_id, -1) + 1
                self.process_layer_data(request)
                continue
            load_spec = request.load_spec
            if load_spec is None or not load_spec.can_load:  # load =0
                continue
            token_len = request.token_len_chunk
            # TODO check this
            # if (load_spec.kvpool_cached_tokens % self.block_size
            #         != 0) and (load_spec.kvpool_cached_tokens
            #                    == token_len - 1):
            #     token_len = request.load_spec.kvpool_cached_tokens + 1
            # else:
            #     token_len = request.load_spec.kvpool_cached_tokens
            # request.token_len_chunk = token_len
            if self.load_async:
                self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                    request, )
            else:
                addr_list = []
                size_list = []
                key_list = []
                mask_num = (request.load_spec.vllm_cached_tokens //
                            self.block_size * self.block_size)
                for start, end, key in self.token_database.process_tokens(
                        token_len, request.block_hashes, mask_num, req_id=request.req_id):
                    addr, size, _ = self.token_database.prepare_value(
                        start, end, request.block_ids)
                    key_list.append(key.to_string())
                    addr_list.append(addr)
                    size_list.append(size)
                key_list_c = key_list[self.tp_rank % len(
                    key_list):] + key_list[:self.tp_rank % len(key_list)]
                addr_list_c = addr_list[self.tp_rank %
                                        len(addr_list
                                            ):] + addr_list[:self.tp_rank %
                                                             len(addr_list)]
                size_list_c = size_list[self.tp_rank %
                                        len(size_list
                                            ):] + size_list[:self.tp_rank %
                                                             len(size_list)]
                self.m_store.get(key_list_c, addr_list_c, size_list_c)
        # For layers in `offload_start_ids`, skip waiting for KV cache save completion.
        # - In prefill: no offloaded data to load (`layer_load_task` is None),
        #   so only signal `layer_load_finished_events` to unblock dependent layers.
        # - In decode: load KV cache from storage (`layer_load_task` is valid),
        #   then signal `layer_load_finished_events` after loading completes.
        # TODO 在结束的时候还会再调用一次这个函数
        if self.use_layerwise and metadata.unfinished_request_ids:
            # if self.tp_rank == 0:
            #     logger.info(f"=====================>  load task {self.layer_load_tasks}")
            #     logger.info(f"=====================>  save task {self.layer_save_tasks}")
            for layer_id in self.offload_start_ids:
                layer_load_task = self.layer_load_tasks[layer_id]
                self.kv_recv_thread.add_request((None, layer_load_task, layer_id))
            # for layer_id in self.offload_start_ids:
            #     is_finish = self.layer_load_finished_events[layer_id].wait(timeout=5)  # try---cache
            #     if not is_finish:
            #         logger.info(f"Layerwise {self.current_layer} load failed")

    def wait_for_layer_load(self) -> None:
        return
        # if self.current_layer in self.independent_layers:
        #     return
        # req_metas = self.layer_load_tasks[self.current_layer]
        # if len(req_metas) == 0:
        #     return
        # # wait_for_save, req_metas, layer_id = layer_load_task
        # addr_list = []
        # size_list = []
        # key_list = []
        # # gvas = []
        #
        # for req_meta in req_metas:
        #     # TODO 优化点：没有必要每次都进行计算，可以存储在CPU当中，这样循环太慢了
        #     # gvas.extend(req_meta.gvas)
        #     # addr_list.extend(req_meta.addr_list)
        #     # size_list.extend(req_meta.size_list)
        #     key_list.extend(req_meta.keys_str)
        #     addr_list.extend(req_meta.addr_list)
        #     size_list.extend(req_meta.size_list)
        # key_list_c = key_list[self.tp_rank %
        #                       len(key_list):] + key_list[:self.tp_rank %
        #                                                  len(key_list)]
        # addr_list_c = addr_list[self.tp_rank %
        #                         len(addr_list):] + addr_list[:self.tp_rank %
        #                                                       len(addr_list)]
        # size_list_c = size_list[self.tp_rank %
        #                         len(size_list):] + size_list[:self.tp_rank %
        #                                                       len(size_list)]
        # self.m_store.get(key_list_c, addr_list_c, size_list_c)
        #
        # # res = self.m_store.store.batch_copy(gvas, addr_list_c, size_list_c, 1)
        # # assert res == 0, "recv failed"
        #
        # req_metas.clear()
        # if self.tp_rank == 0:
        #     logger.info(f"=====================> wait_for_layer_load {self.current_layer}")

        # if self.current_layer in self.independent_layers:
        #     return
        # is_finish = self.layer_load_finished_events[self.current_layer].wait(timeout=5)  #try---cache
        # if not is_finish:
        #     logger.info(f"Layerwise {self.current_layer} load failed")
        # self.layer_load_finished_events[self.current_layer].clear()

        # if self.tp_rank == 0:
        #     logger.info(f"======================> clear {self.current_layer} layer_load_finished_events")

    def save_kv_layer(self,
                      connector_metadata: AscendConnectorMetadata) -> None:
        # if self.tp_rank == 0:
        #     logger.info(f"=====================> save_kv_layer {self.current_layer}")
        # skip independent layers
        if len(self.layer_save_tasks[self.current_layer]) == 0 or self.current_layer in self.independent_layers:
            self.current_layer = self.current_layer + 1
            return
        # self.sync_save_events[self.current_layer].record()
        # req_metas = self.layer_save_tasks[self.current_layer]
        # key_list = []
        # addr_list = []
        # size_list = []
        # layer_id = req_metas[0].layer_id
        # cur_req_ids = set()
        # # gvas = []
        # for req_meta in req_metas:
        #     cur_req_ids.add(req_meta.req_id)
        #     keys = req_meta.keys
        #     is_last_chunk = req_meta.is_last_chunk
        #     if not self.dcp_size > 1:
        #         # gvas_i = req_meta.gvas[self.tp_rank % self.put_step::self.put_step]
        #         addr_list_i = req_meta.addr_list[self.tp_rank % self.put_step::self.put_step]
        #         size_list_i = req_meta.size_list[self.tp_rank % self.put_step::self.put_step]
        #         keys_str = req_meta.keys_str[self.tp_rank % self.put_step::self.put_step]
        #     # TODO there maybe has some problem when only have one block.
        #     if not keys:
        #         if is_last_chunk:
        #             self.set_finished_request(req_meta.req_id)
        #         continue
        #
        #     # gvas.extend(gvas_i)
        #     # addr_list.extend(addr_list_i)
        #     # size_list.extend(size_list_i)
        #
        #     skip_block_num = self.lookup1(keys_str)
        #     # TODO check this
        #     # if skip_block_num == len(keys_str):
        #     #     if is_last_chunk and layer_id == self.final_layer_id:
        #     #         self.set_finished_request(req_meta.req_id)
        #     #     continue
        #     key_list.extend(keys_str[skip_block_num:])
        #     addr_list.extend(addr_list_i[skip_block_num:])
        #     size_list.extend(size_list_i[skip_block_num:])
        #
        # self.sync_save_events[layer_id].synchronize()
        # if len(key_list) > 0:
        #     self.m_store.put(key_list, addr_list, size_list)
        #
        #     # res = self.m_store.store.batch_copy(gvas, addr_list, size_list, 0)
        #     # assert res == 0, "send failed"
        #
        # req_metas.clear()
        # # self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
        # # is_finish = self.layer_save_finished_events[self.current_layer].wait(timeout=5)  # try---cache
        # # if not is_finish:
        # #     logger.info(f"Layerwise {self.current_layer} save failed")
        # # self.layer_save_finished_events[self.current_layer].clear()
        # return

        # Wait for KV cache saving to complete on the final layer that requires offloading.
        if self.current_layer != self.layers_need_to_save[-1]:
            # add current layer save task
            self.sync_save_events[self.current_layer].record()
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
            # add load task, in both prefill and decode stages
            # 1. wait for save, and clear save event
            # 2. start load, for prefill layer_load_tasks is None, skip load in the recv thread.
            # 3. set layer_load_finished_events (both prefill & decode)
            # if self.current_layer < self.num_layers - self.num_reuse_layers:
            if self.current_layer in self.layer_next_map.keys():
                # logger.info(f"=====================> save_kv_layer {self.current_layer} and load {self.layer_next_map[self.current_layer]}")
                self.kv_recv_thread.add_request(
                    (self.current_layer, self.layer_load_tasks[self.layer_next_map[self.current_layer]],
                     self.layer_next_map[self.current_layer]))
        else:
            self.sync_save_events[self.current_layer].record()
            self.kv_send_thread.add_request(self.layer_save_tasks[self.current_layer])
            # is_finish = self.layer_save_finished_events[self.current_layer].wait(timeout=5)  # try---cache
            # if not is_finish:
            #     logger.info(f"Layerwise {self.current_layer} save failed")
            # self.layer_save_finished_events[self.current_layer].clear()
            # # Clear save events for tail layers—no downstream layers exist to reset them.
            # for i in range(self.num_reuse_layers - 1, 0, -1):
            #     self.layer_save_finished_events[self.current_layer - i].clear()

        self.current_layer = self.current_layer + 1

    def lookup1(
            self,
            keys: list[str],
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            res = (self.m_store.
                   exists(keys))  # type: ignore[assignment]
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return index
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
        return len(keys)

    def wait_for_save(self, connector_metadata: AscendConnectorMetadata):
        current_event = None
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

            request.current_event = current_event
            self.kv_send_thread.add_stored_request(  # type: ignore[union-attr]
                request.req_id)
            self.kv_send_thread.add_request(  # type: ignore[union-attr]
                request, )

    def _prepare_req_meta_data(self, req_meta, starts, ends, request, layer_id):
        """
        Prepare common data for both save and load request metadata.

        :param req_meta: The request metadata object to prepare.
        :param starts: List of start positions.
        :param ends: List of end positions.
        :param request: Original request containing block_ids.
        :param layer_id: Current layer ID.
        """
        # Preallocate lists with known size
        keys_count = len(req_meta.keys)
        keys_str = [None] * keys_count
        gvas = []  # This list is currently empty but maintained for compatibility
        addr_list = [None] * keys_count
        size_list = [None] * keys_count

        # Generate keys_str
        for i in range(keys_count):
            keys_str[i] = req_meta.keys[i].to_string()

        # Generate addr_list and size_list
        for i in range(keys_count):
            addr, size = self.token_database.prepare_value_layer(
                starts[i], ends[i], request.block_ids, layer_id)
            addr_list[i] = addr
            size_list[i] = size

        # Avoid deepcopy when possible - use direct assignment since we're creating new lists
        req_meta.keys_str = keys_str
        req_meta.gvas = gvas
        req_meta.addr_list = addr_list
        req_meta.size_list = size_list

    def _get_load_spec_cache_key(self, load_spec):
        """
        Generate a cache key for LoadSpec object.
        """
        if load_spec is None:
            return None
        return (
            load_spec.vllm_cached_tokens,
            load_spec.kvpool_cached_tokens,
            load_spec.can_load
        )

    def _get_cache_key(self, request: ReqMeta):
        """
        Generate a cache key for the process_layer_data method.
        """
        # 将block_hashes转换为元组以便作为字典键
        block_hashes_tuple = tuple(h.hex() if hasattr(h, 'hex') else h for h in request.block_hashes)
        load_spec_key = self._get_load_spec_cache_key(request.load_spec)
        return (
            request.req_id,
            block_hashes_tuple,
            request.token_len_chunk,
            request.can_save,
            load_spec_key,
            request.is_last_chunk
        )

    def _cleanup_cache(self):
        """
        Cleanup the cache when it exceeds the maximum size.
        Implements LRU (Least Recently Used) eviction policy.
        """
        if len(self.process_layer_cache) > self.cache_max_size:
            # 移除最旧的缓存项
            oldest_key, _ = self.cache_lru.popitem(last=False)
            # 从缓存中移除
            del self.process_layer_cache[oldest_key]
            # 从缓存索引中移除
            req_id = oldest_key[0]
            if req_id in self.cache_by_req_id:
                self.cache_by_req_id[req_id].discard(oldest_key)
                # 如果该req_id没有缓存项了，移除该req_id的索引
                if not self.cache_by_req_id[req_id]:
                    del self.cache_by_req_id[req_id]

    def process_layer_data(self, request: ReqMeta) -> Generator[
        Optional[torch.Tensor], None, None]:
        """
        A more efficient version of the layer-wise KV cache retrieval or storage.
        Implements incremental computation: only processes new blocks when they appear.

        :param request: The request containing meta information about the tokens and blocks.

        :return: A generator that yields either None (for store) or a tensor (for retrieve).
        """
        req_id = request.req_id
        current_block_hashes = request.block_hashes
        current_token_len = request.token_len_chunk

        # 生成缓存键
        cache_key = self._get_cache_key(request)

        # 检查缓存中是否存在结果
        if cache_key in self.process_layer_cache:
            # 更新缓存项的使用时间（移到LRU列表末尾）
            self.cache_lru.move_to_end(cache_key)
            # 从缓存中获取结果
            keys, starts, ends, unfull_key_split = self.process_layer_cache[cache_key]
        else:
            # 检查请求历史记录，判断是否有新增的block
            has_new_block = False
            new_block_hashes = None
            new_token_start = None

            if req_id in self.request_history:
                prev_block_hashes, prev_token_len, prev_keys, prev_starts, prev_ends, prev_unfull_key_split = \
                self.request_history[req_id]

                # 判断是否有新的block：block数量增加或token长度超过之前的block容量
                if len(current_block_hashes) > len(prev_block_hashes):
                    has_new_block = True
                    # 新block的信息
                    new_block_hashes = current_block_hashes[-1:]
                    new_token_start = prev_token_len // self.block_size * self.block_size

            if has_new_block and new_block_hashes and new_token_start is not None:
                # 只处理新增的block
                # 获取之前的计算结果
                prev_block_hashes, prev_token_len, prev_keys, prev_starts, prev_ends, prev_unfull_key_split = \
                self.request_history[req_id]

                new_start = prev_token_len // self.block_size * self.block_size
                new_end = current_token_len // self.block_size * self.block_size

                new_hash = new_block_hashes[0]

                if not isinstance(new_hash, str):
                    new_hash = new_hash.hex()

                # 只取新增的block（最后一个）
                new_key = self.token_database._make_key_by_hash(new_hash)
                keys_multi_layer = new_key.split_layers(self.num_layers)
                prev_starts.append(new_start)
                prev_ends.append(new_end)
                prev_keys.append(keys_multi_layer)  # [block_num, layer_num]
            else:
                # 没有新的block或第一次处理该请求，完整计算
                starts, ends, keys = [], [], []
                for start, end, key in self.token_database.process_tokens(
                        current_token_len, current_block_hashes, req_id=f"{req_id}"):
                    keys_multi_layer = key.split_layers(self.num_layers)
                    starts.append(start)
                    ends.append(end)
                    keys.append(keys_multi_layer)  # [block_num, layer_num]

            # 计算unfull_key_split
            unfull_key_split = None
            need_unfull_key = (request.load_spec is not None and request.load_spec.can_load and
                               (current_token_len - 1) % self.block_size != 0)

            if need_unfull_key:
                unfull_key = PoolKey(self.metadata,
                                     f"{req_id}"
                                     f"_lastblock"
                                     )
                unfull_key_split = unfull_key.split_layers(self.num_layers)

            # 更新请求历史记录
            self.request_history[req_id] = (current_block_hashes, current_token_len, keys, starts, ends,
                                            unfull_key_split)
            # 更新缓存
            self.process_layer_cache[cache_key] = (keys, starts, ends, unfull_key_split)
            self.cache_lru[cache_key] = None  # 添加到LRU末尾
            # 更新缓存索引
            self.cache_by_req_id[req_id].add(cache_key)
            # 检查并清理缓存
            self._cleanup_cache()
        self._build_layer_tasks(request, keys, starts, ends, unfull_key_split)

    def _build_layer_tasks(self, request: ReqMeta, keys: list, starts: list, ends: list, unfull_key_split):
        if not keys:
            return
        # Only process further if keys are present
        keys = [list(row) for row in zip(*keys)]

        for layer_id, keys_multi_chunk in enumerate(keys):
            if layer_id in self.independent_layers:
                continue
            # save
            can_save = request.can_save
            req_meta_save = None
            req_meta_load = None

            if can_save is not None and can_save:
                req_meta_save = LasyerMultiBlockReqMeta(
                    request.req_id, keys_multi_chunk, starts, ends,
                    request.block_ids, layer_id, request.is_last_chunk
                )

            load_spec = request.load_spec
            if load_spec is not None and load_spec.can_load:  # load =0
                # 计算token_len
                token_len = request.token_len_chunk
                if (token_len - 1) % self.block_size == 0:
                    req_meta_load = LasyerMultiBlockReqMeta(
                        request.req_id, keys_multi_chunk[:-1], starts[:-1], ends[:-1],
                        request.block_ids, layer_id
                    )
                else:
                    # Use cached unfull_key_split to avoid repeated split_layers calls
                    req_meta_load = LasyerMultiBlockReqMeta(
                        request.req_id, keys_multi_chunk[:-1] +
                                        [unfull_key_split[layer_id]], starts, ends,
                        request.block_ids, layer_id
                    )

            if req_meta_save is not None:
                self._prepare_req_meta_data(req_meta_save, starts, ends, request, layer_id)
                self.layer_save_tasks[layer_id].append(req_meta_save)

            if req_meta_load is not None:
                # For load, we might need to adjust starts and ends if we're using a subset
                token_len = request.token_len_chunk
                load_starts = starts[:-1] if (token_len - 1) % self.block_size == 0 else starts
                load_ends = ends[:-1] if (token_len - 1) % self.block_size == 0 else ends
                self._prepare_req_meta_data(req_meta_load, load_starts, load_ends, request, layer_id)
                self.layer_load_tasks[layer_id].append(req_meta_load)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests(
                # type: ignore[union-attr]
            ) if self.kv_role in ['kv_producer', 'kv_both']
                 or self.consumer_is_to_put else set())

        done_recving = (
            self.kv_recv_thread.
            get_and_clear_finished_requests(  # type: ignore[union-attr]
            ) if self.load_async else set())

        logger.debug(
            "Number of completed KV cache send requests: %d, receive "
            "requests: %d, tp_rank:%d", len(done_sending), len(done_recving),
            self.tp_rank)
        # TODO 可以在这里进行异步删除操作
        for req_id in finished_req_ids:
            self.seq_last_block_id.pop(req_id, None)
            # 清理请求历史记录
            self.request_history.pop(req_id, None)
            # 清理相关缓存 - 使用缓存索引避免遍历所有缓存
            if req_id in self.cache_by_req_id:
                for cache_key in self.cache_by_req_id[req_id]:
                    self.process_layer_cache.pop(cache_key, None)
                    self.cache_lru.pop(cache_key, None)  # O(1)操作
                # 移除req_id的缓存索引
                del self.cache_by_req_id[req_id]
        return done_sending, done_recving

    def get_and_clear_finished_requests(self, finished_req_ids) -> set[str]:
        finished_sending = set()
        for req_id in self.kv_send_thread.stored_requests.copy(  # type: ignore[union-attr]
        ):
            if self.kv_send_thread.stored_requests[  # type: ignore[union-attr]
                req_id] == 0 and req_id in self.finished_store_req:
                self.finished_store_req.remove(req_id)
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id)

        for req_id in finished_req_ids:
            req_remain_jobs = self.kv_send_thread.stored_requests.get(  # type: ignore[union-attr]
                req_id)
            if req_remain_jobs == 0:
                finished_sending.add(req_id)
                self.kv_send_thread.delete_finished_stored_request(  # type: ignore[union-attr]
                    req_id)
            elif req_remain_jobs is not None:
                self.finished_store_req.add(req_id)

        return finished_sending

    def lookup(
            self,
            token_len: int,
            block_hashes: list[BlockHash],
            use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        keys = []
        try:
            starts = []
            for start, end, key in self.token_database.process_tokens(
                    token_len, block_hashes):
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for item in keys_multi_layer:
                        keys.append(item.to_string())
                else:
                    keys.append(key.to_string())
                starts.append(start)

            res = self.m_store.exists(keys)  # type: ignore[assignment]

            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return starts[index]
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return start
        return end

    def lookup_scheduler(
            self,
            token_len: int,
            block_hashes: list[BlockHash],
            use_layerwise: bool,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        end = 0
        keys = []
        try:
            starts = []
            for start, end, key in self.token_database.process_tokens(
                    token_len, block_hashes):
                if use_layerwise:
                    keys_multi_layer = key.split_layers(self.num_layers)
                    for item in keys_multi_layer:
                        keys.append(item.to_string())
                else:
                    keys.append(key.to_string())
                starts.append(start)

            multi_tp_keys = keys[:]
            for i in range(1, min(self.tp_size, self.num_kv_head)):
                for item in keys:
                    new_str = item.replace(  # type: ignore[attr-defined]
                        "@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            for i in range(1, self.pp_size):
                for item in keys:
                    new_str = item.replace(  # type: ignore[attr-defined]
                        "@pp_rank:0", f"@pp_rank:{i}", 1)
                    multi_tp_keys.append(new_str)

            res = self.m_store.exists(
                multi_tp_keys)  # type: ignore[assignment]
            num_block = len(keys)
            if use_layerwise:
                res = self.check_all_layers_exists(res, self.num_layers)
                num_block = len(keys) // self.num_layers
            multi_tp_values = [
                res[i * num_block:(i + 1) * num_block]  # type: ignore[index]
                for i in range(
                    min(self.tp_size, self.num_kv_head) * self.pp_size)
            ]
            index = self.find_min_first_non_one_index(multi_tp_values)
            if index != -1:
                return starts[index]
        # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return start
        return end

    def check_all_layers_exists(self, res: list[int],
                                num_layers: int) -> list[int]:
        total_chunks = len(res) // num_layers
        result = []

        for chunk_idx in range(total_chunks):
            start = chunk_idx * num_layers
            end = start + num_layers
            chunk = res[start:end]
            result.append(1 if all(x == 1 for x in chunk) else 0)

        return result

    def find_min_first_non_one_index(self, arr):
        try:
            return min(idx for row in arr for idx, val in enumerate(row)
                       if val != 1)
        except ValueError:
            return -1
