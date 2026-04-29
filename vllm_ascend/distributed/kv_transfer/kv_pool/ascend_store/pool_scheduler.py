from typing import Any

import vllm.envs as envs
from memcache_hybrid import DistributedObjectStore  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    get_cache_family_granularity,
    infer_group_cache_families,
    normalize_block_ids_by_group,
)


class KVPoolScheduler:
    def __init__(self, vllm_config: "VllmConfig", use_layerwise, page_size_bytes: int):
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        self.use_compress = self.compress_ratios is not None
        self.use_hybrid = self._uses_hybrid_kv_cache(vllm_config, kv_cache_config)
        self.kv_cache_group_ids = (
            list(range(len(kv_cache_config.kv_cache_groups)))
            if kv_cache_config is not None and self.use_hybrid
            else [0]
        )
        self.kv_cache_group_families = self._infer_group_families()
        self.need_truncate = self.use_compress
        self.num_swa_blocks = self._infer_swa_blocks()
        if kv_cache_config is not None:
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                kv_cache_spec = kv_cache_group.kv_cache_spec
                if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                    kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
                if isinstance(kv_cache_spec, MambaSpec) and getattr(kv_cache_spec, "mamba_cache_mode", None) != "align":
                    raise NotImplementedError(
                        "AscendStore hybrid linear-attention support currently requires mamba_cache_mode='align'."
                    )
        if self.use_layerwise and len(self.kv_cache_group_ids) > 1:
            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

        self.mamba_group_ids = self._infer_mamba_groups()
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
        self._block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*self.grouped_block_size)
        self.cache_transfer_granularity = self._infer_cache_transfer_granularity()
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        self.prefill_offload = True
        # Whether to discard partial chunks
        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", not self.prefill_offload))
        self._unfinished_requests: dict[str, tuple[Request, list[int]]] = {}
        self._unfinished_request_ids: set[str] = set()

        self.page_size_bytes = page_size_bytes
        logger.info(f"==============> page_size_bytes {page_size_bytes}")
        self.store_scheduler = DistributedObjectStore()
        self.store_scheduler.init(device_id=0, init_bm=False)

        model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()
        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
        else:
            self.put_step = 1
        self.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model_name = model_config.model.split('/')[-1]

        # Define independent layers (same as pool_worker.py)
        INDEPENDENT_LAYER_INDICES = {0, self.num_layers - 1}
        self.independent_layers = list(INDEPENDENT_LAYER_INDICES)

        keys_per_block_hash = (
            self.pcp_size * self.dcp_size
            * (self.tp_size // self.put_step)
            * (self.num_layers - len(self.independent_layers))
        )
        self.keys_per_block_hash = keys_per_block_hash

    def _get_or_create_request_tracker(self, req_id: str) -> RequestTracker:
        tracker = self._request_trackers.get(req_id)
        if tracker is None:
            tracker = RequestTracker(
                req_id=req_id,
                token_len=0,
                allocated_block_ids=[],
            )
            self._request_trackers[req_id] = tracker
        return tracker

    def _generate_keys_and_alloc(
        self,
        block_hashes,
        request_tracker: RequestTracker,
        has_last_block=False,
    ) -> None:
        keys_to_alloc, last_block_key = self.generate_keys(
            block_hashes,
            req_id=request_tracker.req_id,
            has_last_block=has_last_block,
        )
        alloc_size = self.page_size_bytes * self.keys_per_block_hash

        last_block_gva = request_tracker.last_block_gva
        num_new_chunk_keys= len(keys_to_alloc)
        if last_block_key and last_block_gva is None:
            keys_to_alloc.append(last_block_key)
        if keys_to_alloc:
            new_gvas = self.store_scheduler.batch_alloc(
                keys_to_alloc, [alloc_size] * len(keys_to_alloc))
            if any(gva <= 0 for gva in new_gvas):
                raise ValueError(
                    f"Request {request_tracker.req_id}: batch_alloc failed, "
                    f"gvas={new_gvas}")

            request_tracker.chunk_gvas.extend(new_gvas[:num_new_chunk_keys])
            request_tracker.block_keys.extend(keys_to_alloc[:num_new_chunk_keys])
            if last_block_key is not None and len(new_gvas) > num_new_chunk_keys:
                request_tracker.last_block_key = last_block_key
                request_tracker.last_block_gva = new_gvas[-1]

    def generate_keys(self, chunk_hashes, req_id='', has_last_block=False):
        chunk_keys = []
        for chunk_hash in chunk_hashes:
            key = f"{self.model_name}@{chunk_hash.hex()}"
            chunk_keys.append(key)

        last_block_key = None
        if has_last_block:
            last_block_key = f"{self.model_name}@{req_id}_lastblock"

        return chunk_keys, last_block_key

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
            return 0, False

        if self._discard_partial_chunks:
            token_len = self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
        else:
            token_len = len(request.prompt_token_ids)

        if token_len < self.cache_transfer_granularity:
            return 0, False

        num_blocks = token_len // self._block_size
        block_hashes_to_check = request.block_hashes[:num_blocks]
        keys_to_check = [
            f"{self.model_name}@{bh.hex()}" for bh in block_hashes_to_check
        ]
        remaining_keys = keys_to_check
        tracker = self._get_or_create_request_tracker(request.request_id)
        # cached_gvas = tracker.chunk_gvas
        # if cached_gvas:
        #     cached_keys = keys_to_check[:len(cached_gvas)]
        #     if not all(self.store_scheduler.batch_is_exist(cached_keys)):
        #         raise ValueError(
        #             f"Request {request.request_id}: cached gvas key(s) no longer exist in store")
        #     remaining_keys = remaining_keys[len(cached_gvas):]
        cached_gvas = []
        num_hit_blocks = 0
        if remaining_keys:
            key_infos = self.store_scheduler.batch_get_key_info(remaining_keys)
            for key_info in key_infos:
                sizes = key_info.size()
                if sizes and sizes > 0:
                    cached_gvas.append(key_info.gva_list()[0])
                    num_hit_blocks += 1
                else:
                    break
        num_external_hit_tokens = num_hit_blocks * self._block_size
        tracker.block_keys = keys_to_check[:num_hit_blocks]
        tracker.chunk_gvas = cached_gvas[:num_hit_blocks]
        # TODO 这里没有命中的可以提前申请空间，避免后面申请的时候掩盖不住，这个可以异步进行。
        # 先exists判断是否存在，然后异步获取地址和申请空间，这样是否更高效一点？
        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )
        logger.info(
            "KV pool load spec created req=%s vllm_cached=%d kvpool_cached=%d "
            "need_to_allocate=%d load_async=%s use_layerwise=%s",
            request.request_id,
            num_computed_tokens,
            num_external_hit_tokens,
            need_to_allocate,
            self.load_async,
            self.use_layerwise,
        )

        return need_to_allocate, self.load_async and not self.use_layerwise

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        local_block_ids: list[list[int]] = [[] for _ in self.kv_cache_group_ids]
        if num_external_tokens > 0:
            local_block_ids = normalize_block_ids_by_group(blocks.get_block_ids())

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            logger.debug(
                "KV pool update_state_after_alloc req=%s has no load spec; num_external_tokens=%d",
                request.request_id,
                num_external_tokens,
            )
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            logger.debug(
                "KV pool load spec disabled req=%s because num_external_tokens=0 vllm_cached=%d kvpool_cached=%d",
                request.request_id,
                self.load_specs[request.request_id].vllm_cached_tokens,
                self.load_specs[request.request_id].kvpool_cached_tokens,
            )
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        logger.debug(
            "KV pool load spec enabled req=%s num_external_tokens=%d vllm_cached=%d kvpool_cached=%d groups=%s",
            request.request_id,
            num_external_tokens,
            self.load_specs[request.request_id].vllm_cached_tokens,
            self.load_specs[request.request_id].kvpool_cached_tokens,
            [len(blocks) for blocks in local_block_ids],
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.
        """
        force_skip_save = self.kv_role == "kv_consumer" and not self.consumer_is_to_put

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)

        meta = AscendConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            if load_spec is not None:
                logger.debug(
                    "KV pool build meta attaches load spec req=%s can_load=%s "
                    "vllm_cached=%d kvpool_cached=%d scheduled_tokens=%d "
                    "num_computed=%d",
                    request.req_id,
                    load_spec.can_load,
                    load_spec.vllm_cached_tokens,
                    load_spec.kvpool_cached_tokens,
                    scheduler_output.num_scheduled_tokens[request.req_id],
                    request.num_computed_tokens,
                )
            num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
            request_tuple = self._unfinished_requests.get(request.req_id)
            if request_tuple is None:
                raise ValueError(
                    f"Request {request.req_id} is not in _unfinished_requests, "
                    "but it is scheduled as a new request"
                )
            request_real = request_tuple[0]  # type: ignore[index]
            if not isinstance(request.block_ids[0], list):
                unfolded_block_ids = request.block_ids.copy()
            else:
                unfolded_block_ids = request.block_ids[0].copy()
            previous_tracker = self._request_trackers.get(request.req_id)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids_by_group=normalize_block_ids_by_group(request.block_ids),
                num_saved_tokens=0,
                token_ids=request.prompt_token_ids[:num_tokens_to_compute].copy(),
                block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
                chunk_gvas=(previous_tracker.chunk_gvas.copy() if previous_tracker else []),
            )
            num_hit_blocks = len(request_tracker.block_keys)
            self._request_trackers[request.req_id] = request_tracker
            last_chunk_tokens_num = (
                self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
                if self._discard_partial_chunks
                else len(request.prompt_token_ids)
            )

            num_blocks = num_tokens_to_compute // self._block_size
            has_last_block = num_tokens_to_compute % self._block_size != 0

            self._generate_keys_and_alloc(
                request_real.block_hashes[num_hit_blocks:num_blocks],
                request_tracker=request_tracker,
                has_last_block=has_last_block,
            )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self.cache_transfer_granularity,
                load_spec=load_spec,
                skip_save=force_skip_save,
                block_hashes=request_real.block_hashes,
                is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                discard_partial_chunks=self._discard_partial_chunks,
                original_block_size=self.original_block_size,
                kv_cache_group_families=self.kv_cache_group_families,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                # resumed request
                new_block_ids = cached_reqs.new_block_ids[i]
                # TODO 调试的时候，添加decode，为了验证精度
                if not new_block_ids and not self.prefill_offload:
                    continue
                if req_id in self._preempted_req_ids:
                    self._preempted_req_ids.discard(req_id)
                    load_spec = self.load_specs.pop(req_id, None)
                    if self.prefill_offload:
                        load_spec = LoadSpec(
                            vllm_cached_tokens=0,
                            kvpool_cached_tokens=cached_reqs.num_computed_tokens[i],
                            can_load=True,
                        )
                    request_tuple = self._unfinished_requests.get(req_id)
                    if request_tuple is None:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, "
                            "but it is scheduled as a preempted cached request"
                        )
                    request_real = request_tuple[0]  # type: ignore[index]
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    previous_tracker = self._request_trackers.get(req_id)
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids_by_group=normalize_block_ids_by_group(new_block_ids),
                        num_saved_tokens=0,
                        token_ids=request_real.prompt_token_ids[:num_tokens_to_compute].copy(),
                        block_keys=(previous_tracker.block_keys.copy() if previous_tracker else []),
                        chunk_gvas=(previous_tracker.chunk_gvas.copy() if previous_tracker else []),
                    )
                    self._request_trackers[req_id] = request_tracker
                    num_hit_blocks = len(request_tracker.block_keys)

                    num_blocks = len(new_block_ids)
                    has_last_block = num_tokens_to_compute % self._block_size != 0
                    self._generate_keys_and_alloc(
                        request_real.block_hashes[num_hit_blocks:num_blocks],
                        request_tracker=request_tracker,
                        has_last_block=has_last_block,
                    )

                    last_chunk_tokens_num = (
                        self._floor_to_cache_transfer_granularity(len(request_real.prompt_token_ids))
                        if self._discard_partial_chunks
                        else len(request_real.prompt_token_ids)
                    )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self.cache_transfer_granularity,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request_real.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                        kv_cache_group_families=self.kv_cache_group_families,
                    )

                # decode/chunked request
                else:
                    request_tracker = self._request_trackers.get(req_id)
                    if request_tracker is None:
                        raise ValueError(
                            f"Request {req_id} is not in _request_trackers, "
                            "but it is scheduled to be cached"
                        )
                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        request = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
                        request_tracker.token_len += len(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                        )
                    prev_token_count = request_tracker.token_len - num_new_tokens
                    prev_hash_count = prev_token_count // self._block_size
                    current_hash_count = request_tracker.token_len // self._block_size
                    new_hash_count = current_hash_count - prev_hash_count
                    has_last_block = request_tracker.token_len % self._block_size != 0
                    if new_hash_count > 0 or has_last_block:
                        self._generate_keys_and_alloc(
                            request.block_hashes[prev_hash_count:current_hash_count],
                            request_tracker=request_tracker,
                            has_last_block=has_last_block,
                        )
                    if new_block_ids is not None:
                        request_tracker.update(new_block_ids)
                    last_chunk_tokens_num = (
                        self._floor_to_cache_transfer_granularity(len(request.prompt_token_ids))
                        if self._discard_partial_chunks
                        else len(request.prompt_token_ids)
                    )
                    load_spec = None
                    if self.prefill_offload:
                        load_spec = LoadSpec(
                            vllm_cached_tokens=0,
                            kvpool_cached_tokens=cached_reqs.num_computed_tokens[i],
                            can_load=True,
                        )
                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=force_skip_save,
                        block_hashes=request.block_hashes,
                        is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
                        discard_partial_chunks=self._discard_partial_chunks,
                        original_block_size=self.original_block_size,
                        kv_cache_group_families=self.kv_cache_group_families,
                    )
                if req_meta is not None:
                    self.touch_sending_mamba_blocks(req_meta)
                    meta.add_request(req_meta)
        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request, block_ids) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                if not load_spec:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                if (num_tokens_to_compute % self.cache_transfer_granularity != 0) and (
                    num_tokens_to_compute == len(request.prompt_token_ids) - 1
                ):
                    num_tokens_to_compute = num_tokens_to_compute + 1
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids_by_group=block_ids,
                    num_saved_tokens=0,
                )

                self._request_trackers[request_id] = request_tracker

                num_blocks = num_tokens_to_compute // self._block_size
                has_last_block = num_tokens_to_compute % self._block_size != 0
                block_hashes_for_keys = request.block_hashes[:num_blocks]
                self._generate_keys_and_alloc(
                    block_hashes_for_keys,
                    request_tracker=request_tracker,
                    has_last_block=has_last_block,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self.cache_transfer_granularity,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=request.block_hashes,
                    discard_partial_chunks=self._discard_partial_chunks,
                    kv_cache_group_families=self.kv_cache_group_families,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
        return meta

    def get_sending_event_id(self):
        """
        get a unique event id for a kv store request
        """
        using_id = self.sending_event_id
        # todo: reset sending_event_id, in case infinitely increasing
        self.sending_event_id += 1
        return using_id

    def touch_sending_mamba_blocks(self, req_meta: ReqMeta):
        """
        keep the reference of all non-null mamba blocks that will send to external kv store
        """
        if not self.use_hybrid or len(self.mamba_group_ids) == 0 or not req_meta.can_save:
            return
        using_event_id = self.get_sending_event_id()
        req_meta.event_id = using_event_id
        current_step_sending: list[int] = []
        for group_id in self.mamba_group_ids:
            group_block_ids = req_meta.block_ids_by_group[group_id]
            current_step_sending.extend([block_id for block_id in group_block_ids if block_id > 0])
        logger.debug("event: %s touch blocks: %s", using_event_id, current_step_sending)
        assert self._block_pool is not None
        self._block_pool.touch([self._block_pool.blocks[block_id] for block_id in current_step_sending])
        self.sending_events[using_event_id] = 0
        self.sending_blocks[using_event_id] = current_step_sending

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        hand the connector_output, free non-null mamba blocks and so on.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, AscendStoreKVConnectorWorkerMetadata):
            return
        to_free_block_ids: list[int] = []
        for event_id, count in meta.completed_events.items():
            logger.debug("event %s update with %s", event_id, count)
            total = self.sending_events.get(event_id, -1)
            if total == -1:
                logger.warning("worker reports an invalid event: %s, count %s", event_id, count)
                continue
            total = total + count
            if total >= self._expected_worker_count:
                to_free_block_ids.extend(self.sending_blocks.pop(event_id, []))
                self.sending_events.pop(event_id, None)
            else:
                self.sending_events[event_id] = total

        if to_free_block_ids:
            logger.debug("free blocks: %s", to_free_block_ids)
            assert self._block_pool is not None
            self._block_pool.free_blocks([self._block_pool.blocks[block_id] for block_id in to_free_block_ids])

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            if logger.isEnabledFor(10):
                logger.debug("Delaying free of %d blocks for request %s", len(block_ids), request.request_id)
        return delay_free_blocks, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA path for hybrid KV cache groups."""
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_put:
            return False, None
        tracker = self._request_trackers.get(request.request_id)
        if tracker is not None and tracker.num_saved_tokens <= 0:
            return False, None
        block_ids = cast(tuple[list[int], ...], self.get_sw_clipped_blocks(block_ids))
        valid_group_block_ids = [group_block_ids for group_block_ids in block_ids if group_block_ids]
        delay_free_blocks = bool(valid_group_block_ids)
        if delay_free_blocks:
            logger.debug(
                "Delaying free of %d KV cache groups for request %s",
                len(valid_group_block_ids),
                request.request_id,
            )
        return delay_free_blocks, None

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool


def get_zmq_rpc_path_lookup(vllm_config: "VllmConfig") -> str:
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if "lookup_rpc_port" in extra_config:
            rpc_port = extra_config["lookup_rpc_port"]
        elif "mooncake_rpc_port" in extra_config:
            rpc_port = extra_config["mooncake_rpc_port"]
            logger.warning(
                "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
            )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
