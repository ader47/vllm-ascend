import queue
import threading
from collections import defaultdict
from typing import Any, List

import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerMultiBlockReqMeta,
    ReqMeta,
    get_block_hashes,
)
# isort: on


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.finished_requests: set[str] = set()
        self.max_batch = 512
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            request_data = self.request_queue.get()
            if request_data is None:
                logger.warning("Received a None request!")
                self.request_queue.task_done()
                continue
            try:
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread: %s", e)
                self.request_queue.task_done()

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error("Remote connection failed in contains: %s", e)
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        process_with_block_ids = getattr(self.token_database, "process_tokens_with_block_ids", None)
        if process_with_block_ids is not None:
            return process_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
                block_idx = start // group_block_size
                if block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and cache_role == "kv" and block_id <= 0:
                    continue
                yield start, end, key, block_id

        return iter_with_legacy_process_tokens()

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        group_uses_align_state: list[bool],
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.group_uses_align_state = group_uses_align_state
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            if req_id not in self.stored_requests:
                self.request_queue.task_done()
                return

            for group_id in req_meta.kv_cache_group_ids or [0]:
                starts = []
                ends = []
                keys = []
                block_hashes = []
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )

                for start, end, key, _ in self._process_tokens_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                ):
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_string())
                    block_hashes.append(group_block_hashes[start // group_block_size])

        if not keys:
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

                if not keys:
                    continue

        if skip_block_num == len(keys):
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

                if not missing_indices:
                    continue

                starts = [starts[index] for index in missing_indices]
                ends = [ends[index] for index in missing_indices]
                keys = [keys[index] for index in missing_indices]
                block_hashes = [block_hashes[index] for index in missing_indices]

                logger.info(
                    "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                    len(keys),
                    token_len // group_block_size,
                    len(missing_indices),
                    req_id,
                    group_id,
                )
                logger.debug(
                    "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                    req_id,
                    group_id,
                    token_len,
                    len(keys),
                    keys[:3],
                )

                addrs = []
                sizes = []
                stored_events: list[BlockStored] = []
                prev_key = None
                new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
                for index, start in enumerate(starts):
                    addr, size, _ = self._prepare_value(
                        start,
                        ends[index],
                        block_ids,
                        kv_cache_group_id=group_id,
                    )
                    addrs.append(addr)
                    sizes.append(size)

                    # Create KV event
                    if self.enable_kv_event:
                        token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                        block_size = (
                            req_meta.original_block_size[group_id]
                            if isinstance(req_meta.original_block_size, list)
                            else req_meta.original_block_size
                        )
                        if block_size is not None:
                            stored_event = BlockStored(
                                block_hashes=[new_block_hashes[index]],
                                parent_block_hash=prev_key,
                                token_ids=token_ids,
                                block_size=block_size,
                                lora_id=None,
                                medium="cpu",
                                lora_name=None,
                            )
                            stored_events.append(stored_event)
                            prev_key = new_block_hashes[index]
                            logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

                if self.kv_role == "kv_consumer":
                    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                        keys,
                        addrs,
                        sizes,
                        kv_cache_group_id=group_id,
                    )

                if current_event is not None:
                    current_event.synchronize()
                self.m_store.put(keys, addrs, sizes)

                # TODO Query specific replica info to update the event
                if self.enable_kv_event and stored_events is not None:
                    self.update_kv_event(stored_events)
        finally:
            # always free blocks
            self.mark_completed_events(req_meta.event_id)
        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(start, end, req_meta.block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = _circular_shift(key_list, self.tp_rank % len(key_list))
        addr_list_c = _circular_shift(addr_list, self.tp_rank % len(addr_list))
        size_list_c = _circular_shift(size_list, self.tp_rank % len(size_list))
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: List[threading.Event],
        sync_save_events: List[torch.npu.Event],
        enable_kv_event: bool = False,
        layer_transfer_finished_events = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.stored_requests = defaultdict[str, int](int)
        self.layer_transfer_finished_events = layer_transfer_finished_events

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def try_finish_and_delete_stored_request(self, req_id: str) -> bool:
        with self.done_task_lock:
            if req_id in self.stored_requests and self.stored_requests[req_id] == 0:
                del self.stored_requests[req_id]
                return True
            return False

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_metas: LasyerMultiBlockReqMeta
    ):
        if len(req_metas) == 0:
            self.request_queue.task_done()
            return
        addr_list = []
        gvas_list = []
        size_list = []
        layer_id = req_metas[0].layer_id
        for req_meta in req_metas:
            is_last_chunk = req_meta.is_last_chunk
            if req_meta.addr_list is not None and req_meta.size_list is not None:
                addr_list.extend(req_meta.addr_list[self.tp_rank % self.put_step::self.put_step])
                size_list.extend(req_meta.size_list[self.tp_rank % self.put_step::self.put_step])
            if req_meta.gvas_list is not None:
                gvas_list.extend(req_meta.gvas_list[self.tp_rank % self.put_step::self.put_step])
            if layer_id == self.final_layer_id and is_last_chunk:
                self.set_finished_request(req_meta.req_id)
            self.dec_stored_request(req_meta.req_id)
        self.sync_save_events[layer_id].synchronize()
        res = self.m_store.store.batch_copy(gvas_list, addr_list, size_list, 0)
        # wait for KV transfer (PD)
        # if self.layer_transfer_finished_events is not None:
        #     is_finish = self.layer_transfer_finished_events[layer_id].wait(timeout=10)  # try---cache
        #     if not is_finish:
        #         logger.info(f"Layerwise {layer_id} transfer failed")
        #     self.layer_transfer_finished_events[layer_id].clear()
        if res != 0:
            logger.error("Layerwise %d save batch_copy failed with return code %d", layer_id, res)
        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        self.layer_save_finished_events[layer_id].set()
        req_metas.clear()

        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: List[threading.Event],
        layer_save_finished_events: List[threading.Event],
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events

    def add_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, data: LasyerMultiBlockReqMeta
    ):
        wait_for_save, req_metas, layer_id = data

        if wait_for_save is not None:
            is_finish = self.layer_save_finished_events[wait_for_save].wait(timeout=10)
            if not is_finish:
                logger.info("Layerwise %d save wait timed out", wait_for_save)
            self.layer_save_finished_events[wait_for_save].clear()

        if len(req_metas) == 0:
            assert not self.layer_load_finished_events[layer_id].is_set()
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        addr_list = []
        gvas_list = []
        size_list = []
        layer_id = req_metas[0].layer_id
        for req_meta in req_metas:
            if req_meta.addr_list is not None and req_meta.size_list is not None:
                addr_list.extend(req_meta.addr_list)
                size_list.extend(req_meta.size_list)
            if req_meta.gvas_list is not None:
                gvas_list.extend(req_meta.gvas_list)

        gvas_list_c = _circular_shift(gvas_list, self.tp_rank % len(gvas_list))
        addr_list_c = _circular_shift(addr_list, self.tp_rank % len(addr_list))
        size_list_c = _circular_shift(size_list, self.tp_rank % len(size_list))
        res = self.m_store.store.batch_copy(gvas_list_c, addr_list_c, size_list_c, 1)
        if res != 0:
            logger.error("Layerwise %d load batch_copy failed with return code %d", layer_id, res)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        self.layer_load_finished_events[layer_id].set()
        req_metas.clear()
        self.request_queue.task_done()