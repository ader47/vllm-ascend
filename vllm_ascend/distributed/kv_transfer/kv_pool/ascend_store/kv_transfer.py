import ctypes
import fcntl
import json
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import torch_npu

from vllm.distributed.kv_events import BlockStored
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    ReqMeta,
)

# isort: on
from vllm_ascend.memcache_comm_fence import kv_d2d_comm_stream


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


def _circular_shift_array(value: np.ndarray, offset: int) -> np.ndarray:
    length = len(value)
    if length == 0:
        return value
    offset %= length
    if offset == 0:
        return value
    return np.concatenate((value[offset:], value[:offset]))


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
    ) -> None:
        self.my_key_index = my_key_index
        self.num_ranks_per_layer = num_ranks_per_layer
        self.page_size_bytes = page_size_bytes
        self._block_len_np = np.asarray(token_database.block_len, dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.kv_caches_base_addr,
            dtype=np.int64,
        )
        self._full_block_inner_offsets_np = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(self._block_len_np[:-1], dtype=np.int64),
            )
        )
        self._block_ids_scratch_np: np.ndarray | None = None
        self._block_gvas_scratch_np: np.ndarray | None = None
        self._last_block_ids_scratch_np: np.ndarray | None = None
        self._last_gvas_scratch_np: np.ndarray | None = None

    def _ensure_scratch_array(self, attr_name: str, capacity: int) -> np.ndarray:
        array = getattr(self, attr_name, None)
        if array is None or array.shape[0] < capacity:
            array = np.empty(capacity, dtype=np.int64)
            setattr(self, attr_name, array)
        return array[:capacity]

    def _get_transfer_scratch_arrays(
        self,
        total_blocks: int,
        total_last_blocks: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._ensure_scratch_array("_block_ids_scratch_np", total_blocks),
            self._ensure_scratch_array("_block_gvas_scratch_np", total_blocks),
            self._ensure_scratch_array("_last_block_ids_scratch_np", total_last_blocks),
            self._ensure_scratch_array("_last_gvas_scratch_np", total_last_blocks),
        )

    @staticmethod
    def _concat_transfer_arrays(
        first: np.ndarray,
        second: np.ndarray,
    ) -> np.ndarray:
        if first.size == 0:
            return second
        if second.size == 0:
            return first
        return np.concatenate((first, second))

    @staticmethod
    def _dedupe_transfer_blocks(
        block_ids_arr: np.ndarray,
        block_gvas_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if block_ids_arr.size <= 1:
            return block_ids_arr, block_gvas_arr

        block_transfer_array = np.column_stack((block_ids_arr, block_gvas_arr))
        _, unique_indices = np.unique(
            block_transfer_array,
            axis=0,
            return_index=True,
        )
        if unique_indices.size == block_ids_arr.size:
            return block_ids_arr, block_gvas_arr

        return (
            block_ids_arr[unique_indices],
            block_gvas_arr[unique_indices],
        )

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        block_len_np = self._block_len_np
        length = block_len_np.shape[0]
        base_offset = layer_id * length
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset : base_offset + length]
        rank_layer_offset = (layer_id * self.num_ranks_per_layer + self.my_key_index) * self.page_size_bytes

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * block_len_np[None, :]
        size_arr = np.broadcast_to(block_len_np, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + self._full_block_inner_offsets_np[None, :]

        return (
            addr_arr.ravel(),
            size_arr.ravel(),
            gvas_arr.ravel(),
        )

    @staticmethod
    def _require_request_arrays(
        block_range: LayerBlockRange,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = block_range.request
        if request.block_ids_np is None or request.block_gvas_np is None:
            raise RuntimeError("ReqMeta numpy block metadata is not initialized")
        return request.block_ids_np, request.block_gvas_np

    def build(self, task: LayerTransferTask) -> LayerBatchReqMeta | None:
        if not task.block_ranges:
            return None

        total_blocks = 0
        total_last_blocks = 0
        for block_range in task.block_ranges:
            total_blocks += block_range.end_block - block_range.start_block
            if block_range.partial_block_index is not None:
                total_last_blocks += 1

        (
            block_ids_arr,
            block_gvas_arr,
            last_block_ids_arr,
            last_gvas_arr,
        ) = self._get_transfer_scratch_arrays(total_blocks, total_last_blocks)
        req_ids = []
        is_last_chunks = []
        offset = 0
        last_offset = 0
        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            num_blocks = block_range.end_block - block_range.start_block
            block_ids_np, block_gvas_np = self._require_request_arrays(block_range)
            if num_blocks > 0:
                end = offset + num_blocks
                gva_start = block_range.start_block - request.gva_block_offset
                gva_end = block_range.end_block - request.gva_block_offset
                if gva_start < 0 or gva_end > len(block_gvas_np):
                    raise RuntimeError(
                        "ReqMeta GVA metadata does not cover requested block "
                        f"range [{block_range.start_block}, {block_range.end_block}) "
                        f"with offset {request.gva_block_offset}"
                    )
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block : block_range.end_block]
                block_gvas_arr[offset:end] = block_gvas_np[gva_start:gva_end]
                offset = end

            if block_range.partial_block_index is not None:
                assert request.last_block_gva is not None
                last_block_ids_arr[last_offset] = block_ids_np[block_range.partial_block_index]
                last_gvas_arr[last_offset] = request.last_block_gva
                last_offset += 1

        block_ids_arr = self._concat_transfer_arrays(
            block_ids_arr,
            last_block_ids_arr,
        )
        block_gvas_arr = self._concat_transfer_arrays(
            block_gvas_arr,
            last_gvas_arr,
        )
        block_ids_arr, block_gvas_arr = self._dedupe_transfer_blocks(
            block_ids_arr,
            block_gvas_arr,
        )
        addr_array, size_array, gvas_array = self._build_transfer_arrays(block_ids_arr, block_gvas_arr, task.layer_id)

        return LayerBatchReqMeta(
            req_ids=req_ids,
            layer_id=task.layer_id,
            is_last_chunks=is_last_chunks,
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
            block_ids_array=block_ids_arr.copy(),
            block_gvas_array=block_gvas_arr.copy(),
        )


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.num_addrs_per_block = len(token_database.block_len)
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def add_request(
        self,
        request: ReqMeta | LayerBatchReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(
        self,
        req_ids: set[str] | None = None,
    ) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            if req_ids is None:
                finished_requests = self.finished_requests.copy()
                self.finished_requests.clear()
            else:
                finished_requests = self.finished_requests & req_ids
                self.finished_requests -= finished_requests
        return finished_requests

    def discard_finished_requests(self, req_ids: set[str]) -> None:
        with self.done_task_lock:
            self.finished_requests -= req_ids

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    @staticmethod
    def _split_transfer_packets(
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        max_transfer_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_transfer_bytes <= 0:
            return gvas, addrs, sizes

        split_counts = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
        total_splits = int(split_counts.sum())
        if total_splits == sizes.shape[0]:
            return gvas, addrs, sizes

        split_indices = np.arange(int(split_counts.max()), dtype=np.int64)
        split_mask = split_indices[:, None] < split_counts[None, :]
        entry_indices = np.broadcast_to(
            np.arange(sizes.shape[0], dtype=np.int64),
            split_mask.shape,
        )[split_mask]
        transfer_offsets = np.broadcast_to(
            split_indices[:, None] * max_transfer_bytes,
            split_mask.shape,
        )[split_mask]

        split_gvas = gvas[entry_indices] + transfer_offsets
        split_addrs = addrs[entry_indices] + transfer_offsets
        split_sizes = np.minimum(
            max_transfer_bytes,
            sizes[entry_indices] - transfer_offsets,
        )
        return split_gvas, split_addrs, split_sizes

    def _batch_copy_with_limits(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        direction: int,
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> int:
        if len(gvas) == 0:
            return 0

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                return res
        return 0

    def _set_os_thread_name(self) -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            # Linux task comm is limited to 15 visible bytes plus NUL.
            libc.prctl(15, self.name[:15].encode(), 0, 0, 0)
        except Exception:
            pass

    def _after_set_device(self) -> None:
        pass

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self._set_os_thread_name()
        self.m_store.set_device()
        self._after_set_device()
        self.ready_event.set()
        while True:
            request_data = self.request_queue.get()
            if request_data is None:
                logger.warning("Received a None request!")
                self.request_queue.task_done()
                continue
            # try:
            self._handle_request(request_data)
            # except Exception as e:
            #     logger.error("Error in KVCacheTransferThread: %s", e)
            #     self.request_queue.task_done()

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
            logger.error(f"Remote connection failed in contains: {e}")
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, tp_size, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.enable_kv_event = enable_kv_event

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
        token_len = req_meta.save_end_token
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        starts = []
        ends = []
        keys = []
        block_hashes = []
        if req_id not in self.stored_requests:
            self.request_queue.task_done()
            return

        for index, (start, end, key) in enumerate(self.token_database.process_tokens(token_len, req_meta.block_hashes)):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])

        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]
            block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            self.dec_stored_request(req_id)
            if self.stored_requests.get(req_id, -1) == 0:
                self.delete_finished_stored_request(req_id)
                self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        exists_states = self.lookup(keys)
        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

        if not missing_indices:
            self.dec_stored_request(req_id)
            if self.stored_requests.get(req_id, -1) == 0:
                self.delete_finished_stored_request(req_id)
                self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        starts = [starts[index] for index in missing_indices]
        ends = [ends[index] for index in missing_indices]
        keys = [keys[index] for index in missing_indices]
        block_hashes = [block_hashes[index] for index in missing_indices]

        logger.debug(
            "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            len(missing_indices),
            req_id,
        )

        if keys:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            addrs = []
            sizes = []
            stored_events: list[BlockStored] = []
            prev_key = None
            new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
            for index, start in enumerate(starts):
                addr, size, _ = self.token_database.prepare_value(start, ends[index], block_ids)
                addrs.append(addr)
                sizes.append(size)

                # Create KV event
                if self.enable_kv_event:
                    token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                    stored_event = BlockStored(
                        block_hashes=[new_block_hashes[index]],
                        parent_block_hash=prev_key,
                        token_ids=token_ids,
                        block_size=req_meta.original_block_size,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    stored_events.append(stored_event)
                    prev_key = new_block_hashes[index]
                    logger.debug(f"Added kv cache event '{stored_event}' to kv cache events queue")

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

            if current_event is not None:
                current_event.synchronize()
            self.m_store.put(keys, addrs, sizes)

            # TODO Query specific replica info to update the event
            if self.enable_kv_event and stored_events is not None:
                self.update_kv_event(stored_events)

        self.dec_stored_request(req_id)
        if self.stored_requests.get(req_id, -1) == 0:
            self.delete_finished_stored_request(req_id)
            self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        ready_event: threading.Event,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreRecvingThread",
        )

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size
            * self.block_size
        )
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(token_len, req_meta.block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(start, end, req_meta.block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = _circular_shift(key_list, (self.tp_rank * len(key_list)) // self.tp_size)
        addr_list_c = _circular_shift(addr_list, (self.tp_rank * len(addr_list)) // self.tp_size)
        size_list_c = _circular_shift(size_list, (self.tp_rank * len(size_list)) // self.tp_size)
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        num_layers: int,
        layer_save_finished_events: list[threading.Event],
        sync_save_events: list[torch.npu.Event],
        enable_kv_event: bool = False,
        layer_transfer_finished_events=None,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerSendingThread",
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.stored_requests = defaultdict[str, int](int)
        self.layer_transfer_finished_events = layer_transfer_finished_events
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.layer_batch_builder = LayerBatchBuilder(
            token_database,
            my_key_index,
            num_ranks_per_layer,
            page_size_bytes,
        )

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
        self, req_meta: list[LayerTransferTask]
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, transfer_tasks: list[LayerTransferTask]
    ):
        if len(transfer_tasks) == 0:
            self.request_queue.task_done()
            return
        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")
        req_meta = self.layer_batch_builder.build(transfer_tasks[0])
        if req_meta is None:
            layer_id = transfer_tasks[0].layer_id
            assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
            logger.debug(f">>>>>>>>>>>>>>>>>>>> set save layer {layer_id}")
            self.layer_save_finished_events[layer_id].set()
            self.request_queue.task_done()
            return
        layer_id = req_meta.layer_id
        rank_start = self.tp_rank % self.put_step
        addr_array = req_meta.addr_array[rank_start :: self.put_step]
        size_array = req_meta.size_array[rank_start :: self.put_step]
        gvas_array = req_meta.gvas_array[rank_start :: self.put_step]
        for req_id in req_meta.req_ids:
            self.dec_stored_request(req_id)
        self.sync_save_events[layer_id].synchronize()
        res = self._batch_copy_with_limits(
            gvas_array,
            addr_array,
            size_array,
            0,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        # wait for KV transfer (PD)
        # if self.layer_transfer_finished_events is not None:
        #     is_finish = self.layer_transfer_finished_events[layer_id].wait(timeout=10)  # try---cache
        #     if not is_finish:
        #         logger.info(f"Layerwise {layer_id} transfer failed")
        #     self.layer_transfer_finished_events[layer_id].clear()
        if res != 0:
            logger.error("Layerwise %d save batch_copy failed with return code %d", layer_id, res)
        else:
            for req_id in req_meta.req_ids:
                if self.try_finish_and_delete_stored_request(req_id):
                    self.set_finished_request(req_id)
        assert not self.layer_save_finished_events[layer_id].is_set(), f"thread: {layer_id} save failed "
        logger.debug(f">>>>>>>>>>>>>>>>>>>> set save layer {layer_id}")
        self.layer_save_finished_events[layer_id].set()
        transfer_tasks.clear()

        self.request_queue.task_done()


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        tp_size: int,
        dcp_size: int,
        put_step: int,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        layer_load_finished_events: list[threading.Event],
        layer_save_finished_events: list[threading.Event],
        num_layers: int,
        h2d_stagger_us: int = 0,
        h2d_stagger_group_size: int = 0,
        h2d_stagger_dynamic_addrs_per_us: int = 0,
        h2d_stagger_max_us: int = 0,
        h2d_concurrent_ranks: int = 0,
        h2d_batch_window_us: int = 0,
        h2d_runtime_config_path: str | None = None,
        h2d_runtime_config_check_interval: float = 1.0,
        h2d_max_inflight_ranks: int = 0,
        h2d_token_dir: str | None = None,
        h2d_global_rank: int = 0,
        h2d_global_world_size: int = 1,
        max_transfer_blocks: int = 0,
        max_transfer_bytes: int = 0,
        layer_kv_caches: list[tuple[torch.Tensor, ...]] | None = None,
        h2d_reader_group: GroupCoordinator | None = None,
        h2d_reader_count: int = 0,
        h2d_reader_collective_blocks: int = 16,
    ):
        super().__init__(
            m_store,
            token_database,
            block_size,
            tp_rank,
            tp_size,
            dcp_size,
            ready_event,
            name="KVCacheStoreLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.num_ranks_per_layer = num_ranks_per_layer
        self.page_size_bytes = page_size_bytes
        self.h2d_stagger_us = h2d_stagger_us
        self.h2d_stagger_group_size = h2d_stagger_group_size
        self.h2d_stagger_dynamic_addrs_per_us = h2d_stagger_dynamic_addrs_per_us
        self.h2d_stagger_max_us = h2d_stagger_max_us
        self.h2d_concurrent_ranks = h2d_concurrent_ranks
        self.h2d_batch_window_us = h2d_batch_window_us
        self.h2d_runtime_config_path = h2d_runtime_config_path
        self.h2d_runtime_config_check_interval = max(
            0.0,
            h2d_runtime_config_check_interval,
        )
        self._h2d_runtime_config_last_check = 0.0
        self._h2d_runtime_config_mtime: float | None = None
        self.h2d_max_inflight_ranks = max(0, h2d_max_inflight_ranks)
        self.h2d_token_dir = h2d_token_dir
        self._h2d_token_dir_warning_logged = False
        self._h2d_token_file_warning_logged = False
        self.h2d_global_rank = h2d_global_rank
        self.h2d_global_world_size = max(1, h2d_global_world_size)
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.layer_kv_caches = layer_kv_caches or []
        self.h2d_reader_group = h2d_reader_group
        self.h2d_reader_count = max(0, h2d_reader_count)
        self.h2d_reader_collective_blocks = max(1, h2d_reader_collective_blocks)
        self._block_lens_np = np.asarray(token_database.block_len, dtype=np.int64)
        self._block_inner_offsets_np = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(self._block_lens_np[:-1], dtype=np.int64),
            )
        )
        self._typed_staging_buffers: dict[tuple[Any, ...], torch.Tensor] = {}
        self._slot_mapping_buffers: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._slot_mapping_cache_key: tuple[torch.device, int, bytes] | None = None
        self._slot_mapping_cache_value: torch.Tensor | None = None
        self._h2d_status_tensor = torch.empty(1, dtype=torch.int32, device="cpu")
        self._h2d_gvas_scratch_np: np.ndarray | None = None
        self._h2d_addr_scratch_np: np.ndarray | None = None
        self._h2d_size_scratch_np: np.ndarray | None = None
        self._h2d_block_offsets_scratch_np: np.ndarray | None = None
        self._h2d_base_addrs_scratch_np: np.ndarray | None = None
        self._transfer_stream: Any | None = None
        self._pending_cooperative_loads: dict[
            int,
            list[
                tuple[
                    tuple[torch.Tensor, ...],
                    list[torch.Tensor],
                    np.ndarray,
                    int,
                    int,
                ]
            ],
        ] = {}
        self._inflight_cooperative_layers: set[int] = set()
        self._pending_cooperative_lock = threading.Lock()
        self.layer_batch_builder = LayerBatchBuilder(
            token_database,
            my_key_index,
            num_ranks_per_layer,
            page_size_bytes,
        )

    def _after_set_device(self) -> None:
        self._transfer_stream = torch.npu.Stream()

    def _get_transfer_stream(self) -> Any:
        if self._transfer_stream is None:
            self._transfer_stream = torch.npu.Stream()
        return self._transfer_stream

    def _maybe_update_h2d_runtime_config(self) -> None:
        if not self.h2d_runtime_config_path:
            return
        now = time.monotonic()
        if now - self._h2d_runtime_config_last_check < self.h2d_runtime_config_check_interval:
            return
        self._h2d_runtime_config_last_check = now

        try:
            stat_result = os.stat(self.h2d_runtime_config_path)
        except FileNotFoundError:
            return
        except OSError as err:
            logger.warning(
                "Failed to stat H2D runtime config file %s: %s",
                self.h2d_runtime_config_path,
                err,
            )
            return

        config_mtime = stat_result.st_mtime
        if self._h2d_runtime_config_mtime == config_mtime:
            return

        try:
            with open(self.h2d_runtime_config_path) as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            logger.warning(
                "Failed to load H2D runtime config file %s: %s",
                self.h2d_runtime_config_path,
                err,
            )
            return

        old_concurrent_ranks = self.h2d_concurrent_ranks
        old_batch_window_us = self.h2d_batch_window_us
        old_max_inflight_ranks = self.h2d_max_inflight_ranks
        try:
            if "concurrent_ranks" in config:
                self.h2d_concurrent_ranks = max(
                    0,
                    int(config["concurrent_ranks"]),
                )
            if "batch_window_us" in config:
                self.h2d_batch_window_us = max(
                    0,
                    int(config["batch_window_us"]),
                )
            if "max_inflight_ranks" in config:
                self.h2d_max_inflight_ranks = max(
                    0,
                    int(config["max_inflight_ranks"]),
                )
        except (TypeError, ValueError) as err:
            logger.warning(
                "Invalid H2D runtime config values in %s: %s",
                self.h2d_runtime_config_path,
                err,
            )
            return

        self._h2d_runtime_config_mtime = config_mtime
        if (
            self.h2d_concurrent_ranks != old_concurrent_ranks
            or self.h2d_batch_window_us != old_batch_window_us
            or self.h2d_max_inflight_ranks != old_max_inflight_ranks
        ):
            logger.info(
                "Updated H2D runtime config from %s: "
                "concurrent_ranks %d -> %d, batch_window_us %d -> %d, "
                "max_inflight_ranks %d -> %d",
                self.h2d_runtime_config_path,
                old_concurrent_ranks,
                self.h2d_concurrent_ranks,
                old_batch_window_us,
                self.h2d_batch_window_us,
                old_max_inflight_ranks,
                self.h2d_max_inflight_ranks,
            )

    def add_request(  # type: ignore[override]
        self, req_meta: LayerLoadTask
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _get_h2d_stagger_delay_us(self, layer_id: int, num_addrs: int) -> int:
        self._maybe_update_h2d_runtime_config()
        delay_us = 0
        if self.h2d_concurrent_ranks > 0 and self.h2d_batch_window_us > 0:
            concurrent_ranks = min(
                self.h2d_concurrent_ranks,
                self.h2d_global_world_size,
            )
            num_batches = (self.h2d_global_world_size + concurrent_ranks - 1) // concurrent_ranks
            if num_batches > 1:
                batch_index = self.h2d_global_rank // concurrent_ranks
                # Rotate batch order by layer so the same ranks do not always
                # submit first.
                batch_slot = (batch_index + layer_id) % num_batches
                delay_us += batch_slot * self.h2d_batch_window_us

        if self.h2d_stagger_us <= 0:
            return delay_us
        group_size = self.h2d_stagger_group_size or self.tp_size
        group_size = max(1, group_size)
        slot = (self.tp_rank + layer_id) % group_size

        stagger_us = self.h2d_stagger_us
        if self.h2d_stagger_dynamic_addrs_per_us > 0:
            stagger_us += num_addrs // self.h2d_stagger_dynamic_addrs_per_us
        if self.h2d_stagger_max_us > 0:
            stagger_us = min(stagger_us, self.h2d_stagger_max_us)
        delay_us += slot * stagger_us
        return delay_us

    def _stagger_h2d_submit(self, layer_id: int, num_addrs: int) -> None:
        delay_us = self._get_h2d_stagger_delay_us(layer_id, num_addrs)
        if delay_us > 0:
            time.sleep(delay_us / 1_000_000)

    def _acquire_h2d_token(self) -> tuple[int, str] | None:
        if self.h2d_max_inflight_ranks <= 0:
            return None

        token_dir = self.h2d_token_dir or "/dev/shm/vllm_ascend_h2d_tokens"
        try:
            os.makedirs(token_dir, exist_ok=True)
        except OSError as err:
            if not self._h2d_token_dir_warning_logged:
                logger.warning(
                    "Failed to create H2D token directory %s: %s. Continue without H2D in-flight limit.",
                    token_dir,
                    err,
                )
                self._h2d_token_dir_warning_logged = True
            return None

        while True:
            self._maybe_update_h2d_runtime_config()
            if self.h2d_max_inflight_ranks <= 0:
                return None

            for token_id in range(self.h2d_max_inflight_ranks):
                token_path = os.path.join(token_dir, f"token_{token_id}")
                try:
                    token_fd = os.open(token_path, os.O_CREAT | os.O_RDWR, 0o666)
                except OSError as err:
                    if not self._h2d_token_file_warning_logged:
                        logger.warning("Failed to open H2D token file %s: %s", token_path, err)
                        self._h2d_token_file_warning_logged = True
                    continue

                try:
                    fcntl.flock(token_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return token_fd, token_path
                except BlockingIOError:
                    os.close(token_fd)
                except OSError as err:
                    if not self._h2d_token_file_warning_logged:
                        logger.warning("Failed to lock H2D token file %s: %s", token_path, err)
                        self._h2d_token_file_warning_logged = True
                    os.close(token_fd)

            time.sleep(0.0001)

    @staticmethod
    def _release_h2d_token(token: tuple[int, str] | None) -> None:
        if token is None:
            return
        token_fd, token_path = token
        try:
            fcntl.flock(token_fd, fcntl.LOCK_UN)
        except OSError as err:
            logger.warning("Failed to unlock H2D token file %s: %s", token_path, err)
        finally:
            os.close(token_fd)

    def _batch_copy_h2d_with_token(
        self,
        gvas_array: np.ndarray,
        addr_array: np.ndarray,
        size_array: np.ndarray,
    ) -> int:
        self._maybe_update_h2d_runtime_config()
        token = self._acquire_h2d_token()
        try:
            return self._batch_copy_with_limits(
                gvas_array,
                addr_array,
                size_array,
                1,
                self.max_transfer_blocks,
                self.max_transfer_bytes,
            )
        finally:
            self._release_h2d_token(token)

    def _ensure_h2d_scratch_array(
        self,
        attr_name: str,
        capacity: int,
    ) -> np.ndarray:
        array = getattr(self, attr_name)
        if array is None or array.shape[0] < capacity:
            array = np.empty(capacity, dtype=np.int64)
            setattr(self, attr_name, array)
        return array[:capacity]

    def _get_typed_staging_buffers(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        num_blocks: int,
        cache_key_prefix: tuple[Any, ...] = (),
    ) -> list[torch.Tensor]:
        assert self.h2d_reader_group is not None
        buffers = []
        for cache_index, cache in enumerate(kv_cache):
            block_shape = tuple(cache.shape[1:])
            key = (
                *cache_key_prefix,
                num_blocks,
                cache_index,
                cache.dtype,
                cache.device,
                block_shape,
            )
            buffer = self._typed_staging_buffers.get(key)
            if buffer is None:
                buffer = torch.empty(
                    (
                        num_blocks,
                        *block_shape,
                    ),
                    dtype=cache.dtype,
                    device=cache.device,
                )
                self._typed_staging_buffers[key] = buffer
            buffers.append(buffer)
        return buffers

    def _build_typed_staging_copy_arrays(
        self,
        staging_buffers: list[torch.Tensor],
        layer_id: int,
        block_gvas_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        block_count = len(block_gvas_array)
        block_lens = self._block_lens_np
        cache_count = len(staging_buffers)
        if cache_count > len(block_lens):
            raise RuntimeError(
                "KV cache block length metadata is shorter than staging buffers: "
                f"cache_count={cache_count}, block_lens={len(block_lens)}"
            )
        staging_block_bytes = np.asarray(
            [staging.element_size() * int(np.prod(staging.shape[1:])) for staging in staging_buffers],
            dtype=np.int64,
        )
        if not np.array_equal(staging_block_bytes, block_lens[:cache_count]):
            raise RuntimeError(
                "KV cache staging block bytes mismatch: "
                f"staging={staging_block_bytes.tolist()}, "
                f"metadata={block_lens[:cache_count].tolist()}"
            )
        total_addrs = block_count * cache_count
        rank_layer_offset = layer_id * self.num_ranks_per_layer * self.page_size_bytes
        gvas_array = self._ensure_h2d_scratch_array(
            "_h2d_gvas_scratch_np",
            total_addrs,
        ).reshape(block_count, cache_count)
        addr_array = self._ensure_h2d_scratch_array(
            "_h2d_addr_scratch_np",
            total_addrs,
        ).reshape(block_count, cache_count)
        size_array = self._ensure_h2d_scratch_array(
            "_h2d_size_scratch_np",
            total_addrs,
        ).reshape(block_count, cache_count)
        block_offsets = self._ensure_h2d_scratch_array(
            "_h2d_block_offsets_scratch_np",
            block_count,
        ).reshape(block_count, 1)
        base_addrs = self._ensure_h2d_scratch_array(
            "_h2d_base_addrs_scratch_np",
            cache_count,
        )

        base_addrs[:] = [staging.data_ptr() for staging in staging_buffers]
        block_offsets[:, 0] = np.arange(block_count, dtype=np.int64)
        np.add(
            block_gvas_array[:, None],
            rank_layer_offset + self._block_inner_offsets_np[None, :cache_count],
            out=gvas_array,
        )
        np.multiply(
            block_offsets,
            block_lens[None, :cache_count],
            out=addr_array,
        )
        np.add(addr_array, base_addrs[None, :], out=addr_array)
        size_array[:] = block_lens[None, :cache_count]
        return (
            gvas_array.ravel(),
            addr_array.ravel(),
            size_array.ravel(),
        )

    def _copy_remote_blocks_to_typed_staging(
        self,
        staging_buffers: list[torch.Tensor],
        req_meta: LayerBatchReqMeta,
        block_start: int,
        block_end: int,
    ) -> int:
        assert self.h2d_reader_group is not None
        if self.h2d_reader_group.rank_in_group != 0:
            return 0
        block_gvas = req_meta.block_gvas_array[block_start:block_end]

        gvas_array, addr_array, size_array = self._build_typed_staging_copy_arrays(
            staging_buffers,
            req_meta.layer_id,
            block_gvas,
        )
        self._stagger_h2d_submit(req_meta.layer_id, len(gvas_array))
        return self._batch_copy_h2d_with_token(
            gvas_array,
            addr_array,
            size_array,
        )

    def _broadcast_h2d_status(self, res: int) -> int:
        assert self.h2d_reader_group is not None
        status = self._h2d_status_tensor
        status[0] = res
        torch.distributed.broadcast(
            status,
            src=self.h2d_reader_group.ranks[0],
            group=self.h2d_reader_group.cpu_group,
        )
        return int(status.item())

    def _make_slot_mapping(
        self,
        block_ids_array: np.ndarray,
        tokens_per_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (
            device,
            tokens_per_block,
            block_ids_array.tobytes(),
        )
        if self._slot_mapping_cache_key == cache_key:
            assert self._slot_mapping_cache_value is not None
            return self._slot_mapping_cache_value

        block_ids = torch.from_numpy(block_ids_array.astype(np.int32, copy=False)).to(device=device, non_blocking=True)
        offsets_key = (device, tokens_per_block)
        offsets = self._slot_mapping_buffers.get(offsets_key)
        if offsets is None:
            offsets = torch.arange(
                tokens_per_block,
                device=device,
                dtype=torch.int32,
            )
            self._slot_mapping_buffers[offsets_key] = offsets
        slot_mapping = (block_ids[:, None] * tokens_per_block + offsets[None, :]).reshape(-1)
        self._slot_mapping_cache_key = cache_key
        self._slot_mapping_cache_value = slot_mapping
        return slot_mapping

    def _write_typed_parts_to_kv_cache(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        local_parts: list[torch.Tensor],
        block_ids_array: np.ndarray,
    ) -> None:
        tokens_per_block = local_parts[0].shape[1]
        if block_ids_array.size > 0:
            min_block_id = int(block_ids_array.min())
            max_block_id = int(block_ids_array.max())
            if min_block_id < 0 or max_block_id >= kv_cache[0].shape[0]:
                raise RuntimeError(
                    "KV cache write block id out of range: "
                    f"min={min_block_id}, max={max_block_id}, "
                    f"num_blocks={kv_cache[0].shape[0]}"
                )
        slot_mapping = self._make_slot_mapping(
            block_ids_array,
            tokens_per_block,
            local_parts[0].device,
        )
        key = local_parts[0].reshape(-1, *local_parts[0].shape[2:])
        value = local_parts[1].reshape(-1, *local_parts[1].shape[2:])
        if key.shape[0] != slot_mapping.numel() or value.shape[0] != slot_mapping.numel():
            raise RuntimeError(
                "KV cache write shape mismatch: "
                f"key_rows={key.shape[0]}, value_rows={value.shape[0]}, "
                f"slot_mapping={slot_mapping.numel()}"
            )
        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_mapping=slot_mapping,
        )

        for cache_index in range(2, len(kv_cache)):
            source = local_parts[cache_index].reshape(
                -1,
                local_parts[cache_index].shape[-1],
            )
            if source.shape[0] != slot_mapping.numel():
                raise RuntimeError(
                    "KV cache scatter shape mismatch: "
                    f"cache_index={cache_index}, source_rows={source.shape[0]}, "
                    f"slot_mapping={slot_mapping.numel()}"
                )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[cache_index].view(-1, source.shape[-1]),
                slot_mapping.view(-1, 1),
                source,
            )

    def _submit_cooperative_d2d_and_writeback(
        self,
        layer_id: int,
        chunks: list[
            tuple[
                tuple[torch.Tensor, ...],
                list[torch.Tensor],
                np.ndarray,
                int,
                int,
            ]
        ],
    ) -> None:
        assert self.h2d_reader_group is not None
        group = self.h2d_reader_group
        for kv_cache, staging_buffers, block_ids_array, block_start, block_end in chunks:
            comm_stream = kv_d2d_comm_stream()
            comm_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(comm_stream):
                logger.debug(
                    "Starting H2D broadcast layer=%d chunk=[%d,%d)",
                    layer_id,
                    block_start,
                    block_end,
                )
                for cache_index, staging in enumerate(staging_buffers):
                    logger.debug(
                        "Before H2D broadcast layer=%d chunk=[%d,%d) cache=%d "
                        "rank=%d rank_in_group=%d group=%s shape=%s dtype=%s",
                        layer_id,
                        block_start,
                        block_end,
                        cache_index,
                        self.h2d_reader_group.rank,
                        self.h2d_reader_group.rank_in_group,
                        self.h2d_reader_group.ranks,
                        tuple(staging.shape),
                        staging.dtype,
                    )
                    group.broadcast(staging, src=0)
                logger.debug(
                    "Finished H2D broadcast layer=%d chunk=[%d,%d)",
                    layer_id,
                    block_start,
                    block_end,
                )
            transfer_stream = self._get_transfer_stream()
            transfer_stream.wait_stream(comm_stream)
            with torch.npu.stream(transfer_stream):
                self._write_typed_parts_to_kv_cache(
                    kv_cache,
                    staging_buffers,
                    block_ids_array,
                )
            logger.debug(
                "Submitting H2D writeback layer=%d chunk=[%d,%d)",
                layer_id,
                block_start,
                block_end,
            )
            logger.debug(
                "Submitted H2D writeback layer=%d chunk=[%d,%d)",
                layer_id,
                block_start,
                block_end,
            )

    def _wait_for_inflight_cooperative_loads(self, layer_id: int | None = None) -> None:
        with self._pending_cooperative_lock:
            if layer_id is not None and layer_id not in self._inflight_cooperative_layers:
                return
            if layer_id is None and not self._inflight_cooperative_layers:
                return
        self._get_transfer_stream().synchronize()
        with self._pending_cooperative_lock:
            if layer_id is None:
                self._inflight_cooperative_layers.clear()
            else:
                self._inflight_cooperative_layers.discard(layer_id)

    def commit_pending_cooperative_loads(
        self,
        layer_id: int | None = None,
        wait: bool = False,
    ) -> None:
        with self._pending_cooperative_lock:
            if layer_id is None:
                layer_ids = sorted(self._pending_cooperative_loads)
            elif layer_id in self._pending_cooperative_loads:
                layer_ids = [layer_id]
            else:
                layer_ids = []
            pending_loads = [
                (
                    pending_layer_id,
                    self._pending_cooperative_loads.pop(pending_layer_id),
                )
                for pending_layer_id in layer_ids
            ]
        for pending_layer_id, chunks in pending_loads:
            self._submit_cooperative_d2d_and_writeback(pending_layer_id, chunks)
            with self._pending_cooperative_lock:
                self._inflight_cooperative_layers.add(pending_layer_id)
        if wait:
            if layer_id is None or pending_loads:
                self._wait_for_inflight_cooperative_loads(layer_id)
            else:
                with self._pending_cooperative_lock:
                    is_inflight = layer_id in self._inflight_cooperative_layers
                if is_inflight:
                    self._wait_for_inflight_cooperative_loads(layer_id)

    def _cooperative_h2d_load(self, req_meta: LayerBatchReqMeta) -> int | None:
        group = self.h2d_reader_group
        if group is None or self.h2d_reader_count <= 0 or group.world_size <= 1 or not self.layer_kv_caches:
            return None
        if req_meta.layer_id >= len(self.layer_kv_caches):
            return None
        if len(req_meta.block_ids_array) == 0:
            return 0

        kv_cache = self.layer_kv_caches[req_meta.layer_id]
        if len(kv_cache) < 2:
            return None
        if self.num_ranks_per_layer != 1:
            return None

        block_count = len(req_meta.block_ids_array)
        blocks_per_collective = min(
            self.h2d_reader_collective_blocks,
            block_count,
        )
        chunks = []
        for block_start in range(0, block_count, blocks_per_collective):
            block_end = min(block_start + blocks_per_collective, block_count)
            current_blocks = block_end - block_start
            staging_buffers = self._get_typed_staging_buffers(
                kv_cache,
                current_blocks,
                cache_key_prefix=(req_meta.layer_id, block_start),
            )
            res = self._copy_remote_blocks_to_typed_staging(
                staging_buffers,
                req_meta,
                block_start,
                block_end,
            )
            res = self._broadcast_h2d_status(res)
            if res != 0:
                return res
            chunks.append(
                (
                    kv_cache,
                    staging_buffers,
                    req_meta.block_ids_array[block_start:block_end].copy(),
                    block_start,
                    block_end,
                )
            )
        with self._pending_cooperative_lock:
            self._pending_cooperative_loads[req_meta.layer_id] = chunks
        return 0

    def _handle_request(  # type: ignore[override]
        self, data: LayerLoadTask
    ):
        wait_for_save = data.wait_for_save_layer
        transfer_tasks = data.transfer_tasks
        layer_id = data.layer_id
        attention_start_gate = data.attention_start_gate

        if len(transfer_tasks) == 0:
            if wait_for_save is not None:
                while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                    logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
                logger.debug(f">>>>>>>>>>>>>>>>>>>> clear save layer {wait_for_save}")
                self.layer_save_finished_events[wait_for_save].clear()
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug(f">>>>>>>>>>>>>>>>>>>> set load layer {layer_id}")
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return

        if len(transfer_tasks) > 1:
            raise ValueError(f"Expected at most one layer transfer task, got {len(transfer_tasks)}")
        req_meta = self.layer_batch_builder.build(transfer_tasks[0])
        if req_meta is None:
            assert not self.layer_load_finished_events[layer_id].is_set()
            logger.debug(f">>>>>>>>>>>>>>>>>>>> set load layer {layer_id}")
            self.layer_load_finished_events[layer_id].set()
            self.request_queue.task_done()
            return
        layer_id = req_meta.layer_id

        if wait_for_save is not None:
            while not self.layer_save_finished_events[wait_for_save].wait(timeout=10):
                logger.info("Layerwise %d save wait timed out, keep waiting before load", wait_for_save)
            logger.debug(f">>>>>>>>>>>>>>>>>>>> clear save layer {wait_for_save}")
            self.layer_save_finished_events[wait_for_save].clear()

        res = self._cooperative_h2d_load(req_meta)
        if res is None:
            if attention_start_gate is not None:
                while not attention_start_gate.wait(timeout=10):
                    logger.info("Layerwise %d load waits for attention compute start", layer_id)
            gvas_array = _circular_shift_array(
                req_meta.gvas_array,
                (self.tp_rank * len(req_meta.gvas_array)) // self.tp_size,
            )
            addr_array = _circular_shift_array(
                req_meta.addr_array,
                (self.tp_rank * len(req_meta.addr_array)) // self.tp_size,
            )
            size_array = _circular_shift_array(
                req_meta.size_array,
                (self.tp_rank * len(req_meta.size_array)) // self.tp_size,
            )
            self._stagger_h2d_submit(layer_id, len(gvas_array))
            res = self._batch_copy_h2d_with_token(
                gvas_array,
                addr_array,
                size_array,
            )
        if res != 0:
            logger.error("Layerwise %d load batch_copy failed with return code %d", layer_id, res)
        elif layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_meta.req_ids, req_meta.is_last_chunks):
                if is_last_chunk:
                    self.set_finished_request(req_id)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug(f">>>>>>>>>>>>>>>>>>>> set load layer {layer_id}")
        self.layer_load_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()
        self.get_event.set()
