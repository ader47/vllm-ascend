import threading
import time
from collections import defaultdict

import numpy as np
import torch

from vllm.logger import logger
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVTransferThread,
    _circular_shift_array,
)


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
        self._full_block_inner_offsets_np = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum(self._block_len_np[:-1], dtype=np.int64),
        ))
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
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset:base_offset + length]
        rank_layer_offset = (
            layer_id * self.num_ranks_per_layer + self.my_key_index
        ) * self.page_size_bytes

        addr_arr = (
            layer_base_addrs[None, :]
            + block_ids_arr[:, None] * block_len_np[None, :]
        )
        size_arr = np.broadcast_to(block_len_np, addr_arr.shape)
        gvas_arr = (
            base_gvas_arr[:, None]
            + rank_layer_offset
            + self._full_block_inner_offsets_np[None, :]
        )

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
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block:block_range.end_block]
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
        addr_array, size_array, gvas_array = self._build_transfer_arrays(
            block_ids_arr, block_gvas_arr, task.layer_id)

        return LayerBatchReqMeta(
            req_ids=req_ids,
            layer_id=task.layer_id,
            is_last_chunks=is_last_chunks,
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
        )


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

    def add_request(
        self, req_meta: list[LayerTransferTask]
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(
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
        addr_array = req_meta.addr_array[rank_start::self.put_step]
        size_array = req_meta.size_array[rank_start::self.put_step]
        gvas_array = req_meta.gvas_array[rank_start::self.put_step]
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
            name="KVCacheStoreLayerRecvingThread",
        )
        self.get_event = get_event
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        self.final_layer_id = num_layers - 1
        self.h2d_stagger_us = h2d_stagger_us
        self.h2d_stagger_group_size = h2d_stagger_group_size
        self.h2d_stagger_dynamic_addrs_per_us = h2d_stagger_dynamic_addrs_per_us
        self.h2d_stagger_max_us = h2d_stagger_max_us
        self.max_transfer_blocks = max_transfer_blocks
        self.max_transfer_bytes = max_transfer_bytes
        self.layer_batch_builder = LayerBatchBuilder(
            token_database,
            my_key_index,
            num_ranks_per_layer,
            page_size_bytes,
        )

    def add_request(
        self, req_meta: LayerLoadTask
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _get_h2d_stagger_delay_us(self, layer_id: int, num_addrs: int) -> int:
        if self.h2d_stagger_us <= 0:
            return 0
        group_size = self.h2d_stagger_group_size or self.tp_size
        group_size = max(1, group_size)
        slot = (self.tp_rank + layer_id) % group_size

        stagger_us = self.h2d_stagger_us
        if self.h2d_stagger_dynamic_addrs_per_us > 0:
            stagger_us += num_addrs // self.h2d_stagger_dynamic_addrs_per_us
        if self.h2d_stagger_max_us > 0:
            stagger_us = min(stagger_us, self.h2d_stagger_max_us)
        return slot * stagger_us

    def _stagger_h2d_submit(self, layer_id: int, num_addrs: int) -> None:
        delay_us = self._get_h2d_stagger_delay_us(layer_id, num_addrs)
        if delay_us > 0:
            time.sleep(delay_us / 1_000_000)

    def _handle_request(
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
        res = self._batch_copy_with_limits(
            gvas_array,
            addr_array,
            size_array,
            1,
            self.max_transfer_blocks,
            self.max_transfer_bytes,
        )
        if res != 0:
            logger.error("Layerwise %d load batch_copy failed with return code %d", layer_id, res)
        elif layer_id == self.final_layer_id:
            for req_id, is_last_chunk in zip(req_meta.req_ids,
                                             req_meta.is_last_chunks):
                if is_last_chunk:
                    self.set_finished_request(req_id)
        assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        logger.debug(f">>>>>>>>>>>>>>>>>>>> set load layer {layer_id}")
        self.layer_load_finished_events[layer_id].set()
        transfer_tasks.clear()
        self.request_queue.task_done()
        self.get_event.set()
