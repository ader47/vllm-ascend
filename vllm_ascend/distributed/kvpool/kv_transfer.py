import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import torch
from vllm.logger import logger

from vllm_ascend.distributed.kvpool.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kvpool.config_data import (
    ChunkedTokenDatabase,
    LasyerMultiBlockReqMeta,
    ReqMeta,
)
# isort: on


class KVTransferThread(threading.Thread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 block_size: int, tp_rank: int, dcp_size: int,
                 ready_event: threading.Event, name: str):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.max_batch = 512

    def add_request(
        self,
        request: ReqMeta,
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
            # try:
            request_data = self.request_queue.get()
            if request_data is None:
                logger.warning("Received a None request!")
                self.request_queue.task_done()
                continue
            self._handle_request(request_data)
            # except Exception as e:
            #     logger.error(f"Error in KVCacheTransferThread: {e}")

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        :param tokens: the input tokens, with shape [seq_len]
        :return: An int indicating how many prefix tokens are cached.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            for index, value in enumerate(res):  # type: ignore[arg-type]
                if value != 1:
                    return index
            # all tokens where found, return the maximal end
        except Exception as e:
            logger.error(f"Remote connection failed in contains: {e}")
            return 0
        return len(keys)


class KVCacheStoreSendingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 block_size: int, tp_rank: int, dcp_size: int, put_step: int,
                 kv_role: str, ready_event: threading.Event):
        super().__init__(m_store,
                         token_database,
                         block_size,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheSendingThread")
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        block_ids = req_meta.block_ids
        req_id = req_meta.req_id
        is_last_chunk = req_meta.is_last_chunk
        current_event = req_meta.current_event
        starts = []
        ends = []
        keys = []
        for start, end, key in self.token_database.process_tokens(
                token_len, req_meta.block_hashes):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())

        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step::self.put_step]
            ends = ends[self.tp_rank % self.put_step::self.put_step]
            keys = keys[self.tp_rank % self.put_step::self.put_step]

        if not keys:
            if is_last_chunk:
                self.set_finished_request(req_id)
            return

        skip_block_num = self.lookup(keys)

        if skip_block_num == len(keys):
            if is_last_chunk:
                self.set_finished_request(req_id)
            return

        starts = starts[skip_block_num:]
        ends = ends[skip_block_num:]
        keys = keys[skip_block_num:]

        logger.info(
            "Storing KV cache for %d out of %d blocks "
            "(skip_block_num=%d) for request %s",
            len(keys),
            token_len // self.block_size,
            skip_block_num,
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
            for index, start in enumerate(starts):
                addr, size, _ = self.token_database.prepare_value(
                    start, ends[index], block_ids)
                addrs.append(addr)
                sizes.append(size)

            if self.kv_role == "kv_consumer":
                keys, addrs, sizes = self.token_database.decode_adaptor_prefill_pp(
                    keys, addrs, sizes)

            if current_event is not None:
                current_event.synchronize()
            self.m_store.put(keys, addrs, sizes)

        with self.done_task_lock:
            self.stored_requests[req_id] -= 1
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 block_size: int, tp_rank: int, dcp_size: int,
                 ready_event: threading.Event):
        super().__init__(m_store,
                         token_database,
                         block_size,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheStoreRecvingThread")

    def _handle_request(self, req_meta: ReqMeta):
        req_id = req_meta.req_id
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // self.block_size * self.block_size)
        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(
                req_meta.token_len_chunk, req_meta.block_hashes, mask_num):
            addr, size, _ = self.token_database.prepare_value(
                start, end, req_meta.block_ids)
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
        key_list_c = key_list[self.tp_rank %
                              len(key_list):] + key_list[:self.tp_rank %
                                                         len(key_list)]
        addr_list_c = addr_list[self.tp_rank %
                                len(addr_list):] + addr_list[:self.tp_rank %
                                                             len(addr_list)]
        size_list_c = size_list[self.tp_rank %
                                len(size_list):] + size_list[:self.tp_rank %
                                                             len(size_list)]
        self.m_store.get(key_list_c, addr_list_c, size_list_c)
        self.set_finished_request(req_id)
        self.request_queue.task_done()

class KVCacheStoreLayerSendingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 block_size: int, tp_rank: int, dcp_size: int, put_step: int,
                 ready_event: threading.Event, num_layers: int, layer_save_finished_events: List[threading.Event], sync_save_events: List[torch.npu.Event()]):
        super().__init__(m_store,
                         token_database,
                         block_size,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheStoreLayerSendingThread")
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.layer_save_finished_events = layer_save_finished_events
        self.sync_save_events = sync_save_events
        self.req_ids = set()
        logger.info(f"====================> TP {self.tp_rank} create save thread!!")

    def add_request(  # type: ignore[override]
            self, req_metas: List[ReqMeta]) -> torch.Tensor:
        self.request_queue.put(req_metas)

    def _handle_request(  # type: ignore[override]
            self, req_metas: List[LasyerMultiBlockReqMeta]):
        if len(req_metas) == 0:
            return
        # key_list = []
        # addr_list = []
        # size_list = []
        layer_id = req_metas[0].layer_id
        # key_list_remove = []
        cur_req_ids = set()
        # gvas = []
        for req_meta in req_metas:
            cur_req_ids.add(req_meta.req_id)
            # starts = req_meta.starts
            # ends = req_meta.ends
            keys = req_meta.keys
            # current_event = req_meta.current_event
            # total_block = len(keys)
            is_last_chunk = req_meta.is_last_chunk
            # if not self.dcp_size > 1:
                # gvas_i = req_meta.gvas[self.tp_rank % self.put_step::self.put_step]
                # keys_str = req_meta.keys_str[self.tp_rank % self.put_step::self.put_step]
                # addr_list_i = req_meta.addr_list[self.tp_rank % self.put_step::self.put_step]
                # size_list_i = req_meta.size_list[self.tp_rank % self.put_step::self.put_step]
        #     # TODO there maybe has some problem when only have one block.
            if not keys:
                if is_last_chunk:
                    self.set_finished_request(req_meta.req_id)
                continue
        #     # keys_str = []
        #     # for key in keys:
        #     #     keys_str.append(key.to_string())
        #     # if 'last' not in keys_str[-1]:
        #     #     continue
        #     # logger.info(f"===================> keys_str {keys_str}   =================> addr_list_i {addr_list_i}====================> size_list_i {size_list_i} ======================> gvas_i{gvas_i}")
        #     # skip_block_num = self.lookup(keys_str)
        #     # # TODO check this
        #     if skip_block_num == len(keys_str):
        #         if is_last_chunk and layer_id == self.final_layer_id:
        #             self.set_finished_request(req_meta.req_id)
        #         continue

        #     gvas.extend(gvas_i)
        #     addr_list.extend(addr_list_i)
        #     size_list.extend(size_list_i)
        #
        #     # key_list.extend(keys_str[skip_block_num:])
        #     # size_list.extend(size_list_i[skip_block_num:])
        #     # addr_list.extend(addr_list_i[skip_block_num:])
        #
            if layer_id == self.final_layer_id and is_last_chunk:
                self.set_finished_request(req_meta.req_id)

        # self.sync_save_events[layer_id].synchronize()

        # if len(key_list) > 0:
            # for i in range(0, len(key_list), self.max_batch):
            #     self.m_store.put(key_list[i:i + self.max_batch], addr_list[i:i + self.max_batch],
            #                      size_list[i:i + self.max_batch])

            # res = self.m_store.store.batch_copy(gvas, addr_list, size_list, 0)
            # if res != 0:
            #     logger.info(
            #         f"send failed {res} gvas {gvas} addr_list {addr_list} size_list {size_list} key_list {key_list}")

        # TODO unwait
        # assert not self.layer_save_finished_events                [layer_id].is_set(), f"thread: {layer_id} save failed "
        self.layer_save_finished_events[layer_id].set()
        req_metas.clear()
        self.request_queue.task_done()


MODULE_TRANSFER_SEMAPHORE = threading.Semaphore(1)  # 最多2个线程同时传输


class KVCacheStoreLayerRecvingThread(KVTransferThread):

    def __init__(self, m_store: Backend, token_database: ChunkedTokenDatabase,
                 block_size: int, tp_rank: int, dcp_size: int,
                 ready_event: threading.Event, layer_load_finished_events: List[threading.Event], layer_save_finished_events: List[threading.Event]):
        super().__init__(m_store,
                         token_database,
                         block_size,
                         tp_rank,
                         dcp_size,
                         ready_event,
                         name="KVCacheStoreLayerRecvingThread")
        self.layer_load_finished_events = layer_load_finished_events
        self.layer_save_finished_events = layer_save_finished_events
        # from vllm.distributed.parallel_state import get_tp_group
        # self.tp_group = get_tp_group()
        self.kv_cache = torch.tensor([512,128,512]).to(f"npu:{self.tp_rank}")
        self.count = 0
    def add_request(  # type: ignore[override]
            self, req_metas: List[LasyerMultiBlockReqMeta]) -> torch.Tensor:
        self.request_queue.put(req_metas)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            # try:
            request_data = self.request_queue.get()
            if request_data is None:
                logger.warning("Received a None request!")
                self.request_queue.task_done()
                continue
            self._handle_request(request_data)
            # TODO 不删除，让数据传输一直存在
            self.request_queue.put(request_data)

            time.sleep(0.001)

    def _handle_request(  # type: ignore[override]
            self, data: List[LasyerMultiBlockReqMeta]):
        wait_for_save, req_metas, layer_id = data

        if wait_for_save is not None:
            is_finish = self.layer_save_finished_events[wait_for_save].wait(timeout=5)  # try---cache
            if not is_finish:
                logger.info(f"Layerwise {wait_for_save} save failed")
            # if self.tp_rank == 0:
            #     logger.info(f"======================> clear {layer_id} layer_save_finished_events")
            self.layer_save_finished_events[wait_for_save].clear()

        if len(req_metas) == 0:
            # TODO unwait
            # assert not self.layer_load_finished_events[layer_id].is_set()
            # if self.tp_rank == 0:
            #     logger.info(f"======================> set {layer_id} layer_load_finished_events")
            self.layer_load_finished_events[layer_id].set()
            return

        addr_list = []
        size_list = []
        key_list = []
        layer_id = req_metas[0].layer_id
        gvas = []
        # gvas1 = []

        # for req_meta in req_metas:
        # #     # TODO 优化点：没有必要每次都进行计算，可以存储在CPU当中，这样循环太慢了
        #     gvas.extend(req_meta.gvas)
        #     # gvas1.extend(req_meta.gvas1)
        #     addr_list.extend(req_meta.addr_list)
        #     size_list.extend(req_meta.size_list)
        #     key_list.extend(req_meta.keys_str)
        #     for i in range(1):
        #         addr_list.append(req_meta.addr_list[i*2:i*2+2])
        #         size_list.append(req_meta.size_list[i*2:i*2+2])
        #         key_list.append(req_meta.keys_str[i]+f'@TP{self.tp_rank}@count{self.count}')
        # self.count += 1
        #     # for index, key in enumerate(req_meta.keys):
            #     addr, size = self.token_database.prepare_value_layer(
            #         req_meta.starts[index], req_meta.ends[index],
            #         req_meta.block_ids, req_meta.layer_id)
            #
            #     gvas.append(req_meta.key_gva_mapping[key.to_string()])
            #     gvas.append(req_meta.key_gva_mapping[key.to_string()] + self.token_database.block_len[0])
            #     key_list.append(key.to_string())
            #     addr_list.extend(addr)
            #     size_list.extend(size)

        # gvas = gvas[self.tp_rank %
        #                       len(gvas):] + gvas[:self.tp_rank %
        #                                                  len(gvas)]
        # addr_list_c = addr_list[self.tp_rank %
        #                         len(addr_list):] + addr_list[:self.tp_rank %
        #                                                      len(addr_list)]
        # size_list_c = size_list[self.tp_rank %
        #                         len(size_list):] + size_list[:self.tp_rank %
        #                                                      len(size_list)]
        # logger.info(f"=================> num_trasfer = {num_trasfer}")
        # if len(key_list) > 0:
        #     # for i in range(0, len(key_list), self.max_batch):
        #     for i in range(1):
        #         # logger.info(f"===========> key_list {key_list[:5]}, size_list {size_list[:5]}, addr_list {addr_list[:5]}")
        #         self.m_store.put(key_list, addr_list, size_list)
        #         self.m_store.get(key_list, addr_list, size_list)
                # self.m_store.get(key_list[i:i + self.max_batch], addr_list[i:i + self.max_batch],
                #                  size_list[i:i + self.max_batch])
            # 之前在申请的时候，直接申请一个大的block，可以放下所有的block
            # 然后使用kv_caches_base_addr，这个可以分成k和v，以及host需要是可以放下全部kv 的block，先传输整体的k然后再传输整体的v
        # logger.info(f"=================> {len(gvas)}")
        # else:
        # with MODULE_TRANSFER_SEMAPHORE:
        # if self.tp_rank in [0]:
        # if self.tp_rank in [0,2,4,6,8,10,12,14]:
        # res = self.m_store.store.batch_copy(gvas, addr_list_c, size_list_c, 1)
        # 顺序执行：确保 TP ranks 一个接一个读取
        # if self.tp_rank == 0:
        # res = self.m_store.store.batch_copy(gvas, addr_list_c, size_list_c, 0)
        # logger.info(f"=================> req_metas[0].gvas {len(req_metas[0].gvas)}")
        # if self.tp_rank==0:
        res = self.m_store.store.batch_copy(req_metas[0].gvas, req_metas[0].addr_list, req_metas[0].size_list, 1)
        assert res == 0, "recv failed"
        # self.tp_group.broadcast(self.kv_cache, src=0)
                # for i in range(len(gvas)):
        #     res = self.m_store.store.batch_copy([gvas[i]], [addr_list_c[i]], [size_list_c[i]], 1)
        #     time.sleep(0.001)
        # logger.info(f"transfer {len(gvas)}")
        #     logger.info(f" {gvas} addr_list_c {addr_list_c} size_list_c {size_list_c} ")
                # print(f"recv failed {res}")
        # assert res == 0, "recv failed"
        # self.kv_cache.to(f"npu:{self.tp_rank}")
        # self.kv_cache.to(f"cpu")

        # assert not self.layer_load_finished_events[layer_id].is_set(), f"thread: {layer_id} load failed "
        self.layer_load_finished_events[layer_id].set()
        # req_metas.clear()
        self.request_queue.task_done()
