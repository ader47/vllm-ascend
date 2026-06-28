# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the PD-disaggregated SFA connector.

D (``kv_consumer``): on ``update_state_after_alloc`` allocate the main-MLA CPU
blocks (one-shot, full prompt) via :class:`CPUBlockManager`, and advertise the
split ``remote_block_ids`` (indexer NPU + main MLA CPU) + host/port/engine_id to
the Prefill node through the metaserver rendezvous (mirrors mooncake layerwise).

P (``kv_producer``): send setup — Phase 2.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    ReqMeta,
    RequestTracker,
    SFAKVOffloadConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import (
    CPUBlockManager,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

_INDEXER_GROUP_IDX = 0
_MAIN_GROUP_IDX = 1


class SFAPDCpuOffloadScheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
        is_producer: bool,
        is_consumer: bool,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self.is_producer = is_producer
        self.is_consumer = is_consumer
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        self.block_size = [
            group_spec.kv_cache_spec.block_size
            for group_spec in (kv_cache_config.kv_cache_groups if kv_cache_config else [])
        ]
        # main MLA group block size (group 1) — the CPU offload granularity.
        self._main_block_size = self.block_size[_MAIN_GROUP_IDX] if len(self.block_size) > _MAIN_GROUP_IDX else 128

        self.side_channel_host = get_ip()
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
        )

        # CPU block pool for the main MLA group. Sized to hold the remote
        # prefill of all concurrent requests (4x NPU blocks mirrors sfa offload).
        npu_block_num = kv_cache_config.num_blocks if kv_cache_config else 0
        cpu_block_num = npu_block_num * 4
        self.cpu_block_manager = CPUBlockManager(cpu_block_num)

        self._request_trackers: dict[str, RequestTracker] = {}
        self._remote_prefilled: set[str] = set()
        # req_ids awaiting their first build_connector_meta seed (so the worker
        # can build request_map for get_finished even while async-waiting KV).
        self._reqs_need_recv: set[str] = set()
        # req_id -> CPU block ids pending delayed free (see build_connector_meta).
        self._pending_free: dict[str, list[int]] = {}
        self.executor = ThreadPoolExecutor(32)

    # ------------------------------------------------------------------
    # D side (kv_consumer)
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        # Pull the entire prompt KV from the remote P node into D's CPU pool
        # (main MLA) / HBM (indexer). Async relative to engine execution.
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            assert num_computed_tokens % min(self.block_size) == 0
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return

        # vLLM-allocated NPU block ids per group (indexer + main MLA).
        npu_block_ids_by_group = list(blocks.get_block_ids())
        indexer_npu_ids = (
            npu_block_ids_by_group[_INDEXER_GROUP_IDX]
            if len(npu_block_ids_by_group) > _INDEXER_GROUP_IDX
            else []
        )

        # One-shot CPU block allocation for the main MLA group (full prompt).
        prompt_len = len(request.prompt_token_ids)
        num_main_cpu_blocks = cdiv(prompt_len, self._main_block_size)
        main_cpu_ids = (
            self.cpu_block_manager.allocate_block(num_main_cpu_blocks)
            if num_main_cpu_blocks > 0
            else []
        )

        tracker = RequestTracker(
            req_id=request.request_id,
            allocated_block_ids_npu=list(indexer_npu_ids),
            allocated_block_ids_cpu=list(main_cpu_ids),
        )
        self._request_trackers[request.request_id] = tracker
        self._reqs_need_recv.add(request.request_id)
        self._remote_prefilled.add(request.request_id)

        # Advertise the split destination to P via the metaserver rendezvous.
        # remote_block_ids[0] = D indexer NPU blocks, [1] = D main MLA CPU blocks.
        kv_transfer_params = dict(
            request_id=get_external_request_id(request.request_id),
            do_remote_prefill=False,
            do_remote_decode=True,
            remote_block_ids=[list(indexer_npu_ids), list(main_cpu_ids)],
            remote_block_size=self.block_size,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
            remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            remote_cached_tokens=request.num_computed_tokens,
        )
        params["do_remote_prefill"] = False
        metaserver = params.get("metaserver")
        if metaserver is not None and not params.get("do_virtual", False):
            future = self.executor.submit(
                self._access_metaserver, url=metaserver, message=kv_transfer_params
            )
            future.add_done_callback(self._on_metaserver_done)
        logger.info(
            "SFAPDCpuOffload D advertised req %s: indexer NPU=%d blocks, main CPU=%d blocks -> %s",
            request.request_id,
            len(indexer_npu_ids),
            len(main_cpu_ids),
            metaserver,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # Delayed free: release CPU blocks of requests that finished in a prior
        # step and have now fully left the batch. Keeping them one extra step
        # avoids racing with an in-flight LRU H2D load (batch_copy) issued during
        # the request's last decode step.
        for req_id in list(self._pending_free):
            if req_id not in scheduler_output.num_scheduled_tokens:
                self.cpu_block_manager.free(self._pending_free.pop(req_id))

        meta = SFAKVOffloadConnectorMetadata(set(), scheduler_output.preempted_req_ids)

        def _add_req(req_id: str) -> None:
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                return
            meta.add_request(
                ReqMeta(
                    req_id=tracker.req_id,
                    block_ids_npu=tracker.allocated_block_ids_npu,
                    block_ids_cpu=tracker.allocated_block_ids_cpu,
                    num_new_offload_blocks=0,  # D does not save; CPU pool filled by P
                )
            )

        # Seed every newly-allocated remote-prefill request ONCE, even while it
        # async-waits for KV — the worker needs this to build request_map so
        # get_finished can report done_recving (without it the req never gets
        # scheduled -> stuck "reqs num: 0").
        seeded: set[str] = set()
        for req_id in list(self._reqs_need_recv):
            _add_req(req_id)
            seeded.add(req_id)
        self._reqs_need_recv.clear()

        # Also include currently-scheduled requests (feeds the LRU-load
        # cpu_block_table on the composed SFA worker).
        for req_id in list(self._request_trackers):
            if req_id in seeded:
                continue
            if req_id not in scheduler_output.num_scheduled_tokens:
                continue
            _add_req(req_id)
        return meta

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # Do NOT free the CPU blocks here — defer to the next build_connector_meta
        # after the request has left the batch (see delayed free above).
        tracker = self._request_trackers.pop(request.request_id, None)
        self._remote_prefilled.discard(request.request_id)
        if tracker is not None:
            self._pending_free[request.request_id] = tracker.allocated_block_ids_cpu
        return False, None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _access_metaserver(self, url: str, message: dict[str, Any]):
        client = httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None)
        retry = 0
        while retry < 3:
            retry += 1
            try:
                client.post(url, json=message)
                return
            except Exception as e:
                logger.error("Failed to connect to metaserver: %s, retry %s", url, retry)
                if retry == 3:
                    raise e

    @staticmethod
    def _on_metaserver_done(future):
        if future.exception():
            logger.error("Access metaserver fail: %s", future.exception())

    # ------------------------------------------------------------------
    # P side (kv_producer) — Phase 2
    # ------------------------------------------------------------------
    # Producer send setup (get_num_new_matched_tokens -> 0; update_state_after_alloc
    # -> _reqs_need_send_layerwise; build_connector_meta -> send tasks) will mirror
    # MooncakeLayerwiseConnectorScheduler once Phase 2 lands. The D-side paths above
    # are exercised first because the CPU pool + indexer must be ready before P pushes.
