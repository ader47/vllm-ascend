# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the PD-disaggregated SFA connector.

D (``kv_consumer``): on ``update_state_after_alloc`` allocate indexer NPU
blocks + main-MLA CPU blocks (one-shot, full prompt), store them in
RequestTracker, and send a metaserver rendezvous notification to P carrying
only contact info + ``do_remote_decode`` (NO block ids — D keeps its blocks and
looks them up by req_id when P's READ_READY arrives).

P (``kv_producer``): send setup handled by the mooncake layerwise base.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
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
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
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
    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
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
            npu_block_ids_by_group[_INDEXER_GROUP_IDX] if len(npu_block_ids_by_group) > _INDEXER_GROUP_IDX else []
        )
        main_hbm_ids = npu_block_ids_by_group[_MAIN_GROUP_IDX] if len(npu_block_ids_by_group) > _MAIN_GROUP_IDX else []

        # Part A: the CPU pool stores only FULL main MLA blocks (floor division).
        # The optional partial last block stays in HBM — D's logical-last group1
        # block — so decode can append to it; it is offloaded to CPU once full
        # (decode offload path). num_offloaded_blocks == len(main_cpu_ids) ==
        # num_full, so the threshold auto-excludes the partial (decode reads it
        # from HBM, not the stale CPU copy).
        prompt_len = len(request.prompt_token_ids)
        num_main_cpu_blocks = prompt_len // self._main_block_size
        has_partial = (prompt_len % self._main_block_size) != 0
        main_cpu_ids = self.cpu_block_manager.allocate_block(num_main_cpu_blocks) if num_main_cpu_blocks > 0 else []
        partial_hbm_bid = main_hbm_ids[-1] if (has_partial and main_hbm_ids) else None

        tracker = RequestTracker(
            req_id=request.request_id,
            allocated_block_ids_npu=list(indexer_npu_ids),
            allocated_block_ids_cpu=list(main_cpu_ids),
            num_full=num_main_cpu_blocks,
            partial_hbm_bid=partial_hbm_bid,
            main_hbm_ids=list(main_hbm_ids),
        )
        self._request_trackers[request.request_id] = tracker
        self._reqs_need_recv.add(request.request_id)
        self._remote_prefilled.add(request.request_id)

        # Notify P via the metaserver rendezvous that D is ready to pull this
        # request. D does NOT send its block ids to P — D keeps them (stored in
        # RequestTracker, passed to the D worker via connector_meta) and looks
        # them up by req_id when P's READ_READY arrives. Only contact info +
        # the do_remote_decode "go" flag go to P. (Sending block ids to P was a
        # push-model leftover; in pull mode P only needs P's own source blocks.)
        kv_transfer_params = dict(
            request_id=get_external_request_id(request.request_id),
            do_remote_prefill=False,
            do_remote_decode=True,
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
            future = self.executor.submit(self._access_metaserver, url=metaserver, message=kv_transfer_params)
            future.add_done_callback(self._on_metaserver_done)
        logger.info(
            "SFAPDCpuOffload D advertised req %s: indexer NPU=%d, main CPU(full)=%d, partial_hbm=%s -> %s",
            request.request_id,
            len(indexer_npu_ids),
            len(main_cpu_ids),
            partial_hbm_bid,
            metaserver,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        # Delayed free: release CPU blocks of requests that finished in a prior
        # step and have now fully left the batch. Keeping them one extra step
        # avoids racing with an in-flight LRU H2D load (batch_copy) issued during
        # the request's last decode step.
        for req_id in list(self._pending_free):
            if req_id not in scheduler_output.num_scheduled_tokens:
                self.cpu_block_manager.free(self._pending_free.pop(req_id))

        meta = SFAKVOffloadConnectorMetadata(set(), scheduler_output.preempted_req_ids)

        # B1: maps from scheduled_cached_reqs for decode offload computation.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        num_computed_by_req: dict[str, int] = dict(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
        new_main_hbm_by_req: dict[str, list[int]] = {}
        for i, rid in enumerate(cached_reqs.req_ids):
            nbi = cached_reqs.new_block_ids[i]
            if nbi is None:
                nbi = []
            elif isinstance(nbi, tuple):
                # multi-group: tuple of per-group lists; last = main MLA (group1)
                nbi = nbi[-1] if len(nbi) > 0 else []
            new_main_hbm_by_req[rid] = list(nbi)

        def _add_req(
            req_id: str,
            offload_src: list[int] | None = None,
            offload_dst: list[int] | None = None,
        ) -> None:
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                return
            meta.add_request(
                ReqMeta(
                    req_id=tracker.req_id,
                    block_ids_npu=tracker.main_hbm_ids,
                    block_ids_cpu=tracker.allocated_block_ids_cpu,
                    block_ids_indexer=tracker.allocated_block_ids_npu,
                    num_new_offload_blocks=len(offload_src) if offload_src else 0,
                    num_full=tracker.num_full,
                    partial_hbm_bid=tracker.partial_hbm_bid,
                    offload_src_hbm_ids=offload_src or [],
                    offload_dst_cpu_ids=offload_dst or [],
                )
            )

        # Seed every newly-allocated remote-prefill request ONCE (prefill: no
        # decode offload). The worker needs this to build request_map so
        # get_finished can report done_recving.
        seeded: set[str] = set()
        for req_id in list(self._reqs_need_recv):
            _add_req(req_id)
            seeded.add(req_id)
        self._reqs_need_recv.clear()

        # Decode (cached) requests: extend the main MLA HBM block table, then
        # offload any blocks that newly filled this step HBM->CPU. Part A put the
        # prompt's full blocks in CPU (num_offloaded starts at num_full) and the
        # partial in HBM; as decode fills the partial (and later blocks), they
        # enter [num_offloaded:num_blocks_after_step] and get offloaded here.
        for req_id in list(self._request_trackers):
            if req_id in seeded:
                continue
            if req_id not in scheduler_output.num_scheduled_tokens:
                continue
            tracker = self._request_trackers[req_id]
            tracker.main_hbm_ids.extend(new_main_hbm_by_req.get(req_id, []))
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_computed = num_computed_by_req.get(req_id, 0)
            num_blocks_after_step = (num_computed + num_new_tokens) // self._main_block_size
            num_offloaded = len(tracker.allocated_block_ids_cpu)
            end = min(num_blocks_after_step, len(tracker.main_hbm_ids))
            offload_src = tracker.main_hbm_ids[num_offloaded:end] if end > num_offloaded else []
            offload_dst = self.cpu_block_manager.allocate_block(len(offload_src)) if offload_src else []
            if offload_src:
                tracker.allocated_block_ids_cpu.extend(offload_dst)
                logger.info(
                    "SFAPD B1 offload req %s: %d blocks HBM->CPU (num_offloaded %d->%d)",
                    req_id,
                    len(offload_src),
                    num_offloaded,
                    num_offloaded + len(offload_src),
                )
            _add_req(req_id, offload_src, offload_dst)
        return meta

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
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
