# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the PD-disaggregated SFA connector.

D (``kv_consumer``): on ``update_state_after_alloc`` allocate indexer NPU
blocks + main-MLA CPU blocks (one-shot, full prompt), store them in
RequestTracker, and send a metaserver rendezvous notification to P carrying
only contact info + ``do_remote_decode`` (NO block ids — D keeps its blocks and
looks them up by req_id when P's READ_READY arrives).

P (``kv_producer``): build metadata for layer-wise READ_READY notifications.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    ReqMeta,
    RequestTracker,
    SFAKVOffloadConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import (
    CPUBlockManager,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    SfaPDProducerMetadata,
    get_external_request_id,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

_INDEXER_GROUP_IDX = 0
_MAIN_GROUP_IDX = 1


class _SendReqInfo:
    def __init__(
        self,
        local_block_ids: list[list[int]],
        local_transferred_tokens: int,
        local_computed_tokens: int,
        request: Request,
    ) -> None:
        self.local_block_ids = local_block_ids
        self.local_transferred_tokens = local_transferred_tokens
        self.local_computed_tokens = local_computed_tokens
        self.request = request

    def extend_local_block_ids(self, new_block_ids: list[list[int]]) -> None:
        for i, new_block_id in enumerate(new_block_ids):
            self.local_block_ids[i].extend(new_block_id)

    def update_computed_tokens(self, computed_tokens: int) -> None:
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:
        self.local_transferred_tokens = transferred_tokens


class SFAPDProducerScheduler:
    """P-side scheduler for SFA PD (pull mode).

    D's metaserver rendezvous carries ``do_remote_decode=True`` plus D's ZMQ
    endpoint. P tracks its own local block ids and emits per-step metadata for
    the pull-mode sending thread; D looks up its destination blocks by req_id.
    """

    requires_full_blocks_on_update_after_alloc = True

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self._reqs_need_send_layerwise: dict[str, _SendReqInfo] = {}

    @staticmethod
    def _normalize_block_ids(block_ids: Any) -> list[list[int]]:
        if block_ids is None:
            return []
        if block_ids and isinstance(block_ids[0], int):
            return [list(block_ids)]
        if isinstance(block_ids, tuple):
            return [list(group) for group in block_ids]
        return [list(group) for group in block_ids]

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_decode"):
            return

        local_block_ids = self._normalize_block_ids(blocks.get_block_ids())
        remote_cache_tokens = params["remote_cached_tokens"]
        send_req_info = _SendReqInfo(
            local_block_ids=local_block_ids,
            local_transferred_tokens=remote_cache_tokens,
            local_computed_tokens=0,
            request=request,
        )
        self._reqs_need_send_layerwise[request.request_id] = send_req_info

        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "SFAPD P register remote-decode req %s: local_block_ids=%s, "
                "remote_host=%s, remote_port=%s, remote_tp_size=%s, "
                "remote_cached_tokens=%s",
                request.request_id,
                local_block_ids,
                params.get("remote_host"),
                params.get("remote_port"),
                params.get("remote_tp_size"),
                remote_cache_tokens,
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = SfaPDProducerMetadata()
        cached_reqs = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
            if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                normalized = self._normalize_block_ids(new_blocks)
                self._reqs_need_send_layerwise[req_id].extend_local_block_ids(normalized)
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "SFAPD P extend remote-decode req %s: new_blocks=%s",
                        req_id,
                        normalized,
                    )

        computed_tokens = dict(
            list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
            + [(req.req_id, req.num_computed_tokens) for req in new_reqs]
        )
        min_block_size = min(self.block_size)
        for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            send_req_info = self._reqs_need_send_layerwise.get(req_id)
            if send_req_info is None:
                continue

            send_req_info.update_transferred_tokens(round_down(send_req_info.local_computed_tokens, min_block_size))
            spec_decode_tokens = (
                len(scheduled_spec_decode_tokens[req_id]) if req_id in scheduled_spec_decode_tokens else 0
            )
            send_req_info.update_computed_tokens(computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens)
            request = send_req_info.request
            assert request.kv_transfer_params is not None
            chunk_finish = send_req_info.local_computed_tokens >= len(request.all_token_ids)
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=send_req_info.local_block_ids,
                kv_transfer_params=request.kv_transfer_params,
                token_ids=[],
                chunk_finish=chunk_finish,
                remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens"),
                prompt_len=len(request.all_token_ids),
                local_computed_tokens=send_req_info.local_computed_tokens,
                local_transed_tokens=send_req_info.local_transferred_tokens,
            )
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "SFAPD P add transfer task req %s: local_block_ids=%s, "
                    "local_transed_tokens=%s, local_computed_tokens=%s, "
                    "remote_cache_tokens=%s, prompt_len=%s, chunk_finish=%s, "
                    "remote_host=%s, remote_port=%s",
                    req_id,
                    send_req_info.local_block_ids,
                    send_req_info.local_transferred_tokens,
                    send_req_info.local_computed_tokens,
                    request.kv_transfer_params.get("remote_cached_tokens"),
                    len(request.all_token_ids),
                    chunk_finish,
                    request.kv_transfer_params.get("remote_host"),
                    request.kv_transfer_params.get("remote_port"),
                )
            if chunk_finish:
                self._reqs_need_send_layerwise.pop(req_id)
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class SFAPDCpuOffloadScheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        self.block_size = [
            group_spec.kv_cache_spec.block_size
            for group_spec in (kv_cache_config.kv_cache_groups if kv_cache_config else [])
        ]
        # main MLA group block size (group 1) — the CPU offload granularity.
        # The manager uses its self.block_size to size the null-padded prefix
        # and the SFA kernel uses _main_block_size (via cpu_blocks_map) as the
        # num_offloaded_blocks mask threshold; divergence silently over/under-
        # masks resident KV. Both DERIVE from this group's spec, so they match
        # WHEN DCP*PCP == 1 (the current PD config). Caveat: vLLM core scales
        # SingleTypeKVCacheManager.self.block_size by dcp_world_size*
        # pcp_world_size when either > 1, while _main_block_size here is the RAW
        # spec value — so under DCP/PCP > 1 the two diverge and this connector's
        # null-pad/mask coupling is NOT supported without extra work. Assert the
        # group exists rather than silently falling back to 128.
        assert len(self.block_size) > _MAIN_GROUP_IDX, (
            f"PD offload expects a main-MLA group at index {_MAIN_GROUP_IDX}; got groups={self.block_size}"
        )
        self._main_block_size = self.block_size[_MAIN_GROUP_IDX]

        # Hard-fail the unsupported DCP/PCP>1 config instead of silently
        # mis-masking: under DCP*PCP>1 vLLM core scales the manager's
        # self.block_size (used for the null-pad width) while _main_block_size
        # here stays raw (used for the attention num_offloaded_blocks mask
        # threshold). Divergence -> null-pad width != mask width -> resident KV
        # read from wrong slots (silent corruption). PD with DCP/PCP>1 is not
        # supported by this connector. (Asserted here, on the D-side consumer
        # scheduler, because null-pad only activates for remote-prefilled reqs.)
        pcp = vllm_config.parallel_config.prefill_context_parallel_size
        dcp = vllm_config.parallel_config.decode_context_parallel_size
        assert pcp * dcp == 1, (
            f"SFAPDCpuOffloadConnector null-pad/mask coupling requires "
            f"DCP*PCP == 1 (got pcp={pcp}, dcp={dcp}); the manager block_size "
            f"is scaled by dcp*pcp while _main_block_size stays raw, so the "
            f"null-pad width and the attention mask threshold diverge. PD with "
            f"DCP/PCP>1 is not supported."
        )

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
        # req_ids awaiting their first build_connector_meta seed (so the worker
        # can build request_map for get_finished even while async-waiting KV).
        self._reqs_need_recv: set[str] = set()
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
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "SFAPDCpuOffload D advertised req %s: indexer_npu_ids=%s, "
                "main_cpu_ids=%s, main_hbm_ids=%s, num_full=%s, "
                "partial_hbm=%s, remote_host=%s, remote_port=%s, metaserver=%s",
                request.request_id,
                indexer_npu_ids,
                main_cpu_ids,
                main_hbm_ids,
                num_main_cpu_blocks,
                partial_hbm_bid,
                self.side_channel_host,
                self.side_channel_port,
                metaserver,
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:

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
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "SFAPDCpuOffload D build meta req %s: main_hbm_ids=%s, "
                    "main_cpu_ids=%s, indexer_npu_ids=%s, num_full=%s, "
                    "partial_hbm=%s, offload_src=%s, offload_dst=%s",
                    req_id,
                    tracker.main_hbm_ids,
                    tracker.allocated_block_ids_cpu,
                    tracker.allocated_block_ids_npu,
                    tracker.num_full,
                    tracker.partial_hbm_bid,
                    offload_src or [],
                    offload_dst or [],
                )
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
            # Use finalized (committed) tokens: exclude spec-decode draft
            # tokens, which may be rejected, so we don't offload blocks that
            # still hold uncommitted KV. Matches SFAKVOffloadlScheduler.
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_new_tokens -= len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            num_new_tokens = max(num_new_tokens, 0)
            num_computed = num_computed_by_req.get(req_id, 0)
            num_blocks_after_step = (num_computed + num_new_tokens) // self._main_block_size
            num_offloaded = len(tracker.allocated_block_ids_cpu)
            end = min(num_blocks_after_step, len(tracker.main_hbm_ids))
            offload_src = tracker.main_hbm_ids[num_offloaded:end] if end > num_offloaded else []
            offload_dst = self.cpu_block_manager.allocate_block(len(offload_src)) if offload_src else []
            if envs.VLLM_ASCEND_SFA_DEBUG:
                # Show the slice arithmetic so a hardware run can confirm the
                # offload range skips the null-padded prefix ([0:N]) and catches
                # real decode blocks ([N:end]). main_hbm_ids includes the null
                # prefix (get_block_ids does not filter nulls), so num_offloaded
                # (=N from Part A) correctly offsets past it.
                logger.info(
                    "SFAPD B1 offload slice req %s: num_blocks_after_step=%d, "
                    "len(main_hbm_ids)=%d, slice=[%d:%d] -> %d blocks (null_prefix=%d)",
                    req_id,
                    num_blocks_after_step,
                    len(tracker.main_hbm_ids),
                    num_offloaded,
                    end,
                    len(offload_src),
                    tracker.num_full,
                )
            if offload_src:
                tracker.allocated_block_ids_cpu.extend(offload_dst)
                if envs.VLLM_ASCEND_SFA_DEBUG:
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
        # No need to delay free here: when we reach request_finished in D node here,
        # the request is completely done, no more inference, no more kv transfer.
        # Free cpu blocks immediately to make room for next step scheduling.
        tracker = self._request_trackers.pop(request.request_id, None)
        if tracker is not None:
            self.cpu_block_manager.free(tracker.allocated_block_ids_cpu)
        return False, None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _access_metaserver(self, url: str, message: dict[str, Any]):
        with httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None) as client:
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
