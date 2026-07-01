# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""PD-disaggregated SFA KV-transfer connector.

On the Decode node (``kv_consumer``) the bulk MLA KV is pushed by the remote
Prefill node (``kv_producer``) directly into a CPU pinned offload pool; the
indexer KV lands in HBM. The D-side load path (LRU-resident H2D) is reused
from :class:`SFAKVOffloadWorker`.

See ``/Users/liufeng/.claude/plans/luminous-shimmying-wind.md`` for the design.
"""

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (
    SFAPDCpuOffloadScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (
    SFAPDCpuOffloadProducerWorker,
    SFAPDCpuOffloadWorker,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request


class SFAPDProducerScheduler(MooncakeLayerwiseConnectorScheduler):
    """P-side scheduler for SFA PD (pull mode).

    No override is needed: D's metaserver rendezvous carries only
    ``do_remote_decode=True`` + ZMQ contact info (NO block ids — D keeps its own
    blocks and looks them up by req_id). The mooncake base class's
    ``do_remote_decode`` branch populates ``_reqs_need_send_layerwise`` with P's
    own allocated blocks, which is exactly what ``_transfer_kv_cache`` reads to
    build READ_READY. (An earlier override populated ``_reqs_need_recv`` from
    D's ``remote_block_ids`` — a push-model leftover, now dead.)
    """

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        # Pull mode: D's rendezvous sends only do_remote_decode=True + contact
        # info (no block ids). The mooncake base do_remote_decode branch
        # populates _reqs_need_send_layerwise with P's own blocks, which is what
        # _transfer_kv_cache reads to build READ_READY. No P-side override of
        # _reqs_need_recv is needed (that was a push-model leftover, now dead).
        super().update_state_after_alloc(request, blocks, num_external_tokens)


class SFAPDCpuOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """One connector class branching on ``role`` and ``kv_role``.

    * SCHEDULER + producer : P-side send setup (reuses mooncake layerwise logic).
    * SCHEDULER + consumer : D-side CPU-block allocation / advertisement tracking.
    * WORKER + producer    : P-side layer-wise RDMA push.
    * WORKER + consumer    : D-side : composes ``SFAKVOffloadWorker`` (LRU load +
      CPU pool) + mooncake recv advertisement + indexer/main split registration.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_consumer = vllm_config.kv_transfer_config.is_kv_consumer
        # SFA path is layer-wise on both sides.
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", True)
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        # Guard the asymmetric use_offload assumption (the launch scripts must
        # set it via --additional-config). Fail fast at startup rather than
        # producing confusing mid-run failures.
        #   P (producer)  : use_offload=false — inherited mooncake register
        #                    expects standard paged KV, not the offload 5-tuple.
        #   D (consumer)  : use_offload=true  — drives the SFA offload code path
        #                    (5-tuple kv_cache, num_offloaded_blocks, LRU load).
        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        # AscendConfig may not be initialized yet at connector construction
        # time; init_ascend_config is idempotent (no-op if already done).
        init_ascend_config(vllm_config)
        ascend_use_offload = get_ascend_config().use_offload
        if self.is_producer:
            assert not ascend_use_offload, (
                "SFAPDCpuOffloadConnector producer (P) must run with "
                "use_offload=false (set --additional-config "
                "'{\"use_offload\": false}')."
            )
        if self.is_consumer:
            assert ascend_use_offload, (
                "SFAPDCpuOffloadConnector consumer (D) must run with "
                "use_offload=true (set --additional-config "
                "'{\"use_offload\": true, ...}')."
            )

        if role == KVConnectorRole.SCHEDULER:
            # Producer scheduler reuses mooncake send-setup; consumer scheduler
            # is the D-side CPU-block allocator/advertiser.
            if self.is_producer:
                self.connector_scheduler = SFAPDProducerScheduler(vllm_config, kv_cache_config, str(self.engine_id))
            else:
                self.connector_scheduler = SFAPDCpuOffloadScheduler(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                    self.is_producer,
                    self.is_consumer,
                )
            self.connector_worker = None
        else:
            self.connector_scheduler = None
            if self.is_producer:
                self.connector_worker = SFAPDCpuOffloadProducerWorker(vllm_config, kv_cache_config, str(self.engine_id))
            else:
                self.connector_worker = SFAPDCpuOffloadWorker(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                    self.is_producer,
                    self.is_consumer,
                )

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        # SFA loads inside ``prepare_lru_resident_and_load`` (D) / nothing to wait
        # for on P (push is fire-and-forget per layer).
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        # NOTE: signature diverges by role — mooncake (P) needs the metadata
        # tuple, the composed SFA worker (D) takes none. The worker branches
        # internally; we forward the bound args.
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._get_connector_metadata())

    def wait_for_save(self):
        # No-op (matches mooncake layerwise): P pushes per-layer with completion
        # signalled via the per-layer send events / the DONE callback, not here;
        # D does not save locally (its CPU pool is filled remotely by P). Do NOT
        # forward to the worker — the producer worker (mooncake subclass) has no
        # wait_for_save method.
        return

    # ------------------------------------------------------------------
    # SFA duck-typed hooks (attention/utils.py) — D side only
    # ------------------------------------------------------------------
    def set_req_ids(self, req_ids: list):
        if self.connector_worker is not None:
            self.connector_worker.set_req_ids(req_ids)

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        capturing: bool = False,
    ) -> bool:
        assert self.connector_worker is not None
        return self.connector_worker.prepare_lru_resident_and_load(
            layer_name,
            num_reqs,
            topk_indices,
            current_slots,
            req_ids,
            capturing,
        )

    # Phase 3: real per-req CPU-block count for the solution-1 threshold.
    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_num_cpu_blocks(req_ids)

    # P-side buffer-reuse gate: block until a layer's RDMA push has finished
    # reading its source KV buffer, so the buffer may be reused by a later layer.
    def wait_for_layer_send(self, layer_idx: int) -> None:
        worker = self.connector_worker
        if worker is None or not hasattr(worker, "wait_for_layer_send"):
            return
        worker.wait_for_layer_send(layer_idx)
