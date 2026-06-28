# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Worker side of the PD-disaggregated SFA connector.

D (``kv_consumer``): composes :class:`SFAKVOffloadWorker` for the unchanged
LRU-resident H2D load path + CPU pool, and adds Mooncake recv advertisement so
the remote P node can RDMA-write indexer KV into HBM and main MLA KV into the
CPU pool.

P (``kv_producer``): layer-wise RDMA push, reusing
:class:`MooncakeLayerwiseConnectorWorker`. The split destination (indexer→D HBM,
main MLA→D CPU) is metadata-driven on the sender side, so the only addition over
stock mooncake is a **per-layer send-completion event** — P's prefill reuses KV
buffers across layers and may only reuse a buffer once its RDMA push has finished
reading it (see plan risk #3 / buffer-reuse gating).
"""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as _mlc
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    LayerMetadata,
    MooncakeAgentMetadata,
    MooncakeLayerwiseConnectorWorker,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (
    SFAKVOffloadWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata

# kv_cache_group convention for DeepSeek-V3.2 sparse offload:
# group 0 = indexer (block_size 512), group 1 = main MLA (block_size 128).
_INDEXER_GROUP_IDX = 0
_MAIN_GROUP_IDX = 1


class SFAPDCpuOffloadWorker:
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
        self.tp_rank = get_tensor_model_parallel_rank()  # TP-local rank for the per-rank ZMQ port
        self.side_channel_host = get_ip()
        # Handshake base port (mirrors mooncake layerwise).
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
        )

        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.kv_recv_layer_thread: KVCacheRecvingLayerThread | None = None
        self.engine = None

        # D-side composed SFA worker (LRU load + CPU pool). Lazily built in
        # register_kv_caches once kv_caches are available.
        self.sfa_worker: SFAKVOffloadWorker | None = None
        # per-req CPU-block count for the solution-1 threshold (Phase 3).
        self._cpu_blocks_by_req: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        # external_req_id -> internal_req_id, so get_finished can map the recv
        # thread's done_recving (keyed by external id from P's DONE signal) back
        # to the vLLM-internal id that the scheduler expects.
        self.request_map: dict[str, str] = {}

        # P-side state (Phase 2).
        self.kv_send_layer_thread = None

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------
    def _ensure_engine(self):
        if self.engine is None:
            # device_name=None reuses the process-wide "ascend" engine; we
            # assume host-pinned registration is accepted (user-confirmed).
            self.engine = global_te.get_transfer_engine(self.side_channel_host, None)
        return self.engine

    # ------------------------------------------------------------------
    # D side (kv_consumer) — this class is only instantiated for consumers;
    # producers use :class:`SFAPDCpuOffloadProducerWorker`.
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register indexer (NPU) + main MLA (CPU pool) with Mooncake.

        The sfa model runner hands a 5-tuple per layer:
        ``(k_nope, v_rope, dsa_k_indexer, topk_buf_k, topk_buf_v)``.
        """
        # PDDBG: dump the real kv_caches / group structure so the per-layer_name
        # LayerMetadata packing (indexer->NPU, main->CPU) can be wired to the
        # actual layer names. Remove once packing is finalized.
        groups_dbg = (
            []
            if self.kv_cache_config is None
            else [
                (i, g.kv_cache_spec.block_size, list(g.layer_names))
                for i, g in enumerate(self.kv_cache_config.kv_cache_groups)
            ]
        )
        shapes_dbg = {
            n: [tuple(t.shape) for t in (v if isinstance(v, (list, tuple)) else [v])]
            for n, v in kv_caches.items()
        }
        logger.info("PDDBG keys=%s groups=%s shapes=%s", list(kv_caches.keys()), groups_dbg, shapes_dbg)

        # --- D side: compose the SFA worker for LRU load + CPU pool ---
        self.sfa_worker = SFAKVOffloadWorker(
            self.vllm_config, self.use_layerwise, self.kv_cache_config
        )
        # SFA worker allocates k_caches_cpu/v_caches_cpu + LRU buffers here.
        self.sfa_worker.register_kv_caches(kv_caches)

        # CPU pool owned by the composed SFA worker (filled remotely by P).
        # IMPORTANT: register the EXACT same memory the LRU load reads — the SFA
        # worker captured `gvas_*_bases = [t.data_ptr() ...]` during its own
        # register_kv_caches, so we must advertise those ptrs verbatim (no copy).
        assert self.sfa_worker.k_caches_cpu is not None, (
            "Composed SFA worker did not allocate the CPU pool"
        )
        k_caches_cpu = self.sfa_worker.k_caches_cpu
        v_caches_cpu = self.sfa_worker.v_caches_cpu

        # Build per-layer LayerMetadata keyed by the REAL layer names, matching
        # how mooncake P consumes them in get_transfer_meta: it looks up
        # remote_layer_metadata[layer_name] and zips P's local tensor list with
        # D's by index k (k<->k, v<->v). So:
        #   indexer layer_name (group 0) -> [dsa_k_ptr]            (1 tensor, matches P's indexer [dsa_k])
        #   main layer_name    (group 1) -> [main_k_cpu, main_v_cpu] (2 tensors, matches P's main [k, v])
        # On D the kv_caches dict has only the main layer_name per transformer
        # layer (a 5-tuple with the indexer embedded at [2]); the indexer
        # layer_name comes from kv_cache_config group 0, paired by transformer
        # index (both lists are in layers.0..N order).
        num_blocks = self.kv_cache_config.num_blocks
        indexer_names = list(
            self.kv_cache_config.kv_cache_groups[_INDEXER_GROUP_IDX].layer_names
        )
        main_names = [
            n for n, v in kv_caches.items()
            if len(v if isinstance(v, (list, tuple)) else [v]) == 5
        ]
        assert len(main_names) == len(indexer_names), (
            f"main layers ({len(main_names)}) != indexer layers ({len(indexer_names)})"
        )

        def _region(t: torch.Tensor) -> tuple[int, int, int]:
            # (data_ptr, per-block bytes, block_size_scale = rows // num_blocks)
            scale = t.shape[0] // num_blocks if num_blocks else 1
            return (
                t.data_ptr(),
                t.element_size() * math.prod(t.shape[1:]),
                scale,
            )

        ptrs: list[int] = []
        lengths: list[int] = []
        for pool_idx, (indexer_name, main_name) in enumerate(zip(indexer_names, main_names)):
            main_tuple = list(kv_caches[main_name])
            indexer_t = main_tuple[2]  # dsa_k_indexer, NPU device memory
            k_cpu = k_caches_cpu[pool_idx]
            v_cpu = v_caches_cpu[pool_idx]

            idx_ptr, idx_len, idx_scale = _region(indexer_t)
            k_ptr, k_len, k_scale = _region(k_cpu)
            v_ptr, v_len, v_scale = _region(v_cpu)

            # group 0 indexer (single dsa_k tensor) under its own layer_name
            self.layer_metadata[indexer_name] = LayerMetadata(
                tensor_group_idx=[_INDEXER_GROUP_IDX],
                kv_caches_base_addr=[idx_ptr],
                block_len=[idx_len],
                block_size_scale=[idx_scale],
            )
            # group 1 main MLA (k, v) under the main layer_name — order matches
            # P's main [k_nope, v_rope] so get_transfer_meta pairs them correctly
            self.layer_metadata[main_name] = LayerMetadata(
                tensor_group_idx=[_MAIN_GROUP_IDX, _MAIN_GROUP_IDX],
                kv_caches_base_addr=[k_ptr, v_ptr],
                block_len=[k_len, v_len],
                block_size_scale=[k_scale, v_scale],
            )

            for t in (indexer_t, k_cpu, v_cpu):
                ptrs.append(t.data_ptr())
                lengths.append(t.numel() * t.element_size())

        # CRITICAL: one register_buffer call — global_te.register_buffer has a
        # process-wide latch (is_register_buffer); a second call is a no-op.
        engine = self._ensure_engine()
        # PDDBG: log each region's ptr + 2 MiB alignment + size so an
        # aclrtPointerGetAttributes/register_memory failure pinpoints the ptr.
        for _p, _s in zip(ptrs, lengths):
            logger.info(
                "PDDBG reg ptr=%#x align2M=%s size=%d", _p, _p % (2 * 1024 * 1024), _s
            )
        global_te.register_buffer(ptrs, lengths)

        metadata = MooncakeAgentMetadata(
            te_rpc_port=engine.get_rpc_port(),
            layer_metadata=self.layer_metadata,
        )
        ready_event = threading.Event()
        self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
            self.tp_rank,
            self.side_channel_port,
            self.vllm_config.parallel_config.tensor_parallel_size,
            1,  # pd_head_ratio == 1 for MLA (see plan risk #4)
            " ",  # local_engine_id placeholder (mirrors mooncake)
            metadata,
            ready_event,
        )
        self.kv_recv_layer_thread.start()
        ready_event.wait()
        logger.info(
            "SFAPDCpuOffload D-side registered: %d indexer(NPU) + %d main(CPU) layers",
            sum(1 for m in self.layer_metadata.values() if _INDEXER_GROUP_IDX in m.tensor_group_idx),
            sum(1 for m in self.layer_metadata.values() if _MAIN_GROUP_IDX in m.tensor_group_idx),
        )

    # -- D-side forwards to the composed SFA worker (LRU load path) --
    def start_load_kv(self, metadata: KVConnectorMetadata):
        assert self.sfa_worker is not None
        # Seed external->internal request id map for get_finished. The scheduler
        # includes remote-prefill requests here (even while async-waiting) so the
        # map exists before P's DONE signal arrives.
        for req in getattr(metadata, "requests", []):
            req_id = getattr(req, "req_id", None)
            if req_id is not None:
                self.request_map[get_external_request_id(req_id)] = req_id
        # Refresh the per-req CPU-block count (Phase 3 source of truth) and
        # forward the unchanged load kickoff to the SFA worker.
        self._refresh_cpu_blocks_by_req(metadata)
        self.sfa_worker.start_load_kv(metadata)

    def set_req_ids(self, req_ids: list):
        if self.sfa_worker is not None:
            self.sfa_worker.set_req_ids(req_ids)

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        capturing: bool = False,
    ) -> bool:
        assert self.sfa_worker is not None
        return self.sfa_worker.prepare_lru_resident_and_load(
            layer_name, num_reqs, topk_indices, current_slots, req_ids, capturing
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        # D side does NOT save locally — the CPU pool is filled remotely by P,
        # and decoded tokens stay on HBM. No-op.
        return

    def wait_for_save(self):
        # Nothing to wait for on D (no local save).
        return

    def get_finished(self) -> tuple[set[str], set[str]]:
        # Report done_recving so vLLM schedules async-remote-prefill requests
        # for decode once P has finished pushing their KV. The recv thread keys
        # done by the external req id (from P's DONE_SENDING); map back to the
        # internal id via request_map.
        done_recving: set[str] = set()
        if self.kv_recv_layer_thread is not None:
            done = self.kv_recv_layer_thread.get_and_clear_done_requests()
            done_recving = {self.request_map[s] for s in done if s in self.request_map}
            failed = self.kv_recv_layer_thread.get_and_clear_failed_requests()
            for s in failed:
                internal = self.request_map.get(s)
                if internal is not None:
                    done_recving.add(internal)  # unblock the req; load will error
        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        """Per-req actual main-MLA CPU-block count for the solution-1 threshold."""
        if self.sfa_worker is None:
            return None
        return {rid: self._cpu_blocks_by_req.get(rid, 0) for rid in req_ids}

    def _refresh_cpu_blocks_by_req(self, metadata: KVConnectorMetadata):
        # SFAKVOffloadConnectorMetadata.requests is a list[ReqMeta]. For this
        # connector ReqMeta.block_ids_cpu IS the flat main-MLA CPU block list
        # (the scheduler stores main CPU ids there), so its length is the
        # per-req CPU-block count used by the solution-1 threshold.
        requests = getattr(metadata, "requests", None)
        if requests is None:
            return
        for req in requests:
            req_id = getattr(req, "req_id", None)
            block_ids_cpu = getattr(req, "block_ids_cpu", None)
            if req_id is None or block_ids_cpu is None:
                continue
            self._cpu_blocks_by_req[req_id] = len(block_ids_cpu)


# ======================================================================
# P side (kv_producer) — layer-wise RDMA push with per-layer completion
# ======================================================================
class _SFAPDCpuSendingThread(KVCacheSendingLayerThread):
    """Sending thread that signals per-layer send completion.

    After the synchronous ``batch_transfer_sync_write`` returns, the source KV
    buffer has been fully read by RDMA and may be reused by P's prefill for a
    later layer. We set ``layer_send_done_events[layer_idx]`` so the buffer-reuse
    scheduler can gate on it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        total_layers: int = kwargs.get("total_layers")  # type: ignore[assignment]
        super().__init__(*args, **kwargs)
        self.layer_send_done_events: list[threading.Event] = [
            threading.Event() for _ in range(total_layers)
        ]

    def _transfer_kv_cache(self, send_task: Any) -> None:  # type: ignore[override]
        super()._transfer_kv_cache(send_task)
        # Mark this layer's send complete regardless of per-req failure so a
        # waiter never blocks forever; failures are surfaced via failed_reqs.
        idx = send_task.layer_idx
        if 0 <= idx < len(self.layer_send_done_events):
            self.layer_send_done_events[idx].set()


class SFAPDCpuOffloadProducerWorker(MooncakeLayerwiseConnectorWorker):
    """P-side worker = stock mooncake layerwise push + per-layer send-done gate.

    The split destination (indexer→D HBM, main MLA→D CPU) needs NO sender-side
    branching: ``get_transfer_meta`` derives ``dst`` from the decoder-advertised
    ``remote_layer_metadata[...]`` base addrs, which the D side populated with the
    indexer NPU ptr and the main-MLA CPU-pool ptr respectively.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.layer_send_done_events: list[threading.Event] | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # The base register_kv_caches constructs ``KVCacheSendingLayerThread`` by
        # name from this module's globals. Temporarily swap that global so it
        # builds our subclass (with per-layer completion events), then restore.
        # This avoids duplicating ~140 lines of registration logic.
        # TODO(refactor): give MooncakeLayerwiseConnectorWorker a thread-class
        # hook so this monkeypatch is unnecessary.
        orig = _mlc.KVCacheSendingLayerThread
        _mlc.KVCacheSendingLayerThread = _SFAPDCpuSendingThread  # type: ignore[assignment]
        try:
            super().register_kv_caches(kv_caches)
        finally:
            _mlc.KVCacheSendingLayerThread = orig  # type: ignore[assignment]
        assert isinstance(self.kv_send_layer_thread, _SFAPDCpuSendingThread)
        self.layer_send_done_events = self.kv_send_layer_thread.layer_send_done_events

    def wait_for_layer_send(self, layer_idx: int) -> None:
        """Block until layer ``layer_idx``'s RDMA push has read its source buffer.

        Call this from P's prefill before reusing a KV buffer that layer
        ``layer_idx`` wrote, so the push finishes reading before overwrite.
        """
        if self.layer_send_done_events is None:
            return
        if 0 <= layer_idx < len(self.layer_send_done_events):
            self.layer_send_done_events[layer_idx].wait()

    def get_layer_send_event(self, layer_idx: int) -> threading.Event | None:
        if self.layer_send_done_events is None:
            return None
        if 0 <= layer_idx < len(self.layer_send_done_events):
            return self.layer_send_done_events[layer_idx]
        return None
