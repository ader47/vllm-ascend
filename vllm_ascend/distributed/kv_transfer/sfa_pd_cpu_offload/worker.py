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
import os
import re
import threading
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as _mlc
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    DONE_SENDING_MSG,
    FAILED_SENDING_MSG,
    GET_META_MSG,
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    LayerMetadata,
    MooncakeAgentMetadata,
    MooncakeLayerwiseConnectorWorker,
    ensure_zmq_recv,
    ensure_zmq_send,
    get_external_request_id,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (
    SFAKVOffloadWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.transfer_engine_backend import (
    BACKEND_MEMFABRIC,
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata

# kv_cache_group convention for DeepSeek-V3.2 sparse offload:
# group 0 = indexer (block_size 512), group 1 = main MLA (block_size 128).
_INDEXER_GROUP_IDX = 0
_MAIN_GROUP_IDX = 1
# Matches the transformer-layer index in a kv-cache layer name, e.g.
# "model.layers.5.self_attn" / "model.layers.5.self_attn.indexer" -> 5. Prefer
# this over extract_layer_index(), which asserts the name holds exactly one
# integer and would raise on names carrying an extra index/shard suffix.
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def _layer_idx(layer_name: str) -> int:
    match = _LAYER_IDX_RE.search(layer_name)
    assert match is not None, f"no transformer layer index in layer name {layer_name!r}"
    return int(match.group(1))


STAGING_DONE = b"staging_done"
STAGING_ACK = b"staging_ack"
MF_META = b"mf_meta"        # P→D: (MF_META, p_session, p_layer_meta_serialized)
READ_READY = b"read_ready"   # P→D: (READ_READY, layer_idx, layer_name, p_block_ids_k, p_block_ids_v)
READ_DONE = b"read_done"     # D→P: (READ_DONE, layer_idx)


class SFARecvLayerThread(KVCacheRecvingLayerThread):
    """Extends base recv thread to also handle STAGING_DONE per-layer.

    On STAGING_DONE(layer_idx): invoke the worker's copy callback, then reply
    STAGING_ACK so P can proceed to push the next main MLA layer.
    """

    def __init__(self, *args, on_staging_done=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_staging_done = on_staging_done

    def run(self):
        """Override base run to intercept STAGING_DONE in the ZMQ loop."""
        from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("SFA PD recv (staging-aware) listening on: %s", path)
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        ctx = zmq.Context()
        try:
            sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.ROUTER, bind=True)
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        continue
                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        continue
                    msg = decoder.decode(payload[0])
                    if msg[0] == STAGING_DONE:
                        layer_idx = msg[1]
                        if self._on_staging_done is not None:
                            self._on_staging_done(layer_idx)
                        sock.send_multipart((identity, b"", encoder.encode((STAGING_ACK, layer_idx))))
                    elif msg[0] == GET_META_MSG:
                        sock.send_multipart((identity, b"", encoded_data))
                    elif msg[0] == DONE_SENDING_MSG:
                        request_id = msg[1]
                        trans_count = msg[2]
                        side_channel_path = msg[3]
                        self.update_done_task(request_id, trans_count, side_channel_path)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == FAILED_SENDING_MSG:
                        request_id = msg[1]
                        self.update_failed_task(request_id)
                        sock.send_multipart((identity, b"", b"ACK"))
                    else:
                        logger.error("SFA recv got unexpected message %s", msg)
                except Exception as e:
                    logger.error("SFA recv exception: %s: %s", type(e), e)
        finally:
            ctx.destroy(linger=0)


def _resolve_kv_transfer_backend(vllm_config: VllmConfig) -> str:
    """Pick the KV transfer backend.

    ``kv_connector_extra_config["transfer_backend"]`` overrides the
    ``VLLM_ASCEND_KV_TRANSFER_BACKEND`` env var (default ``mooncake``).
    """
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    return extra.get("transfer_backend") or envs.VLLM_ASCEND_KV_TRANSFER_BACKEND


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
        # External req ids whose DONE/FAILED signal arrived before request_map
        # was seeded (see get_finished). Retried every step until mapped.
        self._pending_done: set[str] = set()
        self._pending_failed: set[str] = set()

        # P-side state (Phase 2).
        self.kv_send_layer_thread = None

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------
    def _ensure_engine(self):
        if self.engine is None:
            # device_name=None reuses the process-wide "ascend" engine; we
            # assume host-pinned registration is accepted (user-confirmed).
            backend = _resolve_kv_transfer_backend(self.vllm_config)
            if backend == BACKEND_MEMFABRIC:
                # D-side unique_id = "<host>:<port>"; memfabric derives its
                # config-store address from it, and the Prefill peer reuses the
                # same "<host>:<port>" (advertised via te_rpc_port) as dest_session.
                # Offset past the ZMQ recv ports (side_channel_port + tp_rank,
                # occupied by KVCacheRecvingLayerThread) to avoid a bind collision.
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                mf_session_port = self.side_channel_port + tp_size + self.tp_rank
                global_te.configure(
                    backend=BACKEND_MEMFABRIC,
                    role=MEMFABRIC_ROLE_DECODE,
                    unique_id=f"{self.side_channel_host}:{mf_session_port}",
                    device_id=torch.npu.current_device(),
                )
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
        # --- D side: compose the SFA worker for LRU load + CPU pool ---
        self.sfa_worker = SFAKVOffloadWorker(
            self.vllm_config, self.use_layerwise, self.kv_cache_config
        )
        # SFA worker allocates k_caches_cpu/v_caches_cpu + LRU buffers here.
        self.sfa_worker.register_kv_caches(kv_caches)

        # CPU pool owned by the composed SFA worker (filled via staging copy).
        # HW limitation: P HBM -> D DRAM cross-node RDMA is not available.
        # Solution: P pushes main MLA to a 1-slot HBM staging buffer, then
        # signals per-layer DONE; D copies staging -> CPU pool locally and ACKs.
        assert self.sfa_worker.k_caches_cpu is not None, (
            "Composed SFA worker did not allocate the CPU pool"
        )
        k_caches_cpu = self.sfa_worker.k_caches_cpu
        v_caches_cpu = self.sfa_worker.v_caches_cpu

        backend = _resolve_kv_transfer_backend(self.vllm_config)
        if backend == BACKEND_MEMFABRIC:
            return self._register_memfabric_pull(
                kv_caches, k_caches_cpu, v_caches_cpu
            )

        # --- mooncake staging path (existing) ---
        # 1-slot HBM staging for main MLA (reused per-layer via ACK handshake).
        # indexer goes direct (HBM->HBM), no staging.
        self.staging_k = torch.empty_like(k_caches_cpu[0], device="npu")
        self.staging_v = torch.empty_like(v_caches_cpu[0], device="npu")
        # Store CPU pool refs for the per-layer copy callback.
        self._cpu_pools = list(zip(k_caches_cpu, v_caches_cpu))

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

        # Pair each indexer layer with its main layer by transformer-layer index.
        # Do NOT rely on kv_caches.items() insertion order: indexer_names comes
        # from kv_cache_config while main_names comes from the model runner's
        # kv_caches dict — zip()-by-position would silently mispair (and write KV
        # to the wrong base addr) the moment the two orderings diverge.
        main_by_layer_idx = {_layer_idx(name): name for name in main_names}
        main_names = [main_by_layer_idx[_layer_idx(name)] for name in indexer_names]

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
        _staging_registered = False
        for pool_idx, (indexer_name, main_name) in enumerate(zip(indexer_names, main_names)):
            main_tuple = list(kv_caches[main_name])
            indexer_t = main_tuple[2]  # dsa_k_indexer, NPU device memory

            idx_ptr, idx_len, idx_scale = _region(indexer_t)
            sk_ptr, sk_len, sk_scale = _region(self.staging_k)
            sv_ptr, sv_len, sv_scale = _region(self.staging_v)

            # group 0 indexer (single dsa_k tensor) under its own layer_name —
            # direct HBM, P pushes straight to D indexer.
            self.layer_metadata[indexer_name] = LayerMetadata(
                tensor_group_idx=[_INDEXER_GROUP_IDX],
                kv_caches_base_addr=[idx_ptr],
                block_len=[idx_len],
                block_size_scale=[idx_scale],
            )
            # group 1 main MLA (k, v) — P pushes to HBM staging (not CPU pool
            # DRAM, which cross-node RDMA can't reach). D copies staging->CPU
            # pool locally after each layer's STAGING_DONE handshake.
            self.layer_metadata[main_name] = LayerMetadata(
                tensor_group_idx=[_MAIN_GROUP_IDX, _MAIN_GROUP_IDX],
                kv_caches_base_addr=[sk_ptr, sv_ptr],
                block_len=[sk_len, sv_len],
                block_size_scale=[sk_scale, sv_scale],
            )

            # Register indexer per-layer (unique HBM tensor per layer).
            # Register staging ONCE (1-slot, shared across all main layers).
            ptrs.append(indexer_t.data_ptr())
            lengths.append(indexer_t.numel() * indexer_t.element_size())
            if not _staging_registered:
                ptrs.append(self.staging_k.data_ptr())
                lengths.append(self.staging_k.numel() * self.staging_k.element_size())
                ptrs.append(self.staging_v.data_ptr())
                lengths.append(self.staging_v.numel() * self.staging_v.element_size())
                _staging_registered = True

        # CRITICAL: one register_buffer call — global_te.register_buffer has a
        # process-wide latch (is_register_buffer); a second call is a no-op.
        self._ensure_engine()
        global_te.register_buffer(ptrs, lengths)

        # Advertise the session port peers reconstruct dest_session from:
        # mooncake -> engine rpc port; memfabric -> the unique_id port baked
        # into the D-side unique_id (see _ensure_engine).
        metadata = MooncakeAgentMetadata(
            te_rpc_port=global_te.get_advertised_rpc_port(),
            layer_metadata=self.layer_metadata,
        )
        ready_event = threading.Event()
        self.kv_recv_layer_thread = SFARecvLayerThread(
            self.tp_rank,
            self.side_channel_port,
            self.vllm_config.parallel_config.tensor_parallel_size,
            1,  # pd_head_ratio == 1 for MLA (see plan risk #4)
            " ",  # local_engine_id placeholder (mirrors mooncake)
            metadata,
            ready_event,
            on_staging_done=self._on_staging_done,
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
        # memfabric pull mode: done comes from MembPullReadThread
        if hasattr(self, "_mf_read_thread") and self._mf_read_thread is not None:
            done = self._mf_read_thread.get_and_clear_done()
            done_recving: set[str] = set()
            still_pending: set[str] = set()
            for ext_id in done | self._pending_done:
                internal = self.request_map.get(ext_id)
                if internal is not None:
                    done_recving.add(internal)
                else:
                    still_pending.add(ext_id)
            self._pending_done = still_pending

            # Failed requests: unblock them (load will error / wrong KV)
            failed = self._mf_read_thread.get_and_clear_failed()
            still_pending_failed: set[str] = set()
            for ext_id in failed | self._pending_failed:
                internal = self.request_map.get(ext_id)
                if internal is not None:
                    done_recving.add(internal)
                else:
                    still_pending_failed.add(ext_id)
            self._pending_failed = still_pending_failed
            return set(), done_recving

        # mooncake staging mode: existing logic
        done_recving: set[str] = set()
        if self.kv_recv_layer_thread is not None:
            done = self.kv_recv_layer_thread.get_and_clear_done_requests()
            still_pending: set[str] = set()
            for ext_id in done | self._pending_done:
                internal = self.request_map.get(ext_id)
                if internal is not None:
                    done_recving.add(internal)
                else:
                    still_pending.add(ext_id)
            self._pending_done = still_pending

            failed = self.kv_recv_layer_thread.get_and_clear_failed_requests()
            still_pending_failed: set[str] = set()
            for ext_id in failed | self._pending_failed:
                internal = self.request_map.get(ext_id)
                if internal is not None:
                    done_recving.add(internal)  # unblock the req; load will error
                else:
                    still_pending_failed.add(ext_id)
            self._pending_failed = still_pending_failed
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

    def _on_staging_done(self, layer_idx: int) -> None:
        """Called by D recv thread when P finishes pushing main MLA layer L
        to the staging HBM buffer. Copy staging -> CPU pool (HBM->DRAM, local),
        then the recv thread sends STAGING_ACK back to P so P can push the
        next layer (1-slot staging, no double buffer needed)."""
        if not self._cpu_pools:
            return
        k_cpu, v_cpu = self._cpu_pools[layer_idx]
        k_cpu.copy_(self.staging_k)
        v_cpu.copy_(self.staging_v)

    def _register_memfabric_pull(
        self,
        kv_caches: dict[str, torch.Tensor],
        k_caches_cpu: list[torch.Tensor],
        v_caches_cpu: list[torch.Tensor],
    ) -> None:
        """memfabric pull mode: D does NOT register anything. Only P registers
        its HBM. D creates a read engine + MembPullReadThread."""
        num_blocks = self.kv_cache_config.num_blocks
        indexer_names = list(
            self.kv_cache_config.kv_cache_groups[_INDEXER_GROUP_IDX].layer_names
        )
        main_names = [
            n for n, v in kv_caches.items()
            if len(v if isinstance(v, (list, tuple)) else [v]) == 5
        ]
        main_by_layer_idx = {_layer_idx(name): name for name in main_names}
        main_names = [main_by_layer_idx[_layer_idx(name)] for name in indexer_names]

        # Store layer info for MembPullReadThread
        self._indexer_names = indexer_names
        self._main_names = main_names
        self._cpu_pools = list(zip(k_caches_cpu, v_caches_cpu))
        self._indexer_tensors = []
        for main_name in main_names:
            main_tuple = list(kv_caches[main_name])
            self._indexer_tensors.append(main_tuple[2])  # dsa_k_indexer

        # Build layer_metadata (D's local addresses, for compatibility)
        for pool_idx, (iname, mname) in enumerate(zip(indexer_names, main_names)):
            indexer_t = self._indexer_tensors[pool_idx]
            k_cpu, v_cpu = self._cpu_pools[pool_idx]
            self.layer_metadata[iname] = LayerMetadata(
                tensor_group_idx=[_INDEXER_GROUP_IDX],
                kv_caches_base_addr=[indexer_t.data_ptr()],
                block_len=[indexer_t.element_size() * math.prod(indexer_t.shape[1:])],
                block_size_scale=[indexer_t.shape[0] // num_blocks if num_blocks else 1],
            )
            self.layer_metadata[mname] = LayerMetadata(
                tensor_group_idx=[_MAIN_GROUP_IDX, _MAIN_GROUP_IDX],
                kv_caches_base_addr=[k_cpu.data_ptr(), v_cpu.data_ptr()],
                block_len=[
                    k_cpu.element_size() * math.prod(k_cpu.shape[1:]),
                    v_cpu.element_size() * math.prod(v_cpu.shape[1:]),
                ],
                block_size_scale=[
                    k_cpu.shape[0] // num_blocks if num_blocks else 1,
                    v_cpu.shape[0] // num_blocks if num_blocks else 1,
                ],
            )

        # Create memfabric engine (no registration)
        self._ensure_engine()
        # Start MembPullReadThread (ZMQ ROUTER + memfabric read)
        self._mf_read_thread = MembPullReadThread(
            tp_rank=self.tp_rank,
            side_channel_port=self.side_channel_port,
            engine=self.engine,
            worker=self,
        )
        self._mf_read_thread.start()
        self._mf_read_thread.ready_event.wait()
        self.kv_recv_layer_thread = None  # not used in pull mode
        logger.info(
            "SFAPDCpuOffload D-side registered (memfabric pull): "
            "%d indexer + %d main layers, zero D-side registration",
            len(indexer_names), len(main_names),
        )

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
# memfabric pull mode — D reads from P HBM
# ======================================================================
class MembPullReadThread(threading.Thread):
    """D-side thread for memfabric pull: receives READ_READY from P,
    reads KV from P's HBM via batch_transfer_sync_read, sends READ_DONE.

    indexer → D HBM (direct), main MLA → D CPU pool DRAM (local target,
    no registration needed per memfabric read semantics).
    """

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        engine: Any,
        worker: SFAPDCpuOffloadWorker,
    ):
        super().__init__(daemon=True, name=f"MembPullReadThread-TP{tp_rank}")
        self.tp_rank = tp_rank
        self.side_channel_port = side_channel_port
        self.engine = engine
        self._worker = worker
        self.ready_event = threading.Event()
        # P session info (set when MF_META received)
        self._p_session: str | None = None
        self._p_layer_meta: dict[str, Any] = {}
        # done tracking (req-level)
        self._done_requests: set[str] = set()
        self._failed_requests: set[str] = set()
        self._lock = threading.Lock()
        self._host = get_ip()

    def get_and_clear_done(self) -> set[str]:
        with self._lock:
            d = self._done_requests
            self._done_requests = set()
            return d

    def get_and_clear_failed(self) -> set[str]:
        with self._lock:
            f = self._failed_requests
            self._failed_requests = set()
            return f

    def run(self):
        from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self._host, handshake_port)
        logger.info("MembPull read thread listening on: %s", path)
        ctx = zmq.Context()
        try:
            sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.ROUTER, bind=True)
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            encoder = msgspec.msgpack.Encoder()
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        continue
                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        continue
                    msg = decoder.decode(payload[0])
                    msg_type = msg[0]

                    if msg_type == MF_META:
                        self._p_session = msg[1]
                        self._p_layer_meta = msgspec.msgpack.decode(msg[2])
                        logger.info("Received MF_META: P session=%s, %d layers",
                                    self._p_session, len(self._p_layer_meta))
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == READ_READY:
                        layer_idx = msg[1]
                        layer_name = msg[2]
                        p_blocks = msg[3]  # P's local_block_ids (list of lists per group)
                        d_blocks = msg[4]  # D's remote_block_ids (list of lists per group)
                        try:
                            self._do_read(layer_name, p_blocks, d_blocks)
                            sock.send_multipart(
                                (identity, b"", encoder.encode((READ_DONE, layer_idx)))
                            )
                        except Exception as e:
                            logger.error("MembPull read failed for layer %d: %s", layer_idx, e)
                            sock.send_multipart(
                                (identity, b"", encoder.encode((READ_DONE, layer_idx)))
                            )

                    elif msg_type == DONE_SENDING_MSG:
                        # P finished all layers for this request
                        request_id = msg[1]
                        with self._lock:
                            self._done_requests.add(request_id)
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == FAILED_SENDING_MSG:
                        request_id = msg[1]
                        with self._lock:
                            self._failed_requests.add(request_id)
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == GET_META_MSG:
                        # D responds with its own metadata (for compatibility)
                        meta_bytes = encoder.encode(
                            MooncakeAgentMetadata(
                                te_rpc_port=0,
                                layer_metadata=self._worker.layer_metadata,
                            )
                        )
                        sock.send_multipart((identity, b"", meta_bytes))
                    else:
                        logger.error("MembPull got unexpected message %s", msg)
                except Exception as e:
                    logger.error("MembPull exception: %s: %s", type(e), e)
        finally:
            ctx.destroy(linger=0)

    def _do_read(
        self,
        layer_name: str,
        p_blocks: list[list[int]],
        d_blocks: list[list[int]],
    ) -> None:
        """Read one layer's KV from P HBM → D local via memfabric.

        p_blocks: P's local_block_ids per group (peer addresses on P).
        d_blocks: D's remote_block_ids per group (local addresses on D).
        """
        if self._p_session is None:
            raise RuntimeError("MF_META not received before READ_READY")

        w = self._worker
        is_indexer = layer_name in w._indexer_names
        pool_idx = None
        for idx, (iname, mname) in enumerate(zip(w._indexer_names, w._main_names)):
            if layer_name == iname:
                is_indexer = True
                pool_idx = idx
                break
            if layer_name == mname:
                is_indexer = False
                pool_idx = idx
                break
        if pool_idx is None:
            logger.warning("MembPull _do_read: layer %s not in indexer/main names, skip", layer_name)
            return  # not a layer we handle

        p_meta = self._p_layer_meta.get(layer_name)
        if p_meta is None:
            logger.warning(
                "MembPull _do_read: layer %s not in P layer_meta (MF_META not received? "
                "have %d layers), skip", layer_name, len(self._p_layer_meta))
            return

        local_ptrs: list[int] = []
        peer_ptrs: list[int] = []
        lengths: list[int] = []

        if is_indexer:
            # indexer (group 0): P HBM → D indexer HBM
            p_bid_list = p_blocks[0] if p_blocks and len(p_blocks) > 0 else []
            d_bid_list = d_blocks[0] if d_blocks and len(d_blocks) > 0 else []
            if not p_bid_list or not d_bid_list:
                return
            block_len = p_meta["block_len"][0]
            p_base = p_meta["base_addrs"][0]
            d_indexer = w._indexer_tensors[pool_idx]
            d_base = d_indexer.data_ptr()
            for p_bid, d_bid in zip(p_bid_list, d_bid_list):
                local_ptrs.append(d_base + d_bid * block_len)
                peer_ptrs.append(p_base + p_bid * block_len)
                lengths.append(block_len)
        else:
            # main MLA (group 1): P HBM → D CPU pool DRAM
            p_bid_list = p_blocks[1] if p_blocks and len(p_blocks) > 1 else []
            d_bid_list = d_blocks[1] if d_blocks and len(d_blocks) > 1 else []
            if not p_bid_list or not d_bid_list:
                return
            k_cpu, v_cpu = w._cpu_pools[pool_idx]
            k_block_len = p_meta["block_len"][0]
            v_block_len = p_meta["block_len"][1]
            p_k_base = p_meta["base_addrs"][0]
            p_v_base = p_meta["base_addrs"][1]
            for p_bid, d_bid in zip(p_bid_list, d_bid_list):
                local_ptrs.append(k_cpu.data_ptr() + d_bid * k_block_len)
                peer_ptrs.append(p_k_base + p_bid * k_block_len)
                lengths.append(k_block_len)
                local_ptrs.append(v_cpu.data_ptr() + d_bid * v_block_len)
                peer_ptrs.append(p_v_base + p_bid * v_block_len)
                lengths.append(v_block_len)

        if local_ptrs:
            # Pre-read checksum (sample first block to detect uninitialized data)
            if is_indexer and d_bid_list:
                d_indexer = w._indexer_tensors[pool_idx]
                pre_sum = d_indexer[d_bid_list[0]].float().sum().item()
            elif not is_indexer and d_bid_list:
                k_cpu, v_cpu = w._cpu_pools[pool_idx]
                pre_sum = k_cpu[d_bid_list[0]].float().sum().item()
            else:
                pre_sum = 0.0

            ret = self.engine.batch_transfer_sync_read(
                self._p_session, local_ptrs, peer_ptrs, lengths
            )
            if ret != 0:
                raise RuntimeError(
                    f"memfabric read failed for layer {layer_name}, ret={ret}"
                )

            # Post-read checksum: compare with pre to confirm data was written.
            # Non-zero post + different from pre = data actually transferred.
            if is_indexer and d_bid_list:
                post_sum = d_indexer[d_bid_list[0]].float().sum().item()
            elif not is_indexer and d_bid_list:
                post_sum = k_cpu[d_bid_list[0]].float().sum().item()
            else:
                post_sum = 0.0
            logger.info(
                "MembPull read layer %s (pool_idx=%d, is_indexer=%s): "
                "%d blocks, %d transfers, pre=%.4f post=%.4f",
                layer_name, pool_idx, is_indexer,
                len(d_bid_list) if d_bid_list else 0,
                len(local_ptrs), pre_sum, post_sum,
            )


# ======================================================================
# P side (kv_producer) — layer-wise RDMA push with per-layer completion
# ======================================================================
class _SFAPDCpuSendingThread(KVCacheSendingLayerThread):
    """Sending thread that signals per-layer send completion + staging ACK.

    For main MLA layers (group 1), after push completes the data lands in D's
    HBM staging buffer. P must wait for D's STAGING_ACK (D copied staging→CPU
    pool) before pushing the next layer, because staging is a 1-slot buffer.
    Indexer layers (group 0) go direct to D HBM — no staging, no ACK needed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        total_layers: int = kwargs.get("total_layers")  # type: ignore[assignment]
        super().__init__(*args, **kwargs)
        self.timeout = 1.0
        self.layer_send_done_events: list[threading.Event] = [
            threading.Event() for _ in range(total_layers)
        ]
        self._poller = zmq.Poller()

    def _transfer_kv_cache(self, send_task: Any) -> None:  # type: ignore[override]
        super()._transfer_kv_cache(send_task)
        idx = send_task.layer_idx
        if 0 <= idx < len(self.layer_send_done_events):
            self.layer_send_done_events[idx].set()

        # For main MLA layers (group 1 = staging path), wait for D's ACK
        # before returning — D has copied staging→CPU pool, staging slot
        # is now free for the next layer.
        layer_name = send_task.layer_name
        local_meta = self.layer_metadata.get(layer_name)
        if local_meta is not None and _MAIN_GROUP_IDX in local_meta.tensor_group_idx:
            transformer_layer_idx = _layer_idx(layer_name)
            self._staging_handshake(send_task, transformer_layer_idx)

    def _staging_handshake(self, send_task: Any, transformer_layer_idx: int) -> None:
        """Send STAGING_DONE to D, block until STAGING_ACK received.

        Uses the first req's remote_host:remote_port as D's ZMQ endpoint
        (same handshake port the base mooncake layerwise uses for DONE_SENDING).

        Args:
            transformer_layer_idx: the transformer layer index (0..N-1), used
              by D to index into _cpu_pools. NOT the mooncake global layer_idx
              (which counts indexer + main layers interleaved).
        """
        if not send_task.send_request:
            return
        req_meta = next(iter(send_task.send_request.values()))
        remote_host = req_meta.remote_host
        remote_port = req_meta.remote_port
        if not remote_host or not remote_port:
            return
        path = make_zmq_path("tcp", remote_host, remote_port)
        encoder = msgspec.msgpack.Encoder()
        encoded = encoder.encode((STAGING_DONE, transformer_layer_idx))
        try:
            with zmq_ctx(zmq.REQ, path) as sock:
                sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                ensure_zmq_send(sock, encoded, path)
                resp = ensure_zmq_recv(sock, self._poller, path, timeout=self.timeout)
                ack_msg = msgspec.msgpack.Decoder(type=tuple).decode(resp)
                if ack_msg[0] != STAGING_ACK:
                    logger.warning(
                        "STAGING handshake: unexpected ACK %s for layer %d",
                        ack_msg, transformer_layer_idx,
                    )
        except Exception as e:
            logger.error(
                "STAGING handshake failed for layer %d (host=%s:%s): %s",
                transformer_layer_idx, remote_host, remote_port, e,
            )


class _MembPullSendingThread(KVCacheSendingLayerThread):
    """P-side sending thread for memfabric pull mode.

    Does NOT push (no batch_transfer_sync_write). Instead, after each layer's
    KV is ready in P HBM, notifies D via READ_READY (ZMQ), waits for READ_DONE.
    First call sends MF_META (P session + layer addresses) to D.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        total_layers: int = kwargs.get("total_layers")  # type: ignore[assignment]
        super().__init__(*args, **kwargs)
        self.timeout = 10.0
        self._mf_meta_sent = False
        # P's memfabric session (unique_id) — set by producer worker
        self.p_session: str = ""
        # Buffer-reuse gate: set after READ_DONE so model runner knows the
        # layer's KV has been read by D and the buffer can be reused.
        self.layer_send_done_events: list[threading.Event] = [
            threading.Event() for _ in range(total_layers)
        ]
        # Persistent ZMQ Context — creating a new Context per message (zmq_ctx)
        # causes I/O thread churn + TCP resource exhaustion when 4 P ranks fire
        # hundreds of messages rapidly on the same machine.
        self._persist_ctx = zmq.Context()

    def _send_recv(self, path: str, encoded: bytes) -> bytes:
        """Send a ZMQ REQ message, block until response. Uses a short-lived
        socket from the persistent Context (clean REQ state per message,
        no Context creation overhead)."""
        from vllm.utils.network_utils import make_zmq_socket
        sock = make_zmq_socket(
            ctx=self._persist_ctx, path=path, socket_type=zmq.REQ, bind=False
        )
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            ensure_zmq_send(sock, encoded, path)
            return ensure_zmq_recv(sock, poller, path, timeout=self.timeout)
        finally:
            sock.close(linger=0)

    def _transfer_kv_cache(self, send_task: Any) -> None:  # type: ignore[override]
        """Override: no write. Just notify D to read + wait."""
        # CRITICAL: wait for the KV to be fully written to P HBM before
        # notifying D to read. Without this, D may read stale/unwritten data.
        if send_task.wait_event is not None:
            send_task.wait_event.synchronize()

        layer_name = send_task.layer_name
        layer_idx = send_task.layer_idx

        if not send_task.send_request:
            return
        req_meta = next(iter(send_task.send_request.values()))
        remote_host = req_meta.remote_host
        # req_meta.remote_port already includes tp_rank offset (set by the
        # mooncake layerwise base class). D binds ROUTER on this port.
        remote_port = req_meta.remote_port
        if not remote_host or not remote_port:
            return

        path = make_zmq_path("tcp", remote_host, remote_port)
        encoder = msgspec.msgpack.Encoder()

        # First call: send MF_META (P session + P layer addresses)
        if not self._mf_meta_sent:
            p_meta_dict = {}
            for ln, meta in self.layer_metadata.items():
                p_meta_dict[ln] = {
                    "base_addrs": list(meta.kv_caches_base_addr),
                    "block_len": list(meta.block_len),
                    "block_size_scale": list(meta.block_size_scale),
                }
            meta_encoded = encoder.encode((MF_META, self.p_session, encoder.encode(p_meta_dict)))
            try:
                self._send_recv(path, meta_encoded)
                self._mf_meta_sent = True
                logger.info("MF_META sent to D (session=%s)", self.p_session)
            except Exception as e:
                logger.error("Failed to send MF_META: %s", e)

        # Aggregate block IDs across all requests in this send_task
        agg_local: list[list[int]] = []
        agg_remote: list[list[int]] = []
        for req_meta in send_task.send_request.values():
            for g, blocks in enumerate(req_meta.local_block_ids):
                while len(agg_local) <= g:
                    agg_local.append([])
                agg_local[g].extend(blocks)
            for g, blocks in enumerate(req_meta.remote_block_ids):
                while len(agg_remote) <= g:
                    agg_remote.append([])
                agg_remote[g].extend(blocks)

        ready_encoded = encoder.encode(
            (READ_READY, layer_idx, layer_name, agg_local, agg_remote)
        )
        try:
            resp = self._send_recv(path, ready_encoded)
            ack_msg = msgspec.msgpack.Decoder(type=tuple).decode(resp)
            if ack_msg[0] != READ_DONE:
                logger.warning(
                    "MembPull: unexpected ACK %s for layer %d (%s)",
                    ack_msg, layer_idx, layer_name,
                )
            # Mark this layer's buffer as safe to reuse (D has read it).
            if 0 <= layer_idx < len(self.layer_send_done_events):
                self.layer_send_done_events[layer_idx].set()
        except Exception as e:
            logger.error(
                "MembPull handshake failed for layer %d (%s): %s",
                layer_idx, layer_name, e,
            )

        # After the last layer, send DONE_SENDING so D reports done_recving.
        # Send directly via ZMQ (can't use callback_func = send_done_send_signal
        # because it accesses req_meta.trans_count which is empty — we skipped
        # super()._transfer_kv_cache() that populates it).
        if layer_idx == self.total_layers - 1:
            for req_id, req_meta in send_task.send_request.items():
                if req_meta.chunk_finish:
                    external_req_id = get_external_request_id(req_id)
                    done_encoded = encoder.encode(
                        (DONE_SENDING_MSG, external_req_id, 0, "")
                    )
                    try:
                        self._send_recv(path, done_encoded)
                    except Exception as e:
                        logger.error("MembPull DONE_SENDING failed for %s: %s", req_id, e)


class SFAPDCpuOffloadProducerWorker(MooncakeLayerwiseConnectorWorker):
    """P-side worker = stock mooncake layerwise push + per-layer send-done gate.

    The split destination (indexer→D HBM, main MLA→D CPU) needs NO sender-side
    branching: ``get_transfer_meta`` derives ``dst`` from the decoder-advertised
    ``remote_layer_metadata[...]`` base addrs, which the D side populated with the
    indexer NPU ptr and the main-MLA CPU-pool ptr respectively.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self._backend = _resolve_kv_transfer_backend(vllm_config)
        if self._backend == BACKEND_MEMFABRIC:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            side_channel_port = (
                vllm_config.kv_transfer_config.kv_port
                + vllm_config.parallel_config.data_parallel_rank * tp_size
            )
            # P's memfabric session port (must be a real TCP port, offset past
            # ZMQ ports just like D side). unique_id = "P_IP:P_store_port".
            mf_session_port = side_channel_port + tp_size + tp_rank
            self._mf_unique_id = f"{get_ip()}:{mf_session_port}"
            # P must connect to D's config store (D is store server). Read D's
            # store URL from env var (set in launch script: D_IP:D_store_port).
            mf_store_url = os.environ.get("MF_CONFIG_STORE_URL")
            if not mf_store_url:
                logger.warning(
                    "MF_CONFIG_STORE_URL not set; memfabric P may not reach D's "
                    "config store. Set it to tcp://<D_IP>:<D_store_port>."
                )
            global_te.configure(
                backend=BACKEND_MEMFABRIC,
                role=MEMFABRIC_ROLE_PREFILL,
                unique_id=self._mf_unique_id,
                store_url=mf_store_url,
                device_id=torch.npu.current_device(),
            )
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.layer_send_done_events: list[threading.Event] | None = None

    def update_decoder_info(self, req_id: str, req_meta: Any) -> Any:
        """Override: in memfabric pull mode, P does NOT need D's metadata
        (P is not pushing to D — D reads from P). Skip GET_META entirely
        to avoid flooding D's ROUTER with 61 unnecessary requests that
        delay MF_META / READ_READY."""
        if self._backend == BACKEND_MEMFABRIC:
            return req_meta
        return super().update_decoder_info(req_id, req_meta)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._backend == BACKEND_MEMFABRIC:
            # memfabric pull: swap in _MembPullSendingThread (notifies D to read,
            # does NOT push). Set its p_session before base class constructs it.
            orig = _mlc.KVCacheSendingLayerThread
            _mlc.KVCacheSendingLayerThread = _MembPullSendingThread  # type: ignore[assignment]
            # Patch __init__ to inject p_session (hack: base class doesn't pass it)
            _orig_init = _MembPullSendingThread.__init__
            _mf_uid = self._mf_unique_id

            def _patched_init(self_inner, *a, **kw):
                _orig_init(self_inner, *a, **kw)
                self_inner.p_session = _mf_uid
            _MembPullSendingThread.__init__ = _patched_init  # type: ignore[assignment]
            try:
                super().register_kv_caches(kv_caches)
            finally:
                _mlc.KVCacheSendingLayerThread = orig  # type: ignore[assignment]
                _MembPullSendingThread.__init__ = _orig_init  # type: ignore[assignment]
        else:
            # mooncake staging push: swap in _SFAPDCpuSendingThread
            orig = _mlc.KVCacheSendingLayerThread
            _mlc.KVCacheSendingLayerThread = _SFAPDCpuSendingThread  # type: ignore[assignment]
            try:
                super().register_kv_caches(kv_caches)
            finally:
                _mlc.KVCacheSendingLayerThread = orig  # type: ignore[assignment]
        if isinstance(self.kv_send_layer_thread, (_SFAPDCpuSendingThread, _MembPullSendingThread)):
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
