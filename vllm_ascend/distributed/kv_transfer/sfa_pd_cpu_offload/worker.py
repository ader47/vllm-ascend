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
    KVCacheSendingLayerThread,
    LayerMetadata,
    MooncakeAgentMetadata,
    MooncakeLayerwiseConnectorWorker,
    ensure_zmq_recv,
    ensure_zmq_send,
    get_external_request_id,
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


MF_META = b"mf_meta"  # P→D: (MF_META, p_session, p_layer_meta_serialized)
# READ_READY (pull model): P sends only its OWN source block ids + the external
# req_id; D looks up its destination blocks by req_id (D's block ids never
# travel to P). One READ_READY per (layer, req).
READ_READY = b"read_ready"  # P→D: (READ_READY, layer_idx, layer_name, ext_req_id, p_block_ids)
READ_DONE = b"read_done"  # D→P: (READ_DONE, layer_idx)


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
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )

        self.layer_metadata: dict[str, LayerMetadata] = {}
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
        # external_req_id -> (indexer_npu_ids, main_cpu_ids, num_full,
        # partial_hbm_bid): D's OWN destination blocks per request, populated in
        # start_load_kv from connector_meta. Part A: the first num_full main
        # blocks → CPU pool; the optional partial last block → D HBM at
        # partial_hbm_bid (None ⇒ no partial).
        self._dest_blocks_by_req: dict[str, tuple[list[int], list[int], int, int | None]] = {}
        # main layer name -> (k_nope HBM, v_rope HBM); the partial block's HBM
        # dest. Populated in register_kv_caches.
        self._hbm_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # B1: per-step decode offload pairs (src HBM block ids, dst CPU block
        # ids); reset in start_load_kv, consumed by save_kv_layer.
        self._offload_pairs: list[tuple[list[int], list[int]]] = []
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
                # Offset 2*tp_size past the ZMQ ports: on a single host P binds
                # ZMQ at P_kv+rank and D at D_kv+rank, and the launch convention
                # is D_kv = P_kv + tp_size, so the combined ZMQ footprint is
                # [P_kv .. P_kv+2*tp_size). Starting memfabric at +tp_size would
                # collide (D store would land on P_kv+tp_size+rank = D's ZMQ
                # range on P's side; P's unique_id would land on D's ZMQ range).
                # +2*tp_size also matches the launch script's MF_CONFIG_STORE_URL
                # formula (D_kv + tp_size + tp_size + rank).
                tp_size = self.vllm_config.parallel_config.tensor_parallel_size
                mf_session_port = self.side_channel_port + 2 * tp_size + self.tp_rank
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
        self.sfa_worker = SFAKVOffloadWorker(self.vllm_config, self.use_layerwise, self.kv_cache_config)
        # SFA worker allocates k_caches_cpu/v_caches_cpu + LRU buffers here.
        self.sfa_worker.register_kv_caches(kv_caches)

        # CPU pool owned by the composed SFA worker (filled via staging copy).
        # HW limitation: P HBM -> D DRAM cross-node RDMA is not available.
        # Solution: P pushes main MLA to a 1-slot HBM staging buffer, then
        # signals per-layer DONE; D copies staging -> CPU pool locally and ACKs.
        assert self.sfa_worker.k_caches_cpu is not None, "Composed SFA worker did not allocate the CPU pool"
        k_caches_cpu = self.sfa_worker.k_caches_cpu
        v_caches_cpu = self.sfa_worker.v_caches_cpu

        # Part A: D's main MLA HBM k/v tensors (group1 paged cache) — the partial
        # last block lands here instead of the CPU pool. Keyed by main layer name
        # (the 5-tuple layers); tuple[0]=k_nope, tuple[1]=v_rope.
        self._hbm_kv = {n: (t[0], t[1]) for n, t in kv_caches.items() if isinstance(t, (list, tuple)) and len(t) == 5}

        # memfabric pull mode only (the mooncake staging path has been removed).
        assert _resolve_kv_transfer_backend(self.vllm_config) == BACKEND_MEMFABRIC, (
            "SFAPDCpuOffloadConnector D side supports memfabric pull only (set transfer_backend=memfabric)."
        )
        self._register_memfabric_pull(kv_caches, k_caches_cpu, v_caches_cpu)

    # -- D-side forwards to the composed SFA worker (LRU load path) --
    def start_load_kv(self, metadata: KVConnectorMetadata):
        assert self.sfa_worker is not None
        # B1: reset decode offload pairs for this step (consumed by save_kv_layer).
        self._offload_pairs: list[tuple[list[int], list[int]]] = []
        # Seed external->internal request id map for get_finished, and store D's
        # own destination blocks per request (keyed by external id, which is what
        # P sends in READ_READY). The scheduler includes remote-prefill requests
        # here (even while async-waiting) so both exist before P's signal arrives.
        for req in getattr(metadata, "requests", []):
            req_id = getattr(req, "req_id", None)
            if req_id is not None:
                ext_id = get_external_request_id(req_id)
                self.request_map[ext_id] = req_id
                indexer_ids = list(getattr(req, "block_ids_npu", []) or [])
                main_ids = list(getattr(req, "block_ids_cpu", []) or [])
                num_full = getattr(req, "num_full", 0) or 0
                partial_hbm_bid = getattr(req, "partial_hbm_bid", None)
                self._dest_blocks_by_req[ext_id] = (indexer_ids, main_ids, num_full, partial_hbm_bid)
                logger.info(
                    "MembPull D stored dest blocks req %s: indexer(HBM)=%d, main(CPU full)=%d, partial_hbm=%s",
                    ext_id,
                    len(indexer_ids),
                    len(main_ids),
                    partial_hbm_bid,
                )
                # B1: collect decode offload pairs (src HBM ids -> dst CPU ids).
                src = list(getattr(req, "offload_src_hbm_ids", []) or [])
                dst = list(getattr(req, "offload_dst_cpu_ids", []) or [])
                if src and dst:
                    self._offload_pairs.append((src, dst))
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
        # B1: offload decode-filled main MLA blocks HBM -> CPU pool. Part A put
        # the prompt's full blocks in CPU (via P) and the partial in HBM; this
        # copies blocks that just filled during decode into the CPU pool, so
        # num_offloaded_blocks grows and decode routes them via LRU. Sync copy
        # (HBM->CPU .to('cpu') is blocking); verified-correct, can go async later.
        pairs = getattr(self, "_offload_pairs", None)
        if not pairs:
            return
        pool_idx = None
        for idx, mname in enumerate(self._main_names):
            if layer_name == mname:
                pool_idx = idx
                break
        if pool_idx is None:
            return
        hbm_kv = self._hbm_kv.get(layer_name)
        if hbm_kv is None:
            return
        k_hbm, v_hbm = hbm_kv
        k_cpu, v_cpu = self._cpu_pools[pool_idx]
        for src, dst in pairs:
            k_cpu[dst] = k_hbm[src].to("cpu")
            v_cpu[dst] = v_hbm[src].to("cpu")

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

        # memfabric pull mode only (mooncake staging path removed). If the read
        # thread isn't up yet, nothing is finished.
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        """Per-req actual main-MLA CPU-block count for the solution-1 threshold."""
        if self.sfa_worker is None:
            return None
        return {rid: self._cpu_blocks_by_req.get(rid, 0) for rid in req_ids}

    def _register_memfabric_pull(
        self,
        kv_caches: dict[str, torch.Tensor],
        k_caches_cpu: list[torch.Tensor],
        v_caches_cpu: list[torch.Tensor],
    ) -> None:
        """memfabric pull mode: D does NOT register anything. Only P registers
        its HBM. D creates a read engine + MembPullReadThread."""
        num_blocks = self.kv_cache_config.num_blocks
        indexer_names = list(self.kv_cache_config.kv_cache_groups[_INDEXER_GROUP_IDX].layer_names)
        main_names = [n for n, v in kv_caches.items() if len(v if isinstance(v, (list, tuple)) else [v]) == 5]
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
        logger.info(
            "SFAPDCpuOffload D-side registered (memfabric pull): %d indexer + %d main layers, zero D-side registration",
            len(indexer_names),
            len(main_names),
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
                        logger.info(
                            "Received MF_META: P session=%s, %d layers", self._p_session, len(self._p_layer_meta)
                        )
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == READ_READY:
                        layer_idx = msg[1]
                        layer_name = msg[2]
                        ext_req_id = msg[3]  # external req_id — D looks up its own dest blocks by this
                        p_block_ids = msg[4]  # P's source block ids (shared by main + indexer on P)
                        try:
                            self._do_read(layer_name, ext_req_id, p_block_ids)
                            sock.send_multipart((identity, b"", encoder.encode((READ_DONE, layer_idx))))
                        except Exception as e:
                            logger.error(
                                "MembPull read failed for layer %d (%s) req %s: %s",
                                layer_idx,
                                layer_name,
                                ext_req_id,
                                e,
                            )
                            sock.send_multipart((identity, b"", encoder.encode((READ_DONE, layer_idx))))

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
        ext_req_id: str,
        p_block_ids: list[int],
    ) -> None:
        """Read one layer's KV (main MLA + indexer) from P HBM → D in one pull.

        P sends only its source block ids + ext_req_id; D looks up its OWN
        destination blocks for this request in
        ``worker._dest_blocks_by_req[ext_req_id]`` = (indexer_npu_ids,
        main_cpu_ids). Those were allocated by D's scheduler and passed to the
        worker via connector_meta (pull model: D's block ids never travel to P).

        Both legs share ``p_block_ids`` (on P the indexer is written into
        dsa_k_cache at the SAME slots as main MLA). Destinations differ:
          main     → D CPU pool at main_cpu_ids (P tensors [0]=k, [1]=v)
          indexer  → D indexer HBM at indexer_npu_ids (P tensor [2]=dsa_k)

        All transfers are issued in a single ``batch_transfer_sync_read``.
        """
        if self._p_session is None:
            raise RuntimeError("MF_META not received before READ_READY")

        w = self._worker
        # D's own destination blocks for this request (allocated by D scheduler).
        dest = w._dest_blocks_by_req.get(ext_req_id)
        if dest is None:
            logger.warning(
                "MembPull _do_read: no dest blocks on D for req %s (layer %s), skip",
                ext_req_id,
                layer_name,
            )
            return
        d_indexer_ids, d_main_ids, num_full, partial_hbm_bid = dest  # (npu, cpu, num_full, partial_hbm_bid)

        # Locate the transformer-layer pool index from the MAIN layer name.
        pool_idx = None
        for idx, mname in enumerate(w._main_names):
            if layer_name == mname:
                pool_idx = idx
                break
        if pool_idx is None:
            logger.warning("MembPull _do_read: layer %s not in main names, skip", layer_name)
            return

        p_meta = self._p_layer_meta.get(layer_name)
        if p_meta is None:
            logger.warning(
                "MembPull _do_read: layer %s not in P layer_meta (MF_META not received? have %d layers), skip",
                layer_name,
                len(self._p_layer_meta),
            )
            return

        if not p_block_ids:
            logger.warning("MembPull _do_read: empty p_block_ids for %s, skip", layer_name)
            return

        p_base_addrs = p_meta["base_addrs"]
        p_block_len = p_meta["block_len"]

        local_ptrs: list[int] = []
        peer_ptrs: list[int] = []
        lengths: list[int] = []
        n_main = 0
        n_indexer = 0

        # Main MLA leg. Part A: the first `num_full` of P's blocks are FULL → D
        # CPU pool (1:1 with d_main_ids); the optional last block is the PARTIAL
        # → D main MLA HBM (group1) at partial_hbm_bid, so decode can append.
        p_k_base, p_v_base = p_base_addrs[0], p_base_addrs[1]
        p_k_len, p_v_len = p_block_len[0], p_block_len[1]
        full_p_blocks = p_block_ids[:num_full]
        if full_p_blocks and d_main_ids:
            k_cpu, v_cpu = w._cpu_pools[pool_idx]
            for p_bid, d_bid in zip(full_p_blocks, d_main_ids):
                peer_ptrs.append(p_k_base + p_bid * p_k_len)
                local_ptrs.append(k_cpu.data_ptr() + d_bid * p_k_len)
                lengths.append(p_k_len)
                peer_ptrs.append(p_v_base + p_bid * p_v_len)
                local_ptrs.append(v_cpu.data_ptr() + d_bid * p_v_len)
                lengths.append(p_v_len)
            n_main = min(len(full_p_blocks), len(d_main_ids))
        # Partial last block → D HBM (whole block_size transfers; the tail beyond
        # prompt_len is garbage and gets overwritten as decode appends).
        if partial_hbm_bid is not None and num_full < len(p_block_ids):
            hbm_kv = w._hbm_kv.get(layer_name)
            if hbm_kv is None:
                logger.warning("MembPull _do_read: no HBM k/v for partial block %s, skip partial", layer_name)
            else:
                k_hbm, v_hbm = hbm_kv
                partial_p_bid = p_block_ids[num_full]
                peer_ptrs.append(p_k_base + partial_p_bid * p_k_len)
                local_ptrs.append(k_hbm.data_ptr() + partial_hbm_bid * p_k_len)
                lengths.append(p_k_len)
                peer_ptrs.append(p_v_base + partial_p_bid * p_v_len)
                local_ptrs.append(v_hbm.data_ptr() + partial_hbm_bid * p_v_len)
                lengths.append(p_v_len)
                n_main += 1

        # Indexer leg: P tensor [2]=dsa_k → D indexer HBM. D's indexer
        # block_size is `scale`× P's (D uses coarser indexer blocks, e.g. 4× →
        # fewer of them: 2 D blocks cover the same tokens as ~7-8 P blocks), so
        # each D indexer block is split into `scale` P-sized sub-addresses to
        # align with P's finer block list. Extra D sub-addresses (D_blocks ×
        # scale > len(P blocks)) are discarded by zip.
        if d_indexer_ids:
            if len(p_base_addrs) < 3:
                logger.error(
                    "MembPull indexer: P layer_meta for %s has %d tensors, need >=3 "
                    "(dsa_k must be registered as the main layer's tensor [2]); skip indexer leg",
                    layer_name,
                    len(p_base_addrs),
                )
            else:
                p_dsa_base = p_base_addrs[2]
                p_dsa_len = p_block_len[2]  # P per-indexer-block bytes (kv-block granularity)
                d_indexer = w._indexer_tensors[pool_idx]
                # D per-indexer-block bytes (coarser — spans scale× P's blocks).
                d_dsa_len = d_indexer.element_size() * math.prod(d_indexer.shape[1:])
                d_base = d_indexer.data_ptr()
                if p_dsa_len <= 0 or d_dsa_len % p_dsa_len != 0:
                    logger.error(
                        "MembPull indexer %s: D dsa_k block_len=%d not a multiple of P=%d; skip indexer leg",
                        layer_name,
                        d_dsa_len,
                        p_dsa_len,
                    )
                else:
                    scale = d_dsa_len // p_dsa_len
                    # Split each D indexer block into `scale` P-sized sub-addresses.
                    d_sub_addrs: list[int] = []
                    for d_bid in d_indexer_ids:
                        d_block_base = d_base + d_bid * d_dsa_len
                        for off in range(scale):
                            d_sub_addrs.append(d_block_base + off * p_dsa_len)
                    logger.debug(
                        "MembPull indexer %s: p_dsa_len=%d d_dsa_len=%d scale=%d, "
                        "D dsa_k shape=%s, D_sub=%d P_blocks=%d",
                        layer_name,
                        p_dsa_len,
                        d_dsa_len,
                        scale,
                        tuple(d_indexer.shape),
                        len(d_sub_addrs),
                        len(p_block_ids),
                    )
                    # Pair with P's blocks; zip truncates to the shorter, discarding
                    # any extra D sub-addresses (e.g. 8 D sub-addrs vs 7 P blocks).
                    for p_bid, d_sub in zip(p_block_ids, d_sub_addrs):
                        peer_ptrs.append(p_dsa_base + p_bid * p_dsa_len)
                        local_ptrs.append(d_sub)
                        lengths.append(p_dsa_len)
                    n_indexer = min(len(p_block_ids), len(d_sub_addrs))

        if not local_ptrs:
            logger.warning(
                "MembPull _do_read: nothing to transfer for %s (main=%d, indexer=%d)",
                layer_name,
                n_main,
                n_indexer,
            )
            return

        ret = self.engine.batch_transfer_sync_read(self._p_session, local_ptrs, peer_ptrs, lengths)
        if ret != 0:
            raise RuntimeError(f"memfabric read failed for layer {layer_name}, ret={ret}")
        # Verify-mode (VLLM_ASCEND_MF_VERIFY=1): log D's destination sums so the
        # user can diff against P's source sums (MFV P ...). main_k/main_v should
        # match P exactly (1:1 byte copy); idx_post is the full D indexer block
        # sum (D blocks are 4× P's and only len(p_block_ids)/scale slots written,
        # so it won't equal P's idx — use it only as a "data landed" check).
        if os.environ.get("VLLM_ASCEND_MF_VERIFY") == "1":
            try:
                k_cpu, v_cpu = w._cpu_pools[pool_idx]
                mk = k_cpu[d_main_ids].float().sum().item() if d_main_ids else 0.0
                mv = v_cpu[d_main_ids].float().sum().item() if d_main_ids else 0.0
                # Part A: the partial last block is in HBM (not the CPU pool) —
                # add it so D's main sum matches P's (full + partial) source sum.
                if partial_hbm_bid is not None:
                    hbm_kv = w._hbm_kv.get(layer_name)
                    if hbm_kv is not None:
                        k_hbm, v_hbm = hbm_kv
                        mk += k_hbm[partial_hbm_bid].float().sum().item()
                        mv += v_hbm[partial_hbm_bid].float().sum().item()
                mi = w._indexer_tensors[pool_idx][d_indexer_ids].float().sum().item() if d_indexer_ids else 0.0
                logger.info(
                    "MFV D layer %s req %s main_k=%.6f main_v=%.6f idx_post=%.6f",
                    layer_name,
                    ext_req_id,
                    mk,
                    mv,
                    mi,
                )
            except Exception as ve:  # noqa: BLE001 - verify must never break the read path
                logger.warning("MFV D checksum failed for %s: %s", layer_name, ve)
        logger.info(
            "MembPull read layer %s req %s (pool_idx=%d): main=%d/%d blocks, indexer=%d/%d (scale split), %d transfers",
            layer_name,
            ext_req_id,
            pool_idx,
            n_main,
            len(d_main_ids),
            n_indexer,
            len(d_indexer_ids),
            len(local_ptrs),
        )


# ======================================================================
# P side (kv_producer) — layer-wise RDMA push with per-layer completion
# ======================================================================
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
        self.layer_send_done_events: list[threading.Event] = [threading.Event() for _ in range(total_layers)]
        # Persistent ZMQ Context — creating a new Context per message (zmq_ctx)
        # causes I/O thread churn + TCP resource exhaustion when 4 P ranks fire
        # hundreds of messages rapidly on the same machine.
        self._persist_ctx = zmq.Context()

    def _send_recv(self, path: str, encoded: bytes) -> bytes:
        """Send a ZMQ REQ message, block until response. Uses a short-lived
        socket from the persistent Context (clean REQ state per message,
        no Context creation overhead)."""
        from vllm.utils.network_utils import make_zmq_socket

        sock = make_zmq_socket(ctx=self._persist_ctx, path=path, socket_type=zmq.REQ, bind=False)
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            ensure_zmq_send(sock, encoded, path)
            return ensure_zmq_recv(sock, poller, path, timeout=self.timeout)
        finally:
            sock.close(linger=0)

    def _send_read_ready(
        self,
        path: str,
        encoder: msgspec.msgpack.Encoder,
        layer_idx: int,
        layer_name: str,
        ext_req_id: str,
        p_block_ids: list[int],
    ) -> bool:
        """Send one READ_READY for (layer, req), block until READ_DONE.

        P sends only its OWN source block ids + the external req_id; D looks up
        its destination blocks by req_id (pull model — D's blocks never travel
        to P). Returns True iff D acknowledged with READ_DONE (or nothing to
        transfer); False on failure, so the caller can withhold the buffer-reuse
        signal instead of silently letting P overwrite a buffer D never read.
        """
        if not p_block_ids:
            logger.warning(
                "MembPull: skip READ_READY layer %d (%s) req %s — empty p_block_ids",
                layer_idx,
                layer_name,
                ext_req_id,
            )
            return True  # nothing to transfer; not a failure
        encoded = encoder.encode((READ_READY, layer_idx, layer_name, ext_req_id, p_block_ids))
        try:
            resp = self._send_recv(path, encoded)
            ack_msg = msgspec.msgpack.Decoder(type=tuple).decode(resp)
            if ack_msg[0] != READ_DONE:
                logger.warning(
                    "MembPull: unexpected ACK %s for layer %d (%s) req %s",
                    ack_msg,
                    layer_idx,
                    layer_name,
                    ext_req_id,
                )
                return False
            return True
        except Exception as e:
            logger.error(
                "MembPull handshake failed for layer %d (%s) req %s: %s",
                layer_idx,
                layer_name,
                ext_req_id,
                e,
            )
            return False

    def _transfer_kv_cache(self, send_task: Any) -> None:  # type: ignore[override]
        """Override: no write. Notify D to read this layer, wait for READ_DONE.

        P has a single kv_cache_group (main MLA); on P the indexer KV is written
        into ``dsa_k_cache`` (= kv_caches[main][2]) at the SAME slots as main
        MLA, so one P block list sources both legs. P sends one READ_READY per
        (layer, req) carrying only P's source block ids + the external req_id;
        D looks up its own destination blocks by req_id (pull model — D's block
        ids never travel to P).
        """
        # CRITICAL: wait for the KV (incl. the dsa_k scatter) to be fully written
        # to P HBM before notifying D to read, else D reads stale/unwritten data.
        if send_task.wait_event is not None:
            send_task.wait_event.synchronize()

        layer_name = send_task.layer_name
        layer_idx = send_task.layer_idx

        if not send_task.send_request:
            logger.warning(
                "MembPull: send_task.send_request EMPTY for layer %d (%s) — "
                "scheduler did not populate requests, skipping",
                layer_idx,
                layer_name,
            )
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

        # First call: send MF_META (P session + P layer addresses). P's
        # layer_metadata[main_name] carries [k, v, dsa_k] (3 tensors) because the
        # mooncake base registers the full 3-tuple, so D reaches the indexer
        # tensor as base_addrs[2] without a separate indexer layer entry.
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

        # One READ_READY per (layer, req): P sends only its OWN source block ids
        # (local_block_ids[0] — P's single group serves both main k/v and the
        # indexer dsa_k, which share slots on P) + the external req_id. D looks
        # up its destination blocks by req_id; P never handles D's block ids.
        # Verify-mode (VLLM_ASCEND_MF_VERIFY=1): log P's source-block sums so the
        # user can diff against D's destination sums (MFV D ...) per (layer, req).
        verify = os.environ.get("VLLM_ASCEND_MF_VERIFY") == "1"
        src_caches = getattr(self, "_source_kv_caches", None)
        all_ok = True
        for req_id, rm in send_task.send_request.items():
            local = rm.local_block_ids
            p_block_ids = local[0] if (local and len(local) > 0) else []
            ext_id = get_external_request_id(req_id)
            if layer_idx == 0:
                logger.info("MembPull P send READ_READY layer 0 req %s: p_blocks=%d", ext_id, len(p_block_ids))
            if verify and src_caches is not None and p_block_ids:
                src = src_caches.get(layer_name)
                if isinstance(src, (list, tuple)) and len(src) >= 3:
                    mk = src[0][p_block_ids].float().sum().item()
                    mv = src[1][p_block_ids].float().sum().item()
                    mi = src[2][p_block_ids].float().sum().item()
                    logger.info(
                        "MFV P layer %s req %s main_k=%.6f main_v=%.6f idx=%.6f",
                        layer_name,
                        ext_id,
                        mk,
                        mv,
                        mi,
                    )
            if not self._send_read_ready(path, encoder, layer_idx, layer_name, ext_id, p_block_ids):
                all_ok = False

        # Only signal "buffer safe to reuse" if D actually read the layer for
        # every request. On failure, withhold the signal so wait_for_layer_send
        # stalls (a visible failure) instead of silent corruption.
        if all_ok and 0 <= layer_idx < len(self.layer_send_done_events):
            self.layer_send_done_events[layer_idx].set()

        # After the last layer, send DONE_SENDING so D reports done_recving.
        # Send directly via ZMQ (can't use callback_func = send_done_send_signal
        # because it accesses req_meta.trans_count which is empty — we skipped
        # super()._transfer_kv_cache() that populates it).
        if layer_idx == self.total_layers - 1:
            for req_id, rm in send_task.send_request.items():
                if rm.chunk_finish:
                    external_req_id = get_external_request_id(req_id)
                    done_encoded = encoder.encode((DONE_SENDING_MSG, external_req_id, 0, ""))
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
                vllm_config.kv_transfer_config.kv_port + vllm_config.parallel_config.data_parallel_rank * tp_size
            )
            # P's memfabric session port. Offset 2*tp_size past P's ZMQ base:
            # with the single-host convention D_kv = P_kv + tp_size, an offset of
            # only tp_size (P_kv+tp_size+rank) would land P's unique_id exactly on
            # D's ZMQ ROUTER ports (D_kv+rank) → bind/listen collision. 2*tp_size
            # clears both P and D ZMQ ranges. unique_id = "P_IP:P_session_port".
            mf_session_port = side_channel_port + 2 * tp_size + tp_rank
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

    def start_load_kv(self, metadata: KVConnectorMetadata) -> None:
        """Override: in memfabric pull mode, skip the mooncake base producer
        branch (it builds push transfer mappings, needs ``remote_block_ids``
        which D no longer sends, and assumes symmetric P/D kv_cache_group
        structure → IndexError on P's 1 group). But preserve what
        ``_transfer_kv_cache`` relies on:

        * reset ``self.current_layer`` — the per-step layer counter that
          ``save_kv_layer`` increments; without the reset it drifts to
          ``>= total_layers`` and every request after the first is skipped.
        * adjust ``remote_port`` by ``tp_rank`` — D's ROUTER binds
          ``side_channel_port + tp_rank`` (one per rank) but D advertises the
          base port, so each P rank must send to ``base + tp_rank``.

        ``remote_host`` / ``local_block_ids`` are already correct from
        ``build_connector_meta`` and need no transformation (P's single group
        is at kernel granularity, scale 1)."""
        if self._backend == BACKEND_MEMFABRIC:
            self.current_layer = 0
            for req_meta in getattr(metadata, "requests", {}).values():
                if req_meta.remote_port is None:
                    continue
                remote_tp_size = req_meta.remote_tp_size or self.tp_size
                tp_ratio = max(1, self.tp_size // remote_tp_size)
                req_meta.remote_port = req_meta.remote_port + self.tp_rank // tp_ratio
            return
        super().start_load_kv(metadata)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # memfabric pull mode only (mooncake staging path removed).
        assert self._backend == BACKEND_MEMFABRIC, "SFAPDCpuOffloadConnector P side supports memfabric pull only."
        # Swap in _MembPullSendingThread (notifies D to read, does NOT push).
        # Set its p_session before the base class constructs it.
        orig = _mlc.KVCacheSendingLayerThread
        _mlc.KVCacheSendingLayerThread = _MembPullSendingThread  # type: ignore[assignment]
        # Patch __init__ to inject p_session (hack: base class doesn't pass it)
        _orig_init = _MembPullSendingThread.__init__

        def _patched_init(self_inner, *a, **kw):
            _orig_init(self_inner, *a, **kw)
            # The sending thread is constructed AFTER the engine is built
            # (base-class order: register_buffer then create thread), so
            # global_te._unique_id now holds the unique_id memfabric ACTUALLY
            # registered under (set in _build_memfabric). Prefer it over the
            # pre-computed _mf_unique_id — memfabric picks its own rpc port
            # (engine.get_rpc_port()), so the computed id is NOT the session
            # D must read from.
            self_inner.p_session = global_te._unique_id or self._mf_unique_id

        _MembPullSendingThread.__init__ = _patched_init  # type: ignore[assignment]
        try:
            super().register_kv_caches(kv_caches)
        finally:
            _mlc.KVCacheSendingLayerThread = orig  # type: ignore[assignment]
            _MembPullSendingThread.__init__ = _orig_init  # type: ignore[assignment]
        # Stash source tensors on the sending thread for env-gated verify
        # checksums (VLLM_ASCEND_MF_VERIFY=1): P sums its source blocks so
        # the user can compare against D's destination sums in the logs.
        self.kv_send_layer_thread._source_kv_caches = kv_caches
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
