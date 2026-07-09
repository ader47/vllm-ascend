# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Worker side of the PD-disaggregated SFA connector (memfabric pull mode).

D (``kv_consumer``): composes :class:`SFAKVOffloadWorker` for the LRU-resident
H2D load path + CPU pool, and runs a memfabric pull read thread. P notifies D
per layer via READ_READY_BATCH; D reads indexer KV into HBM and main MLA KV into the
CPU pool (the partial last block stays in HBM until decode fills it, then the
B1 offload path copies it to CPU).

P (``kv_producer``): registers its HBM KV with memfabric and runs a pull-mode
sending thread that notifies D to read (no RDMA push). A per-layer
send-completion event gates P's KV buffer reuse.
"""

from __future__ import annotations

import math
import re
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    get_shared_layer_transfer_events,
    get_shared_layer_transfer_pending_events,
    set_shared_layer_transfer_events,
    set_shared_layer_transfer_pending_events,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    GET_META_MSG,
    MF_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    LayerMetadata,
    SendTask,
    SfaPDAgentMetadata,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.send_thread import (
    MembPullSendingThread,
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
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    collect_storage_merged_register_regions,
    validate_register_region_count,
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


def _coalesce_desc(
    peer: np.ndarray,
    local: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge adjacent descriptors whose peer AND local addresses are both
    contiguous into one descriptor of summed length. The memfabric engine
    issues one transfer per descriptor, so coalescing a run of R contiguous
    blocks into one descriptor cuts both descriptor count and per-op overhead.
    Lossless: the merged descriptor covers exactly the same bytes."""
    n = peer.shape[0]
    if n <= 1:
        return peer, local, length
    contiguous = (peer[1:] == peer[:-1] + length[:-1]) & (local[1:] == local[:-1] + length[:-1])
    if contiguous.all():
        return peer[:1], local[:1], np.array([int(length.sum())], dtype=np.int64)
    run_start = np.concatenate(([0], np.nonzero(~contiguous)[0] + 1))
    run_end = np.append(run_start[1:] - 1, n - 1)
    cum = np.cumsum(length)
    merged_len = cum[run_end] - cum[run_start] + length[run_start]
    return peer[run_start], local[run_start], merged_len


def _resolve_kv_transfer_backend(vllm_config: VllmConfig) -> str:
    """Pick the KV transfer backend.

    ``kv_connector_extra_config["transfer_backend"]`` overrides the
    ``VLLM_ASCEND_KV_TRANSFER_BACKEND`` env var (default ``mooncake``).
    """
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
    return extra.get("transfer_backend") or envs.VLLM_ASCEND_KV_TRANSFER_BACKEND


class SFAPDCpuOffloadConsumerWorker:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self.tp_rank = get_tensor_model_parallel_rank()  # TP-local rank for the per-rank ZMQ port
        self.side_channel_host = get_ip()
        # D-side ZMQ control-plane base port; each TP rank listens on base + tp_rank.
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
        # External req ids whose DONE signal arrived before request_map
        # was seeded (see get_finished). Retried every step until mapped.
        self._pending_done: set[str] = set()

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------
    def _ensure_engine(self):
        if self.engine is None:
            backend = _resolve_kv_transfer_backend(self.vllm_config)
            if backend == BACKEND_MEMFABRIC:
                # unique_id/store_url are derived by _build_memfabric from
                # engine.get_rpc_port() — no caller-computed port needed.
                global_te.configure(
                    backend=BACKEND_MEMFABRIC,
                    role=MEMFABRIC_ROLE_DECODE,
                    device_id=torch.npu.current_device(),
                )
            self.engine = global_te.get_transfer_engine(self.side_channel_host, None)
        return self.engine

    # ------------------------------------------------------------------
    # D side (kv_consumer) — this class is only instantiated for consumers;
    # Producers use :class:`SFAPDCpuOffloadProducerWorker`.
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Prepare D-side indexer HBM + main MLA CPU-pool destinations.

        The sfa model runner hands a 5-tuple per layer:
        ``(k_nope, v_rope, dsa_k_indexer, topk_buf_k, topk_buf_v)``.
        """
        # --- D side: compose the SFA worker for LRU load + CPU pool ---
        self.sfa_worker = SFAKVOffloadWorker(self.vllm_config, self.use_layerwise, self.kv_cache_config)
        # SFA worker allocates k_caches_cpu/v_caches_cpu + LRU buffers here.
        self.sfa_worker.register_kv_caches(kv_caches)

        # CPU pool owned by the composed SFA worker. P fills it via memfabric
        # pull (D reads main MLA KV from P HBM into this CPU pool).
        assert self.sfa_worker.k_caches_cpu is not None, "Composed SFA worker did not allocate the CPU pool"
        k_caches_cpu = self.sfa_worker.k_caches_cpu
        v_caches_cpu = self.sfa_worker.v_caches_cpu

        # Part A: D's main MLA HBM k/v tensors (group1 paged cache) — the partial
        # last block lands here instead of the CPU pool. Keyed by main layer name
        # (the 5-tuple layers); tuple[0]=k_nope, tuple[1]=v_rope.
        self._hbm_kv = {
            n: (t[0], t[1])
            for n, t in kv_caches.items()
            if isinstance(t, (list, tuple)) and len(t) in (5, 6)
        }

        # memfabric pull mode only (the mooncake staging path has been removed).
        assert _resolve_kv_transfer_backend(self.vllm_config) == BACKEND_MEMFABRIC, (
            "SFAPDCpuOffloadConnector D side supports memfabric pull only (set transfer_backend=memfabric)."
        )
        self._register_memfabric_pull(kv_caches, k_caches_cpu, v_caches_cpu)

    # -- D-side forwards to the composed SFA worker (LRU load path) --
    def start_load_kv(self, metadata: KVConnectorMetadata):
        assert self.sfa_worker is not None
        # B1: reset decode offload — now driven by sfa_worker.process_layer_data
        # (num_new_offload_blocks > 0 triggers it in sfa_worker.start_load_kv).
        # Seed external->internal request id map for get_finished, and store D's
        # own destination blocks per request (keyed by external id, which is what
        # P sends in READ_READY_BATCH). The scheduler includes remote-prefill requests
        # here (even while async-waiting) so both exist before P's signal arrives.
        for req in getattr(metadata, "requests", []):
            req_id = getattr(req, "req_id", None)
            if req_id is not None:
                ext_id = get_external_request_id(req_id)
                self.request_map[ext_id] = req_id
                indexer_ids = list(getattr(req, "block_ids_indexer", []) or [])
                main_ids = list(getattr(req, "block_ids_cpu", []) or [])
                num_full = getattr(req, "num_full", 0) or 0
                partial_hbm_bid = getattr(req, "partial_hbm_bid", None)
                self._dest_blocks_by_req[ext_id] = (indexer_ids, main_ids, num_full, partial_hbm_bid)
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull D stored dest blocks req %s: indexer_hbm_ids=%s, "
                        "main_cpu_ids=%s, num_full=%s, partial_hbm=%s",
                        ext_id,
                        indexer_ids,
                        main_ids,
                        num_full,
                        partial_hbm_bid,
                    )
        # Refresh the per-req CPU-block count (Phase 3 source of truth) and
        # forward the load kickoff to the SFA worker (which also builds offload
        # tasks via process_layer_data when num_new_offload_blocks > 0).
        self._refresh_cpu_blocks_by_req(metadata)
        self.sfa_worker.start_load_kv(metadata)

    def set_req_ids(self, req_ids: list):
        if self.sfa_worker is not None:
            self.sfa_worker.set_req_ids(req_ids)

    def prepare_lru_resident_and_load(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        topk_indices: torch.Tensor,
        current_slots: torch.Tensor,
        req_ids: torch.Tensor,
        token_to_req: torch.Tensor | None = None,
        capturing: bool = False,
    ) -> bool:
        assert self.sfa_worker is not None
        return self.sfa_worker.prepare_lru_resident_and_load(
            layer_name,
            num_tokens,
            num_reqs,
            topk_indices,
            current_slots,
            req_ids,
            token_to_req,
            capturing,
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        # B1: offload decode-filled main MLA blocks HBM -> CPU pool via the
        # composed SFA worker's async background thread (process_layer_data built
        # the per-layer tasks in start_load_kv from num_new_offload_blocks).
        if self.sfa_worker is not None:
            self.sfa_worker.save_kv_layer(layer_name)

    def wait_for_save(self):
        if self.sfa_worker is not None:
            self.sfa_worker.wait_for_save()

    def _cleanup_request_state(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            ext_id = get_external_request_id(req_id)
            self._cpu_blocks_by_req.pop(req_id, None)
            self.request_map.pop(ext_id, None)
            self._dest_blocks_by_req.pop(ext_id, None)
            self._pending_done.discard(ext_id)

    def get_finished(self, finished_req_ids: set[str] | None = None) -> tuple[set[str], set[str]]:
        done_recving: set[str] = set()

        # memfabric pull mode: done comes from MembPullReadThread
        if hasattr(self, "_mf_read_thread") and self._mf_read_thread is not None:
            done = self._mf_read_thread.get_and_clear_done()
            still_pending: set[str] = set()
            for ext_id in done | self._pending_done:
                internal = self.request_map.get(ext_id)
                if internal is not None:
                    done_recving.add(internal)
                else:
                    still_pending.add(ext_id)
            self._pending_done = still_pending

            if done or done_recving or self._pending_done:
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull D get_finished: done_ext=%s, "
                        "done_recving_internal=%s, pending_done_ext=%s",
                        done,
                        done_recving,
                        self._pending_done,
                    )
        # else: read thread not up yet -> nothing finished (done_recving empty).

        # Purge scheduler-finished req state AFTER resolving this step's
        # done signals against request_map. Doing it at the top would pop
        # request_map[ext_id] and discard _pending_done[ext_id] before the
        # resolution loop above, leaking any finished req whose DONE arrives in
        # the same step (unmappable -> stuck in _pending_done forever).
        if finished_req_ids:
            self._cleanup_request_state(finished_req_ids)

        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        """Per-req actual main-MLA CPU-block count for the solution-1 threshold."""
        if self.sfa_worker is None:
            return None
        result = {rid: self._cpu_blocks_by_req[rid] for rid in req_ids if rid in self._cpu_blocks_by_req}
        return result or None

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

        def _offload_tuple_len(v: object) -> int:
            return len(v) if isinstance(v, (list, tuple)) else 1

        main_names = [
            n for n, v in kv_caches.items() if _offload_tuple_len(v) in (5, 6)
        ]
        main_by_layer_idx = {_layer_idx(name): name for name in main_names}
        main_names = [main_by_layer_idx[_layer_idx(name)] for name in indexer_names]

        # Store layer info for MembPullReadThread
        self._indexer_names = indexer_names
        self._main_names = main_names
        self._main_name_to_idx = {n: i for i, n in enumerate(main_names)}
        self._cpu_pools = list(zip(k_caches_cpu, v_caches_cpu))
        self._indexer_tensors = []
        self._indexer_scale_tensors: list[torch.Tensor | None] = []
        for main_name in main_names:
            main_tuple = list(kv_caches[main_name])
            self._indexer_tensors.append(main_tuple[2])  # dsa_k_indexer
            self._indexer_scale_tensors.append(
                main_tuple[5] if len(main_tuple) >= 6 else None
            )

        # Build layer_metadata (D's local addresses, for compatibility)
        for pool_idx, (iname, mname) in enumerate(zip(indexer_names, main_names)):
            indexer_t = self._indexer_tensors[pool_idx]
            indexer_scale_t = self._indexer_scale_tensors[pool_idx]
            k_cpu, v_cpu = self._cpu_pools[pool_idx]
            indexer_addrs = [indexer_t.data_ptr()]
            indexer_block_lens = [indexer_t.element_size() * math.prod(indexer_t.shape[1:])]
            indexer_block_scales = [indexer_t.shape[0] // num_blocks if num_blocks else 1]
            if indexer_scale_t is not None:
                indexer_addrs.append(indexer_scale_t.data_ptr())
                indexer_block_lens.append(
                    indexer_scale_t.element_size() * math.prod(indexer_scale_t.shape[1:])
                )
                indexer_block_scales.append(
                    indexer_scale_t.shape[0] // num_blocks if num_blocks else 1
                )
            self.layer_metadata[iname] = LayerMetadata(
                tensor_group_idx=[_INDEXER_GROUP_IDX],
                kv_caches_base_addr=indexer_addrs,
                block_len=indexer_block_lens,
                block_size_scale=indexer_block_scales,
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
    """D-side thread for memfabric pull: receives READ_READY_BATCH from P,
    reads KV from P's HBM via batch_transfer_sync_read, then replies with
    READ_DONE or READ_FAILED.

    indexer → D HBM (direct), main MLA → D CPU pool DRAM (local target,
    no registration needed per memfabric read semantics).
    """

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        engine: Any,
        worker: SFAPDCpuOffloadConsumerWorker,
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
        self._lock = threading.Lock()
        self._host = get_ip()

    def get_and_clear_done(self) -> set[str]:
        with self._lock:
            d = self._done_requests
            self._done_requests = set()
            return d

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
                        for layer_name, layer_meta in self._p_layer_meta.items():
                            if envs.VLLM_ASCEND_SFA_DEBUG:
                                logger.info(
                                    "MembPull D recv MF_META layer=%s: base_addrs=%s, "
                                    "block_len=%s, block_size_scale=%s",
                                    layer_name,
                                    layer_meta.get("base_addrs"),
                                    layer_meta.get("block_len"),
                                    layer_meta.get("block_size_scale"),
                                )
                        sock.send_multipart((identity, b"", b"ACK"))

                    elif msg_type == READ_READY_BATCH:
                        layer_idx = msg[1]
                        layer_name = msg[2]
                        read_reqs = [(entry[0], list(entry[1])) for entry in msg[3]]
                        done_ext_ids = list(msg[4]) if len(msg) > 4 else []
                        if envs.VLLM_ASCEND_SFA_DEBUG:
                            logger.info(
                                "MembPull D recv READ_READY_BATCH: layer=%d (%s), reqs=%d, done_reqs=%d",
                                layer_idx,
                                layer_name,
                                len(read_reqs),
                                len(done_ext_ids),
                            )
                        try:
                            if read_reqs:
                                self._do_read_batch(layer_name, read_reqs)
                            sock.send_multipart((identity, b"", encoder.encode((READ_DONE, layer_idx))))
                            if envs.VLLM_ASCEND_SFA_DEBUG:
                                logger.info(
                                    "MembPull D sent READ_DONE: layer=%d (%s), reqs=%d, done_reqs=%d",
                                    layer_idx,
                                    layer_name,
                                    len(read_reqs),
                                    len(done_ext_ids),
                                )
                        except Exception as e:
                            logger.error(
                                "MembPull batch read failed for layer %d (%s), reqs=%d: %s",
                                layer_idx,
                                layer_name,
                                len(read_reqs),
                                e,
                            )
                            payload = encoder.encode((READ_FAILED, layer_idx, str(e)))
                            sock.send_multipart((identity, b"", payload))
                        # Mark done outside the try so a last-layer read failure
                        # (READ_FAILED, sent above) still unblocks get_finished.
                        if done_ext_ids:
                            with self._lock:
                                self._done_requests.update(done_ext_ids)

                    elif msg_type == GET_META_MSG:
                        # D responds with its own metadata (for compatibility)
                        meta_bytes = encoder.encode(
                            SfaPDAgentMetadata(
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

    def _resolve_read_layer(self, layer_name: str) -> dict[str, Any] | None:
        """Resolve all layer-constant read state once per layer. The old code
        recomputed this per request (a linear scan of _main_names, an offload_id
        lookup, a p_meta lookup, and all base/length/scale derivation per req).
        Returns None if the whole layer must be skipped (logged once); per-req
        skips stay in _build_req_descriptors."""
        w = self._worker
        pool_idx = w._main_name_to_idx.get(layer_name)
        if pool_idx is None:
            logger.warning("MembPull _do_read: layer %s not in main names, skip", layer_name)
            return None
        # Main-MLA CPU pool is indexed by the resident-load offload_id, NOT pool_idx
        # (_cpu_pools follows offload_layer_names order; _main_names/indexer tensors
        # follow the indexer group). pool_idx stays correct for the indexer tensor.
        offload_id = w.sfa_worker._get_offload_layer_id(layer_name)
        if pool_idx != offload_id and envs.VLLM_ASCEND_SFA_DEBUG:
            logger.warning(
                "MembPull _do_read: layer-order mismatch for %s — pull _main_names "
                "idx=%d != resident offload_id=%d; main MLA was being written to the "
                "wrong CPU-pool slot. Using offload_id.",
                layer_name, pool_idx, offload_id,
            )
        p_meta = self._p_layer_meta.get(layer_name)
        if p_meta is None:
            logger.warning(
                "MembPull _do_read: layer %s not in P layer_meta (MF_META not received? have %d layers), skip",
                layer_name,
                len(self._p_layer_meta),
            )
            return None
        p_base_addrs = p_meta["base_addrs"]
        p_block_len = p_meta["block_len"]

        k_cpu, v_cpu = w._cpu_pools[offload_id]
        hbm_kv = w._hbm_kv.get(layer_name)
        if hbm_kv is None:
            k_hbm_ptr = v_hbm_ptr = None
        else:
            k_hbm_ptr, v_hbm_ptr = hbm_kv[0].data_ptr(), hbm_kv[1].data_ptr()

        # Indexer leg constants (None if the layer can't do the indexer leg).
        indexer = None
        if len(p_base_addrs) < 3:
            logger.error(
                "MembPull indexer: P layer_meta for %s has %d tensors, need >=3 "
                "(dsa_k must be registered as the main layer's tensor [2]); skip indexer leg",
                layer_name,
                len(p_base_addrs),
            )
        else:
            p_dsa_len = p_block_len[2]
            d_indexer = w._indexer_tensors[pool_idx]
            d_dsa_len = d_indexer.element_size() * math.prod(d_indexer.shape[1:])
            if p_dsa_len <= 0 or d_dsa_len % p_dsa_len != 0:
                logger.error(
                    "MembPull indexer %s: D dsa_k block_len=%d not a multiple of P=%d; skip indexer leg",
                    layer_name,
                    d_dsa_len,
                    p_dsa_len,
                )
            else:
                indexer = {
                    "p_dsa_base": p_base_addrs[2],
                    "p_dsa_len": p_dsa_len,
                    "d_base": d_indexer.data_ptr(),
                    "d_dsa_len": d_dsa_len,
                    "scale": d_dsa_len // p_dsa_len,
                    "shape": tuple(d_indexer.shape),
                }

        scale = None
        scale_tensor = (
            w._indexer_scale_tensors[pool_idx]
            if pool_idx < len(w._indexer_scale_tensors)
            else None
        )
        if scale_tensor is not None:
            if len(p_base_addrs) < 4:
                scale = {"error": "p_addr_mismatch", "p_n": len(p_base_addrs)}
            else:
                p_scale_len = p_block_len[3]
                d_scale_len = (
                    scale_tensor.element_size()
                    * math.prod(scale_tensor.shape[1:])
                )
                if p_scale_len <= 0 or d_scale_len % p_scale_len != 0:
                    scale = {
                        "error": "layout_mismatch",
                        "p_scale_len": p_scale_len,
                        "d_scale_len": d_scale_len,
                    }
                else:
                    scale = {
                        "p_scale_base": p_base_addrs[3],
                        "p_scale_len": p_scale_len,
                        "d_scale_base": scale_tensor.data_ptr(),
                        "d_scale_len": d_scale_len,
                        "scale_factor": d_scale_len // p_scale_len,
                    }

        return {
            "layer_name": layer_name,
            "pool_idx": pool_idx,
            "offload_id": offload_id,
            "p_k_base": p_base_addrs[0],
            "p_v_base": p_base_addrs[1],
            "p_k_len": p_block_len[0],
            "p_v_len": p_block_len[1],
            "k_cpu_ptr": k_cpu.data_ptr(),
            "v_cpu_ptr": v_cpu.data_ptr(),
            "k_hbm_ptr": k_hbm_ptr,
            "v_hbm_ptr": v_hbm_ptr,
            "indexer": indexer,
            "scale": scale,
        }

    def _build_req_descriptors(
        self,
        layer: dict[str, Any],
        ext_req_id: str,
        p_block_ids: list[int],
        want_info: bool,
    ) -> tuple[list[int], list[int], list[int], dict[str, Any] | None]:
        """Build memfabric read descriptors for one request using layer-constant
        state from _resolve_read_layer. Empty ptrs (+ None info) for per-req
        skips (no dest / empty p_block_ids / nothing to transfer)."""
        w = self._worker
        layer_name = layer["layer_name"]

        dest = w._dest_blocks_by_req.get(ext_req_id)
        if dest is None:
            logger.warning(
                "MembPull _do_read: no dest blocks on D for req %s (layer %s), skip",
                ext_req_id,
                layer_name,
            )
            return [], [], [], None
        d_indexer_ids, d_main_ids, num_full, partial_hbm_bid = dest  # (npu, cpu, num_full, partial_hbm_bid)
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "MembPull D resolve dest: layer=%s, req=%s, p_block_ids=%s, "
                "d_main_cpu_ids=%s, d_indexer_hbm_ids=%s, num_full=%s, partial_hbm=%s",
                layer_name,
                ext_req_id,
                p_block_ids,
                d_main_ids,
                d_indexer_ids,
                num_full,
                partial_hbm_bid,
            )
        if not p_block_ids:
            logger.warning("MembPull _do_read: empty p_block_ids for %s, skip", layer_name)
            return [], [], [], None

        p_k_base, p_v_base = layer["p_k_base"], layer["p_v_base"]
        p_k_len, p_v_len = layer["p_k_len"], layer["p_v_len"]

        peer_chunks: list[np.ndarray] = []
        local_chunks: list[np.ndarray] = []
        length_chunks: list[np.ndarray] = []
        n_main = 0
        n_indexer = 0

        # Main MLA leg: first `num_full` P blocks FULL -> D CPU pool; optional
        # last PARTIAL -> D HBM.
        full_p_blocks = p_block_ids[:num_full]
        n_full = min(len(full_p_blocks), len(d_main_ids))
        if n_full:
            full_p = np.array(full_p_blocks[:n_full], dtype=np.int64)
            d_main = np.array(d_main_ids[:n_full], dtype=np.int64)
            len_k = np.full(n_full, p_k_len, dtype=np.int64)
            len_v = np.full(n_full, p_v_len, dtype=np.int64)
            # Coalesce the k and v legs separately (each is in block order); a run
            # of contiguous blocks on both P and D becomes one descriptor.
            cp, cl, clen = _coalesce_desc(p_k_base + full_p * p_k_len,
                                          layer["k_cpu_ptr"] + d_main * p_k_len, len_k)
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)
            cp, cl, clen = _coalesce_desc(p_v_base + full_p * p_v_len,
                                          layer["v_cpu_ptr"] + d_main * p_v_len, len_v)
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)
            n_main = n_full
        if partial_hbm_bid is not None and num_full < len(p_block_ids):
            k_hbm_ptr = layer["k_hbm_ptr"]
            if k_hbm_ptr is None:
                logger.warning("MembPull _do_read: no HBM k/v for partial block %s, skip partial", layer_name)
            else:
                partial_p_bid = p_block_ids[num_full]
                peer_chunks.append(np.array(
                    [p_k_base + partial_p_bid * p_k_len,
                     p_v_base + partial_p_bid * p_v_len], dtype=np.int64))
                local_chunks.append(np.array(
                    [k_hbm_ptr + partial_hbm_bid * p_k_len,
                     layer["v_hbm_ptr"] + partial_hbm_bid * p_v_len], dtype=np.int64))
                length_chunks.append(np.array([p_k_len, p_v_len], dtype=np.int64))
                n_main += 1

        # Indexer leg: each D block splits into `scale` P-sized sub-addresses;
        # extra sub-addresses beyond len(p_block_ids) are discarded.
        idx = layer["indexer"]
        if idx is not None and d_indexer_ids:
            p_dsa_base, p_dsa_len = idx["p_dsa_base"], idx["p_dsa_len"]
            d_base, d_dsa_len, scale = idx["d_base"], idx["d_dsa_len"], idx["scale"]
            d_idx_arr = np.array(d_indexer_ids, dtype=np.int64)
            offs = np.arange(scale, dtype=np.int64)
            d_sub = (d_base + d_idx_arr[:, None] * d_dsa_len + offs * p_dsa_len).reshape(-1)
            n_pairs = min(len(p_block_ids), d_sub.shape[0])
            p_idx = np.array(p_block_ids[:n_pairs], dtype=np.int64)
            logger.debug(
                "MembPull indexer %s: p_dsa_len=%d d_dsa_len=%d scale=%d, "
                "D dsa_k shape=%s, D_sub=%d P_blocks=%d",
                layer_name, p_dsa_len, d_dsa_len, scale, idx["shape"],
                int(d_sub.shape[0]), len(p_block_ids),
            )
            cp, cl, clen = _coalesce_desc(
                p_dsa_base + p_idx * p_dsa_len,
                d_sub[:n_pairs],
                np.full(n_pairs, p_dsa_len, dtype=np.int64),
            )
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)
            n_indexer = n_pairs

        # Indexer scale leg (LIC8): P non-offload C8 tensor [3] → D six-tuple [5].
        # Fail loud on any layout mismatch: silently skipping the scale leg would
        # leave D's [5] stale/uninitialized and corrupt indexer dequant.
        scale = layer.get("scale")
        if scale is not None and d_indexer_ids:
            if "error" in scale:
                if scale["error"] == "p_addr_mismatch":
                    raise RuntimeError(
                        f"MembPull indexer scale {layer_name}: D is LIC8 (has scale "
                        f"tensor) but P exposed only {scale['p_n']} base addrs "
                        f"(no scale leg) — P/D LIC8 config mismatch."
                    )
                raise RuntimeError(
                    f"MembPull indexer scale {layer_name}: D scale block_len="
                    f"{scale['d_scale_len']} not a multiple of P={scale['p_scale_len']} — "
                    f"scale layout mismatch; refusing to transfer to avoid silent "
                    f"stale-scale corruption."
                )
            p_scale_base = scale["p_scale_base"]
            p_scale_len = scale["p_scale_len"]
            d_scale_base = scale["d_scale_base"]
            d_scale_len = scale["d_scale_len"]
            scale_factor = scale["scale_factor"]
            d_scale_sub_addrs: list[int] = []
            for d_bid in d_indexer_ids:
                d_block_base = d_scale_base + d_bid * d_scale_len
                for off in range(scale_factor):
                    d_scale_sub_addrs.append(d_block_base + off * p_scale_len)
            n_scale_pairs = min(len(p_block_ids), len(d_scale_sub_addrs))
            if n_scale_pairs:
                p_scale_arr = np.array(p_block_ids[:n_scale_pairs], dtype=np.int64)
                d_scale_arr = np.array(
                    d_scale_sub_addrs[:n_scale_pairs], dtype=np.int64
                )
                cp, cl, clen = _coalesce_desc(
                    p_scale_base + p_scale_arr * p_scale_len,
                    d_scale_arr,
                    np.full(n_scale_pairs, p_scale_len, dtype=np.int64),
                )
                peer_chunks.append(cp)
                local_chunks.append(cl)
                length_chunks.append(clen)

        if not peer_chunks:
            logger.warning(
                "MembPull _do_read: nothing to transfer for %s (main=%d, indexer=%d)",
                layer_name,
                n_main,
                n_indexer,
            )
            return [], [], [], None

        peer_ptrs = np.concatenate(peer_chunks).tolist()
        local_ptrs = np.concatenate(local_chunks).tolist()
        lengths = np.concatenate(length_chunks).tolist()

        # Build read_info only when a consumer is active (_log_read_result is
        # gated on VLLM_ASCEND_MF_VERIFY / VLLM_ASCEND_SFA_DEBUG); otherwise this
        # dict would be allocated per req per layer and immediately discarded.
        info = None
        if want_info:
            info = {
                "layer_name": layer_name,
                "ext_req_id": ext_req_id,
                "pool_idx": layer["pool_idx"],
                "offload_id": layer["offload_id"],
                "d_main_ids": d_main_ids,
                "d_indexer_ids": d_indexer_ids,
                "partial_hbm_bid": partial_hbm_bid,
                "n_main": n_main,
                "n_indexer": n_indexer,
                "num_transfers": len(local_ptrs),
                # pre-coalesce descriptor count (2 per main block incl. partial + 1 per indexer pair)
                "atomic_transfers": 2 * n_main + n_indexer,
            }
        return local_ptrs, peer_ptrs, lengths, info

    def _log_read_result(self, read_info: dict[str, Any]) -> None:
        # Verify-mode (VLLM_ASCEND_MF_VERIFY=1): log D's destination sums so the
        # user can diff against P's source sums (MFV P ...). main_k/main_v should
        # match P exactly (1:1 byte copy); idx_post is the full D indexer block
        # sum (D blocks are 4× P's and only len(p_block_ids)/scale slots written,
        # so it won't equal P's idx — use it only as a "data landed" check).
        w = self._worker
        layer_name = read_info["layer_name"]
        ext_req_id = read_info["ext_req_id"]
        pool_idx = read_info["pool_idx"]
        offload_id = read_info["offload_id"]
        d_main_ids = read_info["d_main_ids"]
        d_indexer_ids = read_info["d_indexer_ids"]
        partial_hbm_bid = read_info["partial_hbm_bid"]
        if envs.VLLM_ASCEND_MF_VERIFY:
            try:
                k_cpu, v_cpu = w._cpu_pools[offload_id]
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
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "MembPull D finished read: layer=%s, req=%s, pool_idx=%d, "
                "main=%d/%d blocks, indexer=%d/%d blocks (scale split), transfers=%d",
                layer_name,
                ext_req_id,
                pool_idx,
                read_info["n_main"],
                len(d_main_ids),
                read_info["n_indexer"],
                len(d_indexer_ids),
                read_info["num_transfers"],
            )

    def _do_read_batch(
        self,
        layer_name: str,
        read_reqs: list[tuple[str, list[int]]],
    ) -> None:
        """Read all requests for one layer in a single memfabric batch."""
        if self._p_session is None:
            raise RuntimeError("MF_META not received before READ_READY_BATCH")

        # Resolve layer-constant state once (pool idx, offload id, p_meta, base
        # addrs/lengths, scale, CPU/HBM pool ptrs); was recomputed per req before.
        layer = self._resolve_read_layer(layer_name)
        if layer is None:
            return  # layer-level skip, already logged once

        want_info = bool(envs.VLLM_ASCEND_MF_VERIFY or envs.VLLM_ASCEND_SFA_DEBUG)
        all_local_ptrs: list[int] = []
        all_peer_ptrs: list[int] = []
        all_lengths: list[int] = []
        read_infos: list[dict[str, Any]] = []
        for ext_req_id, p_block_ids in read_reqs:
            try:
                local_ptrs, peer_ptrs, lengths, read_info = self._build_req_descriptors(
                    layer, ext_req_id, p_block_ids, want_info
                )
            except Exception as e:  # noqa: BLE001 - keep other reqs in the layer moving
                logger.error(
                    "MembPull prepare batch read failed for layer %s req %s: %s",
                    layer_name,
                    ext_req_id,
                    e,
                )
                continue
            if not local_ptrs:
                continue
            all_local_ptrs.extend(local_ptrs)
            all_peer_ptrs.extend(peer_ptrs)
            all_lengths.extend(lengths)
            if want_info:
                read_infos.append(read_info)

        if not all_local_ptrs:
            logger.warning(
                "MembPull _do_read_batch: nothing to transfer for layer %s, reqs=%d",
                layer_name,
                len(read_reqs),
            )
            return

        if envs.VLLM_ASCEND_SFA_DEBUG:
            atomic_total = sum(r.get("atomic_transfers", 0) for r in read_infos)
            logger.info(
                "MembPull D start batched memfabric read: layer=%s, reqs=%d, "
                "p_session=%s, transfers=%d (coalesced from %d)",
                layer_name,
                len(read_infos),
                self._p_session,
                len(all_local_ptrs),
                atomic_total,
            )
        ret = self.engine.batch_transfer_sync_read(self._p_session, all_local_ptrs, all_peer_ptrs, all_lengths)
        if ret != 0:
            raise RuntimeError(f"memfabric batch read failed for layer {layer_name}, ret={ret}")
        for read_info in read_infos:
            self._log_read_result(read_info)


class SFAPDCpuOffloadProducerWorker:
    """P-side worker for memfabric pull mode.

    It registers P's local KV tensors with memfabric and runs a pull-mode
    sending thread. P never pushes KV; it sends READ_READY_BATCH messages so D
    can read P's source blocks and reply with READ_DONE / READ_FAILED.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self._backend = _resolve_kv_transfer_backend(vllm_config)
        if self._backend == BACKEND_MEMFABRIC:
            global_te.configure(
                backend=BACKEND_MEMFABRIC,
                role=MEMFABRIC_ROLE_PREFILL,
                device_id=torch.npu.current_device(),
            )
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + self.dp_rank * self.tp_size
        )
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        set_shared_layer_transfer_events([threading.Event() for _ in range(self.total_layers)])
        set_shared_layer_transfer_pending_events([threading.Event() for _ in range(self.total_layers)])
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()
        self.kv_cache_specs = [
            group_spec.kv_cache_spec
            for group_spec in self.kv_cache_config.kv_cache_groups
        ]
        self.block_size = [spec.block_size for spec in self.kv_cache_specs]
        self.num_kv_cache_groups = len(self.kv_cache_specs)
        self.use_mla = self.vllm_config.model_config.use_mla
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.index_to_name: defaultdict[int, list[str]] = defaultdict(list)
        self.current_layer = 0
        self.kv_send_layer_thread: MembPullSendingThread | None = None
        self.layer_send_done_events: list[threading.Event] | None = None

    def get_finished(self) -> tuple[set[str], set[str]]:
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def set_req_ids(self, req_ids: list) -> None:
        return

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        return None

    def update_decoder_info(self, req_id: str, req_meta: Any) -> Any:
        """Override: in memfabric pull mode, P does NOT need D's metadata
        (P is not pushing to D — D reads from P). Skip GET_META entirely
        to avoid flooding D's ROUTER with 61 unnecessary requests that
        delay MF_META / READ_READY_BATCH."""
        if self._backend == BACKEND_MEMFABRIC:
            return req_meta
        raise RuntimeError("SFAPDCpuOffloadConnector P side supports memfabric pull only.")

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
            for req_id, req_meta in getattr(metadata, "requests", {}).items():
                if req_meta.remote_port is None:
                    continue
                remote_tp_size = req_meta.remote_tp_size or self.tp_size
                tp_ratio = max(1, self.tp_size // remote_tp_size)
                old_remote_port = req_meta.remote_port
                req_meta.remote_port = req_meta.remote_port + self.tp_rank // tp_ratio
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull P start_load_kv req %s: remote_host=%s, "
                        "remote_port=%s->%s, tp_rank=%s, tp_ratio=%s, local_block_ids=%s, "
                        "chunk_finish=%s, local_computed_tokens=%s, local_transed_tokens=%s",
                        req_id,
                        req_meta.remote_host,
                        old_remote_port,
                        req_meta.remote_port,
                        self.tp_rank,
                        tp_ratio,
                        req_meta.local_block_ids,
                        req_meta.chunk_finish,
                        req_meta.local_computed_tokens,
                        req_meta.local_transed_tokens,
                    )
            return
        raise RuntimeError("SFAPDCpuOffloadConnector P side supports memfabric pull only.")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # memfabric pull mode only (mooncake staging path removed).
        assert self._backend == BACKEND_MEMFABRIC, "SFAPDCpuOffloadConnector P side supports memfabric pull only."
        layer2group_ids: dict[str, int] = {}
        for group_idx, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                layer2group_ids[layer_name] = group_idx

        num_blocks = self.kv_cache_config.num_blocks
        for layer_name, kv_cache_tuple in kv_caches.items():
            if not isinstance(kv_cache_tuple, (list, tuple)):
                kv_cache_tuple = [kv_cache_tuple]
            group_idx = layer2group_ids[layer_name]
            layer_meta = LayerMetadata([], [], [], [])
            for single_kv_cache in kv_cache_tuple:
                tensor_num_blocks = single_kv_cache.shape[0]
                assert tensor_num_blocks % num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                block_size_scale = tensor_num_blocks // num_blocks
                block_shape = single_kv_cache.shape[1:]
                layer_meta.tensor_group_idx.append(group_idx)
                layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                layer_meta.block_len.append(single_kv_cache.element_size() * math.prod(block_shape))
                layer_meta.block_size_scale.append(block_size_scale)
            self.layer_metadata[layer_name] = layer_meta
            self.index_to_name[_layer_idx(layer_name)].append(layer_name)

        if self.total_layers < len(self.layer_metadata):
            self.total_layers = len(self.layer_metadata)

        register_regions = collect_storage_merged_register_regions(kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)
        assert global_te._unique_id is not None, "memfabric unique_id was not initialized before send thread setup"

        ready_event = threading.Event()
        self.kv_send_layer_thread = MembPullSendingThread(
            total_layers=self.total_layers,
            ready_event=ready_event,
            layer_metadata=self.layer_metadata,
            p_session=global_te._unique_id,
            layer_transfer_finished_events=get_shared_layer_transfer_events(),
            layer_transfer_pending_events=get_shared_layer_transfer_pending_events(),
        )
        self.kv_send_layer_thread.start()
        ready_event.wait()
        # Stash source tensors on the sending thread for env-gated verify
        # checksums (VLLM_ASCEND_MF_VERIFY=1): P sums its source blocks so
        # the user can compare against D's destination sums in the logs.
        self.kv_send_layer_thread._source_kv_caches = kv_caches
        self.layer_send_done_events = self.kv_send_layer_thread.layer_send_done_events
        logger.info(
            "MembPull P registered kv caches: layers=%d, p_session=%s",
            len(kv_caches),
            global_te._unique_id,
        )

    def _has_memfabric_pull_target(self, connector_metadata: KVConnectorMetadata, layer_idx: int) -> bool:
        for req_meta in getattr(connector_metadata, "requests", {}).values():
            has_endpoint = bool(req_meta.remote_host) and bool(req_meta.remote_port)
            if not has_endpoint:
                continue
            p_block_ids = req_meta.local_block_ids[0] if req_meta.local_block_ids else []
            chunk_done = layer_idx == self.total_layers - 1 and req_meta.chunk_finish
            if p_block_ids or chunk_done:
                return True
        return False

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
        **kwargs,
    ) -> None:
        if self._backend != BACKEND_MEMFABRIC:
            raise RuntimeError("SFAPDCpuOffloadConnector P side supports memfabric pull only.")
        if (
            getattr(connector_metadata, "requests", None)
            and self.current_layer < self.total_layers
        ):
            layer_idx = self.current_layer
            has_pd_target = self._has_memfabric_pull_target(connector_metadata, layer_idx)
            if has_pd_target and self.layer_send_done_events is not None and 0 <= layer_idx < len(
                self.layer_send_done_events
            ):
                self.layer_send_done_events[layer_idx].clear()
            pd_done = getattr(self.kv_send_layer_thread, "layer_transfer_finished_events", None)
            if has_pd_target and pd_done is not None and 0 <= layer_idx < len(pd_done):
                pd_done[layer_idx].clear()
            pd_pending = getattr(self.kv_send_layer_thread, "layer_transfer_pending_events", None)
            if has_pd_target and pd_pending is not None and 0 <= layer_idx < len(pd_pending):
                pd_pending[layer_idx].set()
        # Record a fresh compute-stream event (after the scatter) for the send
        # thread to wait before notify; replaces mooncake's wait_event, which
        # does not capture sfa_v1's scatter on this pull path.
        if self.kv_send_layer_thread is None:
            return
        if not getattr(connector_metadata, "requests", None):
            return
        if self.current_layer >= self.total_layers:
            self.current_layer += 1
            return
        if layer_name == "":
            layer_name = self.index_to_name[self.current_layer][0]

        self.kv_send_layer_thread.record_p_save_event(self.current_layer)
        layer_attn_metadata = None
        if self.use_mla and hasattr(attn_metadata, "__getitem__"):
            try:
                layer_attn_metadata = attn_metadata[layer_name]
            except Exception:
                layer_attn_metadata = None
        if layer_attn_metadata is not None and hasattr(layer_attn_metadata, "reshape_cache_event"):
            wait_event = layer_attn_metadata.reshape_cache_event
        elif hasattr(attn_metadata, "reshape_cache_event"):
            wait_event = attn_metadata.reshape_cache_event
        else:
            wait_event = torch.npu.Event()
            wait_event.record()

        layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
        layer_send_task = SendTask(
            send_request={},
            wait_event=wait_event,
            layer_idx=self.current_layer,
            layer_name=layer_name,
        )
        for req_id, req_meta in connector_metadata.requests.items():
            local_block_ids = req_meta.local_block_ids
            if len(local_block_ids) <= layer_group_idx or not local_block_ids[layer_group_idx]:
                continue
            layer_send_task.send_request[req_id] = self.update_decoder_info(req_id, req_meta)
        if layer_send_task.send_request:
            self.kv_send_layer_thread.send_queue.put(layer_send_task)
        else:
            self.kv_send_layer_thread._signal_layer_done(self.current_layer)
        self.current_layer += 1

    def wait_for_layer_send(self, layer_idx: int) -> None:
        """Block until D has read layer ``layer_idx``'s KV (buffer-reuse gate).

        In pull mode D reads P's KV via memfabric; this waits until D replies
        with READ_DONE or READ_FAILED before P reuses the KV buffer for a later
        layer, so D is no longer reading before P overwrites it.
        """
        if self.layer_send_done_events is None:
            return
        if 0 <= layer_idx < len(self.layer_send_done_events):
            event = self.layer_send_done_events[layer_idx]
            if not event.wait(timeout=10):
                raise RuntimeError(
                    f"Timed out waiting for D to read layer {layer_idx}'s KV before buffer reuse"
                )

    def get_layer_send_event(self, layer_idx: int) -> threading.Event | None:
        if self.layer_send_done_events is None:
            return None
        if 0 <= layer_idx < len(self.layer_send_done_events):
            return self.layer_send_done_events[layer_idx]
        return None
