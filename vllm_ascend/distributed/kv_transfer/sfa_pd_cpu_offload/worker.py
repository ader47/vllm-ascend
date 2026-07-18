# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Worker side of the PD-disaggregated SFA connector (memfabric pull mode).

D (``kv_consumer``): composes :class:`SFAKVOffloadWorker` for the LRU-resident
H2D load path + TP-shared CPU pool, and runs one memfabric pull read thread per
TP rank. Every rank reads Indexer KV and partial Main KV into local HBM; TP0
also reads full Main MLA KV into the shared CPU pool. P notifies D per layer
via READ_READY_BATCH. Once a partial block fills during decode, the B1 offload
path on TP0 copies it to the CPU pool.

P (``kv_producer``): registers its HBM KV with memfabric and runs a pull-mode
sending thread that notifies D to read (no RDMA push). A per-layer
send-completion event gates P's KV buffer reuse.
"""

from __future__ import annotations

import math
import os
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import regex as re
import torch
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
    resize_shared_layer_transfer_events,
    set_shared_layer_transfer_events,
    set_shared_layer_transfer_pending_events,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (
    SFAKVOffloadWorker,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    LayerMetadata,
    SendTask,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.read_thread import (
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.send_thread import (
    MembPullSendingThread,
    ProducerSendState,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.transfer_engine_backend import (
    BACKEND_MEMFABRIC,
    MEMFABRIC_ROLE_DECODE,
    MEMFABRIC_ROLE_PREFILL,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    collect_storage_merged_register_regions,
    get_transfer_timeout_value,
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


def _resolve_kv_transfer_backend(vllm_config: VllmConfig) -> str:
    """Pick the KV transfer backend.

    ``kv_connector_extra_config["transfer_backend"]`` overrides the
    ``VLLM_ASCEND_KV_TRANSFER_BACKEND`` env var.
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

        # The full Main KV CPU pool is TP-shared and allocated only by TP0.
        # Every rank still runs PD receive because Indexer and partial Main KV
        # land in rank-local HBM.
        k_caches_cpu = getattr(self.sfa_worker, "k_caches_cpu", None)
        v_caches_cpu = getattr(self.sfa_worker, "v_caches_cpu", None)

        # Part A: D's main MLA HBM k/v tensors (group1 paged cache) — the partial
        # last block lands here instead of the CPU pool. Keyed by main layer name
        # (the 5-tuple layers); tuple[0]=k_nope, tuple[1]=v_rope.
        self._hbm_kv = {
            n: (t[0], t[1]) for n, t in kv_caches.items() if isinstance(t, (list, tuple)) and len(t) in (5, 6)
        }

        # memfabric pull mode only.
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
                        "MembPull D get_finished: done_ext=%s, done_recving_internal=%s, pending_done_ext=%s",
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

    def _build_consumer_read_state(self) -> ConsumerReadState:
        assert self.sfa_worker is not None
        return ConsumerReadState(
            layer_metadata=self.layer_metadata,
            main_name_to_idx=self._main_name_to_idx,
            cpu_pools=self._cpu_pools,
            hbm_kv=self._hbm_kv,
            indexer_tensors=self._indexer_tensors,
            indexer_scale_tensors=self._indexer_scale_tensors,
            dest_blocks_by_req=self._dest_blocks_by_req,
            get_offload_layer_id=self.sfa_worker._get_offload_layer_id,
        )

    def _register_memfabric_pull(
        self,
        kv_caches: dict[str, torch.Tensor],
        k_caches_cpu: list[torch.Tensor] | None,
        v_caches_cpu: list[torch.Tensor] | None,
    ) -> None:
        """memfabric pull mode: D does NOT register anything. Only P registers
        its HBM. Every D rank reads local HBM legs; TP0 also reads full Main KV
        into the shared CPU pool."""
        num_blocks = self.kv_cache_config.num_blocks
        indexer_names = list(self.kv_cache_config.kv_cache_groups[_INDEXER_GROUP_IDX].layer_names)

        def _offload_tuple_len(v: object) -> int:
            return len(v) if isinstance(v, (list, tuple)) else 1

        main_names = [n for n, v in kv_caches.items() if _offload_tuple_len(v) in (5, 6)]
        main_by_layer_idx = {_layer_idx(name): name for name in main_names}
        main_names = [main_by_layer_idx[_layer_idx(name)] for name in indexer_names]

        # Store layer info for MembPullReadThread
        self._indexer_names = indexer_names
        self._main_names = main_names
        self._main_name_to_idx = {n: i for i, n in enumerate(main_names)}
        if (k_caches_cpu is None) != (v_caches_cpu is None):
            raise RuntimeError("SFA shared CPU K/V pools must either both exist or both be absent")
        has_cpu_pool = k_caches_cpu is not None
        self._cpu_pools: list[tuple[torch.Tensor, torch.Tensor] | None] = (
            list(zip(k_caches_cpu, v_caches_cpu)) if has_cpu_pool else [None] * len(main_names)
        )
        self._indexer_tensors = []
        self._indexer_scale_tensors: list[torch.Tensor | None] = []
        for main_name in main_names:
            main_tuple = list(kv_caches[main_name])
            self._indexer_tensors.append(main_tuple[2])  # dsa_k_indexer
            self._indexer_scale_tensors.append(main_tuple[5] if len(main_tuple) >= 6 else None)

        # Build layer_metadata (D's local addresses, for compatibility)
        for pool_idx, (iname, mname) in enumerate(zip(indexer_names, main_names)):
            indexer_t = self._indexer_tensors[pool_idx]
            indexer_scale_t = self._indexer_scale_tensors[pool_idx]
            indexer_addrs = [indexer_t.data_ptr()]
            indexer_block_lens = [indexer_t.element_size() * math.prod(indexer_t.shape[1:])]
            indexer_block_scales = [indexer_t.shape[0] // num_blocks if num_blocks else 1]
            if indexer_scale_t is not None:
                indexer_addrs.append(indexer_scale_t.data_ptr())
                indexer_block_lens.append(indexer_scale_t.element_size() * math.prod(indexer_scale_t.shape[1:]))
                indexer_block_scales.append(indexer_scale_t.shape[0] // num_blocks if num_blocks else 1)
            self.layer_metadata[iname] = LayerMetadata(
                tensor_group_idx=[_INDEXER_GROUP_IDX],
                kv_caches_base_addr=indexer_addrs,
                block_len=indexer_block_lens,
                block_size_scale=indexer_block_scales,
            )
            # cpu_pools follows the SFA offload-layer order (it is zipped from
            # sfa_worker.k_caches_cpu), which may differ from main_names order
            # -> index by mname's offload id, not pool_idx (matches read_thread).
            _mname_offload_id = self.sfa_worker.layer_name_to_offload_id.get(mname)
            cpu_pool = self._cpu_pools[_mname_offload_id] if _mname_offload_id is not None else None
            if cpu_pool is not None:
                k_cpu, v_cpu = cpu_pool
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
        read_state = self._build_consumer_read_state()
        # Start MembPullReadThread (ZMQ ROUTER + memfabric read)
        self._mf_read_thread = MembPullReadThread(
            tp_rank=self.tp_rank,
            side_channel_port=self.side_channel_port,
            engine=self.engine,
            state=read_state,
        )
        self._mf_read_thread.start()
        self._mf_read_thread.ready_event.wait()
        logger.info(
            "SFAPDCpuOffload D-side registered (memfabric pull): "
            "%d indexer + %d main layers, full-main CPU destination=%s",
            len(indexer_names),
            len(main_names),
            has_cpu_pool,
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


class SFAPDCpuOffloadProducerWorker:
    """P-side worker for memfabric pull mode.

    It registers P's local KV tensors with memfabric and runs a pull-mode
    sending thread. P never pushes KV; it sends READ_READY_BATCH messages so D
    can read P's source blocks and reply with READ_DONE / READ_FAILED.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        # Preserve the Mooncake worker's transfer-engine timeout setup. The
        # memfabric engine reads this during construction.
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
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
        self.side_channel_port = vllm_config.kv_transfer_config.kv_port + self.dp_rank * self.tp_size
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        set_shared_layer_transfer_events([threading.Event() for _ in range(self.total_layers)])
        set_shared_layer_transfer_pending_events([threading.Event() for _ in range(self.total_layers)])
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()
        self.kv_cache_specs = [group_spec.kv_cache_spec for group_spec in self.kv_cache_config.kv_cache_groups]
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
        """Prepare P-side request metadata for memfabric pull mode.

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

    def _build_producer_send_state(self) -> ProducerSendState:
        assert global_te._unique_id is not None, "memfabric unique_id was not initialized before send thread setup"
        return ProducerSendState(
            total_layers=self.total_layers,
            layer_metadata=self.layer_metadata,
            p_session=global_te._unique_id,
            layer_transfer_finished_events=get_shared_layer_transfer_events(),
            layer_transfer_pending_events=get_shared_layer_transfer_pending_events(),
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # memfabric pull mode only.
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
            # Resize in place so a connector that already captured the shared
            # list keeps observing the same event objects.
            resize_shared_layer_transfer_events(self.total_layers)

        register_regions = collect_storage_merged_register_regions(kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        ready_event = threading.Event()
        send_state = self._build_producer_send_state()
        self.kv_send_layer_thread = MembPullSendingThread(
            ready_event=ready_event,
            state=send_state,
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

    def _has_memfabric_pull_target(
        self,
        connector_metadata: KVConnectorMetadata,
        layer_idx: int,
        layer_group_idx: int,
    ) -> bool:
        for req_meta in getattr(connector_metadata, "requests", {}).values():
            has_endpoint = bool(req_meta.remote_host) and bool(req_meta.remote_port)
            if not has_endpoint:
                continue
            # Inspect THIS layer's tensor group (was hardcoded to group 0 /
            # indexer), so a main-MLA layer gates on its own block ids.
            local_block_ids = req_meta.local_block_ids
            if local_block_ids and len(local_block_ids) > layer_group_idx:
                p_block_ids = local_block_ids[layer_group_idx]
            else:
                p_block_ids = []
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
        if getattr(connector_metadata, "requests", None) and self.current_layer < self.total_layers:
            layer_idx = self.current_layer
            # Resolve THIS layer's tensor group so the pull-target gate inspects
            # the right group's block ids (was implicitly group 0 / indexer).
            _gate_layer_name = layer_name if layer_name else self.index_to_name[layer_idx][0]
            layer_group_idx = self.layer_metadata[_gate_layer_name].tensor_group_idx[0]
            has_pd_target = self._has_memfabric_pull_target(connector_metadata, layer_idx, layer_group_idx)
            if (
                has_pd_target
                and self.layer_send_done_events is not None
                and 0 <= layer_idx < len(self.layer_send_done_events)
            ):
                self.layer_send_done_events[layer_idx].clear()
            pd_done = getattr(self.kv_send_layer_thread, "layer_transfer_finished_events", None)
            if has_pd_target and pd_done is not None and 0 <= layer_idx < len(pd_done):
                pd_done[layer_idx].clear()
            pd_pending = getattr(self.kv_send_layer_thread, "layer_transfer_pending_events", None)
            if has_pd_target and pd_pending is not None and 0 <= layer_idx < len(pd_pending):
                pd_pending[layer_idx].set()
        # Record a fresh compute-stream event after the scatter so the send
        # thread waits for SFA's KV write before notifying D.
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
                raise RuntimeError(f"Timed out waiting for D to read layer {layer_idx}'s KV before buffer reuse")

    def get_layer_send_event(self, layer_idx: int) -> threading.Event | None:
        if self.layer_send_done_events is None:
            return None
        if 0 <= layer_idx < len(self.layer_send_done_events):
            return self.layer_send_done_events[layer_idx]
        return None
