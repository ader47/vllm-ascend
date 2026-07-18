# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Worker side of the PD-disaggregated SFA connector (memfabric pull mode).

D (``kv_consumer``): binds to :class:`KVOffloadDecodeManager`'s TP-shared CPU
KV pool and receives indexer KV into rank-local HBM. TP0 pulls main MLA KV;
every TP rank pulls its indexer KV. Decode KV continues to be written directly
to the same CPU pool by the decode-offload manager.

P (``kv_producer``): registers its HBM KV with memfabric and runs a pull-mode
sending thread that notifies D to read (no RDMA push). A per-layer
send-completion event gates P's KV buffer reuse.
"""

from __future__ import annotations

import math
import os
import threading
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
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.kv_offload_decode_manager import (
    get_kv_offload_decode_manager,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    LayerMetadata,
    SendTask,
    get_external_request_id,
    infer_sfa_component_group_ids,
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

        self.decode_manager = None
        self._cpu_blocks_by_req: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        # external_req_id -> internal_req_id, so get_finished can map the recv
        # thread's done_recving (keyed by external id from P's DONE signal) back
        # to the vLLM-internal id that the scheduler expects.
        self.request_map: dict[str, str] = {}
        # external_req_id -> (main CPU block ids, indexer HBM block ids).
        self._dest_blocks_by_req: dict[str, tuple[list[int], list[int]]] = {}
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
        """Bind MemFabric destinations owned by KVOffloadDecodeManager."""
        assert _resolve_kv_transfer_backend(self.vllm_config) == BACKEND_MEMFABRIC, (
            "SFAPDCpuOffloadConnector D side supports memfabric pull only (set transfer_backend=memfabric)."
        )
        self.decode_manager = get_kv_offload_decode_manager()
        if not hasattr(self.decode_manager, "offload_layer_names"):
            raise RuntimeError(
                "KVOffloadDecodeManager.register_kv_caches must run before the PD connector is registered"
            )
        self._register_memfabric_pull(kv_caches)

    # -- D-side forwards to the composed SFA worker (LRU load path) --
    def start_load_kv(self, metadata: KVConnectorMetadata):
        for req in getattr(metadata, "requests", []):
            req_id = getattr(req, "req_id", None)
            if req_id is not None:
                ext_id = get_external_request_id(req_id)
                self.request_map[ext_id] = req_id
                main_ids = list(getattr(req, "main_block_ids", []) or [])
                indexer_ids = list(getattr(req, "indexer_block_ids", []) or [])
                self._dest_blocks_by_req[ext_id] = (main_ids, indexer_ids)
                self._cpu_blocks_by_req[req_id] = len(main_ids)
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull D stored dest blocks req %s: indexer_hbm_ids=%s, "
                        "main_cpu_ids=%s",
                        ext_id,
                        indexer_ids,
                        main_ids,
                    )

    def set_req_ids(self, req_ids: list):
        return

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
        # AscendSFAKVOffloadImpl calls KVOffloadDecodeManager.onload_topk_kv
        # directly.  This legacy duck-typed hook is intentionally inactive.
        return False

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        connector_metadata: KVConnectorMetadata,
    ) -> None:
        return

    def wait_for_save(self):
        return

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
        if hasattr(self, "_mf_read_thread") and self._mf_read_thread is not None:
            for ext_id in self._mf_read_thread.get_and_clear_failed():
                dest = self._dest_blocks_by_req.get(ext_id)
                if dest is None:
                    continue
                main_block_ids, indexer_block_ids = dest
                self._invalid_block_ids.update(main_block_ids)
                self._invalid_block_ids.update(indexer_block_ids)
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        """Per-req actual main-MLA CPU-block count for the solution-1 threshold."""
        if self.decode_manager is None:
            return None
        result = {rid: self._cpu_blocks_by_req[rid] for rid in req_ids if rid in self._cpu_blocks_by_req}
        return result or None

    def _build_consumer_read_state(self) -> ConsumerReadState:
        assert self.decode_manager is not None
        return ConsumerReadState(
            num_blocks=self.kv_cache_config.num_blocks,
            layer_metadata=self.layer_metadata,
            main_name_to_idx=self._main_name_to_idx,
            cpu_pools=self._cpu_pools,
            indexer_tensors=self._indexer_tensors,
            indexer_scale_tensors=self._indexer_scale_tensors,
            dest_blocks_by_req=self._dest_blocks_by_req,
            get_offload_layer_id=self.decode_manager._get_offload_layer_id,
        )

    def _register_memfabric_pull(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Start D pull thread with manager CPU KV and rank-local indexer HBM."""
        assert self.decode_manager is not None
        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks
        main_names = list(self.decode_manager.offload_layer_names)
        indexer_by_layer = {
            _layer_idx(name): name for name in kv_caches if "indexer" in name.lower()
        }

        # Store layer info for MembPullReadThread
        self._main_names = main_names
        self._main_name_to_idx = {n: i for i, n in enumerate(main_names)}
        k_caches_cpu = self.decode_manager.k_caches_cpu
        v_caches_cpu = self.decode_manager.v_caches_cpu
        if self.tp_rank == 0:
            if len(k_caches_cpu) != len(main_names) or len(v_caches_cpu) != len(main_names):
                raise RuntimeError("KVOffloadDecodeManager CPU pool/layer count mismatch")
            self._cpu_pools = list(zip(k_caches_cpu, v_caches_cpu))
        else:
            self._cpu_pools = [None] * len(main_names)
        self._indexer_tensors = []
        self._indexer_scale_tensors: list[torch.Tensor | None] = []
        for main_name in main_names:
            indexer_name = indexer_by_layer.get(_layer_idx(main_name))
            if indexer_name is None:
                self._indexer_tensors.append(None)
                self._indexer_scale_tensors.append(None)
                continue
            indexer_tuple = kv_caches[indexer_name]
            if not isinstance(indexer_tuple, (list, tuple)):
                indexer_tuple = (indexer_tuple,)
            self._indexer_tensors.append(indexer_tuple[0])
            self._indexer_scale_tensors.append(indexer_tuple[1] if len(indexer_tuple) > 1 else None)

        main_group_idx, indexer_group_idx = infer_sfa_component_group_ids(self.kv_cache_config)
        for pool_idx, mname in enumerate(main_names):
            indexer_t = self._indexer_tensors[pool_idx]
            indexer_scale_t = self._indexer_scale_tensors[pool_idx]
            cpu_pool = self._cpu_pools[pool_idx]
            if cpu_pool is not None:
                k_cpu, v_cpu = cpu_pool
                addrs = [k_cpu.data_ptr(), v_cpu.data_ptr()]
                block_lens = [
                    k_cpu.element_size() * math.prod(k_cpu.shape[1:]),
                    v_cpu.element_size() * math.prod(v_cpu.shape[1:]),
                ]
                scales = [k_cpu.shape[0] // num_blocks, v_cpu.shape[0] // num_blocks]
            else:
                addrs, block_lens, scales = [], [], []
            groups = [main_group_idx] * len(addrs)
            if indexer_t is not None:
                addrs.append(indexer_t.data_ptr())
                block_lens.append(indexer_t.element_size() * math.prod(indexer_t.shape[1:]))
                scales.append(indexer_t.shape[0] // num_blocks)
                groups.append(indexer_group_idx)
            if indexer_scale_t is not None:
                addrs.append(indexer_scale_t.data_ptr())
                block_lens.append(indexer_scale_t.element_size() * math.prod(indexer_scale_t.shape[1:]))
                scales.append(indexer_scale_t.shape[0] // num_blocks)
                groups.append(indexer_group_idx)
            self.layer_metadata[mname] = LayerMetadata(
                tensor_group_idx=groups,
                kv_caches_base_addr=addrs,
                block_len=block_lens,
                block_size_scale=scales,
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
            "%d indexer + %d main layers, main CPU destination on this rank=%s",
            sum(t is not None for t in self._indexer_tensors),
            len(main_names),
            self.tp_rank == 0,
        )


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
        self.last_layer_idx = self.total_layers - 1
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()
        self.kv_cache_specs = [group_spec.kv_cache_spec for group_spec in self.kv_cache_config.kv_cache_groups]
        self.block_size = [spec.block_size for spec in self.kv_cache_specs]
        self.num_kv_cache_groups = len(self.kv_cache_specs)
        self.main_group_idx, self.indexer_group_idx = infer_sfa_component_group_ids(self.kv_cache_config)
        self.use_mla = self.vllm_config.model_config.use_mla
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.index_to_name: dict[int, str] = {}
        self.reuse_mate_map: dict[int, int | None] = {}
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
        ``build_connector_meta``; main and indexer group ids remain separate."""
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
            last_layer_idx=self.last_layer_idx,
            layer_metadata=self.layer_metadata,
            p_session=global_te._unique_id,
            main_group_idx=self.main_group_idx,
            indexer_group_idx=self.indexer_group_idx,
            layer_transfer_finished_events=None,
            layer_transfer_pending_events=None,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # memfabric pull mode only.
        assert self._backend == BACKEND_MEMFABRIC, "SFAPDCpuOffloadConnector P side supports memfabric pull only."
        layer2group_ids: dict[str, int] = {}
        for group_idx, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                layer2group_ids[layer_name] = group_idx

        num_blocks = self.kv_cache_config.num_blocks
        main_by_layer = {
            _layer_idx(name): name for name in kv_caches if "indexer" not in name.lower()
        }
        indexer_by_layer = {
            _layer_idx(name): name for name in kv_caches if "indexer" in name.lower()
        }
        if not main_by_layer:
            raise RuntimeError("SFAPD producer did not find main SFA KV cache layers")

        def _append_cache_tensors(
            layer_meta: LayerMetadata,
            cache_or_caches: Any,
            group_idx: int,
        ) -> None:
            if not isinstance(cache_or_caches, (list, tuple)):
                cache_or_caches = (cache_or_caches,)
            for single_kv_cache in cache_or_caches:
                tensor_num_blocks = single_kv_cache.shape[0]
                if tensor_num_blocks % num_blocks != 0:
                    raise ValueError(
                        "The external block size must be an integer multiple "
                        "of the kernel block size."
                    )
                layer_meta.tensor_group_idx.append(group_idx)
                layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                layer_meta.block_len.append(
                    single_kv_cache.element_size() * math.prod(single_kv_cache.shape[1:])
                )
                layer_meta.block_size_scale.append(tensor_num_blocks // num_blocks)

        for physical_idx, main_name in sorted(main_by_layer.items()):
            layer_meta = LayerMetadata([], [], [], [])
            _append_cache_tensors(layer_meta, kv_caches[main_name], layer2group_ids[main_name])
            indexer_name = indexer_by_layer.get(physical_idx)
            if indexer_name is None:
                raise RuntimeError(
                    f"SFAPD producer did not find indexer KV cache for physical layer {physical_idx}"
                )
            _append_cache_tensors(
                layer_meta,
                kv_caches[indexer_name],
                layer2group_ids[indexer_name],
            )
            if len(layer_meta.kv_caches_base_addr) < 3:
                raise RuntimeError(
                    f"SFAPD producer layer {main_name} must expose main K/V and indexer tensors"
                )
            self.layer_metadata[main_name] = layer_meta
            self.index_to_name[physical_idx] = main_name

        self.last_layer_idx = max(main_by_layer)
        self.total_layers = self.last_layer_idx + 1

        # Infer the actual reuse plan from shared source addresses. This keeps
        # the connector synchronized with model-runner tensor merging without
        # relying on an out-of-band setter that may not be invoked.
        layers_by_storage: dict[tuple[int, ...], list[int]] = {}
        for layer_name, metadata in self.layer_metadata.items():
            storage_key = tuple(metadata.kv_caches_base_addr)
            layers_by_storage.setdefault(storage_key, []).append(_layer_idx(layer_name))
        for physical_layers in layers_by_storage.values():
            previous = None
            for physical_idx in sorted(physical_layers):
                self.reuse_mate_map[physical_idx] = previous
                previous = physical_idx

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
            len(self.layer_metadata),
            global_te._unique_id,
        )

    def _has_memfabric_pull_target(
        self,
        connector_metadata: KVConnectorMetadata,
        layer_idx: int,
        layer_group_indices: set[int],
    ) -> bool:
        for req_meta in getattr(connector_metadata, "requests", {}).values():
            has_endpoint = bool(req_meta.remote_host) and bool(req_meta.remote_port)
            if not has_endpoint:
                continue
            local_block_ids = req_meta.local_block_ids
            has_blocks = any(
                len(local_block_ids) > group_idx and bool(local_block_ids[group_idx])
                for group_idx in layer_group_indices
            )
            chunk_done = layer_idx == self.last_layer_idx and req_meta.chunk_finish
            if has_blocks or chunk_done:
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
        resolved_layer_name = layer_name or self.index_to_name.get(self.current_layer)
        if resolved_layer_name is None:
            return
        layer_idx = _layer_idx(resolved_layer_name)
        if getattr(connector_metadata, "requests", None) and layer_idx < self.total_layers:
            layer_group_indices = set(self.layer_metadata[resolved_layer_name].tensor_group_idx)
            has_pd_target = self._has_memfabric_pull_target(
                connector_metadata,
                layer_idx,
                layer_group_indices,
            )
            if (
                has_pd_target
                and self.layer_send_done_events is not None
                and 0 <= layer_idx < len(self.layer_send_done_events)
            ):
                self.layer_send_done_events[layer_idx].clear()
        # Record a fresh compute-stream event after the scatter so the send
        # thread waits for SFA's KV write before notifying D.
        if self.kv_send_layer_thread is None:
            return
        if not getattr(connector_metadata, "requests", None):
            return
        if layer_idx >= self.total_layers:
            self.current_layer += 1
            return
        layer_name = resolved_layer_name

        self.kv_send_layer_thread.record_p_save_event(layer_idx)
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

        layer_send_task = SendTask(
            send_request={},
            wait_event=wait_event,
            layer_idx=layer_idx,
            layer_name=layer_name,
        )
        for req_id, req_meta in connector_metadata.requests.items():
            local_block_ids = req_meta.local_block_ids
            has_main = len(local_block_ids) > self.main_group_idx and bool(local_block_ids[self.main_group_idx])
            has_indexer = (
                len(local_block_ids) > self.indexer_group_idx
                and bool(local_block_ids[self.indexer_group_idx])
            )
            if not (has_main and has_indexer):
                continue
            layer_send_task.send_request[req_id] = self.update_decoder_info(req_id, req_meta)
        if layer_send_task.send_request:
            self.kv_send_layer_thread.send_queue.put(layer_send_task)
        else:
            self.kv_send_layer_thread._signal_layer_done(layer_idx)
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
            error = self.kv_send_layer_thread.get_layer_error(layer_idx) if self.kv_send_layer_thread else None
            if error is not None:
                raise RuntimeError(f"D failed to read layer {layer_idx}'s KV: {error}")

    def get_reuse_mate(self, layer_idx: int) -> int | None:
        return self.reuse_mate_map.get(layer_idx)

    def get_layer_send_event(self, layer_idx: int) -> threading.Event | None:
        if self.layer_send_done_events is None:
            return None
        if 0 <= layer_idx < len(self.layer_send_done_events):
            return self.layer_send_done_events[layer_idx]
        return None
