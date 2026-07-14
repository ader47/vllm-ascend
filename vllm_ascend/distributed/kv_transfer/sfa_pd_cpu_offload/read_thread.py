# SPDX-License-Identifier: Apache-2.0
"""D-side memfabric pull read thread for SFA PD CPU offload."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    GET_META_MSG,
    MF_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    SfaPDAgentMetadata,
)


@dataclass
class ConsumerReadState:
    layer_metadata: dict[str, Any]
    main_name_to_idx: dict[str, int]
    cpu_pools: list[tuple[Any, Any] | None]
    hbm_kv: dict[str, tuple[Any, Any]]
    indexer_tensors: list[Any]
    indexer_scale_tensors: list[Any | None]
    dest_blocks_by_req: dict[str, tuple[list[int], list[int], int, int | None]]
    get_offload_layer_id: Callable[[str], int]


def _coalesce_desc(
    peer: np.ndarray,
    local: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class MembPullReadThread(threading.Thread):
    """D-side thread for memfabric pull.

    Receives READ_READY_BATCH from P, reads KV from P's HBM via
    batch_transfer_sync_read, then replies with READ_DONE or READ_FAILED.
    """

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        engine: Any,
        state: ConsumerReadState,
    ):
        super().__init__(daemon=True, name=f"MembPullReadThread-TP{tp_rank}")
        self.tp_rank = tp_rank
        self.side_channel_port = side_channel_port
        self.engine = engine
        self._state = state
        self.ready_event = threading.Event()
        self._p_session: str | None = None
        self._p_layer_meta: dict[str, Any] = {}
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
                        if done_ext_ids:
                            with self._lock:
                                self._done_requests.update(done_ext_ids)

                    elif msg_type == GET_META_MSG:
                        meta_bytes = encoder.encode(
                            SfaPDAgentMetadata(
                                te_rpc_port=0,
                                layer_metadata=self._state.layer_metadata,
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
        state = self._state
        pool_idx = state.main_name_to_idx.get(layer_name)
        if pool_idx is None:
            logger.warning("MembPull _do_read: layer %s not in main names, skip", layer_name)
            return None
        offload_id = state.get_offload_layer_id(layer_name)
        if pool_idx != offload_id and envs.VLLM_ASCEND_SFA_DEBUG:
            logger.warning(
                "MembPull _do_read: layer-order mismatch for %s -- pull _main_names "
                "idx=%d != resident offload_id=%d. Using offload_id.",
                layer_name,
                pool_idx,
                offload_id,
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

        cpu_pool = state.cpu_pools[offload_id]
        if cpu_pool is None:
            k_cpu_ptr = v_cpu_ptr = None
        else:
            k_cpu, v_cpu = cpu_pool
            k_cpu_ptr, v_cpu_ptr = k_cpu.data_ptr(), v_cpu.data_ptr()
        hbm_kv = state.hbm_kv.get(layer_name)
        if hbm_kv is None:
            k_hbm_ptr = v_hbm_ptr = None
        else:
            k_hbm_ptr, v_hbm_ptr = hbm_kv[0].data_ptr(), hbm_kv[1].data_ptr()

        indexer = None
        if len(p_base_addrs) >= 3:
            p_dsa_len = p_block_len[2]
            d_indexer = state.indexer_tensors[pool_idx]
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
        scale_tensor = state.indexer_scale_tensors[pool_idx] if pool_idx < len(state.indexer_scale_tensors) else None
        if scale_tensor is not None:
            if len(p_base_addrs) < 4:
                scale = {"error": "p_addr_mismatch", "p_n": len(p_base_addrs)}
            else:
                p_scale_len = p_block_len[3]
                d_scale_len = scale_tensor.element_size() * math.prod(scale_tensor.shape[1:])
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
            "k_cpu_ptr": k_cpu_ptr,
            "v_cpu_ptr": v_cpu_ptr,
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
        state = self._state
        layer_name = layer["layer_name"]

        dest = state.dest_blocks_by_req.get(ext_req_id)
        if dest is None:
            logger.warning(
                "MembPull _do_read: no dest blocks on D for req %s (layer %s), skip",
                ext_req_id,
                layer_name,
            )
            return [], [], [], None
        d_indexer_ids, d_main_ids, num_full, partial_hbm_bid = dest
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

        full_p_blocks = p_block_ids[:num_full]
        has_cpu_destination = layer["k_cpu_ptr"] is not None and layer["v_cpu_ptr"] is not None
        n_full = min(len(full_p_blocks), len(d_main_ids)) if has_cpu_destination else 0
        if n_full:
            full_p = np.array(full_p_blocks[:n_full], dtype=np.int64)
            d_main = np.array(d_main_ids[:n_full], dtype=np.int64)
            len_k = np.full(n_full, p_k_len, dtype=np.int64)
            len_v = np.full(n_full, p_v_len, dtype=np.int64)
            cp, cl, clen = _coalesce_desc(p_k_base + full_p * p_k_len, layer["k_cpu_ptr"] + d_main * p_k_len, len_k)
            peer_chunks.append(cp)
            local_chunks.append(cl)
            length_chunks.append(clen)
            cp, cl, clen = _coalesce_desc(p_v_base + full_p * p_v_len, layer["v_cpu_ptr"] + d_main * p_v_len, len_v)
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
                peer_chunks.append(
                    np.array([p_k_base + partial_p_bid * p_k_len, p_v_base + partial_p_bid * p_v_len], dtype=np.int64)
                )
                local_chunks.append(
                    np.array(
                        [k_hbm_ptr + partial_hbm_bid * p_k_len, layer["v_hbm_ptr"] + partial_hbm_bid * p_v_len],
                        dtype=np.int64,
                    )
                )
                length_chunks.append(np.array([p_k_len, p_v_len], dtype=np.int64))
                n_main += 1

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
                "MembPull indexer %s: p_dsa_len=%d d_dsa_len=%d scale=%d, D dsa_k shape=%s, D_sub=%d P_blocks=%d",
                layer_name,
                p_dsa_len,
                d_dsa_len,
                scale,
                idx["shape"],
                int(d_sub.shape[0]),
                len(p_block_ids),
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

        scale = layer.get("scale")
        if scale is not None and d_indexer_ids:
            if "error" in scale:
                if scale["error"] == "p_addr_mismatch":
                    raise RuntimeError(
                        f"MembPull indexer scale {layer_name}: D is LIC8 (has scale "
                        f"tensor) but P exposed only {scale['p_n']} base addrs "
                        f"(no scale leg) -- P/D LIC8 config mismatch."
                    )
                raise RuntimeError(
                    f"MembPull indexer scale {layer_name}: D scale block_len="
                    f"{scale['d_scale_len']} not a multiple of P={scale['p_scale_len']} -- "
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
                d_scale_arr = np.array(d_scale_sub_addrs[:n_scale_pairs], dtype=np.int64)
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
                "atomic_transfers": 2 * n_main + n_indexer,
            }
        return local_ptrs, peer_ptrs, lengths, info

    def _log_read_result(self, read_info: dict[str, Any]) -> None:
        state = self._state
        layer_name = read_info["layer_name"]
        ext_req_id = read_info["ext_req_id"]
        pool_idx = read_info["pool_idx"]
        offload_id = read_info["offload_id"]
        d_main_ids = read_info["d_main_ids"]
        d_indexer_ids = read_info["d_indexer_ids"]
        partial_hbm_bid = read_info["partial_hbm_bid"]
        if envs.VLLM_ASCEND_MF_VERIFY:
            try:
                cpu_pool = state.cpu_pools[offload_id]
                if cpu_pool is None:
                    mk = mv = 0.0
                else:
                    k_cpu, v_cpu = cpu_pool
                    mk = k_cpu[d_main_ids].float().sum().item() if d_main_ids else 0.0
                    mv = v_cpu[d_main_ids].float().sum().item() if d_main_ids else 0.0
                if partial_hbm_bid is not None:
                    hbm_kv = state.hbm_kv.get(layer_name)
                    if hbm_kv is not None:
                        k_hbm, v_hbm = hbm_kv
                        mk += k_hbm[partial_hbm_bid].float().sum().item()
                        mv += v_hbm[partial_hbm_bid].float().sum().item()
                mi = state.indexer_tensors[pool_idx][d_indexer_ids].float().sum().item() if d_indexer_ids else 0.0
                logger.info(
                    "MFV D layer %s req %s main_k=%.6f main_v=%.6f idx_post=%.6f",
                    layer_name,
                    ext_req_id,
                    mk,
                    mv,
                    mi,
                )
            except Exception as ve:
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
        if self._p_session is None:
            raise RuntimeError("MF_META not received before READ_READY_BATCH")

        layer = self._resolve_read_layer(layer_name)
        if layer is None:
            return

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
            except Exception as e:
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
