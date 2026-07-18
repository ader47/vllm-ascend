# SPDX-License-Identifier: Apache-2.0
"""P-side memfabric pull sending thread for SFA PD CPU offload."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    MF_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    LayerMetadata,
    SendTask,
    get_external_request_id,
)


@dataclass
class ProducerSendState:
    total_layers: int
    last_layer_idx: int
    layer_metadata: dict[str, LayerMetadata]
    p_session: str
    main_group_idx: int
    indexer_group_idx: int
    layer_transfer_finished_events: list[threading.Event] | None
    layer_transfer_pending_events: list[threading.Event] | None


class MembPullSendingThread(threading.Thread):
    """P-side sending thread for memfabric pull mode.

    Does NOT push (no batch_transfer_sync_write). Instead, after each layer's
    KV is ready in P HBM, notifies D via READ_READY_BATCH (ZMQ), and drains
    READ_DONE / READ_FAILED replies. First call sends MF_META (P session +
    layer addresses) to D.
    """

    def __init__(
        self,
        *,
        ready_event: threading.Event,
        state: ProducerSendState,
    ) -> None:
        super().__init__(daemon=True, name="SfaPDMembPullSendingThread")
        self.timeout = 10.0
        self._mf_meta_sent_paths: set[str] = set()
        self._state = state
        self.total_layers = state.total_layers
        self.last_layer_idx = state.last_layer_idx
        self.ready_event = ready_event
        self.layer_transfer_finished_events = state.layer_transfer_finished_events
        self.layer_transfer_pending_events = state.layer_transfer_pending_events
        self.send_queue: queue.Queue[SendTask] = queue.Queue()
        # Set means the layer's source buffer has no pending D-side read.
        self.layer_send_done_events: list[threading.Event] = []
        for _ in range(state.total_layers):
            event = threading.Event()
            event.set()
            self.layer_send_done_events.append(event)
        self._persist_ctx = zmq.Context()
        self._dealers: dict[str, Any] = {}
        self._stopped = False
        self._pending_reads_by_layer: dict[int, int] = {}
        self._layer_read_errors: dict[int, str] = {}
        # Per-layer fresh compute-stream events recorded by the producer in
        # save_kv_layer right after KV scatter.
        self._p_save_events: dict[int, Any] = {}

    def _ensure_dealer(self, path: str):
        if path not in self._dealers:
            dealer = self._persist_ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.SNDHWM, 0)
            dealer.setsockopt(zmq.RCVHWM, 0)
            dealer.connect(path)
            self._dealers[path] = dealer
        return self._dealers[path]

    def run(self) -> None:
        try:
            from vllm.distributed import get_world_group

            local_rank = get_world_group().local_rank
            torch.npu.set_device(torch.device(f"npu:{local_rank}"))
        except Exception:
            pass
        self.ready_event.set()

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=tuple)
        try:
            while not self._stopped:
                try:
                    send_task = self.send_queue.get(timeout=0.001)
                    try:
                        self._process_send_task(send_task, encoder)
                    except Exception as e:
                        layer_idx = getattr(send_task, "layer_idx", -1)
                        logger.error(
                            "MembPull send task failed (layer=%s): %s: %s",
                            layer_idx,
                            type(e).__name__,
                            e,
                        )
                        self._fail_layer(layer_idx, str(e))
                except queue.Empty:
                    pass
                try:
                    if self._dealers:
                        self._drain_read_replies(decoder)
                except Exception as e:
                    logger.error("MembPull read-reply drain error: %s: %s", type(e).__name__, e)
        except Exception as e:
            logger.error("MembPull send thread crashed: %s: %s", type(e).__name__, e)
        finally:
            if self._dealers:
                self._drain_read_replies(decoder)
                for dealer in self._dealers.values():
                    dealer.close(linger=0)
                self._dealers.clear()

    def record_p_save_event(self, layer_idx: int) -> None:
        evt = torch.npu.Event()
        evt.record()
        self._p_save_events[layer_idx] = evt

    def _process_send_task(self, send_task: SendTask, encoder: msgspec.msgpack.Encoder) -> None:
        layer_idx = send_task.layer_idx
        p_save_event = self._p_save_events.pop(layer_idx, None)
        if p_save_event is not None:
            p_save_event.synchronize()
        elif send_task.wait_event is not None:
            send_task.wait_event.synchronize()
        layer_name = send_task.layer_name

        endpoint_payloads: dict[
            tuple[str, int],
            tuple[list[tuple[str, list[int], list[int]]], list[str]],
        ] = {}
        for req_id, rm in send_task.send_request.items():
            p_main_block_ids = (
                rm.local_block_ids[self._state.main_group_idx]
                if len(rm.local_block_ids) > self._state.main_group_idx
                else []
            )
            p_indexer_block_ids = (
                rm.local_block_ids[self._state.indexer_group_idx]
                if len(rm.local_block_ids) > self._state.indexer_group_idx
                else []
            )
            ext_id = get_external_request_id(req_id)
            has_endpoint = bool(rm.remote_host) and bool(rm.remote_port)
            chunk_done = layer_idx == self.last_layer_idx and rm.chunk_finish and has_endpoint
            if (p_main_block_ids or p_indexer_block_ids or chunk_done) and has_endpoint:
                endpoint = (rm.remote_host, rm.remote_port)
                read_reqs, done_ext_ids = endpoint_payloads.setdefault(endpoint, ([], []))
                if p_main_block_ids or p_indexer_block_ids:
                    if not p_main_block_ids or not p_indexer_block_ids:
                        raise RuntimeError(
                            f"MembPull P source blocks are incomplete for req {ext_id}: "
                            f"main={len(p_main_block_ids)}, indexer={len(p_indexer_block_ids)}"
                        )
                    read_reqs.append((ext_id, p_main_block_ids, p_indexer_block_ids))
                if chunk_done:
                    done_ext_ids.append(ext_id)
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "MembPull P add READ_READY_BATCH item: layer=%d (%s), req=%s, "
                    "main_blocks=%d, indexer_blocks=%d, done=%s",
                    layer_idx,
                    layer_name,
                    ext_id,
                    len(p_main_block_ids),
                    len(p_indexer_block_ids),
                    chunk_done,
                )

        if endpoint_payloads:
            if 0 <= layer_idx < len(self.layer_send_done_events):
                self.layer_send_done_events[layer_idx].clear()
            if self.layer_transfer_finished_events is not None and 0 <= layer_idx < len(
                self.layer_transfer_finished_events
            ):
                self.layer_transfer_finished_events[layer_idx].clear()
            self._pending_reads_by_layer[layer_idx] = len(endpoint_payloads)
            self._layer_read_errors.pop(layer_idx, None)
            for (remote_host, remote_port), (read_reqs, done_ext_ids) in endpoint_payloads.items():
                path = make_zmq_path("tcp", remote_host, remote_port)
                dealer = self._ensure_dealer(path)
                if path not in self._mf_meta_sent_paths:
                    self._send_mf_meta(path, dealer, encoder)
                dealer.send(
                    encoder.encode(
                        (READ_READY_BATCH, layer_idx, layer_name, read_reqs, done_ext_ids)
                    )
                )
                if envs.VLLM_ASCEND_SFA_DEBUG:
                    logger.info(
                        "MembPull P send READ_READY_BATCH: layer=%d (%s), "
                        "endpoint=%s:%d, reqs=%d, done_reqs=%d",
                        layer_idx,
                        layer_name,
                        remote_host,
                        remote_port,
                        len(read_reqs),
                        len(done_ext_ids),
                    )
        else:
            self._signal_layer_done(layer_idx)

    def _send_mf_meta(self, path: str, dealer, encoder: msgspec.msgpack.Encoder) -> None:
        p_meta_dict = {}
        for ln, meta in self._state.layer_metadata.items():
            p_meta_dict[ln] = {
                "base_addrs": list(meta.kv_caches_base_addr),
                "block_len": list(meta.block_len),
                "block_size_scale": list(meta.block_size_scale),
                "tensor_group_idx": list(meta.tensor_group_idx),
            }
        dealer.send(encoder.encode((MF_META, self._state.p_session, encoder.encode(p_meta_dict))))
        if dealer.poll(timeout=int(self.timeout * 1000)):
            frames = dealer.recv_multipart()
            payload = [f for f in frames if f != b""]
            if payload != [b"ACK"]:
                raise RuntimeError(f"MembPull P MF_META got unexpected reply: {payload!r}")
            self._mf_meta_sent_paths.add(path)
            logger.info(
                "MembPull P sent MF_META: session=%s, layers=%d",
                self._state.p_session,
                len(p_meta_dict),
            )
        else:
            raise RuntimeError("MembPull P MF_META timed out (no reply from D)")

    def _drain_read_replies(self, decoder: msgspec.msgpack.Decoder) -> None:
        for dealer in self._dealers.values():
            while True:
                try:
                    if not dealer.poll(timeout=0):
                        break
                    frames = dealer.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                payload = [f for f in frames if f != b""]
                if len(payload) != 1:
                    continue
                try:
                    msg = decoder.decode(payload[0])
                except Exception:
                    continue
                if len(msg) >= 2 and msg[0] == READ_DONE:
                    self._signal_layer_done(msg[1])
                elif len(msg) >= 2 and msg[0] == READ_FAILED:
                    layer_idx = msg[1]
                    error = msg[2] if len(msg) > 2 else ""
                    logger.error(
                        "MembPull P received READ_FAILED: layer=%s, error=%s",
                        layer_idx,
                        error,
                    )
                    self._layer_read_errors[layer_idx] = str(error)
                    self._signal_layer_done(layer_idx)

    def _signal_layer_done(self, layer_idx: int) -> None:
        pending = self._pending_reads_by_layer.get(layer_idx)
        if pending is not None:
            if pending > 1:
                self._pending_reads_by_layer[layer_idx] = pending - 1
                return
            self._pending_reads_by_layer.pop(layer_idx, None)
        if 0 <= layer_idx < len(self.layer_send_done_events):
            self.layer_send_done_events[layer_idx].set()
        if self.layer_transfer_finished_events is not None and 0 <= layer_idx < len(
            self.layer_transfer_finished_events
        ):
            self.layer_transfer_finished_events[layer_idx].set()
        if self.layer_transfer_pending_events is not None and 0 <= layer_idx < len(self.layer_transfer_pending_events):
            self.layer_transfer_pending_events[layer_idx].clear()
        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info("MembPull P layer send complete: layer=%d", layer_idx)

    def _fail_layer(self, layer_idx: int, error: str) -> None:
        self._pending_reads_by_layer.pop(layer_idx, None)
        self._layer_read_errors[layer_idx] = error
        self._signal_layer_done(layer_idx)

    def get_layer_error(self, layer_idx: int) -> str | None:
        return self._layer_read_errors.get(layer_idx)
