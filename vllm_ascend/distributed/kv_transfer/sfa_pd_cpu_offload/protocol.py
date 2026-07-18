# SPDX-License-Identifier: Apache-2.0
"""Wire protocol helpers for SFA PD CPU offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)

GET_META_MSG = b"get_meta_msg"
MF_META = b"mf_meta"
READ_READY_BATCH = b"read_ready_batch"
READ_DONE = b"read_done"
READ_FAILED = b"read_failed"


@dataclass
class LayerMetadata:
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]


class SfaPDAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    te_rpc_port: int
    layer_metadata: dict[str, LayerMetadata]


@dataclass
class SfaPDProducerReqMeta:
    local_block_ids: list[list[int]]
    token_ids: list[int]
    remote_block_ids: list[list[int]]
    remote_block_size: list[list[int]]
    remote_engine_id: str | None
    remote_host: str | None
    remote_port: int | None
    remote_te_rpc_port: int | None
    remote_layer_metadata: dict[str, LayerMetadata] | None
    metaserver: str | None
    remote_tp_size: int | None
    remote_pcp_size: int | None
    remote_dcp_size: int | None
    chunk_finish: bool = False
    prompt_len: int = 0
    trans_count: list[int] | None = None
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0
    do_virtual: bool = False


class SfaPDProducerMetadata(KVConnectorMetadata):
    def __init__(self) -> None:
        self.requests: dict[str, SfaPDProducerReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        token_ids: list[int] | None = None,
        chunk_finish: bool = False,
        prompt_len: int = 0,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
    ) -> None:
        self.requests[request_id] = SfaPDProducerReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_block_size=kv_transfer_params.get("remote_block_size", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port"),
            remote_layer_metadata=kv_transfer_params.get("remote_layer_metadata"),
            metaserver=kv_transfer_params.get("metaserver"),
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size"),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size"),
            do_virtual=kv_transfer_params.get("do_virtual", False),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
            trans_count=[],
        )


@dataclass
class SendTask:
    send_request: dict[str, SfaPDProducerReqMeta]
    wait_event: Any | None = None
    layer_idx: int = 0
    layer_name: str = ""


def get_external_request_id(request_id: str) -> str:
    # vLLM appends a 9-character EngineCore suffix to request IDs.
    # Guard short / malformed ids so we never return an empty or garbage id
    # (which would corrupt the external_req_id -> internal_req_id map).
    return request_id[:-9] if len(request_id) > 9 else request_id
