# SPDX-License-Identifier: Apache-2.0
"""Wire protocol helpers for SFA PD CPU offload."""

from __future__ import annotations

from dataclasses import dataclass

import msgspec

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


def get_external_request_id(request_id: str) -> str:
    # vLLM appends a 9-character EngineCore suffix to request IDs.
    return request_id[:-9]
