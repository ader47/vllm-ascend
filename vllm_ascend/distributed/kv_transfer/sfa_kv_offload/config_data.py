from __future__ import annotations

from dataclasses import dataclass, field

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class RequestTracker:
    req_id: str
    allocated_indexer_block_ids: list[int]
    allocated_block_ids_cpu: list[int]


@dataclass
class ReqMeta:
    req_id: str
    block_ids_cpu: list[int]
    # The one logical group's NPU block ids address real indexer storage.
    block_ids_indexer: list[int] = field(default_factory=list)
    # Token range produced by this scheduler step. The CPU cache is
    # authoritative at token granularity, so the final partial page is also
    # allocated and updated instead of waiting for a full block.
    write_start: int = 0
    write_count: int = 0


class SFADecodeHostOffloadMetadata(KVConnectorMetadata):
    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str] | None,
    ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)


@dataclass
class LayerMultiBlockReqMeta:
    req_id: str
    layer_id: int
    block_ids_cpu: list[int] | None = None
    cache_npu: tuple[torch.Tensor, torch.Tensor] | None = None
    cache_cpu: tuple[torch.Tensor, torch.Tensor] | None = None
    token_start: int = 0
    token_count: int = 0
    source_rows: list[int] = field(default_factory=list)
    source_slots: list[int] = field(default_factory=list)
    ready_event: object | None = None
