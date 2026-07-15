from abc import ABC
from collections import deque
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.kv_cache_interface import is_direct_sfa_host_offload
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    SFAKVOffloadConnectorMetadata,
    ReqMeta,
    RequestTracker,
)


def _num_touched_blocks(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


def _new_request_prompt_tokens(request: Any) -> int:
    """Read prompt length from the current scheduler-output contract."""
    return length_from_prompt_token_ids_or_embeds(
        request.prompt_token_ids,
        request.prompt_embeds,
    )


class CPUBlockManager(ABC):
    def __init__(self, block_num: int) -> None:
        self.block_num = block_num
        self.block_pool = deque(range(1, block_num))

    def allocate_block(self, new_block_num: int) -> list[int]:
        # logger.info(f'>>>>> pool scheduler allocate cpu block, require: {new_block_num}, resource: {len(self.block_pool)}')
        if len(self.block_pool) < new_block_num:
            raise ValueError("No enough cpu block to allocate")
        allocated_blocks = []
        for _ in range(new_block_num):
            allocated_blocks.append(self.block_pool.popleft())
        return allocated_blocks
    
    def free(self, to_free_blocks: list[int]):
        self.block_pool.extend(to_free_blocks)


class SFAKVOffloadlScheduler:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        use_layerwise,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        init_ascend_config(vllm_config)
        ascend_config = get_ascend_config()
        self.use_offload = ascend_config.use_offload
        self.use_direct_sfa_host_offload = is_direct_sfa_host_offload(
            vllm_config
        )
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
        self.group_block_sizes = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        if self.use_direct_sfa_host_offload:
            # vLLM collapses UniformTypeKVCacheSpecs to a representative spec
            # in the scheduler copy. One group is therefore the durable
            # scheduler-side invariant; the worker validates the Uniform type.
            if len(kv_cache_config.kv_cache_groups) != 1:
                raise ValueError(
                    "Direct SFA host offload scheduler requires one "
                    "logical KV cache group."
                )
            self._block_size = self.group_block_sizes[0]
        else:
            self._block_size = self.group_block_sizes[-1] # only offload kv cache

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._preempted_req_ids: set[str] = set()
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._unfinished_request_ids: set[str] = set()

        # sfa kv offload related
        npu_block_num = self.kv_cache_config.num_blocks
        # we need 4 * npu_blocks of cpu_blocks to fully store all offload blocks (dskv32, 512/128)
        # but you may want to set this to 1 in debug case in case of allocating to much dram
        # TODO remove this and directly compute from model config before merge
        cpu_block_num_multiple = (
            1 if self.use_direct_sfa_host_offload else 4
        )
        cpu_block_num = npu_block_num * cpu_block_num_multiple
        self.cpu_block_manager = CPUBlockManager(cpu_block_num)

    def _infer_group_block_sizes(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: KVCacheConfig | None,
    ) -> list[int]:
        block_sizes: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            block_sizes.append(kv_cache_spec.block_size)
        return block_sizes

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        """
        local_block_ids: list[list[int]] = []

        # TODO check whether these are useless now, delete them if so.
        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.
        """
        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._preempted_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._free_request_cpu_blocks(req_id)
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)
            self._unfinished_request_ids.discard(req_id)

        meta = SFAKVOffloadConnectorMetadata(self._unfinished_request_ids, scheduler_output.preempted_req_ids)

        for request in scheduler_output.scheduled_new_reqs:
            block_ids_npu = request.block_ids[
                0 if self.use_direct_sfa_host_offload else -1
            ].copy()
            write_start = request.num_computed_tokens
            write_count = scheduler_output.num_scheduled_tokens[request.req_id]
            is_prefill = write_start < _new_request_prompt_tokens(request)
            if is_prefill and write_start != 0:
                raise RuntimeError(
                    "SFA decode offload only supports one-shot local prefill; "
                    f"request {request.req_id} resumes at token {write_start}."
                )
            if (
                self.use_direct_sfa_host_offload
                and is_prefill
                and write_count > self._block_size
            ):
                raise RuntimeError(
                    "Direct SFA host offload supports one local prefill page; "
                    f"request {request.req_id} schedules {write_count} tokens "
                    f"with block_size={self._block_size}."
                )
            num_blocks_after_step = _num_touched_blocks(
                write_start + write_count, self._block_size
            )
            num_new_offload_blocks = num_blocks_after_step
            block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_offload_blocks)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                allocated_block_ids_npu=block_ids_npu,
                allocated_block_ids_cpu=block_ids_cpu,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                num_new_offload_blocks=num_new_offload_blocks,
                write_start=write_start,
                write_count=write_count,
                is_prefill=is_prefill,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            # resumed request
            new_block_ids_npu = cached_reqs.new_block_ids[i]
            if isinstance(new_block_ids_npu, tuple):
                new_block_ids_npu = new_block_ids_npu[
                    0 if self.use_direct_sfa_host_offload else -1
                ]
            elif new_block_ids_npu is None:
                new_block_ids_npu = []
            if req_id in self._preempted_req_ids:
                # treat as a new request
                num_computed_tokens = cached_reqs.num_computed_tokens[i]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                assert num_computed_tokens == 0
                num_new_offload_blocks = _num_touched_blocks(
                    num_new_tokens, self._block_size
                )
                block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_offload_blocks)
                request_tracker = RequestTracker(
                    req_id=req_id,
                    allocated_block_ids_npu=new_block_ids_npu,
                    allocated_block_ids_cpu=block_ids_cpu,
                )
                self._request_trackers[req_id] = request_tracker
                self._preempted_req_ids.discard(req_id)
                write_start = 0
                write_count = num_new_tokens
                is_prefill = True
                if (
                    self.use_direct_sfa_host_offload
                    and write_count > self._block_size
                ):
                    raise RuntimeError(
                        "Direct SFA host offload supports one local prefill "
                        f"page; request {req_id} schedules {write_count} "
                        f"tokens with block_size={self._block_size}."
                    )
            # decode/chunked request
            else:
                request_tracker = self._request_trackers[req_id]
                num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                req_tuple = self._unfinished_requests.get(req_id)
                if req_tuple:
                    request = req_tuple[0]
                else:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                    )
                num_computed_token = cached_reqs.num_computed_tokens[i]
                write_start = num_computed_token
                write_count = num_new_tokens
                is_prefill = num_computed_token < request.num_prompt_tokens
                if is_prefill:
                    raise RuntimeError(
                        "SFA decode offload does not support chunked or resumed "
                        f"prefill for request {req_id}."
                    )
                num_tokens_after_step = num_computed_token + num_new_tokens
                num_blocks_after_step = _num_touched_blocks(
                    num_tokens_after_step, self._block_size
                ) # pcp/dcp not considered now
                num_offloaded_blocks = len(request_tracker.allocated_block_ids_cpu)
                num_new_offload_blocks = max(num_blocks_after_step - num_offloaded_blocks, 0)
                new_block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_offload_blocks)
                request_tracker.update(new_block_ids_npu, new_block_ids_cpu)

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                num_new_offload_blocks=num_new_offload_blocks,
                write_start=write_start,
                write_count=write_count,
                is_prefill=is_prefill,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
        if self.use_direct_sfa_host_offload:
            prefill_requests = [
                request for request in meta.requests if request.is_prefill
            ]
            if prefill_requests and len(meta.requests) != 1:
                raise RuntimeError(
                    "Direct SFA host offload supports local prefill for one "
                    "request per batch only."
                )
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        self._free_request_cpu_blocks(request.request_id)
        return False, None
    
    def _free_request_cpu_blocks(
        self,
        request_id: str,
    ):
        tracker = self._request_trackers.get(request_id)
        if tracker is not None:
            self.cpu_block_manager.free(tracker.allocated_block_ids_cpu)
