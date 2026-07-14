from collections import deque
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.core.kv_cache_interface import get_sfa_hybrid_group_ids
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    ReqMeta,
    RequestTracker,
    SFAKVOffloadConnectorMetadata,
)


def _num_finalized_scheduled_tokens(scheduler_output: SchedulerOutput, req_id: str) -> int:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
    draft_tokens = scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
    return max(num_scheduled_tokens - len(draft_tokens), 0)


class CPUBlockManager:
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
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
        self.group_block_sizes = self._infer_group_block_sizes(vllm_config, kv_cache_config)
        self.main_group_ids, self.indexer_group_id = get_sfa_hybrid_group_ids(kv_cache_config.kv_cache_groups)
        main_block_sizes = {self.group_block_sizes[group_id] for group_id in self.main_group_ids}
        assert len(main_block_sizes) == 1, (
            f"All SFA main MLA groups must use the same block size, got {main_block_sizes}."
        )
        self._block_size = main_block_sizes.pop()

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
        cpu_block_num_multiple = 4
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
        """ """
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
            main_block_ids_by_group = {group_id: request.block_ids[group_id].copy() for group_id in self.main_group_ids}
            block_ids_npu = main_block_ids_by_group[self.main_group_ids[0]]
            num_tokens_to_compute = request.num_computed_tokens + _num_finalized_scheduled_tokens(
                scheduler_output, request.req_id
            )
            num_new_offload_blocks = num_tokens_to_compute // self._block_size
            block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_offload_blocks)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                allocated_block_ids_npu=list(block_ids_npu),
                allocated_block_ids_cpu=block_ids_cpu,
                main_block_ids_by_group=main_block_ids_by_group,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                num_new_offload_blocks=num_new_offload_blocks,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            # resumed request
            new_block_ids = cached_reqs.new_block_ids[i]
            if isinstance(new_block_ids, tuple):
                new_main_block_ids_by_group = {
                    group_id: list(new_block_ids[group_id]) for group_id in self.main_group_ids
                }
            else:
                canonical = [] if new_block_ids is None else list(new_block_ids)
                new_main_block_ids_by_group = {self.main_group_ids[0]: canonical}
            new_block_ids_npu = new_main_block_ids_by_group.get(self.main_group_ids[0], [])
            if req_id in self._preempted_req_ids:
                # treat as a new request
                num_computed_tokens = cached_reqs.num_computed_tokens[i]
                num_new_tokens = _num_finalized_scheduled_tokens(scheduler_output, req_id)
                assert num_computed_tokens == 0
                num_new_offload_blocks = num_new_tokens // self._block_size
                block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_offload_blocks)
                request_tracker = RequestTracker(
                    req_id=req_id,
                    allocated_block_ids_npu=list(new_block_ids_npu),
                    allocated_block_ids_cpu=block_ids_cpu,
                    main_block_ids_by_group=new_main_block_ids_by_group,
                )
                self._request_trackers[req_id] = request_tracker
                self._preempted_req_ids.discard(req_id)
            # decode/chunked request
            else:
                request_tracker = self._request_trackers[req_id]
                num_new_tokens = _num_finalized_scheduled_tokens(scheduler_output, req_id)
                req_tuple = self._unfinished_requests.get(req_id)
                if req_tuple:
                    request = req_tuple[0]
                else:
                    raise ValueError(
                        f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached"
                    )
                num_computed_token = cached_reqs.num_computed_tokens[i]
                num_tokens_after_step = num_computed_token + num_new_tokens
                num_blocks_after_step = num_tokens_after_step // self._block_size  # pcp/dcp not considered now
                num_offloaded_blocks = len(request_tracker.allocated_block_ids_cpu)
                num_new_offload_blocks = max(num_blocks_after_step - num_offloaded_blocks, 0)
                new_block_ids_cpu = self.cpu_block_manager.allocate_block(num_new_offload_blocks)
                request_tracker.update(
                    new_block_ids_npu,
                    new_block_ids_cpu,
                    new_main_block_ids_by_group,
                )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                num_new_offload_blocks=num_new_offload_blocks,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
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
        self.cpu_block_manager.free(tracker.allocated_block_ids_cpu)
