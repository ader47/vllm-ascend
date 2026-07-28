# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.scheduler import DSAOffloadScheduler
from vllm_ascend.dsa_offload.scheduler_output import (
    DSAResidentBlockTableReplacement,
)


@dataclass(eq=False)
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_tokens: int
    status: RequestStatus = RequestStatus.RUNNING
    num_output_placeholders: int = 0
    max_tokens: int = 128

    @property
    def num_tokens_with_spec(self) -> int:
        return self.num_tokens


class _AdmissionGate:
    def __init__(self, can_admit: bool) -> None:
        self.can_admit = can_admit
        self.calls = 0

    def can_admit_dense_request(self, **kwargs) -> bool:
        del kwargs
        self.calls += 1
        return self.can_admit


def _make_scheduler() -> DSAOffloadScheduler:
    scheduler = object.__new__(DSAOffloadScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = []
    scheduler.requests = {}
    scheduler.max_num_running_reqs = 4
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.max_model_len = 16384
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
        enable_chunked_prefill=False,
    )
    return scheduler


def test_withhold_decode_restores_original_running_order() -> None:
    scheduler = _make_scheduler()
    decode_a = _Request("decode-a", 1000, 1000, 1, 1001)
    prefill = _Request("prefill", 1000, 512, 0, 1000)
    decode_b = _Request("decode-b", 1000, 1000, 1, 1001)
    scheduler.running = [decode_a, prefill, decode_b]
    scheduler.requests = {request.request_id: request for request in scheduler.running}

    restore = scheduler._withhold_decode_running_for_prefill()

    assert restore is not None
    assert scheduler.running == [prefill]
    assert scheduler.max_num_running_reqs == 2
    restore()
    assert scheduler.running == [decode_a, prefill, decode_b]
    assert scheduler.max_num_running_reqs == 4


def test_waiting_prefill_only_blocks_decode_when_both_pools_can_admit() -> None:
    scheduler = _make_scheduler()
    decode = _Request("decode", 1000, 1000, 1, 1001)
    prefill = _Request(
        "prefill",
        4096,
        0,
        0,
        4096,
        status=RequestStatus.WAITING,
    )
    scheduler.running = [decode]
    scheduler.requests = {
        decode.request_id: decode,
        prefill.request_id: prefill,
    }
    scheduler.waiting.add_request(prefill)

    gate = _AdmissionGate(False)
    scheduler.dsa_coordinator = gate  # type: ignore[assignment]
    assert not scheduler._has_schedulable_waiting_prefill(8192)

    gate.can_admit = True
    assert scheduler._has_schedulable_waiting_prefill(8192)
    assert gate.calls == 2


def test_decode_barrier_temporarily_hides_both_waiting_queues() -> None:
    scheduler = _make_scheduler()
    waiting = _Request(
        "waiting",
        1000,
        0,
        0,
        1000,
        status=RequestStatus.WAITING,
    )
    scheduler.waiting.add_request(waiting)
    original_waiting = scheduler.waiting
    original_skipped = scheduler.skipped_waiting

    restore = scheduler._withhold_waiting_for_decode()

    assert not scheduler.waiting
    assert not scheduler.skipped_waiting
    restore()
    assert scheduler.waiting is original_waiting
    assert scheduler.skipped_waiting is original_skipped


def test_ready_decode_detection_ignores_prefill_rows() -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_Request("prefill", 1000, 512, 0, 1000)]
    assert not scheduler._has_ready_decode_work()

    scheduler.running.append(_Request("decode", 1000, 1000, 1, 1001))
    assert scheduler._has_ready_decode_work()


def test_no_waiting_requests_use_the_upstream_schedule_fast_path(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_Request("decode", 1000, 1000, 1, 1001)]
    marker = object()

    def fail_if_gate_is_evaluated() -> bool:
        raise AssertionError("steady decode should not evaluate the DSA phase gate")

    scheduler._has_running_prefill_work = fail_if_gate_is_evaluated  # type: ignore[method-assign]
    scheduler._attach_dsa_cache_layout = lambda output: output  # type: ignore[method-assign]
    monkeypatch.setattr(Scheduler, "schedule", lambda self: marker)

    assert scheduler.schedule() is marker


def test_projection_carries_enter_resident_table_and_scalar_state() -> None:
    scheduler = _make_scheduler()
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"dense": 1, "enter": 1}
    output.total_num_scheduled_tokens = 2
    states = {
        "dense": SimpleNamespace(
            stage=DSARequestCacheStage.DENSE_DECODE,
            target_resident_budget_tokens=2048,
            sparse_budget_tokens=0,
            resident_valid_tokens=-1,
        ),
        "enter": SimpleNamespace(
            stage=DSARequestCacheStage.ENTER_SPARSE_DECODE,
            target_resident_budget_tokens=4096,
            sparse_budget_tokens=4096,
            resident_valid_tokens=4097,
        ),
    }
    scheduler.dsa_coordinator = SimpleNamespace(
        get_request_cache_state=states.get,
        resident_manager=SimpleNamespace(
            req_to_blocks={
                "enter": [
                    SimpleNamespace(block_id=301),
                    SimpleNamespace(block_id=302),
                ]
            }
        ),
    )

    projection = scheduler._build_dsa_cache_layout_projection(output)

    assert projection.request_ids == ("dense", "enter")
    assert projection.stages == (
        int(DSARequestCacheStage.DENSE_DECODE),
        int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
    )
    assert projection.target_resident_budget_tokens == (2048, 4096)
    assert projection.resident_block_table_replacements == (
        DSAResidentBlockTableReplacement(
            request_id="enter",
            block_ids=(301, 302),
        ),
    )
    assert projection.num_enter_rows == 1
