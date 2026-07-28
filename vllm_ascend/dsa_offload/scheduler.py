# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 稀疏卸载的最薄调度策略适配。

本模块不复制 vLLM ``Scheduler.schedule()``，也不定义平行的
``SchedulerOutput``。基线调度器仍负责 token budget、请求队列、KV block
分配和输出更新；DSA 子类只补充当前首版无法由通用调度器表达的三个约束：

* prefill 与 decode 暂不进入同一个 model forward；
* prefill 数据面完成后，scheduler 才能释放已卸载的 resident 满块；
* preemption/resume 尚未建立 DRAM ledger 恢复协议，发生时显式失败。

阶段与 budget 的语义真源位于 ``request_cache_layout``，物理 block table 由
``DSAKVCacheCoordinator`` 持有。这里不向 vLLM ``Request`` 动态追加字段。
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSAKVCacheCoordinator,
)


def _is_prefill_request(request: Request) -> bool:
    return request.num_output_tokens == 0 and request.num_computed_tokens < request.num_prompt_tokens


class DSAOffloadScheduler(Scheduler):
    """在 vLLM 基线调度循环外增加 DSA 首版生命周期约束。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        coordinator = self.kv_cache_manager.coordinator
        if not isinstance(coordinator, DSAKVCacheCoordinator):
            raise RuntimeError(f"DSAOffloadScheduler requires DSAKVCacheCoordinator, got {type(coordinator).__name__}")
        self.dsa_coordinator = coordinator

    def _has_running_prefill_work(self) -> bool:
        return any(_is_prefill_request(request) for request in self.running)

    def _has_schedulable_waiting_prefill(self, token_budget: int) -> bool:
        """判断队首 prefill 是否真能推进，避免阻塞可运行的 decode。

        仅仅看见 waiting prefill 就暂停 decode 会造成死锁：当两个物理池
        暂时装不下该 prompt 时，必须先允许已有 decode 完成并释放空间。
        """

        if token_budget <= 0 or len(self.running) >= self.max_num_running_reqs:
            return False

        request_queue = self._select_waiting_queue_for_scheduling()
        if request_queue is None:
            return False
        request = request_queue.peek_request()
        if not _is_prefill_request(request) or self._is_blocked_waiting_status(request.status):
            return False

        num_new_tokens = request.num_tokens - request.num_computed_tokens
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
            return False
        if min(num_new_tokens, token_budget) <= 0:
            return False

        return self.dsa_coordinator.can_admit_dense_request(
            request_id=request.request_id,
            num_tokens=request.num_tokens,
            total_computed_tokens=request.num_computed_tokens,
        )

    def _has_ready_decode_work(self) -> bool:
        for request in self.running:
            if request.num_output_tokens <= 0:
                continue
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )
            if num_new_tokens > 0:
                return True
        return False

    def _withhold_decode_running_for_prefill(
        self,
    ) -> Callable[[], None] | None:
        """让基线调度器本轮只看见 prefill，同时保留原始 running 顺序。"""

        withheld: list[tuple[int, Request]] = []
        kept: list[Request] = []
        for index, request in enumerate(self.running):
            if request.num_output_tokens > 0:
                withheld.append((index, request))
            else:
                kept.append(request)
        if not withheld:
            return None

        old_max_num_running_reqs = self.max_num_running_reqs
        self.running = kept
        self.max_num_running_reqs = max(
            0,
            old_max_num_running_reqs - len(withheld),
        )

        def restore() -> None:
            restored = list(self.running)
            for index, request in withheld:
                if (
                    request in restored
                    or request.request_id not in self.requests
                    or request.status != RequestStatus.RUNNING
                ):
                    continue
                restored.insert(min(index, len(restored)), request)
            self.running = restored
            self.max_num_running_reqs = old_max_num_running_reqs

        return restore

    def _withhold_waiting_for_decode(self) -> Callable[[], None]:
        """让基线调度器完成已有 decode，本轮不再接纳新 prefill。"""

        old_waiting = self.waiting
        old_skipped_waiting = self.skipped_waiting
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        def restore() -> None:
            self.waiting = old_waiting
            self.skipped_waiting = old_skipped_waiting

        return restore

    def schedule(self) -> SchedulerOutput:
        # 首版禁用 chunked prefill、async scheduling 和 PP；一次完整
        # prefill 的 schedule/update_from_output 之间不会再次进入 scheduler。
        # 因此没有 waiting 请求时不可能在下一轮遗留可调度 prefill，直接
        # 走上游快路径，避免 steady decode 额外构造空队列或扫描 row。
        if not self.waiting and not self.skipped_waiting:
            return super().schedule()

        token_budget = 0 if self._pause_state == PauseState.PAUSED_ALL else self.max_num_scheduled_tokens
        prefill_barrier = self._has_running_prefill_work() or self._has_schedulable_waiting_prefill(token_budget)

        restore: Callable[[], None] | None = None
        if prefill_barrier:
            restore = self._withhold_decode_running_for_prefill()
        elif self._has_ready_decode_work():
            restore = self._withhold_waiting_for_decode()

        try:
            return super().schedule()
        finally:
            if restore is not None:
                restore()

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        del timestamp
        raise RuntimeError(f"DSA sparse offload does not yet support request preemption: req_id={request.request_id}")

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        outputs = super().update_from_output(
            scheduler_output,
            model_runner_output,
        )

        # full-block dump 与当前 model forward 在同一 NPU stream 上有序。
        # 只有 model output 返回 scheduler 后才释放 HBM resident 满块。
        # 若未来引入异步多 stream dump，必须把这里升级为 event/readiness
        # 协议，不能只凭 host 已返回就释放。
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if (
                request is None
                or request.is_finished()
                or not self.dsa_coordinator.request_cache_layout.should_release_resident_after_prefill(request)
            ):
                continue
            self.dsa_coordinator.release_prefill_resident_blocks(
                request_id,
                preserve_tail_block=(request.num_prompt_tokens % self.block_size != 0),
            )
        return outputs
