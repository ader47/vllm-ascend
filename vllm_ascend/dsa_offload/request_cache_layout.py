# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 请求级 KV-cache 布局规划。

本模块不是一套独立的 vLLM request lifecycle，也不接管 ``RequestStatus``。
它只维护 DSA 双平面 cache 布局在请求维度的最小跨 step 状态：

* ``PREFILL``：Indexer 与 resident MLA 都按完整上下文分配；
* ``DENSE_DECODE``：尚未满足稀疏条件，两个 plane 继续按完整上下文增长；
* ``ENTER_SPARSE_DECODE``：本轮首次把 resident MLA 收缩为 budget + tail；
* ``SPARSE_DECODE``：Indexer 继续保存完整上下文，resident 物理块表保持稳定。

planner 采用 plan/commit 两阶段协议。容量检查和物理分配成功前只生成轻量
候选计划；失败重试不会提前推进请求阶段，也不会留下半更新的 resident
状态。scheduler 当前是单线程同步执行，因此每个请求只持有一个可变的
``DSARequestCacheState``，commit 在物理分配成功后原地刷新，避免每个
decode step 重建状态对象。

该账本位于 scheduler/core 进程，只保存标量语义，不承担 worker tensor 行
状态。P4 会把已 commit 的状态投影到 ``NPUInputBatch``；eager/graph 都应
消费同一个投影，而不是各自重新推导阶段。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


class DSARequestCacheStage(enum.IntEnum):
    """单请求在 scheduler 账本中的 DSA cache 布局阶段。"""

    PREFILL = 0
    DENSE_DECODE = 1
    ENTER_SPARSE_DECODE = 2
    SPARSE_DECODE = 3

    @property
    def is_sparse_decode(self) -> bool:
        return self in (
            DSARequestCacheStage.ENTER_SPARSE_DECODE,
            DSARequestCacheStage.SPARSE_DECODE,
        )


@dataclass(slots=True)
class DSARequestCacheState:
    """一个请求跨 step 持久化的 scheduler cache 布局账本。

    ``sparse_budget_tokens`` 和 ``resident_valid_tokens`` 虽可由当前请求
    长度重新推导，但保留在这里作为 P4 scheduler→worker 投影的单一真源，
    避免 worker、eager 和 graph 分别实现阶段计算。
    """

    stage: DSARequestCacheStage
    target_resident_budget_tokens: int
    sparse_budget_tokens: int = 0
    resident_valid_tokens: int = -1
    prefill_resident_released: bool = False


@dataclass(frozen=True, slots=True)
class DSARequestCachePlan:
    """一次 ``allocate_slots`` 的轻量、不可变候选布局。"""

    request_id: str
    stage: DSARequestCacheStage
    target_resident_budget_tokens: int
    indexer_tokens_need_slot: int
    sparse_budget_tokens: int
    resident_valid_tokens: int
    preserve_resident_tail_block: bool

    @property
    def replace_resident_blocks(self) -> bool:
        """ENTER step 需要把 dense resident 表替换为 budget + tail。"""

        return self.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE

    @property
    def tail_tokens(self) -> int:
        """返回 resident 中预算区之后的有效尾块 token 数。"""

        if not self.stage.is_sparse_decode:
            return 0
        return self.resident_valid_tokens - self.sparse_budget_tokens


class DSARequestCachePlanner:
    """按 prompt 档位和上下文长度规划 DSA 请求 cache 布局。"""

    def __init__(
        self,
        *,
        block_size: int,
        sparse_activation_tokens: int,
        prompt_budget_thresholds: tuple[int, ...],
        resident_budget_tokens: tuple[int, ...],
    ) -> None:
        self.block_size = int(block_size)
        self.sparse_activation_tokens = int(sparse_activation_tokens)
        self.prompt_budget_thresholds = tuple(int(value) for value in prompt_budget_thresholds)
        self.resident_budget_tokens = tuple(int(value) for value in resident_budget_tokens)
        if self.block_size <= 0:
            raise ValueError("DSA cache-layout block_size must be positive")
        if len(self.resident_budget_tokens) != (len(self.prompt_budget_thresholds) + 1):
            raise ValueError(
                "DSA cache-layout resident budgets must contain exactly one more entry than prompt thresholds"
            )
        self._states: dict[str, DSARequestCacheState] = {}

    def select_target_resident_budget_tokens(
        self,
        prompt_tokens: int,
    ) -> int:
        prompt_tokens = max(0, int(prompt_tokens))
        for threshold, budget in zip(
            self.prompt_budget_thresholds,
            self.resident_budget_tokens,
            strict=False,
        ):
            if prompt_tokens <= threshold:
                return budget
        return self.resident_budget_tokens[-1]

    def get_state(
        self,
        request_id: str,
    ) -> DSARequestCacheState | None:
        return self._states.get(request_id)

    def plan(
        self,
        request: Request,
        *,
        num_new_tokens: int,
        max_model_len: int,
    ) -> DSARequestCachePlan:
        """生成本轮布局候选，不修改跨 step 状态。"""

        previous_state = self._states.get(request.request_id)
        if previous_state is None:
            previous_stage = DSARequestCacheStage.PREFILL
            target_budget = self.select_target_resident_budget_tokens(request.num_prompt_tokens)
        else:
            previous_stage = previous_state.stage
            target_budget = previous_state.target_resident_budget_tokens

        indexer_tokens_need_slot = min(
            int(request.num_computed_tokens) + int(num_new_tokens),
            int(max_model_len),
        )
        logical_context_tokens = min(
            int(request.num_tokens),
            int(max_model_len),
        )
        dense_stage = (
            DSARequestCacheStage.PREFILL if int(request.num_output_tokens) == 0 else DSARequestCacheStage.DENSE_DECODE
        )

        sparse_budget_tokens = 0
        resident_valid_tokens = -1
        next_stage = dense_stage

        decode_ready = int(request.num_output_tokens) > 0 and int(request.num_computed_tokens) >= int(
            request.num_prompt_tokens
        )
        if decode_ready and logical_context_tokens > self.sparse_activation_tokens:
            full_blocks_before_tail = (logical_context_tokens - 1) // self.block_size
            candidate_tokens = full_blocks_before_tail * self.block_size
            if candidate_tokens >= target_budget:
                sparse_budget_tokens = target_budget
                tail_tokens = logical_context_tokens - candidate_tokens
                resident_valid_tokens = sparse_budget_tokens + tail_tokens
                next_stage = (
                    DSARequestCacheStage.SPARSE_DECODE
                    if previous_stage.is_sparse_decode
                    else DSARequestCacheStage.ENTER_SPARSE_DECODE
                )

        if previous_stage.is_sparse_decode and not next_stage.is_sparse_decode:
            raise RuntimeError(
                "DSA request cache layout cannot move from sparse decode "
                f"back to dense: req_id={request.request_id}, "
                f"previous={previous_stage.name}, next={next_stage.name}"
            )

        preserve_resident_tail_block = (
            next_stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
            and int(request.num_computed_tokens) % self.block_size != 0
        )
        return DSARequestCachePlan(
            request_id=request.request_id,
            stage=next_stage,
            target_resident_budget_tokens=target_budget,
            indexer_tokens_need_slot=indexer_tokens_need_slot,
            sparse_budget_tokens=sparse_budget_tokens,
            resident_valid_tokens=resident_valid_tokens,
            preserve_resident_tail_block=preserve_resident_tail_block,
        )

    def commit(self, plan: DSARequestCachePlan) -> None:
        """物理分配成功后提交候选布局。"""

        state = self._states.get(plan.request_id)
        if state is None:
            self._states[plan.request_id] = DSARequestCacheState(
                stage=plan.stage,
                target_resident_budget_tokens=(plan.target_resident_budget_tokens),
                sparse_budget_tokens=plan.sparse_budget_tokens,
                resident_valid_tokens=plan.resident_valid_tokens,
            )
            return

        if state.target_resident_budget_tokens != plan.target_resident_budget_tokens:
            raise RuntimeError(
                "DSA request cache budget changed after admission: "
                f"req_id={plan.request_id}, "
                f"committed={state.target_resident_budget_tokens}, "
                f"planned={plan.target_resident_budget_tokens}"
            )
        state.stage = plan.stage
        state.sparse_budget_tokens = plan.sparse_budget_tokens
        state.resident_valid_tokens = plan.resident_valid_tokens

    def should_release_resident_after_prefill(
        self,
        request: Request,
    ) -> bool:
        state = self._states.get(request.request_id)
        if (
            state is None
            or state.prefill_resident_released
            or state.stage != DSARequestCacheStage.PREFILL
            or int(request.num_computed_tokens) < int(request.num_prompt_tokens)
            or int(request.num_prompt_tokens) <= self.sparse_activation_tokens
        ):
            return False
        candidate_full_tokens = (int(request.num_prompt_tokens) // self.block_size) * self.block_size
        return candidate_full_tokens >= state.target_resident_budget_tokens

    def mark_prefill_resident_released(
        self,
        request_id: str,
    ) -> None:
        state = self._states.get(request_id)
        if state is None:
            raise RuntimeError(
                f"Cannot release DSA prefill resident blocks without a cache-layout state: req_id={request_id}"
            )
        state.prefill_resident_released = True

    def free(self, request_id: str) -> None:
        self._states.pop(request_id, None)
