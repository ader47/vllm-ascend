# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA row-mode decode 对原生 FULL graph 的纯准入策略。

本模块只读取已经投影到 ``NPUInputBatch`` 的请求阶段，并回答当前 forward
是否允许复用 vLLM/vLLM-Ascend 原生 FULL decode 图。它不创建图、不持有
buffer，也不修改请求生命周期：

* 原生 uniform decode（普通 Q=1 或 MTP Q=1+D）的
  DENSE/ENTER/SPARSE 任意混排可以入图；
* active batch 可以向上匹配 capture size，额外行由统一 owner 提供 PAD；
* prefill、非 uniform query 和 capture size 未覆盖属于正常 true-eager；
* 状态缺失、行数错位等内部合同破坏必须显式失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.input_batch import (
        DSAInputBatchCacheLayout,
    )


_EXPECTED_EAGER_REASONS = frozenset(
    {
        "empty_batch",
        "non_uniform_decode",
        "non_decode_stage",
        "capture_size_miss",
    }
)


@dataclass(frozen=True, slots=True)
class DSARowModeGraphDecision:
    """一次 DSA FULL-decode 图准入结果。"""

    use_full_graph: bool
    reason: str

    @property
    def is_expected_eager(self) -> bool:
        return self.reason in _EXPECTED_EAGER_REASONS


def evaluate_dsa_row_mode_decode_graph(
    *,
    state: DSAInputBatchCacheLayout,
    num_reqs: int,
    total_num_scheduled_tokens: int,
    max_num_scheduled_tokens: int,
    uniform_decode_query_len: int,
    max_capture_size: int,
) -> DSARowModeGraphDecision:
    """判断当前 forward 是否满足 DSA row-mode FULL graph 合同。"""

    num_reqs = int(num_reqs)
    if not state.valid:
        return DSARowModeGraphDecision(
            False,
            "missing_input_batch_state",
        )
    if state.row_count != num_reqs:
        return DSARowModeGraphDecision(
            False,
            "input_batch_row_mismatch",
        )
    if num_reqs <= 0:
        return DSARowModeGraphDecision(False, "empty_batch")

    uniform_decode_query_len = int(uniform_decode_query_len)
    if uniform_decode_query_len <= 0:
        return DSARowModeGraphDecision(
            False,
            "invalid_uniform_decode_query_len",
        )
    # 直接复用原生 dispatcher 的 uniform query 定义。MTP3 的稳定图是
    # 每请求四条 target query；双尾块覆盖 128-token 边界，不需要为了
    # cache 布局把请求降级为 Q=1。只有基线实际给出的变长 query batch
    # 才按原生范式走 eager。
    if (
        int(total_num_scheduled_tokens)
        != num_reqs * uniform_decode_query_len
        or int(max_num_scheduled_tokens) != uniform_decode_query_len
    ):
        return DSARowModeGraphDecision(
            False,
            "non_uniform_decode",
        )
    if int(max_capture_size) <= 0:
        return DSARowModeGraphDecision(
            False,
            "missing_capture_sizes",
        )
    if int(max_capture_size) < num_reqs:
        return DSARowModeGraphDecision(False, "capture_size_miss")

    stages = state.stages_cpu[:num_reqs]
    first_decode_stage = int(DSARequestCacheStage.DENSE_DECODE)
    last_decode_stage = int(DSARequestCacheStage.SPARSE_DECODE)
    if (
        np.any(stages < first_decode_stage)
        or np.any(stages > last_decode_stage)
    ):
        return DSARowModeGraphDecision(False, "non_decode_stage")

    return DSARowModeGraphDecision(True, "allow_row_mode_decode")
