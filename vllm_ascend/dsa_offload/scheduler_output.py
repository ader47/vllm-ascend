# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA scheduler→worker 的最小类型化投影。

vLLM v0.23 的 ``SchedulerOutput`` 对 cached request 只传新增 block IDs，
无法表达 DSA ENTER step 对 resident MLA block table 的整表替换。这里使用
一个薄子类增加单个 DSA projection 字段：

* 不修改 vLLM；
* 不复制 ``SchedulerOutput`` 的调度或序列化逻辑；
* 所有基线字段仍保持原对象引用，只做一次固定字段数的浅包装；
* request 语义按 ``num_scheduled_tokens`` 的顺序列式传输，避免四份
  request-id→scalar 字典；
* 只有 ENTER 行携带 resident block IDs 全量快照。

多进程执行器使用 pickle 传递 scheduler output，因此该类型化子类会随普通
``SchedulerOutput`` 一起跨进程恢复。worker 最终仍按自己的 ``InputBatch``
行序重排这些列。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)


@dataclass(frozen=True, slots=True)
class DSAResidentBlockTableReplacement:
    """一个 ENTER 请求需要覆盖到 worker 的 resident 全量 block IDs。"""

    request_id: str
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.block_ids:
            raise ValueError("DSA resident block-table replacement cannot be empty")


@dataclass(frozen=True, slots=True)
class DSARequestCacheLayoutProjection:
    """一次 model-forward 中 scheduled request 的 cache 布局列。"""

    request_ids: tuple[str, ...]
    stages: tuple[int, ...]
    target_resident_budget_tokens: tuple[int, ...]
    sparse_budget_tokens: tuple[int, ...]
    resident_valid_tokens: tuple[int, ...]
    resident_block_table_replacements: tuple[
        DSAResidentBlockTableReplacement,
        ...,
    ]

    def __post_init__(self) -> None:
        row_count = len(self.request_ids)
        column_lengths = (
            len(self.stages),
            len(self.target_resident_budget_tokens),
            len(self.sparse_budget_tokens),
            len(self.resident_valid_tokens),
        )
        if any(length != row_count for length in column_lengths):
            raise ValueError(
                "DSA cache-layout projection columns have different lengths: "
                f"request_ids={row_count}, columns={column_lengths}"
            )
        enter_rows = sum(stage == int(DSARequestCacheStage.ENTER_SPARSE_DECODE) for stage in self.stages)
        if len(self.resident_block_table_replacements) != enter_rows:
            raise ValueError(
                "DSA resident replacement count does not match ENTER rows: "
                f"replacements={len(self.resident_block_table_replacements)}, "
                f"enter_rows={enter_rows}"
            )

    @property
    def num_rows(self) -> int:
        return len(self.request_ids)

    @property
    def num_enter_rows(self) -> int:
        return len(self.resident_block_table_replacements)


_BASE_SCHEDULER_OUTPUT_FIELD_NAMES = tuple(field.name for field in fields(SchedulerOutput))


@dataclass
class DSAOffloadSchedulerOutput(SchedulerOutput):
    """保持基线字段语义、仅增加 DSA cache-layout projection 的薄子类。"""

    dsa_cache_layout: DSARequestCacheLayoutProjection | None = None

    @classmethod
    def from_base(
        cls,
        output: SchedulerOutput,
        *,
        dsa_cache_layout: DSARequestCacheLayoutProjection,
    ) -> DSAOffloadSchedulerOutput:
        if isinstance(output, cls):
            output.dsa_cache_layout = dsa_cache_layout
            return output
        base_fields = {name: getattr(output, name) for name in _BASE_SCHEDULER_OUTPUT_FIELD_NAMES}
        return cls(
            **base_fields,
            dsa_cache_layout=dsa_cache_layout,
        )
