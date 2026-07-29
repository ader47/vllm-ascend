# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 在 worker ``NPUInputBatch`` 行序上的固定容量 cache 布局投影。

scheduler/core 的 ``DSARequestCachePlanner`` 是跨 step 语义真源；本模块只在
原生 ``_update_states`` 完成 remove/add/condense/reorder 后，将本轮
projection 写入最终 InputBatch 行序。它不维护第二套请求生命周期。

六列状态共用一个 ``CpuGpuBuffer``：

``stage, target_budget, sparse_budget, resident_valid_tokens, row_mode,
resident_pool_index``

CPU view 服务 host 控制面，固定地址 device view 预留给后续 eager/graph
共同消费。P4 只建立投影并修正 ENTER 的 resident block table，不主动触发
H2D；P5/P6 复用这个 owner，而不是另建 graph-only metadata。稳定请求行序
下六列采用批量刷新；请求增删或重排时，才执行一次 request-id 映射并同步
稳定 resident-pool 行。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.dsa_offload.contracts import (
    DSA_ROW_MODE_DENSE,
    DSA_ROW_MODE_PAD,
    DSA_ROW_MODE_SPARSE,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool
from vllm_ascend.dsa_offload.scheduler_output import (
    DSARequestCacheLayoutProjection,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import CachedRequestState

    from vllm_ascend.worker.npu_input_batch import NPUInputBatch


_INVALID_STAGE = -1
_INVALID_RESIDENT_LENGTH = -1
_STAGE_COLUMN = 0
_TARGET_BUDGET_COLUMN = 1
_SPARSE_BUDGET_COLUMN = 2
_RESIDENT_VALID_COLUMN = 3
_ROW_MODE_COLUMN = 4
_RESIDENT_POOL_INDEX_COLUMN = 5
_NUM_COLUMNS = 6


class DSAInputBatchCacheLayout:
    """与一个 ``NPUInputBatch`` 同生命周期的 DSA 固定容量列。"""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool,
        resident_token_pool: DSAResidentTokenPool,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("DSA InputBatch cache-layout capacity must be positive")
        self.max_num_reqs = int(max_num_reqs)
        if resident_token_pool.max_num_reqs != self.max_num_reqs:
            raise ValueError(
                "DSA InputBatch and resident pool capacities differ: "
                f"input_batch={self.max_num_reqs}, "
                f"resident_pool={resident_token_pool.max_num_reqs}"
            )
        self.resident_token_pool = resident_token_pool
        self.columns = CpuGpuBuffer(
            _NUM_COLUMNS,
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self.columns.np[_STAGE_COLUMN].fill(_INVALID_STAGE)
        self.columns.np[_TARGET_BUDGET_COLUMN].fill(0)
        self.columns.np[_SPARSE_BUDGET_COLUMN].fill(0)
        self.columns.np[_RESIDENT_VALID_COLUMN].fill(_INVALID_RESIDENT_LENGTH)
        self.columns.np[_ROW_MODE_COLUMN].fill(DSA_ROW_MODE_PAD)
        self.columns.np[_RESIDENT_POOL_INDEX_COLUMN].fill(
            resident_token_pool.padding_pool_index
        )
        self._sparse_row_mask = np.empty(
            self.max_num_reqs,
            dtype=np.bool_,
        )
        self._request_ids: tuple[str, ...] = ()
        self._pool_release_callback: Callable[[int], None] | None = None
        self.mapping_version = 0
        self.row_count = 0
        self.valid = False

    @property
    def stages_cpu(self) -> np.ndarray:
        return self.columns.np[_STAGE_COLUMN]

    @property
    def target_resident_budget_tokens_cpu(self) -> np.ndarray:
        return self.columns.np[_TARGET_BUDGET_COLUMN]

    @property
    def sparse_budget_tokens_cpu(self) -> np.ndarray:
        return self.columns.np[_SPARSE_BUDGET_COLUMN]

    @property
    def resident_valid_tokens_cpu(self) -> np.ndarray:
        return self.columns.np[_RESIDENT_VALID_COLUMN]

    @property
    def row_modes_cpu(self) -> np.ndarray:
        return self.columns.np[_ROW_MODE_COLUMN]

    @property
    def resident_pool_indices_cpu(self) -> np.ndarray:
        return self.columns.np[_RESIDENT_POOL_INDEX_COLUMN]

    @property
    def stages(self) -> torch.Tensor:
        return self.columns.gpu[_STAGE_COLUMN]

    @property
    def target_resident_budget_tokens(self) -> torch.Tensor:
        return self.columns.gpu[_TARGET_BUDGET_COLUMN]

    @property
    def sparse_budget_tokens(self) -> torch.Tensor:
        return self.columns.gpu[_SPARSE_BUDGET_COLUMN]

    @property
    def resident_valid_tokens(self) -> torch.Tensor:
        return self.columns.gpu[_RESIDENT_VALID_COLUMN]

    @property
    def row_modes(self) -> torch.Tensor:
        return self.columns.gpu[_ROW_MODE_COLUMN]

    @property
    def resident_pool_indices(self) -> torch.Tensor:
        return self.columns.gpu[_RESIDENT_POOL_INDEX_COLUMN]

    def set_pool_release_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """绑定 P5 DRAM ledger 的稳定 pool-row 释放入口。"""

        self._pool_release_callback = callback

    def clear(self, *, input_batch: NPUInputBatch | None = None) -> None:
        """清空本轮 view，并同步请求增删产生的 resident-pool 行变化。"""

        old_row_count = self.row_count
        if old_row_count:
            rows = slice(0, old_row_count)
            self.stages_cpu[rows] = _INVALID_STAGE
            self.target_resident_budget_tokens_cpu[rows] = 0
            self.sparse_budget_tokens_cpu[rows] = 0
            self.resident_valid_tokens_cpu[rows] = _INVALID_RESIDENT_LENGTH
            self.row_modes_cpu[rows] = DSA_ROW_MODE_PAD
            self.resident_pool_indices_cpu[rows] = (
                self.resident_token_pool.padding_pool_index
            )
        if input_batch is not None:
            request_ids = tuple(input_batch.req_ids[: input_batch.num_reqs])
            self._synchronize_resident_pool_rows(request_ids)
        self.row_count = 0
        self.valid = False

    def refresh(
        self,
        *,
        input_batch: NPUInputBatch,
        projection: DSARequestCacheLayoutProjection,
    ) -> None:
        """按原生 InputBatch 的最终行号写入本轮所有 scheduled request。"""

        row_count = int(input_batch.num_reqs)
        if row_count != projection.num_rows:
            raise RuntimeError(
                "DSA cache-layout projection does not cover the final "
                f"InputBatch: projection_rows={projection.num_rows}, "
                f"input_rows={row_count}"
            )
        if row_count > self.max_num_reqs:
            raise RuntimeError(
                f"DSA InputBatch cache-layout capacity exceeded: rows={row_count}, capacity={self.max_num_reqs}"
            )

        old_row_count = self.row_count
        self.valid = False
        input_request_ids = tuple(input_batch.req_ids[:row_count])
        projection_matches_rows = projection.request_ids == input_request_ids
        if projection_matches_rows:
            # steady decode 的 scheduler 紧凑行序通常与最终 InputBatch 行序
            # 完全一致。此时直接批量刷新四列，避免再做一次逐请求 dict
            # 查找和 Python 标量写入。请求增删、压缩或重排时仍走下方按
            # request_id 映射的精确回退，不依赖两侧行序恰好一致。
            rows = slice(0, row_count)
            self.stages_cpu[rows] = projection.stages
            self.target_resident_budget_tokens_cpu[rows] = (
                projection.target_resident_budget_tokens
            )
            self.sparse_budget_tokens_cpu[rows] = projection.sparse_budget_tokens
            self.resident_valid_tokens_cpu[rows] = (
                projection.resident_valid_tokens
            )
        else:
            if row_count:
                self.stages_cpu[:row_count] = _INVALID_STAGE
            for source_row, request_id in enumerate(projection.request_ids):
                row = input_batch.req_id_to_index.get(request_id)
                if row is None or row >= row_count:
                    raise RuntimeError(
                        "DSA cache-layout projection has no matching "
                        "InputBatch row: "
                        f"request_id={request_id!r}, row={row}"
                    )
                self.stages_cpu[row] = projection.stages[source_row]
                self.target_resident_budget_tokens_cpu[row] = (
                    projection.target_resident_budget_tokens[source_row]
                )
                self.sparse_budget_tokens_cpu[row] = (
                    projection.sparse_budget_tokens[source_row]
                )
                self.resident_valid_tokens_cpu[row] = (
                    projection.resident_valid_tokens[source_row]
                )

        if input_request_ids != self._request_ids:
            self._synchronize_resident_pool_rows(input_request_ids)

        rows = slice(0, row_count)
        stages = self.stages_cpu[rows]
        sparse_row_mask = self._sparse_row_mask[:row_count]
        np.greater_equal(
            stages,
            int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
            out=sparse_row_mask,
        )
        row_modes = self.row_modes_cpu[rows]
        row_modes.fill(DSA_ROW_MODE_DENSE)
        np.copyto(
            row_modes,
            DSA_ROW_MODE_SPARSE,
            where=sparse_row_mask,
        )

        # ENTER 只持续一个 step。仅这些转换行需要为所有层写一次 first-fill
        # 负预算；steady DENSE/SPARSE 不再做逐请求 pool 查询。
        enter_rows = np.flatnonzero(
            stages == int(DSARequestCacheStage.ENTER_SPARSE_DECODE)
        )
        for row in enter_rows:
            self.resident_token_pool.prepare_sparse_request(
                input_request_ids[int(row)],
                target_budget_tokens=int(
                    self.target_resident_budget_tokens_cpu[int(row)]
                ),
            )

        if row_count and np.any(self.stages_cpu[:row_count] == _INVALID_STAGE):
            raise RuntimeError("DSA cache-layout projection left an InputBatch row unset")
        if old_row_count > row_count:
            tail = slice(row_count, old_row_count)
            self.stages_cpu[tail] = _INVALID_STAGE
            self.target_resident_budget_tokens_cpu[tail] = 0
            self.sparse_budget_tokens_cpu[tail] = 0
            self.resident_valid_tokens_cpu[tail] = _INVALID_RESIDENT_LENGTH
            self.row_modes_cpu[tail] = DSA_ROW_MODE_PAD
            self.resident_pool_indices_cpu[tail] = (
                self.resident_token_pool.padding_pool_index
            )
        self.row_count = row_count
        self.valid = True

    def copy_to_device(self, row_count: int | None = None) -> torch.Tensor:
        """供后续 P5/P6 在统一 owner 上执行一次 H2D。"""

        if not self.valid:
            raise RuntimeError("DSA InputBatch cache layout must be refreshed first")
        rows = self.row_count if row_count is None else int(row_count)
        if rows < self.row_count or rows > self.max_num_reqs:
            raise ValueError(
                "DSA cache-layout copy row count must cover active rows and "
                f"fit capacity: rows={rows}, active={self.row_count}, "
                f"capacity={self.max_num_reqs}"
            )
        # PAD 区在 owner 初始化时一次设好；batch 缩小时 refresh() 只清理
        # 由 active 退回 PAD 的尾部。这里不再每 step 重写
        # [active_rows:max_num_reqs]，保持与基线 persistent batch 一样的
        # 增量更新范式。
        # 六列采用 SoA 布局，每一列都是可直接传给自定义算子的连续向量。
        # buffer 只有 6 * max_num_reqs 个 int32，整块复制既保持一次 H2D，
        # 也避免对非连续列 view 逐列 copy 或现场 contiguous。
        self.columns.gpu.copy_(self.columns.cpu, non_blocking=True)
        return self.columns.gpu

    def _synchronize_resident_pool_rows(
        self,
        request_ids: tuple[str, ...],
    ) -> None:
        """仅在请求集合或最终行序变化时重建 batch-row→pool-row 映射。"""

        request_id_set = set(request_ids)
        for request_id in self._request_ids:
            if request_id not in request_id_set:
                released_index = self.resident_token_pool.release(request_id)
                if (
                    released_index is not None
                    and self._pool_release_callback is not None
                ):
                    self._pool_release_callback(released_index)

        for row, request_id in enumerate(request_ids):
            self.resident_pool_indices_cpu[row] = (
                self.resident_token_pool.acquire(request_id)
            )
        self._request_ids = request_ids
        self.mapping_version += 1


def apply_dsa_cache_layout_projection(
    *,
    input_batch: NPUInputBatch,
    requests: dict[str, CachedRequestState],
    state: DSAInputBatchCacheLayout,
    projection: DSARequestCacheLayoutProjection,
    resident_group_id: int,
) -> None:
    """应用标量投影，并在 ENTER 行覆盖 worker resident block table。"""

    state.refresh(
        input_batch=input_batch,
        projection=projection,
    )

    resident_group_id = int(resident_group_id)
    replaced_request_ids: set[str] = set()
    pending_replacements: list[tuple[list[int], int, tuple[int, ...]]] = []
    for replacement in projection.resident_block_table_replacements:
        request_id = replacement.request_id
        row = input_batch.req_id_to_index.get(request_id)
        request = requests.get(request_id)
        if row is None or request is None:
            raise RuntimeError(
                f"DSA ENTER replacement has no matching worker request: request_id={request_id!r}, row={row}"
            )
        if DSARequestCacheStage(int(state.stages_cpu[row])) != DSARequestCacheStage.ENTER_SPARSE_DECODE:
            raise RuntimeError(
                f"DSA resident block-table replacement is only valid for ENTER: request_id={request_id!r}"
            )
        if request_id in replaced_request_ids:
            raise RuntimeError(f"DSA ENTER projection contains a duplicate replacement: request_id={request_id!r}")
        replaced_request_ids.add(request_id)
        if resident_group_id >= len(request.block_ids):
            raise RuntimeError(
                "DSA ENTER replacement resident group is out of range: "
                f"request_id={request_id!r}, group={resident_group_id}, "
                f"groups={len(request.block_ids)}"
            )
        pending_replacements.append(
            (
                request.block_ids[resident_group_id],
                row,
                replacement.block_ids,
            )
        )

    # 所有 ENTER 行先完成校验，再统一改写，避免异常时留下半替换 block 表。
    for resident_block_ids, row, replacement_block_ids in pending_replacements:
        resident_block_ids[:] = replacement_block_ids
        input_batch.block_table[resident_group_id].add_row(
            resident_block_ids,
            row,
        )
