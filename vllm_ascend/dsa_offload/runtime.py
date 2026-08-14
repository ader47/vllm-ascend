# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 数据面的 worker-lifetime tensor owner 与逐层执行上下文。

``DSAInputBatchCacheLayout`` 仍是请求行状态的唯一 worker 投影；本模块只把
其 active-prefix 转成设备算子需要的固定地址 view：

* resident slot-mapping position；
* active DRAM block table；
* 所有层串行复用的 caller-owned LIDU scratch 输出；
* 本轮满块 dump 的紧凑 src/dst 列。

所有大 tensor 在 model runner 初始化期预分配。steady step 只刷新 active
prefix；DRAM 逻辑表未变化时不会重复 H2D。图模式复用这些 owner 的
captured-prefix + PAD view，而不是另建一套 graph-only 语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.dsa_offload.contracts import (
    DSA_A5_ATTENTION_CAPACITY,
    DSA_DRAM_NULL_BLOCK_ID,
    DSA_DUMP_NOOP_DST_BLOCK_ID,
    DSA_LIDU_OUTPUT_CAPACITY,
    DSA_ROW_MODE_SPARSE,
)
from vllm_ascend.dsa_offload.dram_store import DSAHotDRAMStore
from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    DSAOffloadSelectionOutput,
    a5_kvcache_scatter_copy_c8,
    a5_lightning_indexer_decode_update_c8,
    dump_full_kv_cache_blocks,
    dump_full_kv_cache_blocks_c8,
    kvcache_scatter_copy,
    lightning_indexer_decode_update,
    sparse_flash_attention_for_offload,
    sparse_flash_attention_for_offload_c8,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.input_batch import (
        DSAInputBatchCacheLayout,
    )
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch


class DSAOffloadRuntime:
    """持有 eager 与后续 graph 共用的固定地址设备元数据。"""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_num_tokens: int,
        num_layers: int,
        max_model_len: int,
        block_size: int,
        resident_token_pool: DSAResidentTokenPool,
        device: torch.device,
        pin_memory: bool,
        packed_c8: bool = False,
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.max_num_tokens = int(max_num_tokens)
        self.num_layers = int(num_layers)
        self.max_model_len = int(max_model_len)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.packed_c8 = bool(packed_c8)
        self.resident_token_pool = resident_token_pool
        self.max_logical_blocks = cdiv(
            self.max_model_len,
            self.block_size,
        )

        # LIDU 输出只在“当前层 LIDU -> KSC -> SFA-Offload”之间存活。
        # 各层存在 hidden-state 数据依赖，不会并发消费两层输出，因此所有
        # 层复用同一套固定地址 scratch 即可。逐层持久状态只保存在下方
        # resident_token_pool.cache_slots，不能与这里混为一谈。
        output_shape = (
            self.max_num_reqs,
            1,
            DSA_LIDU_OUTPUT_CAPACITY,
        )
        self._lidu_topk_index = torch.empty(
            output_shape,
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_topk_slots = torch.empty_like(self._lidu_topk_index)
        self._lidu_miss_count = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_tail_info = torch.empty(
            (self.max_num_reqs, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_output_views: dict[
            int,
            DSALightningIndexerOutputs,
        ] = {}
        self._a5_attention_slots: torch.Tensor | None = None
        self._a5_resident_seq_lengths: torch.Tensor | None = None
        if self.packed_c8:
            self._a5_attention_slots = torch.empty(
                (
                    self.max_num_reqs,
                    1,
                    DSA_A5_ATTENTION_CAPACITY,
                ),
                dtype=torch.int32,
                device=self.device,
            )
            self._a5_resident_seq_lengths = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        self.resident_positions = torch.empty(
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        self._token_row_modes = torch.empty(
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self._token_sparse_budgets = torch.empty_like(self._token_row_modes)
        self._sparse_token_mask = torch.empty(
            self.max_num_tokens,
            dtype=torch.bool,
            device=self.device,
        )

        self.active_dram_block_table = CpuGpuBuffer(
            self.max_num_reqs,
            self.max_logical_blocks,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.active_dram_block_table.np.fill(DSA_DRAM_NULL_BLOCK_ID)
        self.active_dram_block_table.gpu.zero_()
        max_dump_jobs = cdiv(self.max_num_tokens, self.block_size) + self.max_num_reqs
        self.dump_src_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_dst_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_src_block_ids.np.fill(0)
        self.dump_dst_block_ids.np.fill(DSA_DUMP_NOOP_DST_BLOCK_ID)
        self.dump_src_block_ids.gpu.zero_()
        self.dump_dst_block_ids.gpu.fill_(DSA_DUMP_NOOP_DST_BLOCK_ID)
        self._dump_pool_indices = np.empty(max_dump_jobs, dtype=np.intp)
        self._dump_logical_indices = np.empty(max_dump_jobs, dtype=np.intp)
        self._dump_source_blocks = np.empty(max_dump_jobs, dtype=np.int32)
        self._logical_block_indices = np.arange(
            self.max_logical_blocks,
            dtype=np.intp,
        )
        # 满块边界判定每个 forward 都会执行，即使最终没有 dump。
        # 这些逐请求 scratch 固定复用，避免 steady decode 为几次简单的
        # add/floor-divide/subtract 现场创建 NumPy 临时数组。
        self._scheduled_tokens_i64 = np.empty(
            self.max_num_reqs,
            dtype=np.int64,
        )
        self._computed_tokens_i64 = np.empty_like(self._scheduled_tokens_i64)
        self._tokens_after_schedule = np.empty_like(self._scheduled_tokens_i64)
        self._first_logical_blocks = np.empty_like(self._scheduled_tokens_i64)
        self._completed_after_blocks = np.empty_like(self._scheduled_tokens_i64)
        self._completed_block_counts = np.empty_like(self._scheduled_tokens_i64)
        self._dump_boundary_mask = np.empty(
            self.max_num_reqs,
            dtype=np.bool_,
        )

        self.dram_store: DSAHotDRAMStore | None = None
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self._graph_capture_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature: tuple[int, int, int] | None = None

    def bind_dram_store(self, store: DSAHotDRAMStore) -> None:
        if self.dram_store is not None:
            raise RuntimeError("DSA offload runtime DRAM store was bound twice")
        if store.storage_rows != self.resident_token_pool.storage_rows:
            raise RuntimeError(
                "DSA DRAM/resident pool row capacities differ: "
                f"dram={store.storage_rows}, "
                f"resident={self.resident_token_pool.storage_rows}"
            )
        if store.max_logical_blocks != self.max_logical_blocks:
            raise RuntimeError(
                "DSA DRAM/runtime logical widths differ: "
                f"dram={store.max_logical_blocks}, "
                f"runtime={self.max_logical_blocks}"
            )
        self.dram_store = store

    def release_pool_index(self, pool_index: int) -> None:
        store = self.dram_store
        if store is not None:
            store.release_pool_index(pool_index)

    def prepare_forward(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        num_scheduled_tokens: np.ndarray,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        resident_group_id: int,
    ) -> torch.Tensor:
        """刷新本轮统一元数据并返回 resident slot-mapping positions。"""

        if self.dram_store is None:
            raise RuntimeError("DSA offload runtime has no DRAM store")
        if not state.valid or state.row_count != num_reqs:
            raise RuntimeError(
                "DSA InputBatch state is not ready for eager execution: "
                f"valid={state.valid}, rows={state.row_count}, "
                f"num_reqs={num_reqs}"
            )
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(f"DSA token capacity exceeded: {num_tokens} > {self.max_num_tokens}")

        self.active_num_reqs = int(num_reqs)
        self.execution_num_reqs = 0
        self.dump_launch_count = 0

        self._prepare_dump_plan(
            input_batch=input_batch,
            state=state,
            num_scheduled_tokens=num_scheduled_tokens,
            resident_group_id=resident_group_id,
        )
        self._refresh_candidate_lens(state)
        # 七列共用一个 owner，仅一次 H2D；后续 tensor 均为该 owner 的 view。
        state.copy_to_device(num_reqs)
        dram_table_refreshed = self._refresh_active_dram_table(state)
        self._validate_sparse_dram_rows(
            input_batch=input_batch,
            state=state,
            validate_all_sparse=dram_table_refreshed,
        )

        active_modes = state.row_modes_cpu[:num_reqs]
        if not np.any(active_modes == DSA_ROW_MODE_SPARSE):
            return positions[:num_tokens]

        token_modes = self._token_row_modes[:num_tokens]
        token_budgets = self._token_sparse_budgets[:num_tokens]
        torch.index_select(
            state.row_modes,
            0,
            req_indices[:num_tokens],
            out=token_modes,
        )
        torch.index_select(
            state.sparse_budget_tokens,
            0,
            req_indices[:num_tokens],
            out=token_budgets,
        )
        resident_positions = self.resident_positions[:num_tokens]
        torch.remainder(
            positions[:num_tokens],
            self.block_size,
            out=resident_positions,
        )
        resident_positions.add_(token_budgets)
        torch.eq(
            token_modes,
            DSA_ROW_MODE_SPARSE,
            out=self._sparse_token_mask[:num_tokens],
        )
        torch.where(
            self._sparse_token_mask[:num_tokens],
            resident_positions,
            positions[:num_tokens],
            out=resident_positions,
        )
        return resident_positions

    def _refresh_candidate_lens(
        self,
        state: DSAInputBatchCacheLayout,
    ) -> None:
        """刷新 LI/KSC 共享的历史候选长度，不创建逐 step 临时数组。

        DENSE 行的候选区就是当前完整序列。SPARSE 行最后一个物理块是
        resident tail，因此候选区只覆盖它之前的完整逻辑块：
        ``floor((actual_len - 1) / block_size) * block_size``。
        """

        num_reqs = self.active_num_reqs
        tokens_after = self._tokens_after_schedule[:num_reqs]
        candidate_lens = state.candidate_lens_cpu[:num_reqs]
        np.copyto(candidate_lens, tokens_after, casting="unsafe")

        sparse_mask = self._dump_boundary_mask[:num_reqs]
        np.equal(
            state.row_modes_cpu[:num_reqs],
            DSA_ROW_MODE_SPARSE,
            out=sparse_mask,
        )
        completed_before_tail = self._completed_after_blocks[:num_reqs]
        np.subtract(tokens_after, 1, out=completed_before_tail)
        np.floor_divide(
            completed_before_tail,
            self.block_size,
            out=completed_before_tail,
        )
        np.multiply(
            completed_before_tail,
            self.block_size,
            out=completed_before_tail,
        )
        np.copyto(
            candidate_lens,
            completed_before_tail,
            where=sparse_mask,
            casting="unsafe",
        )

    @property
    def graph_capture_row_count(self) -> int:
        return self._graph_capture_row_count

    def prepare_execution_view(
        self,
        *,
        active_num_reqs: int,
        graph_row_count: int | None,
    ) -> int:
        """把 active-prefix 绑定成 eager 或 FULL-graph 的执行 view。

        eager 仅复制紧凑 dump jobs；graph 固定复制 captured-row 宽度，未
        使用行以 ``dst=-1`` 空转。DRAM table 和请求列仍是同一固定 owner
        的前缀，不创建 graph 专属副本。
        """

        if self._graph_capture_row_count:
            raise RuntimeError("DSA real execution cannot reuse an active capture dummy")
        active_num_reqs = int(active_num_reqs)
        if active_num_reqs != self.active_num_reqs:
            raise RuntimeError(
                "DSA runtime active rows changed before execution view "
                f"was finalized: prepared={self.active_num_reqs}, "
                f"requested={active_num_reqs}"
            )

        if graph_row_count is None:
            execution_num_reqs = active_num_reqs
            launch_count = self.dump_job_count
        else:
            execution_num_reqs = int(graph_row_count)
            if not (active_num_reqs <= execution_num_reqs <= self.max_num_reqs):
                raise RuntimeError(
                    "DSA graph rows must cover active rows and fit runtime "
                    f"capacity: active={active_num_reqs}, "
                    f"graph={execution_num_reqs}, "
                    f"capacity={self.max_num_reqs}"
                )
            if self.dump_job_count > active_num_reqs:
                raise RuntimeError(
                    "DSA single-token graph step produced more dump jobs "
                    f"than active rows: jobs={self.dump_job_count}, "
                    f"rows={active_num_reqs}"
                )
            launch_count = execution_num_reqs
            if self.dump_job_count < launch_count:
                tail = slice(self.dump_job_count, launch_count)
                self.dump_src_block_ids.np[tail] = 0
                self.dump_dst_block_ids.np[tail] = DSA_DUMP_NOOP_DST_BLOCK_ID

        if launch_count:
            self.dump_src_block_ids.gpu[:launch_count].copy_(
                self.dump_src_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
            self.dump_dst_block_ids.gpu[:launch_count].copy_(
                self.dump_dst_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
        self.execution_num_reqs = execution_num_reqs
        self.dump_launch_count = int(launch_count)
        return execution_num_reqs

    def prepare_graph_capture(self, *, row_count: int) -> None:
        """为原生 dummy-run 安装固定地址 DRAM/dump 输入。"""

        if self.dram_store is None:
            raise RuntimeError("DSA graph capture requires an initialized DRAM store")
        row_count = int(row_count)
        if self._graph_capture_row_count:
            raise RuntimeError("DSA runtime graph-capture state was installed twice")
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError(
                "DSA runtime graph-capture row count is outside capacity: "
                f"rows={row_count}, capacity={self.max_num_reqs}"
            )

        self._graph_capture_row_count = row_count
        try:
            self.active_dram_block_table.np[:row_count].fill(DSA_DRAM_NULL_BLOCK_ID)
            self.active_dram_block_table.gpu[:row_count].zero_()
            self.dump_src_block_ids.np[:row_count] = 0
            self.dump_dst_block_ids.np[:row_count] = DSA_DUMP_NOOP_DST_BLOCK_ID
            self.dump_src_block_ids.gpu[:row_count].zero_()
            self.dump_dst_block_ids.gpu[:row_count].fill_(DSA_DUMP_NOOP_DST_BLOCK_ID)
            self.active_num_reqs = row_count
            self.execution_num_reqs = row_count
            self.dump_job_count = 0
            self.dump_launch_count = row_count
        except Exception:
            try:
                self.restore_after_graph_capture()
            except Exception:
                # 保留首次安装失败作为根因，并至少解除 host 侧 installed 状态。
                self.active_num_reqs = 0
                self.execution_num_reqs = 0
                self.dump_job_count = 0
                self.dump_launch_count = 0
                self._graph_capture_row_count = 0
            raise

    def restore_after_graph_capture(self) -> None:
        """恢复 dummy-run 之外的空 runtime 状态。"""

        if not self._graph_capture_row_count:
            return
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self._graph_capture_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature = None

    def _prepare_dump_plan(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        num_scheduled_tokens: np.ndarray,
        resident_group_id: int,
    ) -> None:
        """只遍历本轮跨满块边界的行，构造紧凑 dump jobs。"""

        store = self.dram_store
        assert store is not None
        num_reqs = self.active_num_reqs
        scheduled = self._scheduled_tokens_i64[:num_reqs]
        computed = self._computed_tokens_i64[:num_reqs]
        tokens_after = self._tokens_after_schedule[:num_reqs]
        first_logical = self._first_logical_blocks[:num_reqs]
        completed_after = self._completed_after_blocks[:num_reqs]
        completed_counts = self._completed_block_counts[:num_reqs]
        boundary_mask = self._dump_boundary_mask[:num_reqs]

        np.copyto(
            scheduled,
            num_scheduled_tokens[:num_reqs],
            casting="unsafe",
        )
        np.copyto(
            computed,
            input_batch.num_computed_tokens_cpu[:num_reqs],
            casting="unsafe",
        )
        np.floor_divide(
            computed,
            self.block_size,
            out=first_logical,
        )
        np.add(computed, scheduled, out=tokens_after)
        np.floor_divide(
            tokens_after,
            self.block_size,
            out=completed_after,
        )
        np.subtract(
            completed_after,
            first_logical,
            out=completed_counts,
        )
        np.greater(completed_counts, 0, out=boundary_mask)
        if not np.any(boundary_mask):
            self.dump_job_count = 0
            return
        boundary_rows = np.flatnonzero(boundary_mask)

        resident_table = input_batch.block_table[int(resident_group_id)]
        resident_blocks = resident_table.get_numpy_array()
        resident_row_widths = resident_table.num_blocks_per_row
        job_count = 0
        for row_value in boundary_rows:
            row = int(row_value)
            count = int(completed_counts[row])
            logical_start = int(first_logical[row])
            logical_end = logical_start + count
            next_job_count = job_count + count
            if next_job_count > self._dump_pool_indices.size:
                raise RuntimeError(
                    "DSA full-block dump job capacity exceeded: "
                    f"required={next_job_count}, "
                    f"capacity={self._dump_pool_indices.size}"
                )

            jobs = slice(job_count, next_job_count)
            self._dump_pool_indices[jobs] = state.resident_pool_indices_cpu[row]
            self._dump_logical_indices[jobs] = self._logical_block_indices[logical_start:logical_end]
            if state.row_modes_cpu[row] == DSA_ROW_MODE_SPARSE:
                if count != 1:
                    raise RuntimeError(
                        f"DSA sparse decode completed more than one full block in one step: row={row}, count={count}"
                    )
                tail_column = int(resident_row_widths[row]) - 1
                if tail_column < 0:
                    raise RuntimeError(f"DSA sparse row {row} has no resident tail block")
                self._dump_source_blocks[jobs] = resident_blocks[
                    row,
                    tail_column,
                ]
            else:
                if logical_end > int(resident_row_widths[row]):
                    raise RuntimeError(
                        "DSA dense dump references an unallocated resident "
                        f"block: row={row}, logical_end={logical_end}, "
                        f"resident_blocks={int(resident_row_widths[row])}"
                    )
                self._dump_source_blocks[jobs] = resident_blocks[
                    row,
                    logical_start:logical_end,
                ]
            job_count = next_job_count

        if job_count == 0:
            self.dump_job_count = 0
            return

        reservation = store.reserve_blocks(
            pool_indices=self._dump_pool_indices[:job_count],
            logical_block_indices=self._dump_logical_indices[:job_count],
        )
        new_rows = np.flatnonzero(reservation.new_mask)
        copy_count = int(new_rows.size)
        if copy_count:
            np.take(
                self._dump_source_blocks[:job_count],
                new_rows,
                out=self.dump_src_block_ids.np[:copy_count],
            )
            np.take(
                reservation.physical_block_ids,
                new_rows,
                out=self.dump_dst_block_ids.np[:copy_count],
            )
        self.dump_job_count = copy_count

    def _refresh_active_dram_table(
        self,
        state: DSAInputBatchCacheLayout,
    ) -> bool:
        store = self.dram_store
        assert store is not None
        signature = (
            int(state.mapping_version),
            int(store.table_version),
            self.active_num_reqs,
        )
        if signature == self._dram_table_signature:
            return False
        previous_row_count = self._dram_table_row_count
        active_cpu = self.active_dram_block_table.np[: self.active_num_reqs]
        store.gather_rows(
            pool_indices=state.resident_pool_indices_cpu[: self.active_num_reqs],
            output=active_cpu,
        )
        if self.active_num_reqs < previous_row_count:
            self.active_dram_block_table.np[self.active_num_reqs : previous_row_count].fill(DSA_DRAM_NULL_BLOCK_ID)
        copy_row_count = max(
            self.active_num_reqs,
            previous_row_count,
        )
        self.active_dram_block_table.gpu[:copy_row_count].copy_(
            self.active_dram_block_table.cpu[:copy_row_count],
            non_blocking=True,
        )
        self._dram_table_row_count = self.active_num_reqs
        self._dram_table_signature = signature
        return True

    def _validate_sparse_dram_rows(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        validate_all_sparse: bool,
    ) -> None:
        """在低频布局边界拒绝不完整的 sparse DRAM 映射。

        KSC 的 block 0 是合法可寻址的空 arena，因此缺失逻辑映射若不在
        host 侧拦住，会静默把零 payload 写入 resident HBM。ENTER 每个请求
        只检查一次；稳定 SPARSE 仅在 pool-row 或 DRAM table 版本变化时
        检查，不增加 steady decode 的逐 step 扫描。
        """

        num_reqs = self.active_num_reqs
        stages = state.stages_cpu[:num_reqs]
        if validate_all_sparse:
            rows = np.flatnonzero(stages >= int(DSARequestCacheStage.ENTER_SPARSE_DECODE))
        else:
            rows = np.flatnonzero(stages == int(DSARequestCacheStage.ENTER_SPARSE_DECODE))
        if rows.size == 0:
            return

        dram_table = self.active_dram_block_table.np
        for row_value in rows:
            row = int(row_value)
            # LIDU 把最后一个非空块作为 dense tail；只有它之前的完整块
            # 会成为 KSC 的 DRAM source。
            required_blocks = max(0, int(self._tokens_after_schedule[row]) - 1) // self.block_size
            if required_blocks == 0:
                continue
            missing = np.flatnonzero(dram_table[row, :required_blocks] == DSA_DRAM_NULL_BLOCK_ID)
            if missing.size:
                request_id = input_batch.req_ids[row]
                raise RuntimeError(
                    "DSA sparse decode has an incomplete DRAM block table: "
                    f"request_id={request_id!r}, batch_row={row}, "
                    "resident_pool_row="
                    f"{int(state.resident_pool_indices_cpu[row])}, "
                    f"required_blocks={required_blocks}, "
                    f"first_missing_logical_block={int(missing[0])}. "
                    "Refusing to let KSC read the null DRAM block."
                )

    def get_lidu_outputs(
        self,
        *,
        num_reqs: int,
    ) -> DSALightningIndexerOutputs:
        num_reqs = int(num_reqs)
        view = self._lidu_output_views.get(num_reqs)
        if view is None:
            if not 0 < num_reqs <= self.max_num_reqs:
                raise RuntimeError(f"DSA LIDU row count {num_reqs} is outside capacity")
            view = DSALightningIndexerOutputs(
                topk_index=self._lidu_topk_index[:num_reqs],
                topk_slots=self._lidu_topk_slots[:num_reqs],
                miss_count=self._lidu_miss_count[:num_reqs],
                tail_info=self._lidu_tail_info[:num_reqs],
            )
            self._lidu_output_views[num_reqs] = view
        return view


@dataclass(frozen=True)
class DSALayerOffloadContext:
    """绑定到一个 ``AscendSFAImpl`` 的逐层稳定资源。"""

    layer_id: int
    indexer_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    runtime: DSAOffloadRuntime
    packed_c8: bool = False

    def execute_decode_selection(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        row_modes: torch.Tensor,
        resident_pool_indices: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        indexer_block_table: torch.Tensor,
        resident_cache: tuple[torch.Tensor, ...],
        resident_block_table: torch.Tensor,
        dram_block_table: torch.Tensor,
        candidate_lens: torch.Tensor | None = None,
        query_dequant_scale: torch.Tensor | None = None,
        query_shape: tuple[int, ...] | None = None,
    ) -> DSAOffloadSelectionOutput:
        num_reqs = int(actual_seq_lengths_key.shape[0])
        outputs = self.runtime.get_lidu_outputs(
            num_reqs=num_reqs,
        )
        if self.packed_c8:
            if not isinstance(self.indexer_cache, tuple) or len(self.indexer_cache) != 2 or len(resident_cache) != 1:
                raise RuntimeError("DSA A5 C8 context requires packed resident cache and Indexer key/scale caches")
            if candidate_lens is None or query_dequant_scale is None or query_shape is None:
                raise RuntimeError("DSA A5 C8 decode metadata is incomplete")
            indexer_key, indexer_scale = self.indexer_cache
            attention_slots_buffer = self.runtime._a5_attention_slots
            resident_seq_lengths_buffer = self.runtime._a5_resident_seq_lengths
            if attention_slots_buffer is None or resident_seq_lengths_buffer is None:
                raise RuntimeError("DSA A5 C8 context has no preallocated selection scratch")
            attention_slots = attention_slots_buffer[:num_reqs]
            resident_seq_lengths = resident_seq_lengths_buffer[:num_reqs]
            a5_lightning_indexer_decode_update_c8(
                index_weights=weights,
                query=query.view(query_shape),
                query_dequant_scale=query_dequant_scale.view(query_shape[:-1]),
                actual_seq_lengths_query=actual_seq_lengths_query,
                index_key_cache=indexer_key,
                index_key_dequant_scale=indexer_scale,
                index_block_table=indexer_block_table,
                candidate_lens=candidate_lens,
                final_seq_lengths_kv=actual_seq_lengths_key,
                row_modes=row_modes,
                req_pool_entries=resident_pool_indices,
                cache_slots=self.runtime.resident_token_pool.get_cache_slots(self.layer_id),
                attention_slots=attention_slots,
                resident_seq_lengths=resident_seq_lengths,
                outputs=outputs,
            )
            store = self.runtime.dram_store
            if store is None:
                raise RuntimeError("DSA layer has no bound DRAM store")
            packed_arena = store.get_layer_arenas(self.layer_id).packed
            if packed_arena is None:
                raise RuntimeError("DSA A5 layer has no packed DRAM arena")
            a5_kvcache_scatter_copy_c8(
                resident_packed_cache=resident_cache[0],
                dram_packed_arena=packed_arena,
                resident_block_table=resident_block_table,
                dram_block_table=dram_block_table,
                source_token_ids=outputs.topk_index,
                destination_slots=outputs.topk_slots,
                copy_counts=outputs.miss_count,
            )
            return DSAOffloadSelectionOutput(
                sparse_indices=attention_slots,
                tail_info=outputs.tail_info,
                resident_seq_lengths=resident_seq_lengths,
            )

        if not isinstance(self.indexer_cache, torch.Tensor):
            raise RuntimeError("DSA bf16 Indexer cache must be one tensor")
        if len(resident_cache) < 2:
            raise RuntimeError("DSA bf16 resident cache must contain NOPE and ROPE planes")
        lightning_indexer_decode_update(
            query=query,
            key=self.indexer_cache,
            weights=weights,
            req_pool_entries=resident_pool_indices,
            cache_slots=self.runtime.resident_token_pool.get_cache_slots(self.layer_id),
            row_modes=row_modes,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=indexer_block_table,
            outputs=outputs,
        )
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        kvcache_scatter_copy(
            resident_nope_cache=resident_cache[0],
            resident_rope_cache=resident_cache[1],
            dram_nope_arena=arenas.nope,
            dram_rope_arena=arenas.rope,
            resident_block_table=resident_block_table,
            dram_block_table=dram_block_table,
            src_token_ids=outputs.topk_index,
            dst_slots=outputs.topk_slots,
            copy_counts=outputs.miss_count,
        )
        return DSAOffloadSelectionOutput(
            sparse_indices=outputs.topk_slots,
            tail_info=outputs.tail_info,
        )

    def execute_sparse_attention(
        self,
        *,
        query: torch.Tensor,
        resident_cache: tuple[torch.Tensor, ...],
        selection: DSAOffloadSelectionOutput,
        scale_value: float,
        resident_block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        query_rope: torch.Tensor,
    ) -> torch.Tensor:
        if self.packed_c8:
            if len(resident_cache) != 1:
                raise RuntimeError("DSA A5 C8 attention requires one packed resident cache")
            resident_seq_lengths = selection.resident_seq_lengths
            if resident_seq_lengths is None:
                raise RuntimeError("DSA A5 C8 selection has no resident sequence lengths")
            return sparse_flash_attention_for_offload_c8(
                query=torch.cat([query, query_rope], dim=-1),
                packed_kv=resident_cache[0],
                sparse_indices=selection.sparse_indices,
                scale_value=scale_value,
                block_table=resident_block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                resident_seq_lengths=resident_seq_lengths,
            )
        if len(resident_cache) < 2:
            raise RuntimeError("DSA bf16 attention requires NOPE and ROPE resident caches")
        return sparse_flash_attention_for_offload(
            query=query,
            key=resident_cache[0],
            sparse_indices=selection.sparse_indices,
            tail_info=selection.tail_info,
            scale_value=scale_value,
            block_table=resident_block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=resident_cache[1],
        )

    def dump_full_blocks(
        self,
        *,
        resident_cache: tuple[torch.Tensor, ...],
    ) -> None:
        job_count = self.runtime.dump_launch_count
        if job_count == 0:
            return
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        if self.packed_c8:
            if len(resident_cache) != 1 or arenas.packed is None:
                raise RuntimeError("DSA A5 C8 dump requires packed HBM and DRAM caches")
            dump_full_kv_cache_blocks_c8(
                resident_packed_cache=resident_cache[0],
                dram_packed_arena=arenas.packed,
                src_block_ids=self.runtime.dump_src_block_ids.gpu[:job_count],
                dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
            )
            return
        if len(resident_cache) < 2 or arenas.nope is None or arenas.rope is None:
            raise RuntimeError("DSA bf16 dump requires NOPE and ROPE HBM/DRAM planes")
        dump_full_kv_cache_blocks(
            resident_nope_cache=resident_cache[0],
            resident_rope_cache=resident_cache[1],
            dram_nope_arena=arenas.nope,
            dram_rope_arena=arenas.rope,
            src_block_ids=self.runtime.dump_src_block_ids.gpu[:job_count],
            dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
        )
