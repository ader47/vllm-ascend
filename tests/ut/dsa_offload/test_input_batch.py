# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.dsa_offload.input_batch import (
    DSAInputBatchCacheLayout,
    apply_dsa_cache_layout_projection,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.scheduler_output import (
    DSARequestCacheLayoutProjection,
    DSAResidentBlockTableReplacement,
)


class _BlockTable:
    def __init__(self) -> None:
        self.added_rows: list[tuple[list[int], int]] = []

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.added_rows.append((list(block_ids), row_idx))


class _InputBatch:
    def __init__(self) -> None:
        self.max_num_reqs = 4
        self.req_id_to_index = {"dense": 0, "enter": 1}
        self.block_table = [_BlockTable(), _BlockTable()]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def req_ids(self) -> list[str]:
        return ["dense", "enter"]


def _make_state() -> DSAInputBatchCacheLayout:
    return DSAInputBatchCacheLayout(
        max_num_reqs=4,
        device=torch.device("cpu"),
        pin_memory=False,
    )


def test_projection_follows_final_input_batch_rows_and_replaces_enter_table() -> None:
    input_batch = _InputBatch()
    state = _make_state()
    requests = {
        "dense": SimpleNamespace(block_ids=[[101], [201, 202]]),
        # 模拟基线把 ENTER 新块追加到旧 dense resident 表后的错误中间态。
        "enter": SimpleNamespace(block_ids=[[102], [210, 211, 310, 311]]),
    }
    projection = DSARequestCacheLayoutProjection(
        # 传输顺序与 worker 最终行顺序相反。
        request_ids=("enter", "dense"),
        stages=(
            int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
            int(DSARequestCacheStage.DENSE_DECODE),
        ),
        target_resident_budget_tokens=(2048, 2048),
        sparse_budget_tokens=(2048, 0),
        resident_valid_tokens=(2049, -1),
        resident_block_table_replacements=(
            DSAResidentBlockTableReplacement(
                request_id="enter",
                block_ids=(310, 311, 211),
            ),
        ),
    )

    apply_dsa_cache_layout_projection(
        input_batch=input_batch,
        requests=requests,
        state=state,
        projection=projection,
        resident_group_id=1,
    )

    assert state.valid
    assert state.row_count == 2
    assert state.stages_cpu[:2].tolist() == [
        int(DSARequestCacheStage.DENSE_DECODE),
        int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
    ]
    assert state.sparse_budget_tokens_cpu[:2].tolist() == [0, 2048]
    assert requests["enter"].block_ids[1] == [310, 311, 211]
    assert input_batch.block_table[1].added_rows == [([310, 311, 211], 1)]


def test_projection_rejects_enter_replacement_count_mismatch() -> None:
    try:
        DSARequestCacheLayoutProjection(
            request_ids=("enter", "dense"),
            stages=(
                int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
                int(DSARequestCacheStage.DENSE_DECODE),
            ),
            target_resident_budget_tokens=(2048, 2048),
            sparse_budget_tokens=(2048, 0),
            resident_valid_tokens=(2049, -1),
            resident_block_table_replacements=(),
        )
    except ValueError as error:
        assert "does not match ENTER rows" in str(error)
    else:
        raise AssertionError("ENTER without a replacement must fail")


def test_graph_padding_rows_are_reset_before_copy() -> None:
    input_batch = _InputBatch()
    state = _make_state()
    projection = DSARequestCacheLayoutProjection(
        request_ids=("dense", "enter"),
        stages=(
            int(DSARequestCacheStage.DENSE_DECODE),
            int(DSARequestCacheStage.SPARSE_DECODE),
        ),
        target_resident_budget_tokens=(2048, 4096),
        sparse_budget_tokens=(0, 4096),
        resident_valid_tokens=(-1, 4097),
        resident_block_table_replacements=(),
    )
    state.refresh(input_batch=input_batch, projection=projection)
    state.stages_cpu[2:] = 99

    state.copy_to_device(4)

    assert state.stages.is_contiguous()
    assert state.target_resident_budget_tokens.is_contiguous()
    assert state.stages_cpu.tolist() == [
        int(DSARequestCacheStage.DENSE_DECODE),
        int(DSARequestCacheStage.SPARSE_DECODE),
        -1,
        -1,
    ]
    assert state.stages.tolist() == state.stages_cpu.tolist()
