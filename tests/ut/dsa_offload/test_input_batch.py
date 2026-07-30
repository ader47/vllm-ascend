# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.dsa_offload.input_batch import (
    DSAInputBatchCacheLayout,
    apply_dsa_cache_layout_projection,
    normalize_dsa_enter_updates_before_base,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool
from vllm_ascend.dsa_offload.scheduler_output import (
    DSARequestCacheLayoutProjection,
    DSAResidentBlockTableReplacement,
)


class _BlockTable:
    def __init__(self) -> None:
        self.added_rows: list[tuple[list[int], int]] = []

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.added_rows.append((list(block_ids), row_idx))


class _BoundedBlockTable:
    """模拟原生固定行宽，越界时复现 NumPy broadcast 失败。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.rows: dict[int, list[int]] = {}

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        values = list(block_ids)
        if len(values) > self.capacity:
            raise ValueError("block-table row capacity exceeded")
        self.rows[row_idx] = values

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        values = self.rows.get(row_idx, []) + list(block_ids)
        if len(values) > self.capacity:
            raise ValueError("block-table row capacity exceeded")
        self.rows[row_idx] = values


class _InputBatch:
    def __init__(
        self,
        request_ids: tuple[str, ...] = ("dense", "enter"),
    ) -> None:
        self.max_num_reqs = 4
        self._request_ids = request_ids
        self.req_id_to_index = {
            request_id: row
            for row, request_id in enumerate(request_ids)
        }
        self.block_table = [_BlockTable(), _BlockTable()]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def req_ids(self) -> list[str]:
        return list(self._request_ids)


class _NoLookupRequestRows(dict[str, int]):
    def get(self, key: str, default: int | None = None) -> int | None:
        raise AssertionError(
            f"stable row-order refresh must not look up request {key!r}"
        )


def _make_state() -> DSAInputBatchCacheLayout:
    resident_token_pool = DSAResidentTokenPool(
        max_num_reqs=4,
        num_layers=2,
        max_model_len=8192,
        max_resident_budget_tokens=4096,
        device=torch.device("cpu"),
    )
    return DSAInputBatchCacheLayout(
        max_num_reqs=4,
        device=torch.device("cpu"),
        pin_memory=False,
        resident_token_pool=resident_token_pool,
    )


def _make_enter_projection() -> DSARequestCacheLayoutProjection:
    return DSARequestCacheLayoutProjection(
        request_ids=("enter",),
        stages=(int(DSARequestCacheStage.ENTER_SPARSE_DECODE),),
        target_resident_budget_tokens=(256,),
        sparse_budget_tokens=(256,),
        resident_valid_tokens=(257,),
        resident_block_table_replacements=(
            DSAResidentBlockTableReplacement(
                request_id="enter",
                block_ids=(20, 21, 14),
            ),
        ),
    )


def _make_enter_cached_data() -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["enter"],
        resumed_req_ids=set(),
        # Indexer 仍按原生增量追加；resident 的 20/21 是 replacement 前缀。
        new_block_ids=[([2], [20, 21])],
    )


def test_enter_normalization_prevents_persistent_row_temporary_overflow() -> None:
    input_batch = _InputBatch(("enter",))
    input_batch.block_table = [
        _BoundedBlockTable(capacity=6),
        _BoundedBlockTable(capacity=6),
    ]
    input_batch.block_table[0].add_row([1], 0)
    input_batch.block_table[1].add_row([10, 11, 12, 13, 14], 0)
    requests = {
        "enter": SimpleNamespace(
            block_ids=[[1], [10, 11, 12, 13, 14]],
        ),
    }
    cached_data = _make_enter_cached_data()
    projection = _make_enter_projection()

    # 未归一化时 resident 会先形成 5 + 2 > 6 的错误中间态；
    # 最终 replacement 只有 3 个 block，本身完全合法。
    assert (
        len(requests["enter"].block_ids[1])
        + len(cached_data.new_block_ids[0][1])
        > input_batch.block_table[1].capacity
    )

    normalize_dsa_enter_updates_before_base(
        requests=requests,
        cached_requests=cached_data,
        projection=projection,
        resident_group_id=1,
    )

    # 模拟原生 cached-request update：请求状态与 persistent block table
    # 都只追加剩余的 Indexer 增量。
    for block_ids, new_ids in zip(
        requests["enter"].block_ids,
        cached_data.new_block_ids[0],
        strict=True,
    ):
        block_ids.extend(new_ids)
    for block_table, new_ids in zip(
        input_batch.block_table,
        cached_data.new_block_ids[0],
        strict=True,
    ):
        block_table.append_row(new_ids, 0)

    state = _make_state()
    apply_dsa_cache_layout_projection(
        input_batch=input_batch,
        requests=requests,
        state=state,
        projection=projection,
        resident_group_id=1,
    )

    assert cached_data.new_block_ids == [([2], [])]
    assert requests["enter"].block_ids == [[1, 2], [20, 21, 14]]
    assert input_batch.block_table[0].rows[0] == [1, 2]
    assert input_batch.block_table[1].rows[0] == [20, 21, 14]


def test_enter_normalization_readds_final_table_after_chunk_barrier() -> None:
    input_batch = _InputBatch(("enter",))
    input_batch.block_table = [
        _BoundedBlockTable(capacity=6),
        _BoundedBlockTable(capacity=6),
    ]
    requests = {
        # 请求仍在 worker cache 中，但已被移出 persistent InputBatch。
        "enter": SimpleNamespace(
            block_ids=[[1], [10, 11, 12, 13, 14]],
        ),
    }
    cached_data = _make_enter_cached_data()
    projection = _make_enter_projection()

    normalize_dsa_enter_updates_before_base(
        requests=requests,
        cached_requests=cached_data,
        projection=projection,
        resident_group_id=1,
    )

    # 模拟原生 req_index=None 分支：先消费增量，再用完整请求状态 add row。
    for block_ids, new_ids in zip(
        requests["enter"].block_ids,
        cached_data.new_block_ids[0],
        strict=True,
    ):
        block_ids.extend(new_ids)
    for block_table, block_ids in zip(
        input_batch.block_table,
        requests["enter"].block_ids,
        strict=True,
    ):
        block_table.add_row(block_ids, 0)

    state = _make_state()
    apply_dsa_cache_layout_projection(
        input_batch=input_batch,
        requests=requests,
        state=state,
        projection=projection,
        resident_group_id=1,
    )

    assert requests["enter"].block_ids == [[1, 2], [20, 21, 14]]
    assert input_batch.block_table[1].rows[0] == [20, 21, 14]


def test_stable_projection_skips_cached_request_scan() -> None:
    projection = DSARequestCacheLayoutProjection(
        request_ids=("dense",),
        stages=(int(DSARequestCacheStage.DENSE_DECODE),),
        target_resident_budget_tokens=(256,),
        sparse_budget_tokens=(0,),
        resident_valid_tokens=(-1,),
        resident_block_table_replacements=(),
    )

    class _NoCachedRequestAccess:
        @property
        def req_ids(self):
            raise AssertionError("stable path must not scan cached requests")

    normalize_dsa_enter_updates_before_base(
        requests={},
        cached_requests=_NoCachedRequestAccess(),
        projection=projection,
        resident_group_id=1,
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
    assert state.row_modes_cpu[:2].tolist() == [1, 2]
    assert state.resident_pool_indices_cpu[:2].tolist() == [0, 1]
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


def test_graph_padding_rows_are_reset_when_active_batch_shrinks() -> None:
    state = _make_state()
    full_input_batch = _InputBatch(
        ("dense", "enter", "old-2", "old-3")
    )
    full_projection = DSARequestCacheLayoutProjection(
        request_ids=("dense", "enter", "old-2", "old-3"),
        stages=(
            int(DSARequestCacheStage.DENSE_DECODE),
            int(DSARequestCacheStage.SPARSE_DECODE),
            int(DSARequestCacheStage.DENSE_DECODE),
            int(DSARequestCacheStage.SPARSE_DECODE),
        ),
        target_resident_budget_tokens=(2048, 4096, 2048, 4096),
        sparse_budget_tokens=(0, 4096, 0, 4096),
        resident_valid_tokens=(-1, 4097, -1, 4097),
        resident_block_table_replacements=(),
    )
    state.refresh(
        input_batch=full_input_batch,
        projection=full_projection,
    )

    input_batch = _InputBatch()
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
    assert state.row_modes_cpu.tolist() == [1, 2, 0, 0]
    assert state.resident_pool_indices_cpu[2:].tolist() == [4, 4]


def test_stable_row_order_uses_bulk_column_refresh() -> None:
    input_batch = _InputBatch()
    input_batch.req_id_to_index = _NoLookupRequestRows(
        {
            "dense": 0,
            "enter": 1,
        }
    )
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

    assert state.valid
    assert state.row_count == 2
    assert state.stages_cpu[:2].tolist() == [
        int(DSARequestCacheStage.DENSE_DECODE),
        int(DSARequestCacheStage.SPARSE_DECODE),
    ]
    assert state.target_resident_budget_tokens_cpu[:2].tolist() == [
        2048,
        4096,
    ]
    assert state.sparse_budget_tokens_cpu[:2].tolist() == [0, 4096]
    assert state.resident_valid_tokens_cpu[:2].tolist() == [-1, 4097]
    assert state.resident_pool_indices_cpu[:2].tolist() == [0, 1]


def test_unscheduled_request_keeps_pool_row_until_finished() -> None:
    state = _make_state()
    released_rows: list[int] = []
    state.set_pool_release_callback(released_rows.append)

    both = _InputBatch(("req-a", "req-b"))
    both_projection = DSARequestCacheLayoutProjection(
        request_ids=("req-a", "req-b"),
        stages=(
            int(DSARequestCacheStage.DENSE_DECODE),
            int(DSARequestCacheStage.DENSE_DECODE),
        ),
        target_resident_budget_tokens=(2048, 2048),
        sparse_budget_tokens=(0, 0),
        resident_valid_tokens=(-1, -1),
        resident_block_table_replacements=(),
    )
    state.refresh(
        input_batch=both,
        projection=both_projection,
    )
    req_a_pool_row = state.resident_token_pool.get_index("req-a")
    assert req_a_pool_row is not None
    state.resident_token_pool.get_cache_slots(0)[
        req_a_pool_row,
        0,
    ] = 123

    # vLLM 会把本轮未调度的 req-a 临时移出 persistent InputBatch。
    only_b = _InputBatch(("req-b",))
    only_b_projection = DSARequestCacheLayoutProjection(
        request_ids=("req-b",),
        stages=(int(DSARequestCacheStage.DENSE_DECODE),),
        target_resident_budget_tokens=(2048,),
        sparse_budget_tokens=(0,),
        resident_valid_tokens=(-1,),
        resident_block_table_replacements=(),
    )
    state.refresh(
        input_batch=only_b,
        projection=only_b_projection,
    )

    assert state.resident_token_pool.get_index("req-a") == req_a_pool_row
    assert (
        state.resident_token_pool.get_cache_slots(0)[
            req_a_pool_row,
            0,
        ].item()
        == 123
    )
    assert released_rows == []

    # 再次调度时复用同一请求级状态；只有真正 finished 才回收。
    state.refresh(
        input_batch=both,
        projection=both_projection,
    )
    assert state.resident_pool_indices_cpu[0] == req_a_pool_row
    state.release_request("req-a")
    assert state.resident_token_pool.get_index("req-a") is None
    assert released_rows == [req_a_pool_row]

    # finished 与同一轮同名新请求重叠时，也必须重新绑定 pool row。
    state.refresh(
        input_batch=both,
        projection=both_projection,
    )
    assert state.resident_token_pool.get_index("req-a") is not None


def test_graph_capture_temporarily_reuses_device_owner() -> None:
    state = _make_state()
    cpu_before = state.columns.cpu.clone()

    state.prepare_graph_capture(
        row_count=4,
        target_budget_tokens=2048,
        resident_valid_tokens=2049,
    )

    assert state.graph_capture_row_count == 4
    assert state.stages[:4].tolist() == [
        int(DSARequestCacheStage.SPARSE_DECODE)
    ] * 4
    assert state.row_modes[:4].tolist() == [2, 2, 2, 2]
    assert state.resident_pool_indices[:4].tolist() == [0, 1, 2, 3]
    # capture dummy 不能改写 scheduler/worker 的 CPU 语义真源。
    assert torch.equal(state.columns.cpu, cpu_before)

    state.restore_after_graph_capture()

    assert state.graph_capture_row_count == 0
    assert torch.equal(state.columns.gpu, cpu_before)


def test_graph_capture_reuses_one_input_owner_across_sizes() -> None:
    state = _make_state()
    cpu_before = state.columns.cpu.clone()
    cpu_ptr = state.columns.cpu.data_ptr()
    gpu_ptr = state.columns.gpu.data_ptr()

    for row_count in (1, 2, 4, 2):
        state.prepare_graph_capture(
            row_count=row_count,
            target_budget_tokens=2048,
            resident_valid_tokens=2049,
        )
        assert state.columns.cpu.data_ptr() == cpu_ptr
        assert state.columns.gpu.data_ptr() == gpu_ptr
        assert state.resident_pool_indices[:row_count].tolist() == list(
            range(row_count)
        )

        state.restore_after_graph_capture()
        assert state.graph_capture_row_count == 0
        assert torch.equal(state.columns.gpu, cpu_before)
