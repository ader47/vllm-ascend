# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA worker-local、固定容量的热 DRAM block store。

该 store 只维护两类状态：

* 每层 NOPE/ROPE payload 的 Ascend swapped-memory arena；
* ``resident pool row -> logical full block -> DRAM block id`` 逻辑表。

首版明确关闭 prefix cache、preemption 和 KV connector，因此不同请求不会
共享 DRAM block，也不需要 hash/refcount 体系。block 0 保留为空映射，有效
物理 block id 从 1 开始；初始化后 arena 地址和容量均不可变化。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from typing_extensions import NamedTuple

from vllm_ascend.dsa_offload.contracts import DSA_DRAM_NULL_BLOCK_ID

ArenaFactory = Callable[
    [tuple[int, ...], torch.dtype, int, torch.device],
    torch.Tensor,
]


@dataclass(frozen=True)
class DSALayerDRAMArenas:
    """同一 transformer layer 的两张 NPU 可寻址 DRAM arena。"""

    nope: torch.Tensor
    rope: torch.Tensor


class DSADRAMReservation(NamedTuple):
    """逻辑块解析结果；``new_mask`` 只标记本轮需要真实 dump 的行。"""

    physical_block_ids: np.ndarray
    new_mask: np.ndarray


def calculate_dram_usable_blocks(
    indexer_num_blocks: int,
    block_multiple: float,
) -> int:
    """按浮点倍数向上取整，避免截断用户配置。"""

    indexer_num_blocks = int(indexer_num_blocks)
    block_multiple = float(block_multiple)
    if indexer_num_blocks <= 0:
        raise ValueError(
            "DSA DRAM sizing requires positive Indexer blocks, got "
            f"{indexer_num_blocks}"
        )
    if not math.isfinite(block_multiple) or block_multiple <= 0:
        raise ValueError(
            "DSA hot_cpu_block_multiple must be positive and finite, got "
            f"{block_multiple}"
        )
    return int(math.ceil(indexer_num_blocks * block_multiple))


def _allocate_swapped_arena(
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "DSA sparse offload requires torch_npu swapped-memory arenas"
        ) from exc

    arena = torch_npu.empty_with_swapped_memory(
        (int(capacity), *block_shape),
        dtype=dtype,
        device=device,
    )
    if not arena.is_contiguous():
        raise RuntimeError(
            "DSA swapped-memory arena must be contiguous, got "
            f"shape={tuple(arena.shape)}, stride={tuple(arena.stride())}"
        )
    return arena


class DSAHotDRAMStore:
    """管理 worker 内所有层共享的 DRAM block-id 空间。"""

    def __init__(
        self,
        *,
        usable_blocks: int,
        storage_rows: int,
        max_logical_blocks: int,
        device: torch.device,
        arena_factory: ArenaFactory | None = None,
    ) -> None:
        if usable_blocks <= 0:
            raise ValueError("DSA DRAM store requires positive usable blocks")
        if storage_rows <= 0:
            raise ValueError("DSA DRAM store requires positive row capacity")
        if max_logical_blocks <= 0:
            raise ValueError(
                "DSA DRAM store requires positive logical block width"
            )

        self.usable_blocks = int(usable_blocks)
        self.arena_capacity = self.usable_blocks + 1
        self.storage_rows = int(storage_rows)
        self.max_logical_blocks = int(max_logical_blocks)
        self.device = torch.device(device)
        self._arena_factory = arena_factory or _allocate_swapped_arena
        self._layer_arenas: dict[int, DSALayerDRAMArenas] = {}

        self.logical_block_table = np.full(
            (self.storage_rows, self.max_logical_blocks),
            DSA_DRAM_NULL_BLOCK_ID,
            dtype=np.int32,
        )
        # [0, free_count) 是当前可分配栈。有效 block id 从 1 开始。
        self._free_block_ids = np.arange(
            1,
            self.usable_blocks + 1,
            dtype=np.int32,
        )
        self._free_count = self.usable_blocks
        self.table_version = 0

    @property
    def num_free_blocks(self) -> int:
        return int(self._free_count)

    def add_layer(
        self,
        *,
        layer_id: int,
        resident_nope_cache: torch.Tensor,
        resident_rope_cache: torch.Tensor,
    ) -> None:
        """按 resident cache 的真实 block shape 建立该层 arena。"""

        layer_id = int(layer_id)
        if layer_id in self._layer_arenas:
            raise RuntimeError(
                f"DSA DRAM layer {layer_id} was initialized twice"
            )
        if resident_nope_cache.shape[0] <= 0 or resident_rope_cache.shape[0] <= 0:
            raise ValueError("DSA resident cache must contain physical blocks")
        nope = self._arena_factory(
            tuple(resident_nope_cache.shape[1:]),
            resident_nope_cache.dtype,
            self.arena_capacity,
            self.device,
        )
        rope = self._arena_factory(
            tuple(resident_rope_cache.shape[1:]),
            resident_rope_cache.dtype,
            self.arena_capacity,
            self.device,
        )
        # block 0 是空映射。只清这一块，不触碰大块有效 arena。
        nope[DSA_DRAM_NULL_BLOCK_ID].zero_()
        rope[DSA_DRAM_NULL_BLOCK_ID].zero_()
        self._layer_arenas[layer_id] = DSALayerDRAMArenas(
            nope=nope,
            rope=rope,
        )

    def get_layer_arenas(self, layer_id: int) -> DSALayerDRAMArenas:
        try:
            return self._layer_arenas[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"DSA DRAM arenas are not initialized for layer {layer_id}"
            ) from exc

    def reserve_blocks(
        self,
        *,
        pool_indices: np.ndarray,
        logical_block_indices: np.ndarray,
    ) -> DSADRAMReservation:
        """批量解析 logical block，缺失项一次性分配物理 DRAM block。"""

        pool_indices = np.asarray(pool_indices, dtype=np.intp)
        logical_block_indices = np.asarray(
            logical_block_indices,
            dtype=np.intp,
        )
        if (
            pool_indices.ndim != 1
            or logical_block_indices.ndim != 1
            or pool_indices.shape != logical_block_indices.shape
        ):
            raise ValueError(
                "DSA DRAM reservation requires matching one-dimensional rows"
        )
        if pool_indices.size == 0:
            return DSADRAMReservation(
                physical_block_ids=np.empty(0, dtype=np.int32),
                new_mask=np.empty(0, dtype=np.bool_),
            )
        if (
            int(pool_indices.min()) < 0
            or int(pool_indices.max()) >= self.storage_rows
        ):
            raise IndexError("DSA DRAM pool row is outside table capacity")
        if (
            int(logical_block_indices.min()) < 0
            or int(logical_block_indices.max()) >= self.max_logical_blocks
        ):
            raise IndexError(
                "DSA logical block is outside DRAM table capacity"
            )

        flat_keys = (
            pool_indices.astype(np.int64, copy=False)
            * self.max_logical_blocks
            + logical_block_indices.astype(np.int64, copy=False)
        )
        if np.unique(flat_keys).size != flat_keys.size:
            raise RuntimeError(
                "DSA dump plan contains duplicate request/logical-block rows"
            )

        physical = self.logical_block_table[
            pool_indices,
            logical_block_indices,
        ].copy()
        missing = physical == DSA_DRAM_NULL_BLOCK_ID
        missing_count = int(np.count_nonzero(missing))
        if missing_count:
            if missing_count > self._free_count:
                raise RuntimeError(
                    "DSA hot DRAM block pool is exhausted: "
                    f"required={missing_count}, free={self._free_count}, "
                    f"capacity={self.usable_blocks}"
                )
            start = self._free_count - missing_count
            allocated = self._free_block_ids[start : self._free_count].copy()
            self._free_count = start
            physical[missing] = allocated
            self.logical_block_table[
                pool_indices[missing],
                logical_block_indices[missing],
            ] = allocated
            self.table_version += 1
        return DSADRAMReservation(
            physical_block_ids=physical,
            new_mask=missing,
        )

    def release_pool_index(self, pool_index: int) -> None:
        """请求结束时整行回收，无逐 logical-block Python 调用。"""

        pool_index = int(pool_index)
        if not 0 <= pool_index < self.storage_rows:
            raise IndexError(
                f"DSA DRAM pool row {pool_index} is outside capacity"
            )
        row = self.logical_block_table[pool_index]
        released = row[row != DSA_DRAM_NULL_BLOCK_ID]
        if released.size == 0:
            return
        released = np.unique(released)
        end = self._free_count + int(released.size)
        if end > self.usable_blocks:
            raise RuntimeError(
                "DSA DRAM free-list overflow while releasing pool row "
                f"{pool_index}"
            )
        self._free_block_ids[self._free_count : end] = released
        self._free_count = end
        row.fill(DSA_DRAM_NULL_BLOCK_ID)
        self.table_version += 1

    def gather_rows(
        self,
        *,
        pool_indices: np.ndarray,
        output: np.ndarray,
    ) -> None:
        """将稳定 pool 行批量投影到当前 InputBatch 行序。"""

        pool_indices = np.asarray(pool_indices, dtype=np.intp)
        if output.shape != (
            pool_indices.size,
            self.max_logical_blocks,
        ):
            raise ValueError(
                "DSA active DRAM table output shape mismatch: "
                f"output={output.shape}, rows={pool_indices.size}, "
                f"width={self.max_logical_blocks}"
            )
        np.take(
            self.logical_block_table,
            pool_indices,
            axis=0,
            out=output,
        )
