import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.utils import AscendDeviceType


class _FakeKVCacheSpec:
    def __init__(
        self,
        *,
        page_size_bytes,
        max_blocks_per_request,
        store_on_host,
        block_size=128,
    ):
        self.page_size_bytes = page_size_bytes
        self.max_blocks_per_request = max_blocks_per_request
        self.store_on_host = store_on_host
        self.block_size = block_size

    def max_memory_usage_bytes(self, _vllm_config):
        return self.max_blocks_per_request * self.page_size_bytes


def _make_memory_plan_inputs(max_num_seqs=2):
    specs = {
        "host.0": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "host.1": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "device.0": _FakeKVCacheSpec(
            page_size_bytes=512,
            max_blocks_per_request=100,
            store_on_host=False,
        ),
    }
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    alignment_reserve = 2 * manager_module._CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return specs, vllm_config, alignment_reserve


class TestSparseKVOffloadMemoryPlanning(unittest.TestCase):
    def test_non_a3_is_rejected(self):
        with (
            patch.object(manager_module, "_SPARSE_KV_OFFLOAD_MANAGER", None),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A2),
            self.assertRaisesRegex(RuntimeError, "Sparse KV offload is only support on A3"),
        ):
            manager_module.init_sparse_kv_offload_manager(None, None, None)

    def test_memory_plan_is_limited_by_active_workload(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.npu_limit_blocks, 1000)
        self.assertEqual(budget.dram_limit_blocks, 1000)
        self.assertEqual(budget.workload_limit_blocks, 201)
        self.assertEqual(budget.final_num_blocks, 201)
        self.assertEqual(budget.final_planner_bytes, 201 * (2048 + 512))
        self.assertEqual(budget.planned_host_bytes, 201 * 2048)
        self.assertEqual(budget.planned_device_bytes, 201 * 512)
        self.assertEqual(budget.limiting_factor, "workload")

    def test_memory_plan_is_limited_by_dram_capacity(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 150 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.dram_limit_blocks, 150)
        self.assertEqual(budget.final_num_blocks, 150)
        self.assertEqual(budget.limiting_factor, "dram")

    def test_warning_when_dram_budget_caps_npu_utilization(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with self.assertLogs(manager_module.logger, level="WARNING") as logs:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 500 * 2048,
                keep_device_kv_cache=False,
            )
        self.assertTrue(any("dram_size_per_dp_GB" in line for line in logs.output))

    def test_no_warning_when_dram_budget_not_below_npu(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with patch.object(manager_module.logger, "warning_once") as mock_warn:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 1000 * 2048,
                keep_device_kv_cache=False,
            )
        mock_warn.assert_not_called()

    def test_keep_device_cache_counts_full_npu_page(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        total_page_size = 2048 + 512

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=125 * total_page_size,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=True,
        )

        self.assertEqual(budget.npu_limit_blocks, 125)
        self.assertEqual(budget.final_num_blocks, 125)
        self.assertEqual(budget.planned_device_bytes, 125 * total_page_size)
        self.assertEqual(budget.limiting_factor, "npu")

    def test_non_positive_capacity_produces_zero_blocks(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        for available_device_memory, dram_limit, limiting_factor in (
            (-1, alignment_reserve + 1000 * 2048, "npu"),
            (1000 * 512, alignment_reserve - 1, "dram"),
        ):
            with self.subTest(limiting_factor=limiting_factor):
                budget = plan_sparse_kv_offload_memory(
                    kv_cache_spec=specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=available_device_memory,
                    dram_limit_bytes=dram_limit,
                    keep_device_kv_cache=False,
                )
                self.assertEqual(budget.final_num_blocks, 0)
                self.assertEqual(budget.final_planner_bytes, 0)
                self.assertEqual(budget.limiting_factor, limiting_factor)

    def test_memory_plan_rejects_invalid_spec_layouts(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        invalid_cases = (
            ({"device.0": specs["device.0"]}, "at least one host"),
            ({"host.0": specs["host.0"]}, "at least one device"),
            (
                {
                    **specs,
                    "device.1": _FakeKVCacheSpec(
                        page_size_bytes=512,
                        max_blocks_per_request=100,
                        store_on_host=False,
                        block_size=64,
                    ),
                },
                "one shared block size",
            ),
        )

        for invalid_specs, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plan_sparse_kv_offload_memory(
                    kv_cache_spec=invalid_specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=1000 * 512,
                    dram_limit_bytes=alignment_reserve + 1000 * 2048,
                    keep_device_kv_cache=False,
                )

    def test_cpu_pool_size_includes_per_layer_alignment_reserve(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name],
                    kv_cache_spec=spec,
                )
                for name, spec in specs.items()
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_supports_uniform_specs(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        uniform_specs = UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs=specs,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(specs),
                    kv_cache_spec=uniform_specs,
                )
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_rejects_missing_host_specs(self):
        specs, _, _ = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["device.0"],
                    kv_cache_spec=specs["device.0"],
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "host-resident"):
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)

    def _make_manager_init_inputs(self, dram_size_per_dp_gb=1):
        spec = _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[SimpleNamespace(layer_names=["host.0"], kv_cache_spec=spec)],
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_num_layers=MagicMock(return_value=1),
                max_model_len=128,
            ),
            parallel_config=SimpleNamespace(data_parallel_index=0),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=1,
            ),
            speculative_config=None,
        )
        offload_config = SimpleNamespace(
            topk_buffer_size=1,
            topk=1,
            use_fused_overlap=False,
            use_nano=False,
            dram_size_per_dp_GB=dram_size_per_dp_gb,
        )
        return vllm_config, kv_cache_config, offload_config

    def test_manager_initializes_offload_with_planned_pool_size(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        planned_pool_size = 4096
        for rank, expected_alloc_size in ((0, planned_pool_size), (1, 0)):
            with self.subTest(rank=rank):
                offload_backend = SimpleNamespace(
                    OffloadConfig=lambda: SimpleNamespace(),
                    Scene=SimpleNamespace(SHARED="shared"),
                    initialize=MagicMock(return_value=0),
                )
                tp_group = SimpleNamespace(barrier=MagicMock())

                with (
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_rank",
                        return_value=rank,
                    ),
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_world_size",
                        return_value=2,
                    ),
                    patch.object(
                        manager_module,
                        "get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        manager_module,
                        "get_sparse_kv_offload_cpu_pool_size_bytes",
                        return_value=planned_pool_size,
                    ),
                    patch.object(
                        manager_module,
                        "offload",
                        offload_backend,
                        create=True,
                    ),
                    patch.object(
                        manager_module.torch,
                        "zeros",
                        return_value=MagicMock(),
                    ),
                    patch.object(
                        manager_module.torch,
                        "empty",
                        return_value=MagicMock(),
                    ),
                    patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
                ):
                    SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
                    )

                initialized_config = offload_backend.initialize.call_args.args[0]
                self.assertEqual(
                    initialized_config.reserve_size,
                    planned_pool_size,
                )
                self.assertEqual(
                    initialized_config.alloc_size,
                    expected_alloc_size,
                )
                self.assertEqual(initialized_config.world_size, 2)
                self.assertEqual(initialized_config.rank_id, rank)
                tp_group.barrier.assert_called_once_with()

    def test_manager_rejects_pool_larger_than_dram_limit(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        offload_backend = SimpleNamespace(
            OffloadConfig=MagicMock(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(
                manager_module,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch.object(
                manager_module,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                manager_module,
                "get_tp_group",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=(1 << 30) + 1,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            self.assertRaisesRegex(ValueError, "exceeds DRAM limit"),
        ):
            SparseKVOffloadManager(
                vllm_config,
                kv_cache_config,
                offload_config,
            )

        offload_backend.OffloadConfig.assert_not_called()
        offload_backend.initialize.assert_not_called()


class TestNanoD2HPlanner(unittest.TestCase):
    @staticmethod
    def _manager(max_num_reqs: int = 4) -> SparseKVOffloadManager:
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.max_num_reqs = max_num_reqs
        manager.block_size = 128
        manager.topk_buffer_size = 8192
        manager._allocate_nano_d2h_planner_state()
        return manager

    def test_full_block_boundary_freezes_one_plan_without_advancing_prefix(self):
        manager = self._manager()
        manager.initialize_nano_d2h_slots(
            np.array([0]),
            np.array([10]),
            np.array([0]),
        )
        block_table = np.array([[7, 8]], dtype=np.int32)

        count = manager.plan_nano_d2h_requests(
            np.array([127]),
            np.array([0]),
            np.array([10]),
            block_table,
            np.array([True]),
        )
        self.assertEqual(count, 0)
        self.assertFalse(manager.nano_d2h_inflight_batch_active)

        count = manager.plan_nano_d2h_requests(
            np.array([128]),
            np.array([0]),
            np.array([10]),
            block_table,
            np.array([True]),
        )

        self.assertEqual(count, 1)
        self.assertTrue(manager.nano_d2h_inflight_batch_active)
        self.assertEqual(np.flatnonzero(manager.nano_d2h_inflight_active_cpu).tolist(), [0])
        self.assertEqual(manager.nano_d2h_inflight_generations_cpu[0], 10)
        self.assertEqual(manager.nano_d2h_inflight_source_slots_cpu[0], 8192)
        self.assertEqual(manager.nano_d2h_inflight_destination_slots_cpu[0], 7)
        self.assertEqual(manager.nano_d2h_inflight_target_prefixes_cpu[0], 128)
        self.assertEqual(manager.nano_d2h_stable_prefixes_cpu[0], 0)

    def test_cpu_plan_fields_share_one_pinned_host_staging_buffer(self):
        manager = self._manager()

        manager.nano_d2h_inflight_generations_cpu[1] = 42
        manager.nano_d2h_inflight_target_prefixes_cpu[1] = 128

        self.assertEqual(manager.nano_d2h_inflight_generations_host[1].item(), 42)
        self.assertEqual(manager.nano_d2h_inflight_target_prefixes_host[1].item(), 128)
        plan_storage = manager.nano_d2h_plan_host.untyped_storage().data_ptr()
        for field in (
            manager.nano_d2h_inflight_active_host,
            manager.nano_d2h_inflight_generations_host,
            manager.nano_d2h_inflight_source_slots_host,
            manager.nano_d2h_inflight_destination_slots_host,
            manager.nano_d2h_inflight_target_prefixes_host,
        ):
            self.assertEqual(field.untyped_storage().data_ptr(), plan_storage)

    def test_next_full_block_uses_other_tail_page_and_next_host_block(self):
        manager = self._manager()
        manager.initialize_nano_d2h_slots(
            np.array([2]),
            np.array([21]),
            np.array([128]),
        )

        count = manager.plan_nano_d2h_requests(
            np.array([256]),
            np.array([2]),
            np.array([21]),
            np.array([[30, 31, 32]], dtype=np.int32),
            np.array([True]),
        )

        request_stride = manager.topk_buffer_size + 2 * manager.block_size
        self.assertEqual(count, 1)
        self.assertEqual(
            manager.nano_d2h_inflight_source_slots_cpu[2],
            2 * request_stride + manager.topk_buffer_size + manager.block_size,
        )
        self.assertEqual(manager.nano_d2h_inflight_destination_slots_cpu[2], 31)
        self.assertEqual(manager.nano_d2h_inflight_target_prefixes_cpu[2], 256)
        self.assertEqual(manager.nano_d2h_stable_prefixes_cpu[2], 128)

    def test_multiple_ready_requests_share_one_fixed_capacity_batch(self):
        manager = self._manager(max_num_reqs=3)
        manager.initialize_nano_d2h_slots(
            np.array([0, 4]),
            np.array([5, 9]),
            np.array([0, 128]),
        )

        count = manager.plan_nano_d2h_requests(
            np.array([128, 999, 256]),
            np.array([0, manager.nano_d2h_capacity + 8, 4]),
            np.array([5, -1, 9]),
            np.array([[10, 11], [-1, -1], [20, 21]], dtype=np.int32),
            np.array([True, False, True]),
        )

        self.assertEqual(count, 2)
        self.assertEqual(np.flatnonzero(manager.nano_d2h_inflight_active_cpu).tolist(), [0, 4])
        self.assertEqual(manager.nano_d2h_inflight_destination_slots_cpu[[0, 4]].tolist(), [10, 21])

    def test_rejection_corrected_length_is_the_only_boundary_input(self):
        for corrected_length, expected_count in ((127, 0), (128, 1)):
            with self.subTest(corrected_length=corrected_length):
                manager = self._manager()
                manager.initialize_nano_d2h_slots(
                    np.array([1]),
                    np.array([12]),
                    np.array([0]),
                )
                count = manager.plan_nano_d2h_requests(
                    np.array([corrected_length]),
                    np.array([1]),
                    np.array([12]),
                    np.array([[40]], dtype=np.int32),
                    np.array([True]),
                )
                self.assertEqual(count, expected_count)

    def test_generation_mismatch_cannot_plan_for_reused_slot(self):
        manager = self._manager()
        manager.initialize_nano_d2h_slots(
            np.array([0]),
            np.array([3]),
            np.array([0]),
        )

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            manager.plan_nano_d2h_requests(
                np.array([128]),
                np.array([0]),
                np.array([4]),
                np.array([[2]], dtype=np.int32),
                np.array([True]),
            )
        self.assertFalse(manager.nano_d2h_inflight_batch_active)
        self.assertFalse(manager.nano_d2h_inflight_active_cpu.any())

    def test_two_unretired_blocks_are_rejected(self):
        manager = self._manager()
        manager.initialize_nano_d2h_slots(
            np.array([0]),
            np.array([1]),
            np.array([0]),
        )

        with self.assertRaisesRegex(RuntimeError, "two unretired full blocks"):
            manager.plan_nano_d2h_requests(
                np.array([256]),
                np.array([0]),
                np.array([1]),
                np.array([[2, 3]], dtype=np.int32),
                np.array([True]),
            )

    def test_inflight_slot_cannot_be_rebound_or_replanned(self):
        manager = self._manager()
        manager.initialize_nano_d2h_slots(
            np.array([0]),
            np.array([1]),
            np.array([0]),
        )
        manager.plan_nano_d2h_requests(
            np.array([128]),
            np.array([0]),
            np.array([1]),
            np.array([[2]], dtype=np.int32),
            np.array([True]),
        )

        with self.assertRaisesRegex(RuntimeError, "cannot be rebound"):
            manager.initialize_nano_d2h_slots(
                np.array([0]),
                np.array([2]),
                np.array([0]),
            )
        with self.assertRaisesRegex(RuntimeError, "retire the current inflight batch"):
            manager.plan_nano_d2h_requests(
                np.array([129]),
                np.array([0]),
                np.array([1]),
                np.array([[2]], dtype=np.int32),
                np.array([True]),
            )

    def test_request_plan_expands_all_layer_kv_descriptors(self):
        manager = self._manager(max_num_reqs=2)
        manager.tp_rank = 0
        manager.num_layers = 2
        manager.token_size_bytes_k = 4
        manager.token_size_bytes_v = 6
        manager.addr_k_bases = [1_000, 3_000]
        manager.addr_v_bases = [2_000, 4_000]
        manager.gvas_k_bases = [100_000, 300_000]
        manager.gvas_v_bases = [200_000, 400_000]
        manager._allocate_nano_d2h_descriptor_state(torch.device("cpu"))
        manager.initialize_nano_d2h_slots(
            np.array([1]),
            np.array([9]),
            np.array([0]),
        )
        manager.plan_nano_d2h_requests(
            np.array([0, 128]),
            np.array([manager.nano_d2h_capacity + 1, 1]),
            np.array([-1, 9]),
            np.array([[-1], [7]], dtype=np.int32),
            np.array([False, True]),
        )

        count = manager.prepare_nano_d2h_descriptors()

        capacity = manager.nano_d2h_capacity
        request_stride = manager.topk_buffer_size + 2 * manager.block_size
        source_slot = request_stride + manager.topk_buffer_size
        self.assertEqual(count, manager.num_layers * 2 * capacity)
        self.assertTrue(torch.equal(manager.nano_d2h_plan_npu, manager.nano_d2h_plan_host))
        self.assertEqual(manager.nano_d2h_plan_generations_npu[1].item(), 9)
        self.assertEqual(manager.nano_d2h_plan_target_prefixes_npu[1].item(), 128)
        plan_storage = manager.nano_d2h_plan_npu.untyped_storage().data_ptr()
        for field in (
            manager.nano_d2h_plan_active_npu,
            manager.nano_d2h_plan_generations_npu,
            manager.nano_d2h_plan_source_slots_npu,
            manager.nano_d2h_plan_destination_slots_npu,
            manager.nano_d2h_plan_target_prefixes_npu,
        ):
            self.assertEqual(field.untyped_storage().data_ptr(), plan_storage)
        self.assertEqual(
            manager.nano_d2h_source_ptrs_npu[:, :, 1].tolist(),
            [
                [1_000 + source_slot * 4, 2_000 + source_slot * 6],
                [3_000 + source_slot * 4, 4_000 + source_slot * 6],
            ],
        )
        self.assertEqual(
            manager.nano_d2h_destination_ptrs_npu[:, :, 1].tolist(),
            [
                [100_000 + 7 * 128 * 4, 200_000 + 7 * 128 * 6],
                [300_000 + 7 * 128 * 4, 400_000 + 7 * 128 * 6],
            ],
        )
        self.assertEqual(
            manager.nano_d2h_lengths_npu[:, :, 1].tolist(),
            [[128 * 4, 128 * 6], [128 * 4, 128 * 6]],
        )
        self.assertTrue(torch.all(manager.nano_d2h_lengths_npu[:, :, 0] == 0))
        self.assertTrue(torch.all(manager.nano_d2h_lengths_npu[:, :, 2:] == 0))

    def test_descriptor_expansion_requires_tp0_and_an_inflight_batch(self):
        manager = self._manager()
        manager.tp_rank = 1
        with self.assertRaisesRegex(RuntimeError, "Only TP0"):
            manager.prepare_nano_d2h_descriptors()

        manager.tp_rank = 0
        with self.assertRaisesRegex(RuntimeError, "no inflight"):
            manager.prepare_nano_d2h_descriptors()


if __name__ == "__main__":
    unittest.main()
