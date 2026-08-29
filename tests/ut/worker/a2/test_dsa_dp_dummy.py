# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.worker.model_runner_v1 as runner_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class _MetadataChecked(Exception):
    pass


@pytest.mark.parametrize("num_reqs,padded_reqs", [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4)])
@pytest.mark.parametrize("phase", ["idle", "capture", "warmup"])
def test_dsa_dummy_padding_and_cleanup(monkeypatch, num_reqs, padded_reqs, phase):
    """Exercise the real dummy preparation, stopping before any NPU forward."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.uniform_decode_query_len = 4
    runner.scheduler_config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32)
    runner.model_config = SimpleNamespace(use_mla=True, max_model_len=16384)
    runner.vllm_config = SimpleNamespace(model_config=runner.model_config)
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.ascend_config = SimpleNamespace(
        dsa_offload_config=SimpleNamespace(resident_budget_tokens=[10240]),
    )
    runner.dynamic_eplb = runner.use_cp = runner.use_compress = runner._has_gdn = False
    runner.optimistic_seq_lens_cpu = torch.full((8,), 99, dtype=torch.int32)
    runner.seq_lens = torch.full((8,), 99, dtype=torch.int32)
    runner.query_pos = SimpleNamespace(np=np.arange(32))
    runner.query_start_loc = SimpleNamespace(np=np.zeros(9, dtype=np.int32))
    runner.input_batch = SimpleNamespace(dsa_cache_layout=MagicMock(), block_table=MagicMock())
    runner.dsa_resident_token_pool = MagicMock()
    runner.dsa_offload_runtime = MagicMock()
    for owner in (runner.input_batch.dsa_cache_layout, runner.dsa_resident_token_pool, runner.dsa_offload_runtime):
        owner.graph_capture_row_count = 0
    runner._dsa_row_mode_decode_graph_enabled = lambda: True
    runner._get_cumsum_and_arange = lambda counts, _: np.cumsum(counts)
    runner._pad_query_start_loc_for_fia = lambda *args: args[2]
    mode = CUDAGraphMode.NONE if phase == "warmup" else CUDAGraphMode.FULL
    runner._determine_batch_execution_and_padding = lambda **kwargs: (
        mode,
        SimpleNamespace(num_tokens=padded_reqs * 4, num_reqs=padded_reqs),
        False,
        torch.full((4,), padded_reqs * 4, dtype=torch.int32),
        None,
    )
    monkeypatch.setattr(runner_module, "copy_snapshot_to_gpu", lambda buffer: None)
    monkeypatch.setattr(runner_module, "using_paged_attention", lambda *args: False)

    def check_metadata(**kwargs):
        assert kwargs["num_reqs"] == num_reqs
        assert kwargs["num_reqs_padded"] == padded_reqs
        assert kwargs["for_dsa_idle_dummy"] == (phase == "idle")
        assert kwargs["num_scheduled_tokens_np"].tolist() == [4] * padded_reqs
        assert runner.query_start_loc.np[: padded_reqs + 1].tolist() == list(range(0, 4 * padded_reqs + 1, 4))
        expected_len = 0 if phase == "idle" else 10244
        assert runner.seq_lens[:padded_reqs].eq(expected_len).all()
        assert runner.seq_lens[padded_reqs:].eq(0).all()
        # Fail after preparing owners to also exercise exception cleanup.
        raise _MetadataChecked

    runner._build_attention_metadata = check_metadata
    with pytest.raises(_MetadataChecked):
        runner._dummy_run(
            num_tokens=num_reqs * 4,
            uniform_decode=True,
            is_graph_capturing=phase == "capture",
            force_attention=phase == "warmup",
        )

    state = runner.input_batch.dsa_cache_layout
    runtime = runner.dsa_offload_runtime
    pool = runner.dsa_resident_token_pool
    if phase == "idle":
        state.prepare_idle_dummy.assert_called_once_with(row_count=padded_reqs)
        runtime.prepare_idle_dummy.assert_called_once_with(row_count=padded_reqs)
        state.restore_after_idle_dummy.assert_called_once_with()
        runtime.restore_after_idle_dummy.assert_called_once_with()
        pool.prepare_graph_capture.assert_not_called()
        pool.restore_after_graph_capture.assert_not_called()
    else:
        state.prepare_graph_capture.assert_called_once_with(
            row_count=padded_reqs,
            target_budget_tokens=10240,
            resident_valid_tokens=10244,
        )
        runtime.prepare_graph_capture.assert_called_once_with(row_count=padded_reqs)
        pool.prepare_graph_capture.assert_called_once_with(row_count=padded_reqs, target_budget_tokens=10240)
        state.restore_after_graph_capture.assert_called_once_with()
        runtime.restore_after_graph_capture.assert_called_once_with()
        pool.restore_after_graph_capture.assert_called_once_with()
        state.prepare_idle_dummy.assert_not_called()


@pytest.mark.parametrize("is_graph_capturing", [False, True])
@pytest.mark.parametrize("fail_forward", [False, True])
def test_dsa_dummy_scope_always_restores_matching_owners(is_graph_capturing, fail_forward):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(dsa_cache_layout=MagicMock())
    runner.dsa_offload_runtime = MagicMock()
    runner.dsa_resident_token_pool = MagicMock()
    try:
        with runner._dsa_graph_dummy_state_scope(True, is_graph_capturing=is_graph_capturing):
            if fail_forward:
                raise _MetadataChecked
    except _MetadataChecked:
        pass
    method = "restore_after_graph_capture" if is_graph_capturing else "restore_after_idle_dummy"
    getattr(runner.input_batch.dsa_cache_layout, method).assert_called_once_with()
    getattr(runner.dsa_offload_runtime, method).assert_called_once_with()
    if is_graph_capturing:
        runner.dsa_resident_token_pool.restore_after_graph_capture.assert_called_once_with()
    else:
        runner.dsa_resident_token_pool.restore_after_graph_capture.assert_not_called()


@pytest.mark.parametrize("idle,execution_rows", [(True, 2), (False, 2), (True, 1)])
def test_idle_metadata_uses_decode_path_without_weakening_real_row_checks(monkeypatch, idle, execution_rows):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())] * 2)
    runner.dsa_kv_cache_group_ids = SimpleNamespace(indexer=0, resident_mla=1)
    runner.pcp_size = 1
    runner.use_cp = runner.use_async_spec_decode = runner.use_compress = False
    runner.dsa_offload_enabled = True
    runner._dsa_decode_forward = False  # May be stale from the last real prefill.
    runner.optimistic_seq_lens_cpu = runner.seq_lens = torch.zeros(8, dtype=torch.int32)
    query_ends = torch.arange(0, 36, 4, dtype=torch.int32)
    runner.query_start_loc = SimpleNamespace(cpu=query_ends, gpu=query_ends.clone())
    runner.actual_seq_lengths_q = [4, 8]
    runner.positions = torch.zeros(32, dtype=torch.int64)
    runner.attn_state = runner_module.AscendAttentionState.SpecDecoding
    runner.decode_token_per_req = 4
    state = SimpleNamespace(
        graph_capture_row_count=0,
        valid=False,
        row_count=0,
        row_modes=torch.zeros(8, dtype=torch.int32),
        resident_pool_indices=torch.full((8,), 8, dtype=torch.int32),
        sparse_budget_tokens=torch.zeros(8, dtype=torch.int32),
        candidate_lens=torch.full((8,), 2048, dtype=torch.int32),
    )
    block_table = SimpleNamespace(
        slot_mapping=SimpleNamespace(gpu=torch.zeros(32, dtype=torch.int64)),
        get_device_tensor=lambda: torch.zeros((8, 128), dtype=torch.int32),
    )
    runner.input_batch = SimpleNamespace(
        dsa_cache_layout=state,
        block_table=[block_table, block_table],
        num_computed_tokens_cpu_tensor=torch.zeros(8, dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.full((8,), 20480, dtype=torch.int32),
    )
    runner.dsa_resident_token_pool = SimpleNamespace(graph_capture_row_count=0)
    runner.dsa_offload_runtime = SimpleNamespace(
        graph_capture_row_count=0,
        active_num_reqs=0,
        execution_num_reqs=execution_rows,
        active_dram_block_table=SimpleNamespace(gpu=torch.zeros((8, 128), dtype=torch.int32)),
    )

    def check_common(**kwargs):
        assert kwargs["dsa_decode_forward"]
        assert not kwargs["is_prefilling"].any()
        assert kwargs["num_reqs"] == 2
        assert kwargs["seq_lens"].eq(0).all()
        assert kwargs["dsa_row_modes"].eq(0).all()
        assert kwargs["dsa_candidate_lens"].eq(2048).all()
        raise _MetadataChecked

    monkeypatch.setattr(runner_module, "AscendCommonAttentionMetadata", check_common)
    error = _MetadataChecked if idle and execution_rows == 2 else RuntimeError
    with pytest.raises(error):
        runner._build_attention_metadata(
            num_tokens=4,
            num_reqs=1,
            max_query_len=4,
            num_tokens_padded=8,
            num_reqs_padded=2,
            for_dsa_idle_dummy=idle,
        )


@pytest.mark.parametrize("capture_owner", ["state", "runtime", "pool"])
def test_idle_dummy_cannot_overwrite_an_active_capture(capture_owner):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    owners = {name: MagicMock(graph_capture_row_count=0) for name in ("state", "runtime", "pool")}
    owners[capture_owner].graph_capture_row_count = 2
    runner.input_batch = SimpleNamespace(dsa_cache_layout=owners["state"])
    runner.dsa_offload_runtime = owners["runtime"]
    runner.dsa_resident_token_pool = owners["pool"]

    with pytest.raises(RuntimeError, match="cannot overwrite capture"):
        runner._prepare_dsa_graph_dummy_state(row_count=2, is_graph_capturing=False)
    for owner in owners.values():
        owner.prepare_idle_dummy.assert_not_called()
        owner.restore_after_idle_dummy.assert_not_called()
        owner.prepare_graph_capture.assert_not_called()
        owner.restore_after_graph_capture.assert_not_called()
