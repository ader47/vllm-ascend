import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.utils.cpp_extension

from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.config import get_offload_tp_rank


def _make_config(extra_config: dict | None = None, tp_size: int = 4):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config=extra_config or {},
            kv_port=14579,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=tp_size,
        ),
    )


def test_get_offload_tp_rank_defaults_to_rank_zero():
    assert get_offload_tp_rank(_make_config()) == 0


def test_get_offload_tp_rank_reads_connector_extra_config():
    cfg = _make_config({"offload_tp_rank": "2"})
    assert get_offload_tp_rank(cfg) == 2


def test_get_offload_tp_rank_rejects_out_of_range_rank():
    cfg = _make_config({"offload_tp_rank": 4}, tp_size=4)
    with pytest.raises(ValueError, match="offload_tp_rank"):
        get_offload_tp_rank(cfg)


def test_consumer_non_active_rank_skips_cpu_pool(monkeypatch):
    fake_memfabric = types.ModuleType("memfabric_hybrid")
    fake_memfabric.offload = SimpleNamespace(
        empty=MagicMock(),
        initialize=MagicMock(),
        sparse_copy=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", fake_memfabric)
    monkeypatch.setattr(torch.utils.cpp_extension, "load", MagicMock(return_value=MagicMock()))

    module_name = "vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker"
    sys.modules.pop(module_name, None)
    worker_module = importlib.import_module(module_name)
    monkeypatch.setattr(worker_module, "get_tensor_model_parallel_rank", lambda: 1)

    worker = worker_module.SFAPDCpuOffloadConsumerWorker(
        _make_config({"offload_tp_rank": 0}),
        use_layerwise=True,
        kv_cache_config=None,
    )

    worker.register_kv_caches({})
    assert worker.sfa_worker is None
    fake_memfabric.offload.initialize.assert_not_called()

    metadata = SimpleNamespace(requests=[SimpleNamespace(req_id="req-0")])
    worker.start_load_kv(metadata)

    assert worker.get_num_cpu_blocks(["req-0", "req-1"]) == {
        "req-0": 0,
        "req-1": 0,
    }
    assert worker.get_finished()[1] == {"req-0"}


def test_consumer_non_active_rank_reports_each_req_once(monkeypatch):
    """A non-active rank must vote each req into finished_recving at most once.
    Re-reporting every step would satisfy vLLM's aggregator countdown
    (world_size votes) using only non-active ranks, releasing the req before
    the active rank finishes pulling its KV."""
    fake_memfabric = types.ModuleType("memfabric_hybrid")
    fake_memfabric.offload = SimpleNamespace(
        empty=MagicMock(),
        initialize=MagicMock(),
        sparse_copy=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", fake_memfabric)
    monkeypatch.setattr(torch.utils.cpp_extension, "load", MagicMock(return_value=MagicMock()))

    module_name = "vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker"
    sys.modules.pop(module_name, None)
    worker_module = importlib.import_module(module_name)
    monkeypatch.setattr(worker_module, "get_tensor_model_parallel_rank", lambda: 1)

    worker = worker_module.SFAPDCpuOffloadConsumerWorker(
        _make_config({"offload_tp_rank": 0}),
        use_layerwise=True,
        kv_cache_config=None,
    )
    worker.register_kv_caches({})

    md = SimpleNamespace(requests=[SimpleNamespace(req_id="req-0")])
    worker.start_load_kv(md)
    assert worker.get_finished()[1] == {"req-0"}  # first sight: voted once
    worker.start_load_kv(md)
    assert worker.get_finished()[1] == set()  # not re-reported next step
    md2 = SimpleNamespace(requests=[SimpleNamespace(req_id="req-1")])
    worker.start_load_kv(md2)
    assert worker.get_finished()[1] == {"req-1"}  # other req independent
    worker._cleanup_request_state({"req-0"})  # finished -> may vote again on retry
    worker.start_load_kv(md)
    assert worker.get_finished()[1] == {"req-0"}
