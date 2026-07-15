"""Unit tests for SFAKVOffloadWorker layer registration.

Covers the BF16 five-entry resident tuple used by the PD decode consumer.

The worker module JIT-builds a C++ extension and imports memfabric_hybrid at
module load time, neither of which is available in the UT sandbox; both are
stubbed before the import below.
"""

from unittest.mock import MagicMock

# Stub heavy module-level dependencies BEFORE importing the worker.
# 1. cpu_sparse_attn cpp extension JIT build (torch.utils.cpp_extension.load).
import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())  # noqa: E402

# 2. memfabric_hybrid.offload is not exported in the sandbox install.
import memfabric_hybrid  # noqa: E402

if not hasattr(memfabric_hybrid, "offload"):  # noqa: E402
    memfabric_hybrid.offload = MagicMock()  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (  # noqa: E402
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (  # noqa: E402
    SFAKVOffloadWorker,
)


def _make_worker_without_init() -> SFAKVOffloadWorker:
    """Bypass __init__ (heavy); set only the attrs _register_offload_layers reads."""
    w = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    w.num_target_layers = 0
    w.tp_rank = 0
    w.pending_save_layer_ids = set()
    w.submitted_save_layer_ids = set()
    return w


def _tuple(n: int) -> tuple:
    return tuple(torch.zeros(1) for _ in range(n))


def test_register_selects_offload_tuples_and_skips_others():
    w = _make_worker_without_init()
    kv_caches = {
        "layer.0": _tuple(5),
        "layer.1": _tuple(5),
        "indexer.layer.0": torch.zeros(1),
        "layer.2": _tuple(6),
    }
    w._register_offload_layers(kv_caches)
    assert w.offload_layer_names == ["layer.0", "layer.1"]
    assert w.num_offload_layers == 2


def test_register_raises_when_no_offload_layers():
    w = _make_worker_without_init()
    with pytest.raises(ValueError, match="did not find SFA KV cache layers"):
        w._register_offload_layers({"layer.0": _tuple(3)})


def test_register_all_five_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(5), "layer.1": _tuple(5)})
    assert w.num_offload_layers == 2


def _make_worker_for_save_tasks() -> SFAKVOffloadWorker:
    worker = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    worker.num_layers = 2
    worker.decode_width = 4
    worker.lru_managed_capacity = 4092
    worker.layer_save_tasks = [[], []]
    worker.topk_buffers_k = [object(), object()]
    worker.topk_buffers_v = [object(), object()]
    worker.k_caches_cpu = [object(), object()]
    worker.v_caches_cpu = [object(), object()]
    return worker


def test_decode_save_tasks_read_reserved_resident_slots():
    worker = _make_worker_for_save_tasks()
    request = ReqMeta(
        req_id="req-0",
        block_ids_cpu=[21, 22],
        write_start=127,
        write_count=4,
    )

    worker.process_layer_data(request, request_row=3)

    for layer_id, tasks in enumerate(worker.layer_save_tasks):
        assert len(tasks) == 1
        task = tasks[0]
        assert task.layer_id == layer_id
        assert task.cache_npu == (
            worker.topk_buffers_k[layer_id],
            worker.topk_buffers_v[layer_id],
        )
        assert task.token_start == 127
        assert task.token_count == 4
        assert task.source_rows == [3, 3, 3, 3]
        assert task.source_slots == [4092, 4093, 4094, 4095]
