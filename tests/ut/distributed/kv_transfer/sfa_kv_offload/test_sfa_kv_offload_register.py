"""Unit tests for SFAKVOffloadWorker layer registration.

Covers:
- offload layer selection by tuple length (five/six-tuple in, others out)
- the mixed LIC8 / non-LIC8 guard: under offload, C8 must be uniform across
  sparse layers (the attention path gates the quant indexer on a GLOBAL flag),
  so coexisting five- and six-tuple layers must raise.

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

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (  # noqa: E402
    SFAKVOffloadWorker,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (  # noqa: E402
    ReqMeta,
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
    # Five- and six-tuple layers are offload candidates; single tensors and
    # other lengths are skipped. (5- and 6-tuple cannot coexist — see the
    # mixed guard test below — so exercise them in separate dicts.)
    for offload_len in (5, 6):
        w = _make_worker_without_init()
        kv_caches = {
            "layer.0": _tuple(offload_len),
            "layer.1": _tuple(offload_len),
            "indexer.layer.0": torch.zeros(1),  # single tensor, not an offload tuple
            "layer.2": _tuple(3),  # neither five- nor six-tuple
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


def test_register_all_six_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(6), "layer.1": _tuple(6)})
    assert w.num_offload_layers == 2


def test_register_rejects_mixed_five_and_six_tuple():
    """Mixed LIC8 / non-LIC8 layers under offload would route a non-C8 layer
    through the quant indexer (global flag) — must raise at registration."""
    w = _make_worker_without_init()
    kv_caches = {"layer.0": _tuple(5), "layer.1": _tuple(6)}
    with pytest.raises(ValueError, match="mixed LIC8 / non-LIC8"):
        w._register_offload_layers(kv_caches)


def _make_worker_for_save_tasks() -> SFAKVOffloadWorker:
    worker = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    worker.num_layers = 2
    worker.decode_width = 4
    worker.use_direct_sfa_host_offload = False
    worker.lru_managed_capacity = 4092
    worker.layer_save_tasks = [[], []]
    worker.k_caches_npu = [object(), object()]
    worker.v_caches_npu = [object(), object()]
    worker.topk_buffers_k = [object(), object()]
    worker.topk_buffers_v = [object(), object()]
    worker.k_caches_cpu = [object(), object()]
    worker.v_caches_cpu = [object(), object()]
    return worker


def test_decode_save_tasks_read_reserved_resident_slots():
    worker = _make_worker_for_save_tasks()
    request = ReqMeta(
        req_id="req-0",
        block_ids_npu=[11, 12],
        block_ids_cpu=[21, 22],
        write_start=127,
        write_count=4,
        is_prefill=False,
    )

    worker.process_layer_data(request, row_start=3)

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
        assert task.source_rows == [3, 4, 5, 6]
        assert task.source_slots == [4092, 4093, 4094, 4095]
        assert task.uses_resident


def test_prefill_save_tasks_read_normal_paged_cache():
    worker = _make_worker_for_save_tasks()
    request = ReqMeta(
        req_id="req-0",
        block_ids_npu=[11],
        block_ids_cpu=[21],
        write_start=0,
        write_count=37,
        is_prefill=True,
    )

    worker.process_layer_data(request, row_start=0)

    for layer_id, tasks in enumerate(worker.layer_save_tasks):
        task = tasks[0]
        assert task.cache_npu == (
            worker.k_caches_npu[layer_id],
            worker.v_caches_npu[layer_id],
        )
        assert task.token_start == 0
        assert task.token_count == 37
        assert task.source_rows is None
        assert task.source_slots is None
        assert not task.uses_resident


def test_direct_prefill_save_tasks_read_compact_stage_page_zero():
    worker = _make_worker_for_save_tasks()
    worker.use_direct_sfa_host_offload = True
    request = ReqMeta(
        req_id="req-0",
        block_ids_npu=[11],
        block_ids_cpu=[21],
        write_start=0,
        write_count=37,
        is_prefill=True,
    )

    worker.process_layer_data(request, row_start=0)

    for tasks in worker.layer_save_tasks:
        assert tasks[0].block_ids_npu == [0]
