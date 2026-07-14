"""Unit tests for SFAKVOffloadWorker layer registration.

Covers:
- offload layer selection from the true-hybrid main groups
- split main-cache tuple validation
- unrelated four-tuples are not mistaken for SFA main caches

The worker module JIT-builds a C++ extension and imports memfabric_hybrid at
module load time, neither of which is available in the UT sandbox; both are
stubbed before the import below.
"""

from types import SimpleNamespace
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


def _make_worker_without_init(
    main_layer_names: tuple[str, ...] = ("layer.0", "layer.1"),
) -> SFAKVOffloadWorker:
    """Bypass __init__ (heavy); set only the attrs _register_offload_layers reads."""
    w = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    w.num_target_layers = 0
    w.tp_rank = 0
    w.pending_save_layer_ids = set()
    w.submitted_save_layer_ids = set()
    w.main_group_ids = [0]
    w.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=list(main_layer_names)),
            SimpleNamespace(layer_names=["indexer.layer.0"]),
        ]
    )
    return w


def _tuple(n: int) -> tuple:
    return tuple(torch.zeros(1) for _ in range(n))


def test_register_selects_main_group_layers_and_skips_unrelated_four_tuple():
    w = _make_worker_without_init()
    kv_caches = {
        "layer.0": _tuple(4),
        "unrelated.layer": _tuple(4),
        "indexer.layer.0": _tuple(1),
        "layer.1": _tuple(4),
    }

    w._register_offload_layers(kv_caches)

    assert w.offload_layer_names == ["layer.0", "layer.1"]
    assert w.num_offload_layers == 2


def test_register_raises_when_main_layer_is_missing():
    w = _make_worker_without_init()
    with pytest.raises(ValueError, match="missing main MLA cache layers"):
        w._register_offload_layers({"layer.0": _tuple(4)})


def test_register_rejects_invalid_main_tuple_length():
    w = _make_worker_without_init()
    kv_caches = {"layer.0": _tuple(4), "layer.1": _tuple(5)}
    with pytest.raises(ValueError, match="expected split main tuple length 4"):
        w._register_offload_layers(kv_caches)
