# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.dsa_offload.ops as dsa_ops
from vllm_ascend.dsa_offload.ops import (
    _normalize_lidu_weights_layout,
    quant_lightning_indexer_topk,
)


def test_packed_c8_op_check_loads_extension_before_schema_check(monkeypatch) -> None:
    load_calls = 0

    def _load_custom_op_library() -> bool:
        nonlocal load_calls
        load_calls += 1
        return True

    monkeypatch.setattr(dsa_ops, "load_custom_op_library", _load_custom_op_library)
    monkeypatch.setattr(dsa_ops, "_REQUIRED_A5_C8_OPS", ())

    dsa_ops.require_dsa_offload_ops(packed_c8=True)

    assert load_calls == 1


def test_packed_c8_op_check_reports_extension_load_failure(monkeypatch) -> None:
    monkeypatch.setattr(dsa_ops, "load_custom_op_library", lambda: False)

    with pytest.raises(RuntimeError, match="operator library failed to load"):
        dsa_ops.require_dsa_offload_ops(packed_c8=True)


def test_lidu_weights_normalizes_fused_projection_suffix_view() -> None:
    fused_projection = torch.arange(
        4 * 192,
        dtype=torch.bfloat16,
    ).view(4, 192)
    weights = fused_projection[:, 128:]

    assert weights.shape == (4, 64)
    assert weights.stride() == (192, 1)
    assert not weights.is_contiguous()

    normalized = _normalize_lidu_weights_layout(weights)

    assert normalized.shape == weights.shape
    assert normalized.stride() == (64, 1)
    assert normalized.is_contiguous()
    torch.testing.assert_close(normalized, weights)


def test_lidu_weights_keeps_already_contiguous_storage() -> None:
    weights = torch.empty((1, 64), dtype=torch.bfloat16)

    normalized = _normalize_lidu_weights_layout(weights)

    assert normalized.is_contiguous()
    assert normalized.data_ptr() == weights.data_ptr()


def test_quant_li_normalizes_native_tuple_output(monkeypatch) -> None:
    captured: dict[str, object] = {}
    raw_topk = (
        torch.arange(2 * 2048, dtype=torch.int32)
        .view(
            2048,
            2,
        )
        .t()
    )

    def _fake_quant_li(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, dtype=torch.int32), raw_topk

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_quant_lightning_indexer=_fake_quant_li),
    )
    weights_storage = torch.empty((2, 64), dtype=torch.bfloat16)
    weights = weights_storage[:, ::2]
    assert not weights.is_contiguous()

    result = quant_lightning_indexer_topk(
        query=torch.empty((2, 32, 128), dtype=torch.float8_e4m3fn),
        key=torch.empty((4, 128, 1, 128), dtype=torch.float8_e4m3fn),
        weights=weights,
        query_dequant_scale=torch.empty((2, 32), dtype=torch.float32),
        key_dequant_scale=torch.empty((4, 128, 1, 1), dtype=torch.float32),
        actual_seq_lengths_query=torch.tensor([1, 2], dtype=torch.int32),
        candidate_lens=torch.tensor([2048, 2048], dtype=torch.int32),
        block_table=torch.zeros((2, 16), dtype=torch.int32),
    )

    assert result.shape == (2, 1, 2048)
    assert result.dtype == torch.int32
    assert result.is_contiguous()
    assert captured["sparse_count"] == 2048
    assert captured["key_dequant_scale"].shape == (4, 128, 1)
    assert captured["weights"] is weights
