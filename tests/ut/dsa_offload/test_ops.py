# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.dsa_offload.ops as dsa_ops
from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    _normalize_lidu_weights_layout,
    a5_lightning_indexer_decode_update_c8,
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
    monkeypatch.setattr(dsa_ops, "_REQUIRED_A5_NATIVE_OPS", ())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    dsa_ops.require_dsa_offload_ops(packed_c8=True)

    assert load_calls == 1


def test_packed_c8_op_check_reports_extension_load_failure(monkeypatch) -> None:
    monkeypatch.setattr(dsa_ops, "load_custom_op_library", lambda: False)

    with pytest.raises(RuntimeError, match="operator library failed to load"):
        dsa_ops.require_dsa_offload_ops(packed_c8=True)


def test_packed_c8_op_check_reports_missing_native_ops(monkeypatch) -> None:
    monkeypatch.setattr(dsa_ops, "load_custom_op_library", lambda: True)
    monkeypatch.setattr(dsa_ops, "_REQUIRED_A5_C8_OPS", ())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    with pytest.raises(RuntimeError, match="native operators are unavailable"):
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


@pytest.mark.parametrize(
    "key_scale_shape",
    [(4, 128, 1), (4, 128, 1, 1)],
)
def test_quant_li_normalizes_native_tuple_output(
    monkeypatch,
    key_scale_shape: tuple[int, ...],
) -> None:
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
        key_dequant_scale=torch.empty(key_scale_shape, dtype=torch.float32),
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


def test_a5_fused_lidu_preserves_weight_stride_and_squeezes_key_scale(
    monkeypatch,
) -> None:
    captured: dict[str, tuple[torch.Tensor, ...]] = {}

    def _fake_fused_op(*args: torch.Tensor) -> None:
        captured["args"] = args

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_dsa_a5_li_manage_nomtp_c8_out",
        _fake_fused_op,
        raising=False,
    )
    batch = 2
    weights_storage = torch.empty((batch, 160), dtype=torch.bfloat16)
    weights = weights_storage[:, 128:]
    assert weights.shape == (batch, 32)
    assert weights.stride() == (160, 1)
    assert not weights.is_contiguous()

    outputs = DSALightningIndexerOutputs(
        topk_index=torch.empty((batch, 1, 16384), dtype=torch.int32),
        topk_slots=torch.empty((batch, 1, 16384), dtype=torch.int32),
        miss_count=torch.empty((batch,), dtype=torch.int32),
        tail_info=torch.empty((batch, 2), dtype=torch.int32),
    )
    key_scale = torch.empty((8, 128, 1, 1), dtype=torch.float32)
    a5_lightning_indexer_decode_update_c8(
        index_weights=weights,
        query=torch.empty((batch, 32, 128), dtype=torch.float8_e4m3fn),
        query_dequant_scale=torch.empty((batch, 32), dtype=torch.float32),
        actual_seq_lengths_query=torch.tensor([1, 2], dtype=torch.int32),
        index_key_cache=torch.empty((8, 128, 1, 128), dtype=torch.float8_e4m3fn),
        index_key_dequant_scale=key_scale,
        index_block_table=torch.zeros((batch, 64), dtype=torch.int32),
        candidate_lens=torch.tensor([4096, 6144], dtype=torch.int32),
        final_seq_lengths_kv=torch.tensor([4096, 6145], dtype=torch.int32),
        row_modes=torch.tensor([1, 2], dtype=torch.int32),
        req_pool_entries=torch.tensor([-1, 0], dtype=torch.int32),
        cache_slots=torch.empty((2, 65537), dtype=torch.int32),
        attention_slots=torch.empty((batch, 1, 2176), dtype=torch.int32),
        resident_seq_lengths=torch.empty((batch,), dtype=torch.int32),
        outputs=outputs,
    )

    args = captured["args"]
    assert args[0] is weights
    assert args[0].stride() == (160, 1)
    assert args[5].shape == (8, 128, 1)
    assert args[5].data_ptr() == key_scale.data_ptr()
    assert args[14].data_ptr() == outputs.topk_index.data_ptr()
    assert args[15].data_ptr() == outputs.topk_slots.data_ptr()
    assert args[16].data_ptr() == outputs.miss_count.data_ptr()
