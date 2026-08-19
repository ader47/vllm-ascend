from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def _a5_qsfa_inputs():
    impl = SimpleNamespace(
        scale=0.125,
        sfa_qsfa_tile_size=128,
        qk_rope_head_dim=16,
    )
    return dict(
        sfa_impl=impl,
        ql_nope=torch.randn(3, 2, 32),
        q_pe=torch.randn(3, 2, 16),
        kv=torch.empty(4, 16, 1, 80, dtype=torch.int8),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        topk_indices=torch.zeros(3, 1, dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([3], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([3], dtype=torch.int32),
    )


def test_a5_kv_quant_sparse_attention_uses_native_binding():
    inputs = _a5_qsfa_inputs()
    expected = torch.randn(3, 2, 32)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
            return_value=expected,
        ) as mock_native_qsfa,
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_flash_attention",
            create=True,
            side_effect=AssertionError("A5 must use the torch_npu binding"),
        ),
    ):
        output = A5DeviceAdaptor.execute_kv_quant_sparse_flash_attention(
            **inputs,
        )

    assert output is expected
    call_kwargs = mock_native_qsfa.call_args.kwargs
    assert call_kwargs["key"] is inputs["kv"]
    assert call_kwargs["value"] is inputs["kv"]
    assert call_kwargs["query"].shape == (3, 2, 48)
    assert "return_softmax_lse" not in call_kwargs


def test_a5_kv_quant_sparse_attention_rejects_lse():
    inputs = _a5_qsfa_inputs()

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
        return_value=torch.randn(3, 2, 32),
    ):
        with pytest.raises(RuntimeError, match="cannot return softmax"):
            A5DeviceAdaptor.execute_kv_quant_sparse_flash_attention(
                **inputs,
                return_lse=True,
            )


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
