# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.attention import sfa_v1
from vllm_ascend.core import kv_cache_interface
from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.dsa_offload.config import DSAOffloadConfig
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker import model_runner_v1


@pytest.fixture
def sfa_mtp_setup(monkeypatch):
    hf_config = SimpleNamespace(model_type="glm_moe_dsa", kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        get_num_layers=lambda _: 78,
        get_total_num_hidden_layers=lambda: 78,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(num_nextn_predict_layers=1)),
        ),
    )
    ascend_config = SimpleNamespace(
        dsa_offload_config=DSAOffloadConfig(),
        enable_sparse_sfa_c8=True,
        is_sparse_li_c8_layer=lambda _: True,
        enable_mlapo=True,
    )
    monkeypatch.setattr(sfa_v1, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(sfa_v1, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(sfa_v1, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(sfa_v1, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(sfa_v1, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    monkeypatch.setattr(kv_cache_interface, "get_ascend_device_type", lambda: AscendDeviceType.A5)
    for name in ("enable_dsa_cp", "enable_sp", "enable_dsa_cp_with_layer_shard", "enable_dsa_cp_with_o_proj_tp"):
        monkeypatch.setattr(sfa_v1, name, lambda: False)

    def make_impl(layer_index):
        return sfa_v1.AscendSFAImpl(
            num_heads=32,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            qk_head_dim=192,
            v_head_dim=128,
            rotary_emb=MagicMock(),
            q_b_proj=MagicMock(),
            kv_b_proj=MagicMock(),
            o_proj=MagicMock(),
            indexer=MagicMock(n_head=64, head_dim=128),
            layer_name=f"model.layers.{layer_index}.self_attn.attn" if layer_index is not None else None,
        )

    return config, ascend_config, make_impl


@pytest.mark.parametrize("offload_enabled", [False, True])
@pytest.mark.parametrize("layer_index, is_mtp", [(0, False), (77, False), (78, True), (79, False), (None, False)])
def test_only_mtp_disables_both_c8_caches(sfa_mtp_setup, offload_enabled, layer_index, is_mtp):
    _, ascend_config, make_impl = sfa_mtp_setup
    ascend_config.dsa_offload_config = DSAOffloadConfig(
        enabled=offload_enabled,
        mtp_num_speculative_tokens=3 if offload_enabled else 0,
        mtp_cache_layer_count=1 if offload_enabled else 0,
    )
    impl = make_impl(layer_index)
    assert impl.is_mtp_full_cache_layer is is_mtp
    assert impl.enable_sparse_sfa_c8 is not is_mtp
    assert impl.enable_sparse_li_c8 is not is_mtp
    assert impl.enable_mlapo


@pytest.mark.parametrize("method", [None, "eagle"])
def test_other_speculation_keeps_c8(sfa_mtp_setup, method):
    config, _, make_impl = sfa_mtp_setup
    if method is None:
        config.speculative_config = None
    else:
        config.speculative_config.method = method
    impl = make_impl(78)
    assert not impl.is_mtp_full_cache_layer
    assert impl.enable_sparse_sfa_c8
    assert impl.enable_sparse_li_c8


def test_mtp_uses_physical_layers_not_draft_token_count(sfa_mtp_setup):
    config, _, make_impl = sfa_mtp_setup
    config.speculative_config.draft_model_config.hf_config.num_nextn_predict_layers = 2
    assert make_impl(79).is_mtp_full_cache_layer
    assert not make_impl(80).is_mtp_full_cache_layer


def test_pipeline_parallel_target_layers_keep_c8(sfa_mtp_setup):
    config, _, make_impl = sfa_mtp_setup
    config.model_config.get_num_layers = lambda _: 39
    assert not make_impl(39).is_mtp_full_cache_layer
    assert make_impl(39).enable_sparse_sfa_c8
    assert make_impl(78).is_mtp_full_cache_layer


@pytest.mark.parametrize("offload_enabled", [False, True])
def test_mtp_cache_allocation_stays_bf16(sfa_mtp_setup, monkeypatch, offload_enabled):
    config, ascend_config, make_impl = sfa_mtp_setup
    ascend_config.dsa_offload_config = DSAOffloadConfig(
        enabled=offload_enabled,
        mtp_num_speculative_tokens=3 if offload_enabled else 0,
        mtp_cache_layer_count=1 if offload_enabled else 0,
    )
    impl = make_impl(78)
    attn = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(attn)
    attn.impl = impl
    attn.kv_lora_rank = impl.kv_lora_rank
    attn.qk_rope_head_dim = impl.qk_rope_head_dim
    layers = {impl.layer_name: attn}
    if not offload_enabled:
        target_impl = make_impl(77)
        target_attn = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(target_attn)
        target_attn.impl = target_impl
        target_attn.kv_lora_rank = target_impl.kv_lora_rank
        target_attn.qk_rope_head_dim = target_impl.qk_rope_head_dim
        layers[target_impl.layer_name] = target_attn
    monkeypatch.setattr(model_runner_v1, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(model_runner_v1, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(model_runner_v1, "enable_fa_quant", lambda *_: False)
    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner.vllm_config = config
    runner.model_config = config.model_config
    runner.parallel_config = config.parallel_config
    runner.ascend_config = ascend_config
    runner.dsa_offload_enabled = offload_enabled
    runner.use_sparse = True
    runner.use_compress = False
    runner.use_hybrid_blocks = False
    runner.is_kv_consumer = False
    runner.device = torch.device("cpu")
    runner.block_size = 128
    runner.kv_cache_dtype = torch.bfloat16
    runner.sfa_dcp_replicated_indexer_size = 1
    runner.runner_only_attn_layers = set()
    # Global target C8 settings must not override the MTP layer's cache spec.
    runner.c8_k_cache_dtype = torch.float8_e4m3fn
    runner.c8_k_scale_cache_dtype = torch.float32

    specs = runner.get_kv_cache_spec()
    spec = specs[impl.layer_name]
    assert spec.sparse_head_dim == (512, 64, 128)
    assert not spec.cache_sparse_sfa_c8
    assert not spec.cache_sparse_li_c8
    assert spec.dtype == torch.bfloat16
    # Native baseline groups can contain target C8 and MTP BF16 together;
    # allocation and reshape must retain each layer's individual spec.
    group_spec = UniformTypeKVCacheSpecs.from_specs(specs) if not offload_enabled else spec
    assert group_spec is not None
    cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=2 * layer_spec.page_size_bytes, shared_by=[name]) for name, layer_spec in specs.items()
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=list(specs), kv_cache_spec=group_spec)],
    )
    raw_caches = runner._allocate_kv_cache_tensors(cache_config)
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(kv_cache_spec=layer_spec, backend=sfa_v1.AscendSFABackend, layer_names=[name])
        for name, layer_spec in specs.items()
    ]
    all_caches = runner._reshape_kv_cache_tensors(cache_config, raw_caches)
    caches = all_caches[impl.layer_name]
    assert len(caches) == 3
    for cache, head_dim in zip(caches, (512, 64, 128)):
        assert cache.dtype == torch.bfloat16
        assert cache.shape == (2, 128, 1, head_dim)
    if not offload_enabled:
        target_caches = all_caches[target_impl.layer_name]
        assert [cache.dtype for cache in target_caches] == [torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32]

    impl.mlapo_is_quantized = False
    prolog = MagicMock(
        return_value=(
            torch.empty(1, impl.num_heads, impl.kv_lora_rank),
            torch.empty(1, impl.num_heads, impl.qk_rope_head_dim),
            None,
            torch.empty(1, impl.q_lora_rank),
            None,
        )
    )
    monkeypatch.setattr(BaseDeviceAdaptor, "_execute_sfa_mla_prolog_v3_op", prolog)
    A5DeviceAdaptor.sfa_preprocess_with_mlapo(
        impl,
        hidden_states=torch.zeros(1, 16, dtype=torch.bfloat16),
        kv_cache=caches,
        cos=torch.ones(1, impl.qk_rope_head_dim, dtype=torch.bfloat16),
        sin=torch.zeros(1, impl.qk_rope_head_dim, dtype=torch.bfloat16),
        slot_mapping=torch.tensor([0]),
        num_input_tokens=1,
    )
    kwargs = prolog.call_args.kwargs
    assert kwargs["token_x"].dtype == torch.bfloat16
    assert kwargs["kv_cache"] is caches[0]
    assert kwargs["kr_cache"] is caches[1]
    assert kwargs["weight_quant_mode"] == kwargs["kv_cache_quant_mode"] == 0
    assert kwargs["ckvkr_repo_mode"] == 0
