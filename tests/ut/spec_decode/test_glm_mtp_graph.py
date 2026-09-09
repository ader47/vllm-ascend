# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for GLM MTP graph inputs, DP selection and merged execution.

Can also run without vLLM/CANN: pytest --noconftest <this file>. We load the
actual buffer module and proposer methods, with only their runtime globals
substituted. This is not a substitute for NPU capture/replay accuracy testing.
"""

import ast
import copy
import importlib.util
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import IntEnum
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "vllm_ascend/spec_decode"


def load_rope(cos_cache=None, sin_cache=None, cos_buffer=None, sin_buffer=None):
    source = ROOT / "vllm_ascend/ops/rotary_embedding.py"
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_cos_and_sin_mla")
    if cos_cache is None:
        cos_cache = torch.arange(2048, dtype=torch.float32).reshape(-1, 1)
        sin_cache = -cos_cache
    ns = {
        "torch": torch,
        "_cos_cache": cos_cache,
        "_sin_cache": sin_cache,
        "_cos_mla": cos_buffer,
        "_sin_mla": sin_buffer,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    return ns["get_cos_and_sin_mla"]


def load_buffers():
    spec = importlib.util.spec_from_file_location("glm_mtp_graph_buffers", SPEC_DIR / "glm_mtp_graph.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules, {"vllm_ascend.ops.rotary_embedding": SimpleNamespace(get_cos_and_sin_mla=load_rope())}
    ):
        spec.loader.exec_module(module)
    return module.GLMMTPGraphBuffers


def prepare_metadata(p, common, count, batch, indices):
    metadata = p._glm_mtp_graph.prepare(common, count, batch, indices)
    p._glm_mtp_graph.preprocess(count, metadata)
    return metadata


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_kind", ["fresh", "cached", "explicit", "cast", "strided"])
def test_rope_gather_matches_advanced_index_and_preserves_output_storage(dtype, output_kind):
    # Include a strided cache (interleaved RoPE) and repeated/boundary positions.
    source = torch.arange(17 * 2 * 8, dtype=torch.float32).reshape(17, 2, 8).to(dtype)
    cos_cache, sin_cache = source[:, 0], source[:, 1]
    out_dtype = torch.float32 if output_kind == "cast" else dtype
    shape = (8, 1, 1, 8)
    cos_buf, sin_buf = torch.empty(shape, dtype=out_dtype), torch.empty(shape, dtype=out_dtype)
    if output_kind == "strided":
        cos_buf = torch.empty(8, 1, 2, 8, dtype=dtype)[:, :, :1]
        sin_buf = torch.empty_like(cos_buf)
    gather = load_rope(cos_cache, sin_cache, cos_buf, sin_buf)
    for index in ([16, 0, 7, 7], [1, 15, 0, 16]):
        positions = torch.tensor(index, dtype=torch.long)
        views = (cos_buf[:4], sin_buf[:4])
        if output_kind == "fresh":
            result = gather(positions)
        elif output_kind == "cached":
            result = gather(positions, use_cache=True)
        else:
            result = gather(positions, out=views)
        for value, cache, view in zip(result, (cos_cache, sin_cache), views):
            torch.testing.assert_close(value, cache[positions, None, None].to(value.dtype), rtol=0, atol=0)
            if output_kind != "fresh":
                assert value.data_ptr() == view.data_ptr()
    # The direct gather also handles zero-token prefixes without reshaping -1.
    assert gather(torch.empty(0, dtype=torch.long), use_cache=True)[0].shape == (0, 1, 1, 8)


class Mode(IntEnum):
    NONE = 0
    FULL = 2
    FULL_DECODE_ONLY = 3

    def has_full_cudagraphs(self):
        return self in (Mode.FULL, Mode.FULL_DECODE_ONLY)


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    uniform: bool = True


def load_methods(context):
    """Execute production method bodies; do not maintain a second algorithm."""
    source = SPEC_DIR / "llm_base_proposer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    names = {
        "_sync_glm_mtp_graph",
        "_run_glm_mtp_graph",
        "_run_merged_draft",
        "dummy_run",
        "_propose",
        "maybe_pad_and_reduce",
        "maybe_all_gather_and_unpad",
        "compute_draft_token_ids",
    }
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    methods += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "greedy_sample"]

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        context.attn_metadata = metadata
        context.cudagraph_runtime_mode = kwargs["aclgraph_runtime_mode"]
        context.batch_descriptor = kwargs["batch_descriptor"]
        tp = config.parallel_config.tensor_parallel_size
        padded_tokens = (kwargs["num_tokens"] + tp - 1) // tp * tp
        context.mc2_mask = torch.arange(padded_tokens) < kwargs["num_actual_tokens"]
        context.flash_comm_v1_enabled = False
        yield

    ns = {
        "torch": torch,
        "CUDAGraphMode": Mode,
        "get_forward_context": lambda: context,
        "set_ascend_forward_context": forward_context,
        "_EXTRA_CTX": context,
        "get_ascend_config": lambda: SimpleNamespace(enable_reduce_sample=getattr(context, "reduce_sample", False)),
        "get_tp_group": lambda: context.tp_group,
        "lmhead_tp_enable": lambda: False,
        "logger": Mock(),
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + methods, type_ignores=[]), str(source), "exec"), ns)
    return {name: ns[name] for name in names}


@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "tp8",
        "tp8_draft1",
        "tp8_dp2",
        "tp2",
        "tp2_draft2",
        "tp1_draft8",
        "eager",
        "mtp1",
        "async",
        "other_glm",
        "dynamic_eplb",
        "shared_expert_dp",
        "sp",
        "flashcomm2",
        "lmhead_tp",
    ],
)
@pytest.mark.parametrize("tp,dp", [(1, 1), (1, 16), (2, 4), (4, 2), (8, 1), (8, 2), (16, 2)])
def test_graph_gate_during_runner_initialization(case, tp, dp):
    # Runner has not set dynamic_eplb yet when it constructs the drafter.
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1, data_parallel_size=dp),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.FULL_DECODE_ONLY),
    )
    p = SimpleNamespace(
        use_cuda_graph=True,
        method="mtp",
        num_speculative_tokens=3,
        speculative_config=SimpleNamespace(draft_tensor_parallel_size=tp),
        pcp_size=1,
        dcp_size=1,
        parallel_drafting=False,
        use_async_scheduling=False,
        enable_shared_expert_dp=case == "shared_expert_dp",
        runner=SimpleNamespace(
            dsa_offload_enabled=True,
            enable_enpu=False,
            ascend_config=SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False)),
        ),
    )
    if case == "eager":
        p.use_cuda_graph = False
    elif case.startswith("tp8"):
        config.parallel_config.tensor_parallel_size = 8
        config.parallel_config.data_parallel_size = 2 if case == "tp8_dp2" else 1
        p.speculative_config.draft_tensor_parallel_size = 1 if case == "tp8_draft1" else 8
    elif case == "tp2":
        config.parallel_config.tensor_parallel_size = 2
        p.speculative_config.draft_tensor_parallel_size = 1
    elif case == "tp2_draft2":
        config.parallel_config.tensor_parallel_size = 2
        p.speculative_config.draft_tensor_parallel_size = 2
    elif case == "tp1_draft8":
        config.parallel_config.tensor_parallel_size = 1
        p.speculative_config.draft_tensor_parallel_size = 8
    elif case == "mtp1":
        p.num_speculative_tokens = 1
    elif case == "async":
        p.use_async_scheduling = True
    elif case == "other_glm":
        config.model_config.hf_text_config.model_type = "glm4_moe"
    elif case == "dynamic_eplb":
        p.runner.ascend_config.eplb_config.dynamic_eplb = True
    tree = ast.parse((SPEC_DIR / "llm_base_proposer.py").read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "use_glm_mtp_graph" for target in node.targets)
    )
    allowed = eval(
        compile(ast.Expression(assignment.value), "<graph gate>", "eval"),
        {
            "self": p,
            "vllm_config": config,
            "CUDAGraphMode": Mode,
            "get_ascend_device_type": lambda: 950,
            "AscendDeviceType": SimpleNamespace(A5=950),
            "enable_sp": lambda _: case == "sp",
            "flashcomm2_enable": lambda: case == "flashcomm2",
            "lmhead_tp_enable": lambda: case == "lmhead_tp",
        },
    )
    assert allowed == (case in ("supported", "tp8", "tp8_dp2", "tp2_draft2"))


class Builder:
    def __init__(self):
        self.calls = []

    def build(self, _, common, model):
        return self.build_for_drafting(common, 0)

    def build_for_drafting(self, common, step):
        self.calls.append((step, common))
        return SimpleNamespace(
            slot_mapping=common.slot_mapping,
            seq_lens=common.seq_lens,
            seq_lens_cpu=common.seq_lens_cpu,
            cum_query_lens=common.query_start_loc[1:],
            block_table=common.block_table_tensor,
            cos=common.positions.float().reshape(-1, 1, 1, 1),
            sin=-common.positions.float().reshape(-1, 1, 1, 1),
            num_actual_tokens=common.num_actual_tokens,
            num_input_tokens=common.num_input_tokens,
            dcp_context=None,
            dsa_cp_context=None,
        )


def proposer(context=None, tp_size=1):
    builder = Builder()
    table = torch.arange(8 * 8, dtype=torch.int32).reshape(8, 8) + 10
    p = SimpleNamespace(
        decode_threshold=4,
        num_speculative_tokens=3,
        max_model_len=1024,
        block_size=128,
        device="cpu",
        input_ids=torch.zeros(32, dtype=torch.int32),
        positions=torch.zeros(32, dtype=torch.int32),
        hidden_states=torch.zeros(32, 4),
        token_indices_to_sample=torch.zeros(32, dtype=torch.int32),
        arange=torch.arange(33, dtype=torch.int32),
        slot_mapping_group=[torch.zeros(32, dtype=torch.int32) for _ in range(3)],
        seq_lens_group=[torch.zeros(32, dtype=torch.int32) for _ in range(3)],
        query_start_loc_group=[torch.zeros(33, dtype=torch.int32) for _ in range(3)],
        kv_cache_gid=1,
        draft_attn_groups=[SimpleNamespace(get_metadata_builder=lambda: builder)],
        attn_layer_names=["mtp"],
        runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                block_table=[None, SimpleNamespace(get_device_tensor=lambda: table)], lora_id_to_lora_request={}
            ),
            get_model=lambda: None,
            pcp_manager=None,
        ),
        method="mtp",
        _share_mtp_indices=False,
        parallel_drafting=False,
        pcp_size=1,
        dcp_size=1,
        pass_hidden_states_to_model=True,
        supports_mm_inputs=False,
        uses_mrope=False,
        use_cuda_graph=True,
        use_glm_mtp_graph=True,
        enable_shared_expert_dp=False,
        is_multimodal_model=False,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=1024, use_mla=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        ),
        use_compress=False,
        model_returns_tuple=lambda: True,
        maybe_pad_and_reduce=lambda hidden, positions: (hidden, positions),
        maybe_all_gather_and_unpad=lambda last, positions, hidden: (last, positions, hidden),
    )
    p._get_positions = lambda count: p.positions[:count]
    p._set_positions = lambda count, value: p.positions[:count].copy_(value)
    p._glm_mtp_graph = load_buffers()(p, SimpleNamespace, "SpecDecoding")
    if context is not None:
        for name, method in load_methods(context).items():
            setattr(p, name, MethodType(method, p))
    return p


def inputs(p, batch_size=2, rejected=0, offset=0):
    count = 4 * batch_size
    positions = torch.arange(124 + offset, 124 + offset + count, dtype=torch.int32)
    p.positions[:count].copy_(positions)
    p.input_ids[:count].copy_(torch.arange(count) + 5 + offset)
    p.hidden_states[:count].fill_(2 + offset)
    indices = torch.arange(batch_size, dtype=torch.int32) * 4 + 3 - rejected
    lengths = positions[indices] + 1
    table = p.runner.input_batch.block_table[1].get_device_tensor().clone()[:batch_size]
    table += offset
    return SimpleNamespace(
        num_actual_tokens=count,
        num_reqs=batch_size,
        batch_size=lambda: batch_size,
        max_query_len=4,
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32) * 4,
        query_start_loc_cpu=torch.arange(batch_size + 1, dtype=torch.int32) * 4,
        block_table_tensor=table,
        slot_mapping=torch.arange(count, dtype=torch.int32) + offset,
        seq_lens=lengths,
        _seq_lens_cpu=lengths.clone(),
        seq_lens_cpu=lengths.clone(),
    ), indices


def tensors(metadata):
    attn = metadata["mtp"]
    return [getattr(attn, name) for name in ("slot_mapping", "seq_lens", "cum_query_lens", "block_table", "cos", "sin")]


def test_three_steps_have_distinct_query_layout_and_rope():
    p = proposer()
    common, indices = inputs(p)
    result = prepare_metadata(p, common, 8, 2, indices)
    assert [m["mtp"].cum_query_lens.tolist() for m in result] == [[4, 8], list(range(1, 9)), list(range(1, 9))]
    assert [m["mtp"].num_actual_tokens for m in result] == [8, 8, 8]
    assert result[1]["mtp"].cos[:2].flatten().tolist() == [128, 132]
    assert result[2]["mtp"].cos[:2].flatten().tolist() == [129, 133]
    assert result[1]["mtp"].slot_mapping[0] == common.block_table_tensor[0, 1] * 128
    for step in (1, 2):
        assert result[step]["mtp"].slot_mapping[2:].eq(-1).all()
        assert result[step]["mtp"].seq_lens[2:].eq(0).all()
    assert len({m["mtp"].cos.data_ptr() for m in result}) == 3


def test_capture_buffers_follow_request_replacement_shrinking_and_idle():
    p = proposer()
    captured = prepare_metadata(p, None, 16, 0, None)
    addresses = [[t.data_ptr() for t in tensors(m)] for m in captured]
    for batch_size, offset in ((4, 0), (1, 17), (0, 0), (2, 31)):
        common, indices = inputs(p, batch_size, offset=offset)
        before = common.seq_lens.clone()
        result = prepare_metadata(p, common if batch_size else None, 16, batch_size, indices)
        assert addresses == [[t.data_ptr() for t in tensors(m)] for m in result]
        for step in range(3):
            for old, new in zip(tensors(captured[step]), tensors(result[step])):
                torch.testing.assert_close(old, new)
            active_tokens = batch_size * 4 if step == 0 else batch_size
            assert captured[step]["mtp"].slot_mapping[active_tokens:].eq(-1).all()
            assert captured[step]["mtp"].seq_lens[batch_size:].eq(0).all()
        torch.testing.assert_close(common.seq_lens, before)
        assert captured[0]["mtp"].block_table[batch_size:].eq(0).all()


@pytest.mark.parametrize("rejected", [0, 1, 2, 3])
def test_accepted_position_drives_next_slots_not_fixed_fourth_token(rejected):
    p = proposer()
    common, indices = inputs(p, rejected=rejected)
    result = prepare_metadata(p, common, 8, 2, indices)
    for step in (1, 2):
        position = 127 - rejected + step
        expected = common.block_table_tensor[0, position // 128] * 128 + position % 128
        assert result[step]["mtp"].slot_mapping[0] == expected
        assert result[step]["mtp"].cos[0].item() == position


def test_max_length_masks_writes_and_preserves_cpu_gpu_lengths():
    p = proposer()
    common, indices = inputs(p, batch_size=1, offset=896)
    result = prepare_metadata(p, common, 4, 1, indices)
    for step in (1, 2):
        assert result[step]["mtp"].slot_mapping.eq(-1).all()
        assert result[step]["mtp"].cos[0].eq(0).all()
        assert result[step]["mtp"].seq_lens[0] == step
        torch.testing.assert_close(result[step]["mtp"].seq_lens, result[step]["mtp"].seq_lens_cpu)


def test_replay_staging_does_not_rebuild_metadata_or_gather_rope():
    p = proposer()
    buffers = p._glm_mtp_graph
    builder = p.draft_attn_groups[0].get_metadata_builder()
    metadata = prepare_metadata(p, None, 16, 0, None)
    assert len(builder.calls) == 3
    pointers = [[t.data_ptr() for t in tensors(m)] for m in metadata]
    common, indices = inputs(p, batch_size=2, rejected=2)
    # These tensors remain untouched until the captured prelude runs.
    later_slots = metadata[1]["mtp"].slot_mapping.clone()
    rope = metadata[0]["mtp"].cos.clone()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        staged = buffers.prepare(common, 16, 2, indices)
    names = {event.key for event in prof.key_averages()}
    assert not names.intersection({"aten::index", "aten::index_select", "aten::gather", "aten::clone"})
    assert staged is metadata and len(builder.calls) == 3
    assert pointers == [[t.data_ptr() for t in tensors(m)] for m in staged]
    torch.testing.assert_close(staged[1]["mtp"].slot_mapping, later_slots)
    torch.testing.assert_close(staged[0]["mtp"].cos, rope)
    buffers.preprocess(16, staged)
    assert staged[1]["mtp"].slot_mapping[:2].ne(-1).all()
    assert staged[0]["mtp"].cos[:8].ne(0).all()


@pytest.mark.parametrize("count", [4, 16, 32])
def test_captured_preprocessing_matches_scalar_reference_and_preserves_sources(count):
    p = proposer()
    for batch, rejection, offset in ((count // 4, 0, 0), (1, 3, 20), (0, 0, 0), (1, 0, 896)):
        common, indices = inputs(p, batch, rejection, offset)
        positions = p.positions[:count].clone()
        source_slots, source_lens, source_table = (
            common.slot_mapping.clone(),
            common.seq_lens.clone(),
            common.block_table_tensor.clone(),
        )
        result = prepare_metadata(p, common if batch else None, count, batch, indices)
        for step, per_layer in enumerate(result):
            attn = per_layer["mtp"]
            expected_lens = torch.zeros_like(attn.seq_lens)
            expected_slots = torch.full_like(attn.slot_mapping, -1)
            for row in range(batch):
                length = int(source_lens[row])
                for _ in range(step):
                    length = 1 if length + 1 > p.max_model_len else length + 1
                expected_lens[row] = length
                if step == 0:
                    expected_slots[row * 4 : (row + 1) * 4] = source_slots[row * 4 : (row + 1) * 4]
                    torch.testing.assert_close(
                        attn.cos[row * 4 : (row + 1) * 4].flatten(), positions[row * 4 : (row + 1) * 4].float()
                    )
                else:
                    position = int(positions[indices[row]]) + step
                    if position < p.max_model_len:
                        expected_slots[row] = (
                            source_table[row, position // p.block_size] * p.block_size + position % p.block_size
                        )
                    assert attn.cos[row].item() == (position if position < p.max_model_len else 0)
            torch.testing.assert_close(attn.seq_lens, expected_lens)
            torch.testing.assert_close(attn.seq_lens_cpu, expected_lens)
            torch.testing.assert_close(attn.slot_mapping, expected_slots)
        torch.testing.assert_close(common.slot_mapping, source_slots)
        torch.testing.assert_close(common.seq_lens, source_lens)
        torch.testing.assert_close(common.block_table_tensor, source_table)


def test_c8_group_outputs_and_graph_keys_do_not_alias_builder_scratch(monkeypatch):
    p = proposer()
    scratch = [torch.zeros(32, dtype=torch.int32) for _ in range(3)]

    class GroupingBuilder(Builder):
        def build_for_drafting(self, common, step):
            attn = super().build_for_drafting(common, step)
            attn.group_len, attn.group_key_idx, attn.group_key_cache_idx = [
                t[: common.num_input_tokens] for t in scratch
            ]
            attn.block_size = p.block_size
            return attn

    builder = GroupingBuilder()
    p.draft_attn_groups[0].get_metadata_builder = lambda: builder

    def store(slots, lengths, key_idx, cache_idx, block_size):
        lengths.copy_((slots >= 0).int())
        key_idx.copy_(slots % block_size)
        cache_idx.copy_(slots // block_size)

    # CPU stand-in verifies the real call sites/order and independent outputs,
    # not the CANN grouping kernel implementation.
    store_mock = Mock(side_effect=store)
    monkeypatch.setattr(torch.ops._C_ascend, "store_kv_block_metadata", store_mock, raising=False)
    cached = {}
    for count in (4, 16, 4, 16):
        common, indices = inputs(p, 1, rejected=2, offset=count)
        metadata = prepare_metadata(p, common, count, 1, indices)
        if count in cached:
            assert cached[count] is metadata
        cached[count] = metadata
        for per_layer in metadata:
            attn = per_layer["mtp"]
            assert attn.group_len.data_ptr() not in [t.data_ptr() for t in scratch]
            torch.testing.assert_close(attn.group_len, (attn.slot_mapping >= 0).int())
            torch.testing.assert_close(attn.group_key_idx, attn.slot_mapping % p.block_size)
            torch.testing.assert_close(attn.group_key_cache_idx, attn.slot_mapping // p.block_size)
        for tensor in scratch + p.query_start_loc_group + p.seq_lens_group + p.slot_mapping_group:
            tensor.fill_(999)  # emulate target/eager scratch reuse
    assert len(builder.calls) == 6  # once per step per distinct graph key
    assert store_mock.call_count == 12
    for name in ("seq_lens", "slot_mapping", "cos", "sin", "group_len", "cum_query_lens"):
        addresses = [getattr(m["mtp"], name).data_ptr() for metadata in cached.values() for m in metadata]
        assert len(set(addresses)) == 6


@pytest.mark.parametrize("c8_grouping", [False, True])
def test_cached_metadata_with_production_sfa_builder(monkeypatch, c8_grouping):
    # Exercise the production builder's field wiring as well as the graph
    # helper. Only runtime/CANN dependencies are substituted on CPU.
    p = proposer()
    source = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAMetadataBuilder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build")
    ns = {
        "torch": torch,
        "get_ascend_config": lambda: SimpleNamespace(c8_enable_reshape_optim=c8_grouping),
        "get_cos_and_sin_mla": load_rope(cos_buffer=torch.zeros(32, 1, 1, 1), sin_buffer=torch.zeros(32, 1, 1, 1)),
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), ns)
    builder = SimpleNamespace(
        kernel_block_size=128,
        enable_dsa_cp=False,
        model_config=SimpleNamespace(get_head_size=lambda: 64),
        attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *args: None),
        metadata_cls=lambda **kwargs: SimpleNamespace(dcp_context=None, **kwargs),
    )
    for name in ("group_len", "group_key_idx", "group_key_cache_idx"):
        setattr(builder, name, torch.zeros(32, dtype=torch.int32))
        setattr(builder, f"spec_{name}", [torch.zeros(32, dtype=torch.int32) for _ in range(2)])
    build = MethodType(ns["_build"], builder)
    builder.build = Mock(side_effect=lambda _, common, model: build(common))
    builder.build_for_drafting = Mock(side_effect=lambda common, step: build(common, draft_index=step))
    monkeypatch.setattr(torch.ops._C_ascend, "store_kv_block_metadata", Mock(), raising=False)
    p.draft_attn_groups[0].get_metadata_builder = lambda: builder
    defaults = dict.fromkeys(
        (
            "dsa_indexer_block_table",
            "dsa_indexer_slot_mapping",
            "dsa_row_modes",
            "dsa_resident_pool_indices",
            "dsa_sparse_budget_tokens",
            "dsa_candidate_lens",
            "dsa_dram_block_table",
            "dsa_decode_forward",
        )
    )
    p._glm_mtp_graph.metadata_cls = lambda **kwargs: SimpleNamespace(causal=True, **defaults, **kwargs)
    for batch in (0, 2, 1, 0):
        common, indices = inputs(p, batch, rejected=1)
        result = prepare_metadata(p, common if batch else None, 8, batch, indices)
        for step, per_layer in enumerate(result):
            attn = per_layer["mtp"]
            assert attn.block_size == 128 and attn.head_dim == 64
            assert attn.dsa_decode_forward is None and attn.dsa_row_modes is None
            assert (attn.group_len is not None) == c8_grouping
            torch.testing.assert_close(attn.seq_lens, attn.seq_lens_cpu)
            assert attn.slot_mapping[(batch * 4 if step == 0 else batch) :].eq(-1).all()
    builder.build.assert_called_once()
    assert builder.build_for_drafting.call_count == 2


@pytest.mark.parametrize(
    "graph,padded,aux", [(True, True, False), (True, True, True), (True, False, False), (False, True, False)]
)
def test_runner_contiguous_draft_inputs_use_views_only_in_padded_glm_path(graph, padded, aux):
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    # Execute the actual input-selection block rather than duplicating it.
    parent = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.If) and "use_glm_mtp_graph" in ast.unparse(child.test) for child in node.orelse)
    )
    ids, positions, hidden = torch.arange(16), torch.arange(16) + 9, torch.randn(16, 4)
    indices = torch.arange(4) if padded else torch.tensor([3, 0, 2, 1])
    p = SimpleNamespace(
        drafter=SimpleNamespace(use_glm_mtp_graph=graph),
        vllm_config=SimpleNamespace(speculative_config=SimpleNamespace(disable_padded_drafter_batch=not padded)),
        input_ids=SimpleNamespace(gpu=ids),
        use_aux_hidden_state_outputs=aux,
        _get_positions=lambda index: positions[:index] if isinstance(index, int) else positions[index],
    )
    ns = {
        "self": p,
        "torch": torch,
        "token_indices": indices,
        "common_attn_metadata": SimpleNamespace(num_actual_tokens=4),
        "hidden_states": hidden,
        "aux_hidden_states": [hidden, hidden + 1],
    }
    exec(compile(ast.Module(body=parent.orelse, type_ignores=[]), str(source), "exec"), ns)
    torch.testing.assert_close(ns["target_token_ids"], ids[indices])
    torch.testing.assert_close(ns["target_positions"], positions[indices])
    expected = torch.cat([hidden[indices], (hidden + 1)[indices]], dim=-1) if aux else hidden[indices]
    torch.testing.assert_close(ns["target_hidden_states"], expected)
    if graph and padded:
        assert ns["target_token_ids"].data_ptr() == ids.data_ptr()
        assert ns["target_positions"].data_ptr() == positions.data_ptr()
        if not aux:
            assert ns["target_hidden_states"].data_ptr() == hidden.data_ptr()


@pytest.mark.parametrize(
    "eligible,local_tokens,peer_mode,peer_tokens,expected",
    [
        (True, 4, Mode.FULL, 16, Mode.FULL),
        (True, 8, Mode.NONE, 16, Mode.NONE),
        (False, 4, Mode.FULL, 16, Mode.NONE),
        (True, 3, Mode.FULL, 16, Mode.NONE),
    ],
)
def test_dp_graph_decision_uses_one_collective_and_global_mode(
    eligible, local_tokens, peer_mode, peer_tokens, expected
):
    p = proposer(SimpleNamespace())

    def dispatch(num_tokens, **kwargs):
        return Mode.FULL, SimpleNamespace(num_tokens=num_tokens)

    def sync(num_tokens, *, is_draft_model, cudagraph_mode):
        assert is_draft_model
        return max(num_tokens, peer_tokens), None, min(cudagraph_mode, peer_mode)

    p.runner.cudagraph_dispatcher = SimpleNamespace(dispatch=Mock(side_effect=dispatch))
    p.runner._sync_metadata_across_dp = Mock(side_effect=sync)
    count, _, mode, descriptor = p._sync_glm_mtp_graph(local_tokens, eligible=eligible)
    assert count == peer_tokens and mode == expected
    assert (descriptor is not None) == (expected == Mode.FULL)
    p.runner._sync_metadata_across_dp.assert_called_once()


class ToyMTP:
    """Tuple-output drafter with deterministic logits; no mocked tensor math."""

    def __init__(self, context, vocab_size=17):
        self.context = context
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, input_ids, positions, hidden_states, **kwargs):
        attn = self.context.attn_metadata["mtp"]
        self.calls.append((input_ids.shape[0], self.context.mc2_mask.clone()))
        value = input_ids.float() + hidden_states[:, 0] + attn.cos.flatten()
        # Changes of the block table must affect replay, not just fresh capture.
        value = value + attn.block_table[0, 0]
        last = value.unsqueeze(1).repeat(1, 4)
        return last, last + 2  # distinct logit and recycled hidden states

    def compute_logits(self, hidden):
        token = hidden[:, 0].long() % self.vocab_size
        return torch.zeros(hidden.shape[0], self.vocab_size).scatter_(1, token[:, None], 1)


class FrozenInputsGraph:
    """Freeze capture-time scalar arguments/tensor views, as replay requires.

    CPU execution reruns the op body using those old views; it intentionally
    ignores replacement metadata and kwargs. No claim of testing ACL itself.
    """

    def __init__(self, p, context):
        self.p, self.context = p, context
        self.concrete_aclgraph_entries = {}

    def __call__(self, **kwargs):
        key = self.context.batch_descriptor
        if key not in self.concrete_aclgraph_entries:
            self.concrete_aclgraph_entries[key] = SimpleNamespace(aclgraph=True, kwargs=kwargs)
        captured = self.concrete_aclgraph_entries[key].kwargs
        self.context.attn_metadata = captured["multi_steps_attn_metadata"][0]
        return self.p._run_merged_draft(**captured)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("graph_tokens", [4, 8, 16, 32])
def test_merged_replay_matches_eager_through_batch_and_acceptance_changes(tp_size, graph_tokens):
    ctx = SimpleNamespace()
    graph = proposer(ctx, tp_size=tp_size)
    graph.model = ToyMTP(ctx)
    graph._runnable = FrozenInputsGraph(graph, ctx)
    # Capture on an idle rank, then change request count, accepted positions,
    # sequence lengths and block assignments without changing the graph key.
    graph._run_glm_mtp_graph(None, graph_tokens, 0, None, None, graph_tokens)
    for batch, rejected, offset in ((4, 0, 0), (1, 2, 19), (0, 0, 0), (2, 3, 27), (4, 1, 0)):
        batch = min(batch, graph_tokens // 4)
        common, indices = inputs(graph, batch, rejected, offset)
        result = graph._run_glm_mtp_graph(common if batch else None, graph_tokens, batch, indices, None, graph_tokens)
        assert result.shape == (batch, 3)
        assert len(graph._runnable.concrete_aclgraph_entries) == 1
        calls = graph.model.calls[-3:]
        assert [n for n, _ in calls] == [graph_tokens] * 3
        assert [int(mask.sum()) for _, mask in calls] == [batch * 4, batch, batch]
        for _, mask in calls:
            assert mask.numel() == (graph_tokens + tp_size - 1) // tp_size * tp_size
            assert not mask[graph_tokens:].any()
        if batch == 0:
            continue

        ref_ctx = SimpleNamespace(
            cudagraph_runtime_mode=Mode.NONE,
            mc2_mask=torch.ones(graph_tokens, dtype=torch.bool),
            flash_comm_v1_enabled=False,
        )
        ref = proposer(ref_ctx)
        ref.model = ToyMTP(ref_ctx)
        ref_common, ref_indices = inputs(ref, batch, rejected, offset)
        metadata = prepare_metadata(ref, ref_common if batch else None, graph_tokens, batch, ref_indices)
        ref_ctx.attn_metadata = metadata[0]
        expected = ref._run_merged_draft(
            graph_tokens, batch, ref.token_indices_to_sample[:batch], ref.positions, None, metadata, batch * 4, False
        )
        torch.testing.assert_close(result, expected)


@torch.inference_mode()
def test_dummy_and_busy_proposer_reuse_the_same_graph_and_collective_sequence():
    ctx = SimpleNamespace()
    p = proposer(ctx)
    p.model = ToyMTP(ctx)
    p._runnable = FrozenInputsGraph(p, ctx)
    p.runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda num_tokens, **kwargs: (Mode.FULL, Descriptor(num_tokens))
    )
    p.runner._sync_metadata_across_dp = Mock(side_effect=lambda count, **kwargs: (count, None, Mode.FULL))
    p.dummy_run(8, aclgraph_runtime_mode=Mode.FULL, num_reqs=2, batch_descriptor=Descriptor(8))
    common, indices = inputs(p, batch_size=2, rejected=2)
    p.set_inputs_first_pass = Mock(return_value=(8, indices, common, None))
    output = p._propose(
        p.input_ids[:8],
        p.positions[:8],
        p.hidden_states[:8],
        torch.ones(2),
        indices,
        common,
        Descriptor(8),
        None,
    )
    assert output.shape == (2, 3)
    assert len(p._runnable.concrete_aclgraph_entries) == 1
    assert p.runner._sync_metadata_across_dp.call_count == 2  # exactly one per invocation
    assert len(p.model.calls) == 6  # three MTP iterations per invocation


@pytest.mark.parametrize("prefill", [False, True])
def test_busy_proposer_keeps_eager_when_local_or_peer_requires_it(prefill):
    p = proposer(SimpleNamespace())
    common, indices = inputs(p, batch_size=2)
    p.set_inputs_first_pass = Mock(return_value=(8, indices, common, None))
    p.runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda num_tokens, **kwargs: (Mode.FULL, Descriptor(num_tokens))
    )
    p.runner._sync_metadata_across_dp = Mock(return_value=(16, None, Mode.NONE))
    p._runnable = Mock()

    class ReachedEagerBuilder(Exception):
        pass

    p.draft_attn_groups[0].get_metadata_builder().build = Mock(side_effect=ReachedEagerBuilder)
    with pytest.raises(ReachedEagerBuilder):
        p._propose(
            p.input_ids[:8],
            p.positions[:8],
            p.hidden_states[:8],
            torch.ones(2),
            indices,
            common,
            Descriptor(8),
            None,
            num_prefill_reqs=int(prefill),
        )
    p._runnable.assert_not_called()
    p.runner._sync_metadata_across_dp.assert_called_once()


def test_non_sp_mtp_preserves_distinct_logits_and_recycled_hidden_states():
    p = proposer(SimpleNamespace(flash_comm_v1_enabled=False), tp_size=8)
    pre_norm, post_norm, positions = torch.randn(4, 4), torch.randn(4, 4), torch.arange(4)
    hidden, pos = p.maybe_pad_and_reduce(post_norm, positions)
    assert hidden is post_norm and pos is positions
    logits_hidden, pos, recycled = p.maybe_all_gather_and_unpad(pre_norm, positions, post_norm)
    assert logits_hidden is pre_norm and recycled is post_norm and pos is positions


class GlooTPGroup:
    """Real CPU collectives; never presented as HCCL graph validation."""

    def __init__(self, rank, size, process_group=None):
        self.rank_in_group, self.world_size = rank, size
        self.process_group = process_group
        self.gather_shapes = []

    def all_gather(self, tensor, dim=-1):
        self.gather_shapes.append(tuple(tensor.shape))
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(parts, tensor.contiguous(), group=self.process_group)
        return torch.cat(parts, dim=dim)


class ShardedToyMTP(ToyMTP):
    def __init__(self, context):
        super().__init__(context, vocab_size=64)
        self.group = context.tp_group
        # Use the actual Ascend normal (non-finegrained-lmhead-TP) logits path.
        source = ROOT / "vllm_ascend/ops/vocab_parallel_embedding.py"
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_logits_normal")
        ns = {"get_ascend_config": lambda: SimpleNamespace(enable_reduce_sample=context.reduce_sample)}
        future = ast.parse("from __future__ import annotations").body
        exec(compile(ast.Module(body=future + [method], type_ignores=[]), str(source), "exec"), ns)
        self._get_logits_normal = MethodType(ns["_get_logits_normal"], self)
        self.org_vocab_size = self.vocab_size
        shard_size = self.vocab_size // self.group.world_size
        self.head = SimpleNamespace(
            num_org_embeddings_per_partition=shard_size,
            quant_method=SimpleNamespace(apply=self.local_logits),
        )
        self._gather_logits = self.group.all_gather

    def __call__(self, *args, **kwargs):
        pre_norm, _ = super().__call__(*args, **kwargs)
        pre_norm /= self.group.world_size
        torch.distributed.all_reduce(pre_norm, group=self.group.process_group)
        return pre_norm, pre_norm + 2

    def local_logits(self, head, hidden, bias=None):
        full_logits = super().compute_logits(hidden)
        start = self.group.rank_in_group * head.num_org_embeddings_per_partition
        return full_logits[:, start : start + head.num_org_embeddings_per_partition].contiguous()

    def compute_logits(self, hidden):
        return self._get_logits_normal(hidden, self.head, None)


def _tp_dp_collective_worker(rank, rendezvous, tp_size):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=8, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=45)
    )
    try:
        dp_rank = rank // tp_size
        tp_group = None
        for start in range(0, 8, tp_size):
            group = torch.distributed.new_group(list(range(start, start + tp_size)))
            if start == dp_rank * tp_size:
                tp_group = group
        for reduce_sample in (False, True):
            ctx = SimpleNamespace(tp_group=GlooTPGroup(rank % tp_size, tp_size, tp_group), reduce_sample=reduce_sample)
            p = proposer(ctx, tp_size=tp_size)
            p.model = ShardedToyMTP(ctx)
            p._runnable = FrozenInputsGraph(p, ctx)
            for graph_tokens in (4, 32):
                p._run_glm_mtp_graph(None, graph_tokens, 0, None, None, graph_tokens)
                for batch, rejected in ((graph_tokens // 4, 0), (1, 3), (0, 0)):
                    # Different DP replicas shrink/idle while their TP peers
                    # keep the same padded collective shapes on every step.
                    batch = max(0, batch - dp_rank)
                    common, indices = inputs(p, batch, rejected)
                    ctx.tp_group.gather_shapes.clear()
                    result = p._run_glm_mtp_graph(
                        common if batch else None, graph_tokens, batch, indices, None, graph_tokens
                    )
                    # All three iterations use fixed padded request shapes on
                    # every TP rank, including for a shrinking local batch.
                    expected_shape = (graph_tokens // 4, 1 if reduce_sample else 64 // tp_size)
                    assert ctx.tp_group.gather_shapes == [expected_shape] * (6 if reduce_sample else 3)
                    assert [int(mask.sum()) for _, mask in p.model.calls[-3:]] == [batch * 4, batch, batch]
                    if batch == 0:
                        assert result.shape == (0, 3)
                        continue

                    ref_ctx = SimpleNamespace(
                        cudagraph_runtime_mode=Mode.NONE,
                        flash_comm_v1_enabled=False,
                        mc2_mask=torch.ones(graph_tokens, dtype=torch.bool),
                    )
                    ref = proposer(ref_ctx)
                    ref.model = ToyMTP(ref_ctx, vocab_size=64)
                    common_ref, indices_ref = inputs(ref, batch, rejected)
                    metadata = prepare_metadata(ref, common_ref, graph_tokens, batch, indices_ref)
                    ref_ctx.attn_metadata = metadata[0]
                    expected = ref._run_merged_draft(
                        graph_tokens,
                        batch,
                        ref.token_indices_to_sample[:batch],
                        ref.positions,
                        None,
                        metadata,
                        batch * 4,
                        False,
                    )
                    # Each rank checks against full-vocabulary eager output,
                    # not just rank 0 (a wrong local argmax must fail).
                    torch.testing.assert_close(result, expected)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires CPU Gloo")
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_tp_dp_collectives_and_both_logits_paths_match_full_vocab_eager(tmp_path, tp_size):
    torch.multiprocessing.spawn(
        _tp_dp_collective_worker, args=(str(tmp_path / "tp_dp_init"), tp_size), nprocs=8, join=True
    )


@pytest.mark.parametrize("mtp_eager", [False, True])
def test_tp8_smoke_entry_config_without_initializing_vllm(monkeypatch, mtp_eager):
    examples = ROOT / "examples/dsa_demo"
    monkeypatch.syspath_prepend(str(examples))
    # The smoke module sets runtime env defaults; isolate them from other tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in ("simple_prompt_test_dp16", "simple_prompt_test_tp8_mtp_graph"):
        spec = importlib.util.spec_from_file_location(name, examples / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    entry = module
    monkeypatch.setattr(sys, "argv", ["tp8-smoke"] + (["--mtp-eager"] if mtp_eager else []))
    entry.smoke.run_node_dp = Mock()
    entry.main()
    launch = entry.smoke.run_node_dp.call_args.args[0]
    assert launch.node_size == 1 and launch.node_rank == 0 and launch.sync_port == 0
    assert entry.smoke.DATA_PARALLEL_SIZE == 1
    assert len(entry.smoke.PROMPTS) == 1 and not entry.smoke.ENABLE_PROFILE
    expected_result = "tp8_mtp_eager.json" if mtp_eager else "tp8_mtp_graph.json"
    assert expected_result == entry.smoke.RESULT_JSON
    config = entry.smoke.build_llm_kwargs(0, False)
    assert config["model"] == "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
    assert config["tensor_parallel_size"] == 8 and config["enable_expert_parallel"]
    assert not config["enforce_eager"] and not config["enable_chunked_prefill"]
    assert config["max_num_seqs"] == 1
    assert config["max_num_batched_tokens"] == config["max_model_len"] == 25600
    assert config["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 8,
        "enforce_eager": mtp_eager,
    }
    assert config["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert config["compilation_config"]["cudagraph_capture_sizes"] == [4]
    assert not config["compilation_config"]["pass_config"]["enable_sp"]
    for option in (
        "enable_flashcomm1",
        "enable_shared_expert_dp",
        "enable_flashcomm2_parallel_size",
        "enable_reduce_sample",
    ):
        assert not config["additional_config"][option]


@pytest.mark.parametrize("tp,dp", [(1, 16), (2, 4), (4, 2), (8, 1), (8, 2)])
@pytest.mark.parametrize("mtp_graph", [False, True])
def test_generic_smoke_matches_draft_tp_and_keeps_control_layout(monkeypatch, tp, dp, mtp_graph):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    source = ROOT / "examples/dsa_demo/simple_prompt_test_dp16.py"
    spec = importlib.util.spec_from_file_location("generic_smoke", source)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    smoke.TENSOR_PARALLEL_SIZE = tp
    smoke.DATA_PARALLEL_SIZE = dp
    smoke.RUN_MODE = "graph"
    smoke.ENABLE_MTP = True
    smoke.ENABLE_MTP_GRAPH = mtp_graph
    smoke.ENABLE_PROFILE = False
    smoke.PROMPTS = ["short prompt"] * dp
    config = smoke.build_llm_kwargs(0, False)
    assert config["speculative_config"]["draft_tensor_parallel_size"] == tp
    assert config["speculative_config"]["enforce_eager"] == (not mtp_graph)
    assert not config["enforce_eager"]
    assert config["compilation_config"]["pass_config"] == {"enable_sp": False}
    for key in ("enable_flashcomm1", "enable_shared_expert_dp", "enable_flashcomm2_parallel_size"):
        assert not config["additional_config"][key]
    if dp > 1:
        assert config["additional_config"]["dp_allreduce_on_npu"]
    assert all(len(smoke.shard_prompts(rank)) == 1 for rank in range(dp))


@pytest.mark.parametrize("graph,load_failure", [(True, False), (True, True), (False, False)])
def test_draft_loading_validates_before_disabling_compile(monkeypatch, graph, load_failure):
    """Execute production loading/context and DSA guard with a validating
    dataclass replacement, without claiming to initialize real VllmConfig/NPU.
    """
    compile_mode = SimpleNamespace(NONE=0, VLLM_COMPILE=3)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(CompilationMode=compile_mode, CUDAGraphMode=Mode))
    monkeypatch.setitem(
        sys.modules, "vllm.compilation.backends", SimpleNamespace(set_model_tag=lambda _: nullcontext())
    )
    guard_tree = ast.parse((ROOT / "vllm_ascend/dsa_offload/config.py").read_text())
    guard = next(
        n
        for n in ast.walk(guard_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "validate_finalized_graph_contract"
    )
    future = ast.parse("from __future__ import annotations").body
    ns = {}
    exec(compile(ast.Module(body=future + [guard], type_ignores=[]), "<DSA guard>", "exec"), ns)
    contract = SimpleNamespace(enabled=True, enable_row_mode_decode_graph=True)
    validate = lambda cfg: ns["validate_finalized_graph_contract"](contract, cfg, phase="platform")

    @dataclass
    class Compilation:
        mode: int = compile_mode.VLLM_COMPILE
        cudagraph_mode: Mode = Mode.FULL_DECODE_ONLY
        cudagraph_capture_sizes: tuple = (4,)
        static_forward_context: dict = field(default_factory=dict)
        static_all_moe_layers: list = field(default_factory=list)

    @dataclass
    class Config:
        compilation_config: Compilation = field(default_factory=Compilation)

        def __post_init__(self):
            validate(self)

    target = Config()
    original_compilation = target.compilation_config
    created = []

    def create_config():
        # Upstream _create_draft_vllm_config uses validating replace(); it
        # must still see the target's VLLM_COMPILE mode here.
        cfg = replace(target)
        created.append(cfg)
        return cfg

    loaded = []

    def get_model(*, vllm_config, **kwargs):
        loaded.append(vllm_config)
        assert target.compilation_config is original_compilation
        validate(target)
        cc = vllm_config.compilation_config
        assert cc.mode == (compile_mode.NONE if graph else compile_mode.VLLM_COMPILE)
        assert cc.static_forward_context is original_compilation.static_forward_context
        assert cc.static_all_moe_layers is original_compilation.static_all_moe_layers
        cc.static_forward_context["mtp.attn"] = "registered"
        cc.static_all_moe_layers.append("mtp.moe")
        if load_failure:
            raise RuntimeError("test weight loading failure")
        return "loaded model"

    tree = ast.parse((SPEC_DIR / "llm_base_proposer.py").read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_eager_context")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSpecDecodeBaseProposer")
    get_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_model")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    context_if = next(
        n
        for n in init.body
        if isinstance(n, ast.If)
        and any(
            isinstance(a, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "maybe_eager_context" for t in a.targets)
            for a in n.body
        )
    )
    p = SimpleNamespace(
        use_glm_mtp_graph=graph,
        use_cuda_graph=graph,
        maybe_eager_context=nullcontext(),
        vllm_config=target,
        _create_draft_vllm_config=create_config,
        method="mtp",
        speculative_config=SimpleNamespace(draft_model_config=None, draft_load_config=None),
    )
    runtime = {
        "copy": copy,
        "replace": replace,
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "CompilationMode": compile_mode,
        "get_model": get_model,
        "logger": Mock(),
        "self": p,
        "vllm_config": target,
        "enable_sp": lambda _: False,
    }
    exec(
        compile(ast.Module(body=future + [helper, get_method, context_if], type_ignores=[]), "<draft loading>", "exec"),
        runtime,
    )
    with p.maybe_eager_context:
        if load_failure:
            with pytest.raises(RuntimeError, match="test weight loading failure"):
                runtime["_get_model"](p)
        else:
            assert runtime["_get_model"](p) == "loaded model"
    assert target.compilation_config is original_compilation
    assert original_compilation.mode == compile_mode.VLLM_COMPILE
    assert original_compilation.static_forward_context["mtp.attn"] == "registered"
    assert original_compilation.static_all_moe_layers == ["mtp.moe"]
    assert created[0].compilation_config.mode == compile_mode.VLLM_COMPILE
    if graph:
        assert loaded[0] is not target and loaded[0] is not created[0]
        assert loaded[0].compilation_config.mode == compile_mode.NONE
    # No weakening of target validation, including after a load failure.
    with pytest.raises(ValueError, match="requires compilation_config.mode=VLLM_COMPILE"):
        replace(target, compilation_config=replace(original_compilation, mode=compile_mode.NONE))
