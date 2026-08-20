# SPDX-License-Identifier: Apache-2.0
"""Source-level checks for the Ascend Mamba precision-kernel override."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
POSTPROCESS = ROOT / "vllm_ascend" / "ops" / "triton" / "mamba" / "postprocess.py"
PATCH_MAMBA_UTILS = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_mamba_utils.py"


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def test_postprocess_keeps_only_existing_ascend_precision_kernel() -> None:
    functions = _top_level_functions(POSTPROCESS)

    assert set(functions) == {"postprocess_mamba_fused_kernel"}
    postprocess_source = ast.unparse(functions["postprocess_mamba_fused_kernel"])
    assert "src_ptr = src_addr.to(tl.pointer_type(tl.uint8))" in postprocess_source
    assert "dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))" in postprocess_source
    assert "PRECOMPUTED_NEW_COMPUTED" in postprocess_source
    assert "tl.store(num_accepted_tokens_ptr + req_idx, 1)" in postprocess_source


def test_patch_only_installs_existing_ascend_postprocess_kernel() -> None:
    patch_source = PATCH_MAMBA_UTILS.read_text()

    assert "mamba_utils.postprocess_mamba_fused_kernel = postprocess_mamba_fused_kernel" in patch_source
    assert "MambaBase.bind_kv_cache" not in patch_source
    assert "mamba_utils._copy_mamba_state_block" not in patch_source
    assert "mamba_utils.precopy_mamba_align_fused_kernel" not in patch_source


def test_layerwise_mamba_copy_is_grouped_by_layer() -> None:
    functions = _top_level_functions(PATCH_MAMBA_UTILS)
    patch_source = PATCH_MAMBA_UTILS.read_text()

    assert "_collect_mamba_copy_meta_with_layers" in functions
    assert "prepare_mamba_copy_by_layer" in functions
    assert "do_mamba_copy_block_for_layer" in functions
    assert "finish_mamba_copy_by_layer" in functions
    assert ("mamba_utils.prepare_mamba_copy_by_layer = prepare_mamba_copy_by_layer") in patch_source
    assert ("mamba_utils.do_mamba_copy_block_for_layer = do_mamba_copy_block_for_layer") in patch_source
    assert "mamba_utils.finish_mamba_copy_by_layer = finish_mamba_copy_by_layer" in patch_source


def test_layerwise_mamba_copy_stages_pointer_metadata_once() -> None:
    functions = _top_level_functions(PATCH_MAMBA_UTILS)
    selected = [
        functions["prepare_mamba_copy_by_layer"],
        functions["do_mamba_copy_block_for_layer"],
    ]

    copy_calls = []
    namespace = {
        "MambaCopyBuffers": object,
        "_can_launch_triton_batch_memcpy": lambda: True,
        "mamba_utils": SimpleNamespace(
            batch_memcpy=lambda src, dst, sizes: copy_calls.append((src, dst, sizes)),
        ),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PATCH_MAMBA_UTILS), "exec"), namespace)

    class FakeBuffer:
        def __init__(self):
            self.np = [0] * 8
            self.gpu = [0] * 8
            self.copy_count = 0

        def copy_to_gpu(self, n):
            self.copy_count += 1
            self.gpu[:n] = self.np[:n]
            return self.gpu[:n]

    copy_bufs = SimpleNamespace(
        src_ptrs=FakeBuffer(),
        dst_ptrs=FakeBuffer(),
        sizes=FakeBuffer(),
        offset=4,
        _layer_copy_metadata={
            "layers.0.linear_attn": ([11, 12], [21, 22], [31, 32]),
            "layers.1.linear_attn": ([13, 14], [23, 24], [33, 34]),
        },
        _layer_copy_slices={},
        _layer_copy_staged=False,
        _layer_tensor_copy_pairs={},
    )

    namespace["prepare_mamba_copy_by_layer"](copy_bufs)
    assert copy_bufs.src_ptrs.copy_count == 1
    assert copy_bufs.dst_ptrs.copy_count == 1
    assert copy_bufs.sizes.copy_count == 1
    assert copy_bufs.src_ptrs.gpu[:4] == [11, 12, 13, 14]

    namespace["do_mamba_copy_block_for_layer"](copy_bufs, "layers.0.linear_attn")
    namespace["do_mamba_copy_block_for_layer"](copy_bufs, "layers.1.linear_attn")

    assert copy_calls == [
        ([11, 12], [21, 22], [31, 32]),
        ([13, 14], [23, 24], [33, 34]),
    ]
    assert copy_bufs.src_ptrs.copy_count == 1
