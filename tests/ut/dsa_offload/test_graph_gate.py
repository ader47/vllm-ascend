# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.dsa_offload.graph_gate import (
    evaluate_dsa_row_mode_decode_graph,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)


def _state(*stages: DSARequestCacheStage):
    return SimpleNamespace(
        valid=True,
        row_count=len(stages),
        stages_cpu=np.asarray(stages, dtype=np.int32),
    )


@pytest.mark.parametrize(
    "stages",
    [
        (DSARequestCacheStage.DENSE_DECODE,),
        (DSARequestCacheStage.ENTER_SPARSE_DECODE,),
        (DSARequestCacheStage.SPARSE_DECODE,),
        (
            DSARequestCacheStage.DENSE_DECODE,
            DSARequestCacheStage.ENTER_SPARSE_DECODE,
            DSARequestCacheStage.SPARSE_DECODE,
        ),
    ],
)
def test_all_single_token_decode_stages_share_full_graph(stages) -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(*stages),
        num_reqs=len(stages),
        total_num_scheduled_tokens=len(stages),
        max_num_scheduled_tokens=1,
        max_capture_size=8,
    )

    assert decision.use_full_graph
    assert not decision.is_expected_eager


@pytest.mark.parametrize(
    ("state", "num_reqs", "total_tokens", "max_tokens", "capture", "reason"),
    [
        (
            _state(DSARequestCacheStage.PREFILL),
            1,
            1,
            1,
            8,
            "non_decode_stage",
        ),
        (
            _state(DSARequestCacheStage.DENSE_DECODE),
            1,
            2,
            2,
            8,
            "non_single_token_decode",
        ),
        (
            _state(
                DSARequestCacheStage.DENSE_DECODE,
                DSARequestCacheStage.SPARSE_DECODE,
            ),
            2,
            2,
            1,
            1,
            "capture_size_miss",
        ),
    ],
)
def test_normal_non_graph_phases_are_expected_eager(
    state,
    num_reqs: int,
    total_tokens: int,
    max_tokens: int,
    capture: int,
    reason: str,
) -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=state,
        num_reqs=num_reqs,
        total_num_scheduled_tokens=total_tokens,
        max_num_scheduled_tokens=max_tokens,
        max_capture_size=capture,
    )

    assert not decision.use_full_graph
    assert decision.is_expected_eager
    assert decision.reason == reason


def test_stale_input_batch_state_is_not_a_fallback() -> None:
    state = _state(DSARequestCacheStage.DENSE_DECODE)
    state.row_count = 2

    decision = evaluate_dsa_row_mode_decode_graph(
        state=state,
        num_reqs=1,
        total_num_scheduled_tokens=1,
        max_num_scheduled_tokens=1,
        max_capture_size=8,
    )

    assert not decision.use_full_graph
    assert not decision.is_expected_eager
    assert decision.reason == "input_batch_row_mismatch"


def test_enabled_graph_without_capture_sizes_is_not_a_fallback() -> None:
    decision = evaluate_dsa_row_mode_decode_graph(
        state=_state(DSARequestCacheStage.DENSE_DECODE),
        num_reqs=1,
        total_num_scheduled_tokens=1,
        max_num_scheduled_tokens=1,
        max_capture_size=0,
    )

    assert not decision.use_full_graph
    assert not decision.is_expected_eager
    assert decision.reason == "missing_capture_sizes"
