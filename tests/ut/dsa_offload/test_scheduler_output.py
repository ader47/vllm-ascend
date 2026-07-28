# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import pytest
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.scheduler_output import (
    DSAOffloadSchedulerOutput,
    DSARequestCacheLayoutProjection,
)


def _projection() -> DSARequestCacheLayoutProjection:
    return DSARequestCacheLayoutProjection(
        request_ids=("req",),
        stages=(int(DSARequestCacheStage.DENSE_DECODE),),
        target_resident_budget_tokens=(2048,),
        sparse_budget_tokens=(0,),
        resident_valid_tokens=(-1,),
        resident_block_table_replacements=(),
    )


def test_scheduler_output_wrap_is_shallow_and_pickle_safe() -> None:
    base = SchedulerOutput.make_empty()
    base.num_scheduled_tokens["req"] = 1
    base.total_num_scheduled_tokens = 1

    wrapped = DSAOffloadSchedulerOutput.from_base(
        base,
        dsa_cache_layout=_projection(),
    )

    assert isinstance(wrapped, SchedulerOutput)
    assert wrapped.num_scheduled_tokens is base.num_scheduled_tokens
    assert wrapped.scheduled_cached_reqs is base.scheduled_cached_reqs
    restored = pickle.loads(pickle.dumps(wrapped))
    assert isinstance(restored, DSAOffloadSchedulerOutput)
    assert restored.dsa_cache_layout == wrapped.dsa_cache_layout
    assert restored.num_scheduled_tokens == {"req": 1}


def test_projection_rejects_columns_with_different_lengths() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        DSARequestCacheLayoutProjection(
            request_ids=("req",),
            stages=(),
            target_resident_budget_tokens=(2048,),
            sparse_budget_tokens=(0,),
            resident_valid_tokens=(-1,),
            resident_block_table_replacements=(),
        )
