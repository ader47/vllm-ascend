from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_scheduler import (
    _new_request_prompt_tokens,
    _num_touched_blocks,
)


def test_num_touched_blocks_includes_partial_page():
    assert _num_touched_blocks(0, 128) == 0
    assert _num_touched_blocks(1, 128) == 1
    assert _num_touched_blocks(127, 128) == 1
    assert _num_touched_blocks(128, 128) == 1
    assert _num_touched_blocks(129, 128) == 2


def test_new_request_prompt_tokens_uses_scheduler_output_fields():
    request = SimpleNamespace(
        prompt_token_ids=[11, 12, 13],
        prompt_embeds=None,
    )

    assert _new_request_prompt_tokens(request) == 3
