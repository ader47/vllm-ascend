import pytest
import torch

from vllm_ascend.ascend_config import LRUResidentCacheConfig
from vllm_ascend.attention.sfa_v1 import (
    _partition_decode_topk,
)


def test_lru_resident_cache_config_defaults_disabled():
    cfg = LRUResidentCacheConfig({})
    assert cfg.enabled is False
    assert cfg.buffer_size == 2048
    assert cfg.topk == 2048


def test_lru_resident_cache_config_accepts_4096_capacity():
    cfg = LRUResidentCacheConfig({
        "enabled": True,
        "buffer_size": 4096,
        "topk": 2048,
    })
    assert cfg.enabled is True
    assert cfg.buffer_size == 4096
    assert cfg.topk == 2048


def test_lru_resident_cache_config_rejects_capacity_smaller_than_topk():
    with pytest.raises(ValueError, match="buffer_size must be >= topk"):
        LRUResidentCacheConfig({
            "enabled": True,
            "buffer_size": 1024,
            "topk": 2048,
        })


def test_partition_decode_topk_maps_variable_mtp_window_to_fresh_slots():
    topk_indices = torch.tensor(
        [
            [0, 4, -1, 3],
            [9, 10, 12, -1],
            [10, 11, 4, -1],
            [12, 5, 9, -1],
        ],
        dtype=torch.int32,
    )
    token_to_req = torch.tensor([0, 1, 1, 1], dtype=torch.int32)
    seq_lens = torch.tensor([5, 13], dtype=torch.int32)
    tokens_per_req = torch.tensor([1, 3], dtype=torch.int32)

    host_indices, fresh_mask, fresh_slots = _partition_decode_topk(
        topk_indices,
        token_to_req,
        seq_lens,
        tokens_per_req,
        fresh_slot_start=4092,
    )

    torch.testing.assert_close(
        host_indices,
        torch.tensor(
            [
                [0, -1, -1, 3],
                [9, -1, -1, -1],
                [-1, -1, 4, -1],
                [-1, 5, 9, -1],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        fresh_mask,
        torch.tensor(
            [
                [False, True, False, False],
                [False, True, True, False],
                [True, True, False, False],
                [True, False, False, False],
            ]
        ),
    )
    expected_fresh_slots = torch.tensor(
        [
            [4088, 4092, 4087, 4091],
            [4091, 4092, 4094, 4081],
            [4092, 4093, 4086, 4081],
            [4094, 4087, 4091, 4081],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(fresh_slots, expected_fresh_slots)
