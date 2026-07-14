from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (
    ReqMeta,
)


def test_req_meta_resolves_main_blocks_per_hybrid_group():
    req = ReqMeta(
        req_id="req-0",
        block_ids_npu=[10, 11],
        block_ids_cpu=[1, 2],
        main_block_ids_by_group={0: [10, 11], 2: [20, 21]},
    )

    assert req.get_main_block_ids(0) == [10, 11]
    assert req.get_main_block_ids(2) == [20, 21]


def test_req_meta_prefers_step_offload_slice_for_hybrid_group():
    req = ReqMeta(
        req_id="req-0",
        block_ids_npu=[10, 11],
        block_ids_cpu=[1],
        main_block_ids_by_group={0: [10, 11], 2: [20, 21]},
        offload_src_hbm_ids_by_group={0: [11], 2: [21]},
    )

    assert req.get_main_block_ids(0) == [11]
    assert req.get_main_block_ids(2) == [21]
