"""Regression tests for SFA PD transfer into KVOffloadDecodeManager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (  # noqa: E402
    READ_READY_BATCH,
    SendTask,
    infer_sfa_component_group_ids,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.read_thread import (  # noqa: E402
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.send_thread import (  # noqa: E402
    MembPullSendingThread,
)


def test_infer_separate_main_and_indexer_groups():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.self_attn.indexer"]),
            SimpleNamespace(layer_names=["model.layers.0.self_attn"]),
        ]
    )

    assert infer_sfa_component_group_ids(config) == (1, 0)


def test_infer_uniform_group_uses_same_block_ids():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    "model.layers.0.self_attn",
                    "model.layers.0.self_attn.indexer",
                ]
            )
        ]
    )

    assert infer_sfa_component_group_ids(config) == (0, 0)


def _make_read_thread() -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        num_blocks=16,
        layer_metadata={},
        main_name_to_idx={},
        cpu_pools=[],
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={"req-0": ([3, 4], [8])},
        get_offload_layer_id=lambda _: 0,
    )
    return thread


def _make_layer(k_cpu_ptr: int | None, v_cpu_ptr: int | None) -> dict:
    return {
        "layer_name": "model.layers.0.self_attn",
        "pool_idx": 0,
        "offload_id": 0,
        "p_k_base": 1000,
        "p_v_base": 2000,
        "p_k_len": 10,
        "p_v_len": 20,
        "k_cpu_ptr": k_cpu_ptr,
        "v_cpu_ptr": v_cpu_ptr,
        "indexer": {
            "p_dsa_base": 7000,
            "block_len": 5,
            "d_base": 8000,
            "shape": (16, 1, 1, 5),
        },
        "scale": None,
    }


def test_read_descriptors_use_independent_main_and_indexer_block_ids():
    thread = _make_read_thread()

    local, peer, lengths, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7],
        want_info=True,
    )

    assert local == [3030, 4060, 8040]
    assert peer == [1010, 2020, 7035]
    assert lengths == [20, 40, 5]
    assert info is not None
    assert info["n_main"] == 2
    assert info["n_indexer"] == 1


def test_non_tp0_read_descriptors_still_transfer_indexer():
    thread = _make_read_thread()

    local, peer, lengths, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None),
        "req-0",
        p_main_block_ids=[1, 2],
        p_indexer_block_ids=[7],
        want_info=True,
    )

    assert local == [8040]
    assert peer == [7035]
    assert lengths == [5]
    assert info is not None
    assert info["n_main"] == 0
    assert info["n_indexer"] == 1


def test_read_descriptor_rejects_incomplete_indexer_transfer():
    thread = _make_read_thread()

    with pytest.raises(RuntimeError, match="indexer block count mismatch"):
        thread._build_req_descriptors(
            _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
            "req-0",
            p_main_block_ids=[1, 2],
            p_indexer_block_ids=[7, 9],
            want_info=False,
        )


def test_send_thread_wires_both_cache_group_block_lists():
    layer_name = "model.layers.0.self_attn"
    thread = MembPullSendingThread.__new__(MembPullSendingThread)
    thread._state = SimpleNamespace(main_group_idx=1, indexer_group_idx=0)
    thread.last_layer_idx = 0
    thread._p_save_events = {}
    thread.layer_send_done_events = [MagicMock()]
    thread.layer_transfer_finished_events = None
    thread._pending_reads_by_layer = {}
    thread._layer_read_errors = {}
    thread._mf_meta_sent_paths = set()
    thread._send_mf_meta = MagicMock()
    dealer = MagicMock()
    thread._ensure_dealer = MagicMock(return_value=dealer)
    encoder = MagicMock()
    encoder.encode.side_effect = lambda value: value
    req_meta = SimpleNamespace(
        local_block_ids=[[7], [3, 4]],
        remote_host="127.0.0.1",
        remote_port=1234,
        chunk_finish=True,
    )

    thread._process_send_task(
        SendTask(
            send_request={"req-0": req_meta},
            layer_idx=0,
            layer_name=layer_name,
        ),
        encoder,
    )

    sent_message = dealer.send.call_args.args[0]
    assert sent_message[0] == READ_READY_BATCH
    assert sent_message[3] == [("req-0", [3, 4], [7])]
    assert sent_message[4] == ["req-0"]
