"""Regression tests for the TP-shared SFA PD CPU pool."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())

memfabric_hybrid = pytest.importorskip("memfabric_hybrid")
if not hasattr(memfabric_hybrid, "offload"):
    memfabric_hybrid.offload = MagicMock()

from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload import read_thread as read_thread_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload import worker as worker_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.read_thread import (  # noqa: E402
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (  # noqa: E402
    SFAPDProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (  # noqa: E402
    SFAPDCpuOffloadConsumerWorker,
    SFAPDCpuOffloadProducerWorker,
)


def _make_read_thread(partial_hbm_bid: int | None = 9) -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={},
        cpu_pools=[],
        hbm_kv={},
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={"req-0": ([7], [3, 4], 2, partial_hbm_bid)},
        get_offload_layer_id=lambda _: 0,
    )
    return thread


def test_non_owner_still_registers_memfabric_pull():
    consumer = SFAPDCpuOffloadConsumerWorker.__new__(SFAPDCpuOffloadConsumerWorker)
    consumer.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"transfer_backend": "memfabric"},
        ),
    )
    consumer.use_layerwise = True
    consumer.kv_cache_config = MagicMock()
    consumer._register_memfabric_pull = MagicMock()
    sfa_worker = SimpleNamespace(register_kv_caches=MagicMock())
    kv_caches = {"model.layers.0.self_attn.attn": tuple(MagicMock() for _ in range(5))}

    with patch.object(worker_module, "SFAKVOffloadWorker", return_value=sfa_worker):
        consumer.register_kv_caches(kv_caches)

    consumer._register_memfabric_pull.assert_called_once_with(kv_caches, None, None)


def test_non_owner_resolves_layer_without_cpu_destination():
    indexer = MagicMock()
    indexer.shape = (16, 2, 1, 128)
    indexer.element_size.return_value = 2
    indexer.data_ptr.return_value = 8000
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={"model.layers.0.self_attn.attn": 0},
        cpu_pools=[None],
        hbm_kv={},
        indexer_tensors=[indexer],
        indexer_scale_tensors=[None],
        dest_blocks_by_req={},
        get_offload_layer_id=lambda _: 0,
    )
    thread._p_layer_meta = {
        "model.layers.0.self_attn.attn": {
            "base_addrs": [1000, 2000, 7000],
            "block_len": [10, 20, 256],
        }
    }

    layer = thread._resolve_read_layer("model.layers.0.self_attn.attn")

    assert layer is not None
    assert layer["k_cpu_ptr"] is None
    assert layer["v_cpu_ptr"] is None
    assert layer["indexer"]["d_base"] == 8000


def test_reuse_layer_accepts_two_tensor_producer_metadata():
    layer_name = "model.layers.1.self_attn.attn"
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={layer_name: 0},
        cpu_pools=[None],
        hbm_kv={},
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={},
        get_offload_layer_id=lambda _: 0,
    )
    thread._p_layer_meta = {
        layer_name: {
            "base_addrs": [1000, 2000],
            "block_len": [10, 20],
        }
    }

    with patch.object(read_thread_module.logger, "error") as error_log:
        layer = thread._resolve_read_layer(layer_name)

    assert layer is not None
    assert layer["indexer"] is None
    assert layer["scale"] is None
    error_log.assert_not_called()


def _make_layer(
    k_cpu_ptr: int | None,
    v_cpu_ptr: int | None,
    with_scale: bool = False,
) -> dict:
    return {
        "layer_name": "model.layers.0.self_attn.attn",
        "pool_idx": 0,
        "offload_id": 0,
        "p_k_base": 1000,
        "p_v_base": 2000,
        "p_k_len": 10,
        "p_v_len": 20,
        "k_cpu_ptr": k_cpu_ptr,
        "v_cpu_ptr": v_cpu_ptr,
        "k_hbm_ptr": 5000,
        "v_hbm_ptr": 6000,
        "indexer": {
            "p_dsa_base": 7000,
            "p_dsa_len": 5,
            "d_base": 8000,
            "d_dsa_len": 10,
            "scale": 2,
            "shape": (16, 2, 1, 128),
        },
        "scale": (
            {
                "p_scale_base": 9000,
                "p_scale_len": 4,
                "d_scale_base": 10000,
                "d_scale_len": 8,
                "scale_factor": 2,
            }
            if with_scale
            else None
        ),
    }


def test_cpu_pool_owner_reads_full_main_partial_and_indexer():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 3
    assert info["n_indexer"] == 2
    assert 3030 in local
    assert 4060 in local
    assert 5090 in local
    assert 6180 in local
    assert 8070 in local


def test_non_owner_reads_only_partial_and_indexer_hbm():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 1
    assert info["n_indexer"] == 2
    assert local == [5090, 6180, 8070]


def test_non_owner_reads_indexer_scale_without_partial_or_cpu_pool():
    thread = _make_read_thread(partial_hbm_bid=None)

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None, with_scale=True),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 0
    assert info["n_indexer"] == 2
    assert local == [8070, 10056]


def test_producer_scheduler_keeps_all_block_groups_and_finishes_chunk():
    scheduler = SFAPDProducerScheduler.__new__(SFAPDProducerScheduler)
    scheduler.block_size = [512, 128]
    scheduler._reqs_need_send_layerwise = {}
    request = SimpleNamespace(
        request_id="req-0-internal",
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_cached_tokens": 0,
            "remote_host": "127.0.0.1",
            "remote_port": 1234,
        },
        all_token_ids=list(range(128)),
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([7], [3, 4]))

    scheduler.update_state_after_alloc(request, blocks, 0)
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id, num_computed_tokens=0)],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={request.request_id: 128},
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    req_meta = metadata.requests[request.request_id]
    assert req_meta.local_block_ids == [[7], [3, 4]]
    assert req_meta.local_computed_tokens == 128
    assert req_meta.chunk_finish is True
    assert request.request_id not in scheduler._reqs_need_send_layerwise


def test_producer_worker_preserves_transfer_timeout_setup(monkeypatch):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"transfer_backend": "memfabric"},
            kv_port=14579,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            tensor_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            get_num_layers=MagicMock(return_value=1),
            use_mla=True,
        ),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    engine = MagicMock()
    monkeypatch.delenv("ASCEND_TRANSFER_TIMEOUT", raising=False)

    with (
        patch.object(worker_module, "get_transfer_timeout_value", return_value=4321),
        patch.object(worker_module, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(
            worker_module.torch,
            "npu",
            SimpleNamespace(current_device=MagicMock(return_value=0)),
            create=True,
        ),
        patch.object(worker_module.global_te, "configure"),
        patch.object(worker_module.global_te, "get_transfer_engine", return_value=engine),
        patch.object(worker_module, "set_shared_layer_transfer_events"),
        patch.object(worker_module, "set_shared_layer_transfer_pending_events"),
    ):
        SFAPDCpuOffloadProducerWorker(config, kv_cache_config, "engine-0")

    assert worker_module.os.environ["ASCEND_TRANSFER_TIMEOUT"] == "4321"
