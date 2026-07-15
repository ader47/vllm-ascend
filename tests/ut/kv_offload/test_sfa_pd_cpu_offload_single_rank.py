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

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.config_data import (  # noqa: E402
    RequestTracker,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload import worker as worker_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.read_thread import (  # noqa: E402
    ConsumerReadState,
    MembPullReadThread,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (  # noqa: E402
    SFAPDCpuOffloadScheduler,
    SFAPDProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.worker import (  # noqa: E402
    SFAPDCpuOffloadConsumerWorker,
    SFAPDCpuOffloadProducerWorker,
)


def _make_read_thread() -> MembPullReadThread:
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={},
        cpu_pools=[],
        indexer_tensors=[],
        dest_blocks_by_req={"req-0": ([7, 8, 9], [3, 4, 5])},
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
    indexer.shape = (16, 1, 1, 128)
    indexer.element_size.return_value = 2
    indexer.data_ptr.return_value = 8000
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={"model.layers.0.self_attn.attn": 0},
        cpu_pools=[None],
        indexer_tensors=[indexer],
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


def test_resolve_read_layer_accepts_main_only_prefill_manifest():
    layer_name = "model.layers.0.self_attn.attn"
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={layer_name: 0},
        cpu_pools=[None],
        indexer_tensors=[None],
        dest_blocks_by_req={},
        get_offload_layer_id=lambda _: 0,
    )
    thread._p_layer_meta = {
        layer_name: {
            "base_addrs": [1000, 2000],
            "block_len": [10, 20],
        }
    }

    layer = thread._resolve_read_layer(layer_name)

    assert layer is not None
    assert layer["p_k_base"] == 1000
    assert layer["p_v_base"] == 2000
    assert layer["indexer"] is None


def _fake_contiguous_cache(base_addr: int, num_blocks: int = 8):
    cache = MagicMock()
    cache.shape = (num_blocks, 4, 1, 8)
    cache.element_size.return_value = 2
    cache.stride.side_effect = lambda dim: (32, 8, 8, 1)[dim]
    cache.data_ptr.return_value = base_addr
    return cache


def test_producer_composes_split_prefill_manifest_by_main_layer():
    main_0 = "model.layers.0.self_attn.attn"
    main_1 = "model.layers.1.self_attn.attn"
    indexer_0 = "model.layers.0.self_attn.indexer.k_cache"
    layer_specs = {
        main_0: SimpleNamespace(page_size_bytes=128),
        main_1: SimpleNamespace(page_size_bytes=128),
        indexer_0: SimpleNamespace(page_size_bytes=64),
    }
    group = SimpleNamespace(
        layer_names=list(layer_specs),
        kv_cache_spec=SimpleNamespace(kv_cache_specs=layer_specs),
    )
    worker = SFAPDCpuOffloadProducerWorker.__new__(
        SFAPDCpuOffloadProducerWorker
    )
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "AscendStoreConnector",
                        "kv_connector_extra_config": {
                            "backend": "memcache",
                            "use_layerwise": True,
                            "layerwise_num_shared_buffers": 1,
                        },
                    }
                ]
            },
        )
    )
    kv_caches = {
        main_0: (
            _fake_contiguous_cache(1000),
            _fake_contiguous_cache(2000),
        ),
        main_1: (
            _fake_contiguous_cache(3000),
            _fake_contiguous_cache(4000),
        ),
        indexer_0: (_fake_contiguous_cache(5000),),
    }

    result = worker._build_split_prefill_layer_metadata(
        kv_caches,
        {name: 0 for name in layer_specs},
        num_blocks=8,
    )

    assert result is not None
    metadata, ordered_names = result
    assert ordered_names == [main_0, main_1]
    assert set(metadata) == {main_0, main_1}
    assert metadata[main_0].kv_caches_base_addr == [1000, 2000, 5000]
    assert metadata[main_0].tensor_group_idx == [0, 0, 0]
    assert metadata[main_0].block_size_scale == [1, 1, 1]
    assert metadata[main_1].kv_caches_base_addr == [3000, 4000]


def _make_layer(
    k_cpu_ptr: int | None,
    v_cpu_ptr: int | None,
    with_indexer: bool = True,
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
        "indexer": (
            {
                "p_dsa_base": 7000,
                "p_dsa_len": 5,
                "d_base": 8000,
                "shape": (16, 1, 1, 128),
            }
            if with_indexer
            else None
        ),
    }


def test_cpu_pool_owner_reads_all_main_pages_and_real_indexer():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=3000, v_cpu_ptr=4000),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 3
    assert info["n_indexer"] == 3
    assert 3030 in local
    assert 4060 in local
    assert 8035 in local


def test_non_tp0_rank_reads_only_real_indexer_hbm():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is not None
    assert info["n_main"] == 0
    assert info["n_indexer"] == 3
    assert local == [8035]


def test_skip_topk_non_tp0_rank_has_no_transfer_leg():
    thread = _make_read_thread()

    local, _, _, info = thread._build_req_descriptors(
        _make_layer(k_cpu_ptr=None, v_cpu_ptr=None, with_indexer=False),
        "req-0",
        [1, 2, 3],
        want_info=True,
    )

    assert info is None
    assert local == []


def test_mf_meta_requires_matching_real_indexer_owners():
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = ConsumerReadState(
        layer_metadata={},
        main_name_to_idx={layer_0: 0, layer_1: 1},
        cpu_pools=[None, None],
        indexer_tensors=[MagicMock(), None],
        dest_blocks_by_req={},
        get_offload_layer_id=lambda _: 0,
    )
    thread._p_layer_meta = {
        layer_0: {"base_addrs": [1, 2], "block_len": [4, 4]},
        layer_1: {"base_addrs": [1, 2, 3], "block_len": [4, 4, 4]},
    }

    with pytest.raises(ValueError, match="missing_on_d.*extra_on_d"):
        thread._validate_indexer_owners()


def test_producer_scheduler_keeps_all_block_groups_and_finishes_chunk():
    scheduler = SFAPDProducerScheduler.__new__(SFAPDProducerScheduler)
    scheduler.block_size = [128]
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
    blocks = SimpleNamespace(get_block_ids=lambda: ([3, 4],))

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
    assert req_meta.local_block_ids == [[3, 4]]
    assert req_meta.local_computed_tokens == 128
    assert req_meta.chunk_finish is True
    assert request.request_id not in scheduler._reqs_need_send_layerwise


def test_decode_scheduler_uses_scheduled_new_request_position():
    scheduler = SFAPDCpuOffloadScheduler.__new__(SFAPDCpuOffloadScheduler)
    scheduler._main_block_size = 128
    scheduler._reqs_need_recv = set()
    scheduler._request_trackers = {
        "req-0": RequestTracker(
            req_id="req-0",
            allocated_indexer_block_ids=[10],
            allocated_block_ids_cpu=[3],
        )
    }
    scheduler.cpu_block_manager = MagicMock()
    scheduler.cpu_block_manager.allocate_block.return_value = [4]
    scheduler_output = SimpleNamespace(
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req-0", num_computed_tokens=127)
        ],
        num_scheduled_tokens={"req-0": 2},
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert len(metadata.requests) == 1
    assert metadata.requests[0].write_start == 127
    assert metadata.requests[0].write_count == 2
    assert metadata.requests[0].block_ids_cpu == [3, 4]
    scheduler.cpu_block_manager.allocate_block.assert_called_once_with(1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (([3, 4],), [3, 4]),
        ([[3, 4]], [3, 4]),
        ([3, 4], [3, 4]),
        (None, []),
    ],
)
def test_decode_scheduler_normalizes_one_group_block_ids(raw, expected):
    assert SFAPDCpuOffloadScheduler._group_zero_block_ids(raw) == expected


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
