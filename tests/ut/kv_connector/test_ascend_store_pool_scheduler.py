from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)


class _FakeBlockHash:

    def __init__(self, value: str):
        self.value = value

    def hex(self) -> str:
        return self.value


class _FakeStoreScheduler:

    def __init__(self, exists_states: list[int]):
        self.exists_states = exists_states
        self.queried_keys: list[str] = []

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        self.queried_keys = keys
        return self.exists_states


def _make_scheduler(
    exists_states: list[int],
    *,
    use_layerwise: bool = False,
    backend_name: str = "mooncake",
) -> KVPoolScheduler:
    scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
    scheduler.kv_role = "kv_producer"
    scheduler.consumer_is_to_load = False
    scheduler._discard_partial_chunks = True
    scheduler._block_size = 16
    scheduler.model_name = "test-model"
    scheduler.pcp_size = 2
    scheduler.dcp_size = 2
    scheduler.tp_size = 4
    scheduler.put_step = 2
    scheduler.pp_size = 2
    scheduler.num_layers = 3
    scheduler.use_layerwise = use_layerwise
    scheduler.backend_name = backend_name
    scheduler.use_gva_layerwise = use_layerwise and backend_name == "memcache"
    scheduler.load_async = False
    scheduler.store_scheduler = _FakeStoreScheduler(exists_states)
    scheduler.load_specs = {}
    return scheduler


def test_non_layerwise_lookup_uses_all_rank_key_combinations():
    scheduler = _make_scheduler([1] * 16)
    request = SimpleNamespace(
        request_id="req1",
        prompt_token_ids=list(range(16)),
        block_hashes=[_FakeBlockHash("hash0")],
        num_tokens=17,
    )

    tokens, load_async = scheduler.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,
    )

    assert tokens == 16
    assert not load_async
    assert len(scheduler.store_scheduler.queried_keys) == 16
    assert "test-model@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@hash0" in (
        scheduler.store_scheduler.queried_keys
    )
    assert "test-model@pcp1@dcp1@head_or_tp_rank:1@pp_rank:1@hash0" in (
        scheduler.store_scheduler.queried_keys
    )


def test_non_layerwise_lookup_requires_all_rank_keys_for_block_hit():
    scheduler = _make_scheduler([1] * 16 + [1, 0] + [1] * 14)
    request = SimpleNamespace(
        request_id="req1",
        prompt_token_ids=list(range(32)),
        block_hashes=[_FakeBlockHash("hash0"), _FakeBlockHash("hash1")],
        num_tokens=33,
    )

    tokens, load_async = scheduler.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,
    )

    assert tokens == 16
    assert not load_async


def test_mooncake_layerwise_lookup_uses_layer_keys():
    scheduler = _make_scheduler([1] * 24, use_layerwise=True)
    request = SimpleNamespace(
        request_id="req1",
        prompt_token_ids=list(range(16)),
        block_hashes=[_FakeBlockHash("hash0")],
        num_tokens=17,
    )

    tokens, load_async = scheduler.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,
    )

    assert tokens == 16
    assert not load_async
    assert len(scheduler.store_scheduler.queried_keys) == 24
    assert "test-model@pcp0@dcp0@head_or_tp_rank:0@hash0@0" in (
        scheduler.store_scheduler.queried_keys
    )
    assert "test-model@pcp1@dcp1@head_or_tp_rank:1@hash0@2" in (
        scheduler.store_scheduler.queried_keys
    )


def test_non_memcache_layerwise_lookup_uses_layer_keys():
    scheduler = _make_scheduler(
        [1] * 24,
        use_layerwise=True,
        backend_name="yuanrong",
    )
    request = SimpleNamespace(
        request_id="req1",
        prompt_token_ids=list(range(16)),
        block_hashes=[_FakeBlockHash("hash0")],
        num_tokens=17,
    )

    tokens, load_async = scheduler.get_num_new_matched_tokens(
        request,
        num_computed_tokens=0,
    )

    assert tokens == 16
    assert not load_async
    assert len(scheduler.store_scheduler.queried_keys) == 24
    assert "test-model@pcp0@dcp0@head_or_tp_rank:0@hash0@0" in (
        scheduler.store_scheduler.queried_keys
    )
    assert "test-model@pcp1@dcp1@head_or_tp_rank:1@hash0@2" in (
        scheduler.store_scheduler.queried_keys
    )
