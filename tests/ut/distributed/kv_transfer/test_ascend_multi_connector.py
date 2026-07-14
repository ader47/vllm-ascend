# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (
    AscendMultiConnector,
)


class _Blocks:
    def __init__(self):
        self.empty_blocks = object()

    def new_empty(self):
        return self.empty_blocks


def test_update_state_after_alloc_forwards_real_blocks_to_marked_sender():
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._requests_to_connector = {"req": 0}

    chosen = SimpleNamespace(
        connector_scheduler=object(),
        update_state_after_alloc=MagicMock(),
    )
    marked_sender = SimpleNamespace(
        connector_scheduler=SimpleNamespace(
            requires_full_blocks_on_update_after_alloc=True,
        ),
        update_state_after_alloc=MagicMock(),
    )
    ordinary = SimpleNamespace(
        connector_scheduler=object(),
        update_state_after_alloc=MagicMock(),
    )
    connector._connectors = [chosen, marked_sender, ordinary]

    request = SimpleNamespace(request_id="req")
    blocks = _Blocks()

    connector.update_state_after_alloc(request, blocks, 7)

    chosen.update_state_after_alloc.assert_called_once_with(request, blocks, 7)
    marked_sender.update_state_after_alloc.assert_called_once_with(request, blocks, 7)
    ordinary.update_state_after_alloc.assert_called_once_with(
        request,
        blocks.empty_blocks,
        0,
    )


def test_configures_sibling_with_ascend_store_gva_reuse_plan():
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    sibling = SimpleNamespace(set_gva_layerwise_reuse_plan=MagicMock())
    connector._connectors = [sibling]
    kv_transfer_config = SimpleNamespace(
        kv_connector="MultiConnector",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_connector_extra_config": {
                        "backend": "memcache",
                        "use_layerwise": True,
                        "layerwise_num_shared_buffers": 2,
                    },
                }
            ]
        },
    )
    main_names = [f"model.layers.{i}.self_attn.attn" for i in range(6)]
    indexer_names = [
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.4.self_attn.indexer.k_cache",
    ]
    layer_specs = {
        **{name: SimpleNamespace(page_size_bytes=128) for name in main_names},
        **{name: SimpleNamespace(page_size_bytes=64) for name in indexer_names},
    }
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[*main_names, *indexer_names],
                kv_cache_spec=SimpleNamespace(kv_cache_specs=layer_specs),
            )
        ]
    )

    connector._configure_gva_layerwise_reuse(kv_transfer_config, kv_cache_config)

    # Indexer owners share the scheduler group but do not add callbacks.
    sibling.set_gva_layerwise_reuse_plan.assert_called_once_with({3: 1, 4: 2})
