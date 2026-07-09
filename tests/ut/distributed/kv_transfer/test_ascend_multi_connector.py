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
