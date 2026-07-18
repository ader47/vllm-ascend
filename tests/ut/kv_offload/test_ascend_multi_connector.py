"""Tests for Ascend-specific MultiConnector allocation fan-out."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
)


class _FakeBlocks:
    def __init__(self) -> None:
        self.empty = object()

    def new_empty(self):
        return self.empty


def _make_connector(*, requires_full_blocks: bool = False):
    return SimpleNamespace(
        requires_full_blocks_on_update_after_alloc=requires_full_blocks,
        update_state_after_alloc=MagicMock(),
    )


def test_update_state_after_alloc_forwards_full_blocks_to_observer():
    chosen = _make_connector()
    full_blocks_observer = _make_connector(requires_full_blocks=True)
    unrelated = _make_connector()
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [chosen, full_blocks_observer, unrelated]
    connector._requests_to_connector = {"req-0": 0}
    request = SimpleNamespace(request_id="req-0")
    blocks = _FakeBlocks()

    connector.update_state_after_alloc(request, blocks, num_external_tokens=16)

    chosen.update_state_after_alloc.assert_called_once_with(request, blocks, 16)
    full_blocks_observer.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        16,
    )
    unrelated.update_state_after_alloc.assert_called_once_with(request, blocks.empty, 0)


def test_update_state_after_alloc_forwards_observer_without_chosen_connector():
    full_blocks_observer = _make_connector(requires_full_blocks=True)
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = [full_blocks_observer]
    connector._requests_to_connector = {}
    request = SimpleNamespace(request_id="req-0")
    blocks = _FakeBlocks()

    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)

    full_blocks_observer.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        0,
    )
