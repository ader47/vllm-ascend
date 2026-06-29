# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-export of the pluggable transport-engine singleton.

The concrete factory now lives in
:mod:`vllm_ascend.distributed.kv_transfer.utils.transfer_engine_backend`,
which supports both the mooncake and memfabric TransferEngine backends
(selected via :func:`global_te.configure` or the
``VLLM_ASCEND_KV_TRANSFER_BACKEND`` env var). This module keeps the historical
``global_te`` import path working unchanged for the mooncake layerwise base
classes and existing callers.
"""
from vllm_ascend.distributed.kv_transfer.utils.transfer_engine_backend import (
    GlobalTE,
    global_te,
)

__all__ = ["GlobalTE", "global_te"]
