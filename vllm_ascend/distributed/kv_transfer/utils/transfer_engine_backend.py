# SPDX-License-Identifier: Apache-2.0
"""Pluggable transport-engine backend for the layerwise KV connectors.

The mooncake layerwise base classes (``mooncake_layerwise_connector.py``) drive
the transport engine through a small duck-typed surface:

* ``get_rpc_port()``
* ``register_memory(ptr, size)``
* ``batch_transfer_sync_write(session, src_list, dst_list, length_list)``

A memfabric ``TransferEngine`` exposes the same shape (see
``memfabric_hybrid/src/smem/.../pytransfer.cpp``), with two caveats that
:class:`MemfabricBackend` papers over so the two backends are interchangeable
without touching the base classes:

1. memfabric write calls return a non-zero (possibly positive) error code on
   failure, whereas the base classes gate on ``if ret < 0``. The adapter
   normalises to ``0`` (success) / ``-1`` (failure).
2. memfabric's advertised "session port" is the port embedded in the D-side
   ``unique_id`` (the peer reconstructs ``dest_session`` from it), not
   memfabric's internally-found rpc port.

Backend selection: :meth:`GlobalTE.configure` (called by the SFA connector
workers before the engine is built) or the ``VLLM_ASCEND_KV_TRANSFER_BACKEND``
env var (default ``mooncake``). Only the selected backend's library is imported.
"""
from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

from vllm.logger import logger

from vllm_ascend import envs

BACKEND_MOONCAKE = "mooncake"
BACKEND_MEMFABRIC = "memfabric"
_VALID_BACKENDS = (BACKEND_MOONCAKE, BACKEND_MEMFABRIC)

# memfabric TransferEngine roles (validated by pytransfer.cpp Initialize).
MEMFABRIC_ROLE_PREFILL = "Prefill"
MEMFABRIC_ROLE_DECODE = "Decode"


@runtime_checkable
class TransferEngineBackend(Protocol):
    """Subset of the mooncake TransferEngine API the layerwise base classes use."""

    def get_rpc_port(self) -> int: ...

    def register_memory(self, ptr: int, size: int) -> int: ...

    def batch_transfer_sync_write(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
    ) -> int: ...


class MemfabricBackend:
    """Adapter around a memfabric ``TransferEngine``.

    Normalises the return-value semantics (memfabric returns non-zero, possibly
    positive, on failure) and exposes the advertised session port (the unique_id
    port) as ``get_rpc_port`` so the mooncake base classes can treat it exactly
    like a mooncake engine.
    """

    def __init__(self, engine: Any, advertised_rpc_port: int | None):
        self._engine = engine
        # The port peers reconstruct dest_session from (D side). None on the
        # Prefill side, which never advertises — falls back to the underlying
        # engine's rpc port there.
        self._advertised_rpc_port = advertised_rpc_port

    def get_rpc_port(self) -> int:
        if self._advertised_rpc_port is not None:
            return self._advertised_rpc_port
        return self._engine.get_rpc_port()

    def register_memory(self, ptr: int, size: int) -> int:
        ret = self._engine.register_memory(ptr, size)
        return 0 if ret == 0 else -1

    def batch_transfer_sync_write(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
    ) -> int:
        ret = self._engine.batch_transfer_sync_write(session_id, src_list, dst_list, length_list)
        if ret != 0:
            logger.error(
                "memfabric batch_transfer_sync_write failed (ret=%s) for session %s",
                ret,
                session_id,
            )
            return -1
        return 0

    def batch_transfer_sync_read(
        self,
        session_id: str,
        local_buffers: list[int],
        peer_buffers: list[int],
        length_list: list[int],
    ) -> int:
        ret = self._engine.batch_transfer_sync_read(
            session_id, local_buffers, peer_buffers, length_list
        )
        if ret != 0:
            logger.error(
                "memfabric batch_transfer_sync_read failed (ret=%s) for session %s",
                ret,
                session_id,
            )
            return -1
        return 0


class GlobalTE:
    """Process-wide, config-aware transport-engine factory.

    Preserves the ``global_te.get_transfer_engine / register_buffer`` surface
    the mooncake base classes import, but routes to mooncake or memfabric based
    on :meth:`configure` (or the env-var fallback).
    """

    def __init__(self):
        self.engine: TransferEngineBackend | None = None
        self.is_register_buffer = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        # Backend configuration captured before the engine is built.
        self._backend: str | None = None
        self._role: str | None = None  # memfabric only: Prefill / Decode
        self._store_url: str | None = None  # memfabric store url (protocol source only)
        self._unique_id: str | None = None  # memfabric D-side unique_id ("IP:PORT")
        self._device_id: int = 0
        self._advertised_rpc_port: int | None = None  # memfabric: unique_id port

    @property
    def backend(self) -> str:
        return self._backend or envs.VLLM_ASCEND_KV_TRANSFER_BACKEND

    def configure(
        self,
        backend: str,
        role: str | None = None,
        store_url: str | None = None,
        unique_id: str | None = None,
        device_id: int = 0,
    ) -> None:
        """Set the backend before ``get_transfer_engine`` builds the engine.

        Must run before the mooncake base ``__init__`` calls
        ``get_transfer_engine`` — e.g. in the SFA worker ``__init__`` ahead of
        ``super().__init__``. No-op once the engine is built.
        """
        with self.transfer_engine_lock:
            if self.engine is not None:
                if backend != self._backend:
                    logger.warning(
                        "global_te engine already built as %s; configure(%s) ignored",
                        self._backend,
                        backend,
                    )
                return
            if backend not in _VALID_BACKENDS:
                raise ValueError(
                    f"Invalid KV transfer backend {backend!r}; expected one of {_VALID_BACKENDS}"
                )
            if role is not None and role not in (MEMFABRIC_ROLE_PREFILL, MEMFABRIC_ROLE_DECODE):
                raise ValueError(
                    f"Invalid memfabric role {role!r}; expected "
                    f"{MEMFABRIC_ROLE_PREFILL!r} or {MEMFABRIC_ROLE_DECODE!r}"
                )
            self._backend = backend
            self._role = role
            self._store_url = store_url
            self._unique_id = unique_id
            self._device_id = device_id

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.engine is None:
            with self.transfer_engine_lock:
                if self.engine is None:
                    backend = self.backend
                    if backend not in _VALID_BACKENDS:
                        raise ValueError(
                            f"Invalid KV transfer backend {backend!r}; "
                            f"expected one of {_VALID_BACKENDS}"
                        )
                    self.engine = self._build(backend, hostname, device_name)
                    logger.info("KV transfer backend = %s, role = %s", backend, self._role)
        return self.engine

    def get_advertised_rpc_port(self) -> int:
        """Port to advertise in ``MooncakeAgentMetadata.te_rpc_port``.

        mooncake: the engine's rpc port. memfabric: the unique_id port, which
        the peer reconstructs ``dest_session`` from (see plan §3). Only the
        Decode side advertises; the Prefill side never calls this.
        """
        if self.backend == BACKEND_MEMFABRIC:
            assert self._advertised_rpc_port is not None, (
                "memfabric backend requires a configured unique_id to advertise a port"
            )
            return self._advertised_rpc_port
        assert self.engine is not None
        return self.engine.get_rpc_port()

    def _build(self, backend: str, hostname: str, device_name: str | None):
        if backend == BACKEND_MOONCAKE:
            return self._build_mooncake(hostname, device_name)
        return self._build_memfabric(hostname)

    def _build_mooncake(self, hostname: str, device_name: str | None):
        try:
            from mooncake.engine import TransferEngine  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run vLLM with MooncakeConnector."
            ) from e
        engine = TransferEngine()
        device_name = device_name if device_name is not None else ""
        ret_value = engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
        if ret_value != 0:
            raise RuntimeError(
                f"TransferEngine initialization failed with ret_value: {ret_value}"
            )
        return engine

    def _build_memfabric(self, hostname: str):
        if self._role not in (MEMFABRIC_ROLE_PREFILL, MEMFABRIC_ROLE_DECODE):
            raise RuntimeError(
                "memfabric backend requires role='Prefill'/'Decode'; "
                "call global_te.configure() first"
            )
        # Decode side must own a unique_id — it determines the config-store
        # address (pytransfer.cpp derives store url from unique_id). Prefill
        # never starts a store, so its unique_id is informational only.
        if self._role == MEMFABRIC_ROLE_DECODE and not self._unique_id:
            raise RuntimeError(
                "memfabric Decode role requires unique_id (e.g. '<host>:<port>'); "
                "call global_te.configure(unique_id=...) first"
            )
        unique_id = self._unique_id or f"{hostname}:{self._device_id}"
        # store_url carries the protocol; memfabric derives the real store
        # address from unique_id. Use the full tcp://<unique_id> form — memfabric
        # examples require tcp://IP:PORT, and a bare tcp://<host> (no port) can
        # trip the store-url parser (smem ExtractIpPortFromUrl).
        store_url = self._store_url or f"tcp://{unique_id}"
        if self._role == MEMFABRIC_ROLE_DECODE:
            self._advertised_rpc_port = self._port_from_unique_id(unique_id)
        logger.info(
            "memfabric TransferEngine initialize: store_url=%s unique_id=%s "
            "role=%s device_id=%s",
            store_url, unique_id, self._role, self._device_id,
        )
        try:
            from memfabric_hybrid import TransferEngine as MFTransferEngine  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memfabric_hybrid (memfabric-hybrid) to use the "
                "memfabric KV transfer backend; see "
                "https://gitee.com/ascend/memfabric_hybrid"
            ) from e
        engine = MFTransferEngine()
        ret = engine.initialize(store_url, unique_id, self._role, self._device_id)
        if ret != 0:
            raise RuntimeError(
                f"memfabric TransferEngine initialize failed (ret={ret}); "
                f"store_url={store_url!r} unique_id={unique_id!r}. memfabric "
                f"requires the unique_id host to be a numeric IPv4 reachable by "
                f"the peer (smem's inet_pton rejects 0.0.0.0/IPv6/hostname). "
                f"Set VLLM_HOST_IP to the node's IPv4 if get_ip() returned a "
                f"non-IPv4 address."
            )
        return MemfabricBackend(engine, self._advertised_rpc_port)

    @staticmethod
    def _port_from_unique_id(unique_id: str) -> int:
        # unique_id = "IP:PORT[_PID]" → port is the segment after the last ':'
        # and before any optional '_PID' suffix.
        core = unique_id.rsplit("_", 1)[0]
        return int(core.rsplit(":", 1)[-1])

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.engine.register_memory(ptr, size)
                # mooncake returns !=0 on failure; MemfabricBackend returns -1.
                # Treat any non-zero as failure.
                if ret_value != 0:
                    raise RuntimeError(
                        f"Memory registration failed (backend={self.backend}, "
                        f"ret={ret_value})."
                    )
            self.is_register_buffer = True


global_te = GlobalTE()
