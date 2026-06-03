import sys
import types
import unittest
from unittest.mock import MagicMock, patch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine
fake_store = types.ModuleType("mooncake.store")
fake_store.ReplicateConfig = MagicMock()  # type: ignore[attr-defined]


class FakeMooncakeDistributedStore:
    instances = []

    def __init__(self):
        self.setup_calls = []
        FakeMooncakeDistributedStore.instances.append(self)

    def setup(self, *args):
        self.setup_calls.append(args)
        return 0


fake_store.MooncakeDistributedStore = FakeMooncakeDistributedStore  # type: ignore[attr-defined]
sys.modules["mooncake.store"] = fake_store

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (  # noqa: E402
    MooncakeBackend,
    _convert_to_bytes,
    _parse_global_segment_size,
)


class TestParseGlobalSegmentSize(unittest.TestCase):
    def test_int_input(self):
        self.assertEqual(_parse_global_segment_size(1024), 1024)
        self.assertEqual(_parse_global_segment_size(0), 0)

    def test_gb_unit(self):
        self.assertEqual(_parse_global_segment_size("2GB"), 2 * 1024**3)
        self.assertEqual(_parse_global_segment_size("1.5GB"), int(1.5 * 1024**3))
        self.assertEqual(_parse_global_segment_size(" 2 GB "), 2 * 1024**3)

    def test_gb_unit_edge_cases(self):
        with self.assertRaises(ValueError):
            _parse_global_segment_size("GB")
        with self.assertRaises(ValueError):
            _parse_global_segment_size("abcGB")

    def test_mb_unit(self):
        self.assertEqual(_parse_global_segment_size("512MB"), 512 * 1024**2)
        self.assertEqual(_parse_global_segment_size("0.5MB"), int(0.5 * 1024**2))
        self.assertEqual(_parse_global_segment_size("1024MB"), 1024 * 1024**2)

    def test_kb_unit(self):
        self.assertEqual(_parse_global_segment_size("256KB"), 256 * 1024)
        self.assertEqual(_parse_global_segment_size("1.25KB"), int(1.25 * 1024))

    def test_b_unit(self):
        self.assertEqual(_parse_global_segment_size("4096B"), 4096)
        self.assertEqual(_parse_global_segment_size("1024b"), 1024)

    def test_no_unit(self):
        self.assertEqual(_parse_global_segment_size("2048"), 2048)
        self.assertEqual(_parse_global_segment_size("0"), 0)

    def test_non_string_non_int_input(self):
        self.assertEqual(_parse_global_segment_size(2048.0), 2048)
        self.assertEqual(_parse_global_segment_size(True), 1)

        with self.assertRaises(TypeError):
            _parse_global_segment_size(None)

        with self.assertRaises(TypeError):
            _parse_global_segment_size({"size": 1024})


class TestConvertToBytes(unittest.TestCase):
    def test_valid_conversion(self):
        self.assertEqual(_convert_to_bytes("10", 1, "10"), 10)
        self.assertEqual(_convert_to_bytes("1.5", 1024, "1.5KB"), int(1.5 * 1024))
        self.assertEqual(_convert_to_bytes("0", 1024**3, "0GB"), 0)

    def test_invalid_numbers(self):
        with self.assertRaises(ValueError):
            _convert_to_bytes("abc", 1, "abc")

        with self.assertRaises(ValueError):
            _convert_to_bytes("1.2.3", 1024, "1.2.3KB")


class TestMooncakeSchedulerClient(unittest.TestCase):

    def setUp(self):
        FakeMooncakeDistributedStore.instances.clear()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te")
    def test_scheduler_client_does_not_contribute_memory(self, mock_global_te, mock_get_ip):
        mock_get_ip.return_value = "127.0.0.1"
        transfer_engine = MagicMock()
        transfer_engine.get_rpc_port.return_value = 12345
        transfer_engine.get_engine.return_value = object()
        mock_global_te.get_transfer_engine.return_value = transfer_engine

        config = MagicMock()
        config.metadata_server = "P2PHANDSHAKE"
        config.global_segment_size = 1024
        config.local_buffer_size = 2048
        config.protocol = "ascend"
        config.device_name = ""
        config.master_server_address = "127.0.0.1:50051"

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend."
            "MooncakeStoreConfig.load_from_env",
            return_value=config,
        ):
            MooncakeBackend.create_scheduler_client(MagicMock())

        store = FakeMooncakeDistributedStore.instances[-1]
        setup_args = store.setup_calls[-1]
        self.assertEqual(setup_args[2], 0)
        self.assertEqual(setup_args[3], 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te")
    def test_worker_client_uses_configured_memory(self, mock_global_te, mock_get_ip):
        mock_get_ip.return_value = "127.0.0.1"
        transfer_engine = MagicMock()
        transfer_engine.get_rpc_port.return_value = 12345
        transfer_engine.get_engine.return_value = object()
        mock_global_te.get_transfer_engine.return_value = transfer_engine

        config = MagicMock()
        config.metadata_server = "P2PHANDSHAKE"
        config.global_segment_size = 1024
        config.local_buffer_size = 2048
        config.protocol = "ascend"
        config.device_name = ""
        config.master_server_address = "127.0.0.1:50051"

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend."
            "MooncakeStoreConfig.load_from_env",
            return_value=config,
        ):
            MooncakeBackend(MagicMock())

        store = FakeMooncakeDistributedStore.instances[-1]
        setup_args = store.setup_calls[-1]
        self.assertEqual(setup_args[2], 1024)
        self.assertEqual(setup_args[3], 2048)
