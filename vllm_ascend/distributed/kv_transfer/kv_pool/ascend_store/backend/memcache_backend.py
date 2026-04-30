# Standard
from enum import Enum
from typing import Any

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3


class MemcacheBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, init_buffer_manager: bool = True):
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memcache by following the instructions at "
                "https://gitee.com/ascend/memfabric_hybrid "  # noqa: E501
                "to run vLLM with MemcacheConnector."
            ) from e
        try:
            self.store = DistributedObjectStore()
            if init_buffer_manager:
                soc_version = get_ascend_device_type()
                if soc_version in {AscendDeviceType.A2}:
                    tmp_tensor = torch.zeros(1, device="npu")
                    output_tensor_list = [
                        torch.empty_like(tmp_tensor) for _ in range(torch.distributed.get_world_size())
                    ]
                    torch.distributed.all_gather(output_tensor_list, tmp_tensor, group=get_world_group().device_group)
                self.local_rank = get_world_group().local_rank
                res = self.store.init(self.local_rank)
            else:
                self.local_rank = 0
                res = self.store.init(device_id=0, init_bm=False)
            assert res == 0
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    def set_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        soc_version = get_ascend_device_type()
        if soc_version in {AscendDeviceType.A2}:
            for ptr, size in zip(ptrs, sizes):
                self.store.register_buffer(ptr, size)
        else:
            pass

    def exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        return self.store.batch_alloc(keys, sizes)

    def get_key_info(self, keys: list[str]) -> list[Any]:
        return self.store.batch_get_key_info(keys)

    def copy_to_global(self, gvas: list[int], addrs: list[int], sizes: list[int]) -> int:
        return self.store.batch_copy(gvas, addrs, sizes, MmcDirect.COPY_L2G.value)

    def copy_to_local(self, gvas: list[int], addrs: list[int], sizes: list[int]) -> int:
        return self.store.batch_copy(gvas, addrs, sizes, MmcDirect.COPY_G2L.value)

    def get(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        try:
            res = self.store.batch_get_into_layers(key, addr, size, MmcDirect.COPY_G2L.value)
            for value in res:
                if value != 0:
                    logger.error(f"Failed to get key {key},res:{res}")
        except Exception as e:
            logger.error(f"Failed to get key {key}. {e}")

    def put(self, key: list[str], addr: list[list[int]], size: list[list[int]]):
        try:
            res = self.store.batch_put_from_layers(key, addr, size, MmcDirect.COPY_L2G.value)
            for value in res:
                if value != 0:
                    logger.error(f"Failed to get key {key},res:{res}")
        except Exception as e:
            logger.error(f"Failed to put key {key},error:{e}")
