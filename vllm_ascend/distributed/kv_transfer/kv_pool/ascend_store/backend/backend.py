from abc import ABC, abstractmethod
from typing import Any

from vllm.config import ParallelConfig


class Backend(ABC):
    @abstractmethod
    def __init__(self, parallel_config: ParallelConfig):
        pass

    @abstractmethod
    def set_device(self):
        pass

    @abstractmethod
    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        pass

    @abstractmethod
    def exists(self, keys: list[str]) -> list[int]:
        pass

    @abstractmethod
    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    @abstractmethod
    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        pass

    def alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        raise NotImplementedError("Current backend does not support allocation.")

    def get_key_info(self, keys: list[str]) -> list[Any]:
        raise NotImplementedError("Current backend does not support key info lookup.")

    def copy_to_global(self, gvas: list[int], addrs: list[int], sizes: list[int]) -> int:
        raise NotImplementedError("Current backend does not support global copy.")

    def copy_to_local(self, gvas: list[int], addrs: list[int], sizes: list[int]) -> int:
        raise NotImplementedError("Current backend does not support local copy.")
