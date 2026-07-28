# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 稀疏卸载的 Ascend 框架集成模块。

这里维护 DSA offload 自身的配置、能力契约和后续运行时组件，避免与
vLLM-Ascend 已有的原生 DSA/SFA 实现混为一谈。
"""

from vllm_ascend.dsa_offload.config import (
    DSA_SPARSE_CONFIG_KEY,
    DSAOffloadConfig,
    DSAOffloadTraceConfig,
)

__all__ = [
    "DSA_SPARSE_CONFIG_KEY",
    "DSAOffloadConfig",
    "DSAOffloadTraceConfig",
]
