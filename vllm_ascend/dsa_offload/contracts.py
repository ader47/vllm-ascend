# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA offload 框架与算子共享的静态接口契约。

本文件只放不会随请求或 decode step 变化的 ABI 常量。后续接入
LIDU/KSC/SFA-offload 时，框架配置校验和算子适配层应共同引用这些定义，
避免同一约束在多处以魔法数字重复出现。
"""

DSA_SFA_COMPUTE_TOPK = 2048
DSA_LIDU_OUTPUT_CAPACITY = 12288
DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS = (6144, 10240, 12288)
DSA_REQUIRED_CACHE_BLOCK_SIZE = 128
