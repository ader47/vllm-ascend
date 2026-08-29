#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""双机各 4 卡的 DP4TP2EP8 离线 DSA 冒烟测试（vLLM 0.23）。

复用同目录 simple_prompt_test_dp16.py 的启动、同步和 DSA 配置，不修改框架。
默认只做 cache-init；通过后将 RUN_MODE 改为 eager，测试短 prompt 生成。
这两步均关闭 MTP、chunked prefill 和 graph，不验证长序列 sparse/ENTER。

在两机分别设置 ASCEND_RT_VISIBLE_DEVICES 为本机准备使用的 4 张物理卡，
例如 node 0 使用 4,5,6,7，node 1 使用 0,1,2,3。然后分别运行：

    python examples/dsa_demo/simple_prompt_test_2p_dp4tp2.py \
        --node-size 2 --node-rank 0 --master-addr <NODE0_IP> \
        --master-port 29520 --sync-port 29620

    python examples/dsa_demo/simple_prompt_test_2p_dp4tp2.py \
        --node-size 2 --node-rank 1 --master-addr <NODE0_IP> \
        --master-port 29520 --sync-port 29620

IP 和通信网卡沿用当前容器的 VLLM_HOST_IP、HCCL_IF_IP、HCCL_SOCKET_IFNAME、
GLOO_SOCKET_IFNAME、TP_SOCKET_IFNAME 设置。两机都必须同步此文件及其依赖脚本。
"""

from __future__ import annotations

import os
from typing import Any

import simple_prompt_test_dp16 as smoke

MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
RUN_MODE = "cache-init"  # cache-init / eager
TENSOR_PARALLEL_SIZE = 2
DATA_PARALLEL_SIZE = 4
NODE_SIZE = 2
NPUS_PER_NODE = TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE // NODE_SIZE

# 必须在模块加载时设置：multiprocessing spawn 子进程会重新导入主脚本。
# 这里只修改本次进程中的脚本配置，不修改原脚本文件或 vLLM 实现。
smoke.MODEL_PATH = MODEL_PATH
smoke.RUN_MODE = RUN_MODE
smoke.TENSOR_PARALLEL_SIZE = TENSOR_PARALLEL_SIZE
smoke.DATA_PARALLEL_SIZE = DATA_PARALLEL_SIZE
smoke.PROMPTS = ["请用一句话介绍你自己。"] * DATA_PARALLEL_SIZE
smoke.MAX_NUM_SEQS = 1
smoke.MAX_MODEL_LEN = 4096
smoke.MAX_NUM_BATCHED_TOKENS = 4096
smoke.MAX_TOKENS = 32
smoke.GPU_MEMORY_UTILIZATION = 0.82
smoke.ENABLE_EXPERT_PARALLEL = True
smoke.ENABLE_A5_PACKED_C8_DSA = True
smoke.ENABLE_CHUNKED_PREFILL = False
smoke.ENABLE_MTP = False
smoke.ENABLE_PROFILE = False
smoke.RESULT_JSON = None

_BASE_BUILD_LLM_KWARGS = smoke.build_llm_kwargs


def build_llm_kwargs(dp_rank: int, multi_node: bool) -> dict[str, Any]:
    kwargs = _BASE_BUILD_LLM_KWARGS(dp_rank, multi_node)
    # 共用宿主机时，不让 950 自动绑核逻辑修改共享 UVB 线程的 CPU affinity。
    kwargs["additional_config"]["enable_cpu_binding"] = False
    return kwargs


smoke.build_llm_kwargs = build_llm_kwargs


def get_device_groups(visible_devices: str) -> list[list[int]]:
    """Validate the explicit node-wide device list without initializing an NPU."""
    try:
        devices = [int(value.strip()) for value in visible_devices.split(",")]
    except ValueError as exc:
        raise ValueError("Set ASCEND_RT_VISIBLE_DEVICES to 4 physical IDs, e.g. 4,5,6,7") from exc
    if (
        len(devices) != NPUS_PER_NODE
        or len(set(devices)) != NPUS_PER_NODE
        or any(device < 0 or device >= 8 for device in devices)
    ):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain exactly 4 distinct IDs in [0, 7]")
    return [devices[start : start + TENSOR_PARALLEL_SIZE] for start in range(0, NPUS_PER_NODE, TENSOR_PARALLEL_SIZE)]


def main() -> None:
    args = smoke.parse_args()
    if args.node_size != NODE_SIZE:
        raise ValueError("This script requires --node-size 2")
    if RUN_MODE not in {"cache-init", "eager"}:
        raise ValueError("This first-stage test only supports RUN_MODE=cache-init/eager")
    _, _, _, rank_map = smoke.resolve_node_launch(args)
    groups = get_device_groups(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""))
    # 仅父进程规范化本机 4 卡列表。不要在 spawn 导入时覆盖引擎已切好的 2 卡列表。
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(device) for group in groups for device in group)
    for (local_rank, global_rank), devices in zip(rank_map, groups, strict=True):
        print(
            f"[dsa-smoke] node={args.node_rank} local_dp={local_rank} "
            f"global_dp={global_rank} physical_npus={devices}"
        )
    smoke.run_node_dp(args)


if __name__ == "__main__":
    main()
