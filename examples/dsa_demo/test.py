#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单机 TP8DP1EP8：target graph + 三轮 MTP 合并 graph。

默认 BS1、短 prompt、关闭 chunked prefill/profile/SP。
加 --mtp-eager 可运行 target graph + MTP eager 对照组。
复用 dp16 脚本的配置构建与启动逻辑，不改动原脚本的用户配置。
"""

import argparse

import simple_prompt_test_dp16 as smoke

from novel_dataset import chinese_20k
import os
os.environ["HCCL_IF_IP"] = "141.61.33.21"
os.environ["GLOO_SOCKET_IFNAME"] = "eth4"
os.environ["TP_SOCKET_IFNAME"] = "eth4"
os.environ["HCCL_SOCKET_IFNAME"] = "eth4"

smoke.MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
smoke.RUN_MODE = "graph"   # disabled / cache-init / eager / graph
# smoke.RUN_MODE = "disabled"   # disabled / cache-init / eager / graph
smoke.TENSOR_PARALLEL_SIZE = 2
smoke.DATA_PARALLEL_SIZE = 4
smoke.MAX_NUM_SEQS = 16
smoke.PROMPTS = [((chinese_20k[0]*3)[:92800]),] * (smoke.DATA_PARALLEL_SIZE * smoke.MAX_NUM_SEQS)
smoke.MAX_MODEL_LEN = 131072
smoke.MAX_NUM_BATCHED_TOKENS = 4096
smoke.ENABLE_CHUNKED_PREFILL = True
smoke.MAX_TOKENS = 256
smoke.GPU_MEMORY_UTILIZATION = 0.93
smoke.ENABLE_EXPERT_PARALLEL = True
smoke.ENABLE_A5_PACKED_C8_DSA = True
smoke.ENABLE_MTP = True
smoke.MTP_NUM_SPECULATIVE_TOKENS = 3
smoke.DSA_INDEXER_MLA_BLOCK_RATIO = 4
smoke.ENABLE_PROFILE = True
smoke.PROFILE_DIR = "/home/l00948936/profiling/benchmark"

_BASE_BUILD_LLM_KWARGS = smoke.build_llm_kwargs


def build_llm_kwargs(dp_rank: int, multi_node: bool) -> dict:
    kwargs = _BASE_BUILD_LLM_KWARGS(dp_rank, multi_node)
    kwargs["speculative_config"]["draft_tensor_parallel_size"] = smoke.TENSOR_PARALLEL_SIZE
    kwargs["additional_config"].update(
        enable_flashcomm1=False,
        enable_shared_expert_dp=False,
        enable_flashcomm2_parallel_size=0,
        enable_reduce_sample=False,
        enable_mlapo=True
    )
    kwargs["compilation_config"]["pass_config"] = {"enable_sp": False}
    return kwargs


smoke.build_llm_kwargs = build_llm_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp-eager", action="store_true", help="仅关闭 MTP 图, target 仍入图")
    args = parser.parse_args()
    smoke.ENABLE_MTP_GRAPH = not args.mtp_eager
    smoke.RESULT_JSON = "tp8_mtp_eager.json" if args.mtp_eager else "tp8_mtp_graph.json"
    smoke.run_node_dp(argparse.Namespace(node_size=1, node_rank=0, master_addr="", master_port=0, sync_port=0))


if __name__ == "__main__":
    main()