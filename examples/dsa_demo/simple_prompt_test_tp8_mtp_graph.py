#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单机 TP8DP1EP8：target graph + 三轮 MTP 合并 graph。

默认 BS1、短 prompt、关闭 chunked prefill/profile/SP。
加 --mtp-eager 可运行 target graph + MTP eager 对照组。
复用 dp16 脚本的配置构建与启动逻辑，不改动原脚本的用户配置。
"""

import argparse

import simple_prompt_test_dp16 as smoke

smoke.MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
smoke.RUN_MODE = "graph"
smoke.TENSOR_PARALLEL_SIZE = 8
smoke.DATA_PARALLEL_SIZE = 1
smoke.PROMPTS = ["请用一句话介绍你自己。"]
smoke.MAX_NUM_SEQS = 1
smoke.MAX_MODEL_LEN = 25600
smoke.MAX_NUM_BATCHED_TOKENS = 25600
smoke.MAX_TOKENS = 64
smoke.GPU_MEMORY_UTILIZATION = 0.85
smoke.ENABLE_EXPERT_PARALLEL = True
smoke.ENABLE_A5_PACKED_C8_DSA = True
smoke.ENABLE_MTP = True
smoke.MTP_NUM_SPECULATIVE_TOKENS = 3
smoke.ENABLE_CHUNKED_PREFILL = False
smoke.ENABLE_PROFILE = False

_BASE_BUILD_LLM_KWARGS = smoke.build_llm_kwargs


def build_llm_kwargs(dp_rank: int, multi_node: bool) -> dict:
    kwargs = _BASE_BUILD_LLM_KWARGS(dp_rank, multi_node)
    kwargs["speculative_config"]["draft_tensor_parallel_size"] = 8
    kwargs["additional_config"].update(
        enable_flashcomm1=False,
        enable_shared_expert_dp=False,
        enable_flashcomm2_parallel_size=0,
        enable_reduce_sample=False,
    )
    kwargs["compilation_config"]["pass_config"] = {"enable_sp": False}
    return kwargs


smoke.build_llm_kwargs = build_llm_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp-eager", action="store_true", help="仅关闭 MTP 图，target 仍入图")
    args = parser.parse_args()
    smoke.ENABLE_MTP_GRAPH = not args.mtp_eager
    smoke.RESULT_JSON = "tp8_mtp_eager.json" if args.mtp_eager else "tp8_mtp_graph.json"
    smoke.run_node_dp(argparse.Namespace(node_size=1, node_rank=0, master_addr="", master_port=0, sync_port=0))


if __name__ == "__main__":
    main()
