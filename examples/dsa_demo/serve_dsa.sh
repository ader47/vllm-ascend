#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.1 DSA 稀疏卸载在线服务最小启动脚本。
# 修改“用户配置”后直接执行：
#
#   bash examples/dsa_demo/serve_dsa.sh

set -euo pipefail

# =========================
# 用户配置
# =========================

MODEL_PATH="/mnt/kv_dpc/weight/GLM-5.1-w4a8"
SERVED_MODEL_NAME="glm-5.1-dsa"
RUN_MODE="eager"  # eager / graph

HOST="0.0.0.0"
PORT="8000"
API_KEY="EMPTY"

TENSOR_PARALLEL_SIZE="16"
MAX_NUM_SEQS="4"
MAX_MODEL_LEN="131072"
MAX_NUM_BATCHED_TOKENS="16384"
GPU_MEMORY_UTILIZATION="0.90"
ENABLE_CHUNKED_PREFILL="true"

DSA_SPARSE_ACTIVATION_TOKENS="6144"
DSA_INDEXER_MLA_BLOCK_RATIO="3"
DSA_MAX_ACTIVE_REQS="256"
DSA_HOT_CPU_BLOCK_MULTIPLE="3.0"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE="200"
export OMP_NUM_THREADS="10"
export OMP_PROC_BIND="false"
export PYTHONHASHSEED="114514"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_MLAPO="1"
export VLLM_LOGGING_LEVEL="INFO"

case "${RUN_MODE}" in
    eager)
        ENABLE_DSA_GRAPH="false"
        EXECUTION_ARGS=(
            --enforce-eager
        )
        ;;
    graph)
        ENABLE_DSA_GRAPH="true"
        EXECUTION_ARGS=(
            --no-enforce-eager
            --compilation-config
            '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
        )
        ;;
    *)
        echo "RUN_MODE must be eager or graph, got: ${RUN_MODE}" >&2
        exit 2
        ;;
esac

if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
    PREFILL_ARGS=(
        --enable-chunked-prefill
        --scheduler-reserve-full-isl
    )
else
    PREFILL_ARGS=(
        --no-enable-chunked-prefill
    )
fi

ADDITIONAL_CONFIG="$(
    cat <<JSON
{"dsa_sparse_config":{"enabled":true,"split_indexer_cache":true,"indexer_mla_block_ratio":${DSA_INDEXER_MLA_BLOCK_RATIO},"sparse_activation_tokens":${DSA_SPARSE_ACTIVATION_TOKENS},"prompt_budget_thresholds":[32768,65536],"resident_budget_tokens":[6144,10240,12288],"max_active_reqs":${DSA_MAX_ACTIVE_REQS},"hot_cpu_block_multiple":${DSA_HOT_CPU_BLOCK_MULTIPLE},"enable_row_mode_decode_graph":${ENABLE_DSA_GRAPH},"trace_points":{"enabled":false,"points":["first_sample"],"ranks":[0]}}}
JSON
)"

echo "[dsa-online] mode=${RUN_MODE} model=${MODEL_PATH}"
echo "[dsa-online] endpoint=http://${HOST}:${PORT}/v1"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --quantization ascend \
    --seed 1024 \
    --enable-expert-parallel \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size 128 \
    --no-async-scheduling \
    --stream-interval 1 \
    --generation-config vllm \
    --reasoning-parser glm45 \
    --additional-config "${ADDITIONAL_CONFIG}" \
    "${PREFILL_ARGS[@]}" \
    "${EXECUTION_ARGS[@]}"
