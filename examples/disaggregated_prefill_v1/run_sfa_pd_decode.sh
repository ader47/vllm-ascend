#!/bin/bash
# =============================================================================
# SFA PD-disaggregated CPU-offload — Decode (D) node launcher.
# Connector: SFAPDCpuOffloadConnector, kv_role = kv_consumer.
#
# D-side worker composes SFAKVOffloadWorker (LRU H2D load + CPU pool) and:
#   - allocates main-MLA CPU blocks one-shot (full prompt) in the scheduler
#   - registers indexer NPU + main MLA CPU pool with Mooncake in ONE call
#   - advertises per-layer split base addrs via KVCacheRecvingLayerThread
# The remote P then RDMA-writes indexer KV -> HBM, main MLA KV -> CPU pool.
# D reuses the existing SFA LRU-resident H2D load path unchanged.
#
# START ORDER: launch D first (its recving thread must be up before P can
# GET_META its layer base addrs), then P, then the proxy.
# =============================================================================
set -euo pipefail

# ---------------------------- CONFIG (edit me) -------------------------------
MODEL_PATH="/mnt/weight/GLM-5.2-W4A8-0628"     # MUST match the P node
SERVE_HOST="80.5.17.112"                    # external HTTP listen addr (proxy connects here)
SERVE_PORT=8200                         # external HTTP port
TP_SIZE=16                               # tensor parallel size
VISIBLE_DEVICES=0,1,2,3                       # NPU cards for the D node (use a different card than P)
NET_IFACE="enp48s3u1u1"                          # NIC for gloo/tp/hccl; multi-host -> real iface

KV_PORT=10001                           # Mooncake side-channel base port (different from P)
KV_RANK=1                               # D node kv_rank (P=0, D=1)
export VLLM_VERSION=0.23.0
# export ASCEND_RT_VISIBLE_DEVICES=
# D MUST run with use_offload=true: it drives the SFA offload code path in the
# model runner (5-tuple kv_cache, num_offloaded_blocks, indexer_block_table, the
# LRU-resident H2D load). Without it the connector registers but the model never
# drives it. lru_resident_cache_config.{buffer_size,topk} default to 2048.
ADDITIONAL_CONFIG='{"use_offload": true, "lru_resident_cache_config": {"enabled": true, "buffer_size": 4096, "topk":2048}}'
export VLLM_ASCEND_KV_TRANSFER_BACKEND="memfabric"
export VLLM_ASCEND_MF_VERIFY="0"
export VLLM_ASCEND_SFA_DEBUG="0"
export  MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.1.2/aarch64-linux/lib64
# ----------------------------------------------------------------------------

export HCCL_IF_IP="80.5.17.112"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
# export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
# export PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-$VISIBLE_DEVICES}"

exec vllm serve "$MODEL_PATH" \
  --host "$SERVE_HOST" \
  --port "$SERVE_PORT" \
  --served-model-name glm \
  --max-num-seqs 4 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len 1048576 \
  --max-num-batched-tokens 4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --quantization ascend \
  --additional-config "$ADDITIONAL_CONFIG" \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4]}' \
  --no-enforce-eager \
  --safetensors-load-strategy 'prefetch' \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{
    \"kv_connector\": \"SFAPDCpuOffloadConnector\",
    \"kv_buffer_device\": \"npu\",
    \"kv_role\": \"kv_consumer\",
    \"kv_parallel_size\": 1,
    \"kv_port\": ${KV_PORT},
    \"kv_rank\": ${KV_RANK},
    \"kv_connector_extra_config\": {\"use_layerwise\": true}
  }"
