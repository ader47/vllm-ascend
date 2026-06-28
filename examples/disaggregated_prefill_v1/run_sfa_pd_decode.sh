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
MODEL_PATH="/path/to/DeepSeek-V3.2"     # MUST match the P node
SERVE_HOST="0.0.0.0"                    # external HTTP listen addr (proxy connects here)
SERVE_PORT=8200                         # external HTTP port
TP_SIZE=1                               # tensor parallel size
VISIBLE_DEVICES=1                       # NPU cards for the D node (use a different card than P)
NET_IFACE="lo"                          # NIC for gloo/tp/hccl; multi-host -> real iface

KV_PORT=20002                           # Mooncake side-channel base port (different from P)
KV_RANK=1                               # D node kv_rank (P=0, D=1)
# ----------------------------------------------------------------------------

export HCCL_IF_IP="${HCCL_IF_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export PHYSICAL_DEVICES="${PHYSICAL_DEVICES:-$VISIBLE_DEVICES}"

exec vllm serve "$MODEL_PATH" \
  --host "$SERVE_HOST" \
  --port "$SERVE_PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config "{
    \"kv_connector\": \"SFAPDCpuOffloadConnector\",
    \"kv_buffer_device\": \"npu\",
    \"kv_role\": \"kv_consumer\",
    \"kv_parallel_size\": 1,
    \"kv_port\": ${KV_PORT},
    \"kv_rank\": ${KV_RANK},
    \"kv_connector_extra_config\": {\"use_layerwise\": true}
  }"
