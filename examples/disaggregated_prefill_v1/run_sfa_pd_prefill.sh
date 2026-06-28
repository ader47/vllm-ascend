#!/bin/bash
# =============================================================================
# SFA PD-disaggregated CPU-offload — Prefill (P) node launcher.
# Connector: SFAPDCpuOffloadConnector, kv_role = kv_producer.
#
# P computes prefill KV layer-wise and pushes it to D via Mooncake RDMA:
#   - indexer KV  -> D HBM
#   - main MLA KV -> D CPU pool
# (the split destination is metadata-driven on P; no sender-side branching.)
#
# Model: use an MLA + sparse model such as DeepSeek-V3.2 — SFA backend is
# selected automatically when (use_mla, use_sparse) == (True, True)
# (see platform.py:get_attn_backend_cls). No extra enable flag needed.
#
# Bring-up notes for this connector (still pending hardware verification):
#   - 2 MiB alignment of the D-side CPU pool (worker.py NOTE)
#   - per-layer 5-tuple -> LayerMetadata packing correctness (watch PDDBG log)
#   - P-side buffer-reuse gating (wait_for_layer_send) wiring
# =============================================================================
set -euo pipefail

# ---------------------------- CONFIG (edit me) -------------------------------
MODEL_PATH="/path/to/DeepSeek-V3.2"     # MLA + sparse (SFA) model
SERVE_HOST="0.0.0.0"                    # external HTTP listen addr (proxy connects here)
SERVE_PORT=8100                         # external HTTP port
TP_SIZE=4                               # tensor parallel size (P and D MUST match)
VISIBLE_DEVICES=0,1,2,3                 # NPU cards for the P node (must NOT overlap D)
NET_IFACE="lo"                          # NIC for gloo/tp/hccl; multi-host -> real iface

# Mooncake gives each TP rank its own ZMQ port = KV_PORT + tp_rank. So with
# TP_SIZE=4 this node occupies KV_PORT+0..+3 (here 20001-20004). On a single
# host D's KV_PORT must be >= this + TP_SIZE to avoid a port collision
# (run_sfa_pd_decode.sh uses 20005). Multi-host (different IPs) can reuse ports.
KV_PORT=20001                           # Mooncake side-channel base port (occupies +0..+3)
KV_RANK=0                               # P node kv_rank (P=0, D=1; inert for mooncake)

# P MUST run with use_offload=false: the producer worker inherits mooncake's
# register_kv_caches, which expects standard paged KV tensors (not the 5-tuple
# that only exists when use_offload=true). Default is false; set explicitly as a
# guard against misconfiguration.
ADDITIONAL_CONFIG='{"use_offload": false}'
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
  --additional-config "$ADDITIONAL_CONFIG" \
  --kv-transfer-config "{
    \"kv_connector\": \"SFAPDCpuOffloadConnector\",
    \"kv_buffer_device\": \"npu\",
    \"kv_role\": \"kv_producer\",
    \"kv_parallel_size\": 1,
    \"kv_port\": ${KV_PORT},
    \"kv_rank\": ${KV_RANK},
    \"kv_connector_extra_config\": {\"use_layerwise\": true}
  }"
