#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly NNODES=2
readonly LOCAL_WORLD_SIZE=8
readonly EXPECTED_WORLD_SIZE=16

NODE0_IP=""
NODE1_IP=""
NODE_RANK="auto"
MASTER_PORT=29688
TIMEOUT_SECONDS=300
PAYLOAD_MIB=8
RUN_MC2=1
LOG_DIR="./two_node_preflight_logs"
IFACE=""

usage() {
    cat <<'USAGE'
Run the same command on both 8-NPU nodes:

  bash examples/dsa_demo/check_two_node_dp16_ep16.sh \
    --node0-ip 141.61.141.22 \
    --node1-ip 141.61.141.26

The script detects whether it is node 0 or node 1 from local IPv4 addresses,
then launches an HCCL world of 2 nodes x 8 NPUs. It checks all_reduce,
all_to_all_single, and the default MC2 dispatch/combine path.

Options:
  --node0-ip IP          Node 0 host IP used as torchrun master (required)
  --node1-ip IP          Node 1 host IP (required)
  --node-rank auto|0|1   Override automatic rank detection (default: auto)
  --iface NAME           Override interface detection for HCCL sockets
  --master-port PORT     torchrun rendezvous port (default: 29688)
  --timeout-seconds SEC  Distributed initialization timeout (default: 300)
  --payload-mib MIB      all_reduce diagnostic payload per rank (default: 8)
  --skip-mc2             Run only HCCL all_reduce/all_to_all
  --log-dir DIR          Directory for the per-node log
  -h, --help             Show this help
USAGE
}

die() {
    printf '[FATAL] %s\n' "$*" >&2
    exit 2
}

warn() {
    printf '[WARN] %s\n' "$*"
}

section() {
    printf '\n========== %s ==========\n' "$1"
}

require_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_ipv4() {
    local ip=$1
    local octets
    local octet
    IFS='.' read -r -a octets <<<"$ip"
    [[ ${#octets[@]} -eq 4 ]] || return 1
    for octet in "${octets[@]}"; do
        [[ "$octet" =~ ^[0-9]+$ ]] || return 1
        ((10#$octet >= 0 && 10#$octet <= 255)) || return 1
    done
}

while (($# > 0)); do
    case "$1" in
        --node0-ip)
            require_value "$@"
            NODE0_IP=$2
            shift 2
            ;;
        --node1-ip)
            require_value "$@"
            NODE1_IP=$2
            shift 2
            ;;
        --node-rank)
            require_value "$@"
            NODE_RANK=$2
            shift 2
            ;;
        --iface)
            require_value "$@"
            IFACE=$2
            shift 2
            ;;
        --master-port)
            require_value "$@"
            MASTER_PORT=$2
            shift 2
            ;;
        --timeout-seconds)
            require_value "$@"
            TIMEOUT_SECONDS=$2
            shift 2
            ;;
        --payload-mib)
            require_value "$@"
            PAYLOAD_MIB=$2
            shift 2
            ;;
        --skip-mc2)
            RUN_MC2=0
            shift
            ;;
        --log-dir)
            require_value "$@"
            LOG_DIR=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$NODE0_IP" ]] || die "--node0-ip is required"
[[ -n "$NODE1_IP" ]] || die "--node1-ip is required"
is_ipv4 "$NODE0_IP" || die "invalid --node0-ip: $NODE0_IP"
is_ipv4 "$NODE1_IP" || die "invalid --node1-ip: $NODE1_IP"
[[ "$NODE0_IP" != "$NODE1_IP" ]] || die "node0-ip and node1-ip must differ"
[[ "$NODE_RANK" == "auto" || "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] \
    || die "--node-rank must be auto, 0, or 1"
is_positive_integer "$MASTER_PORT" || die "master-port must be a positive integer"
((MASTER_PORT <= 65535)) || die "master-port must be <= 65535"
is_positive_integer "$TIMEOUT_SECONDS" || die "timeout-seconds must be a positive integer"
is_positive_integer "$PAYLOAD_MIB" || die "payload-mib must be a positive integer"

command -v ip >/dev/null 2>&1 || die "the ip command is required"

PYTHON_BIN=${PYTHON_BIN:-python}
command -v "$PYTHON_BIN" >/dev/null 2>&1 \
    || die "Python executable not found: $PYTHON_BIN"

LOCAL_IPV4=$(
    ip -o -4 addr show \
        | awk '{split($4, address, "/"); print address[1]}'
)
HAS_NODE0=0
HAS_NODE1=0
grep -Fxq "$NODE0_IP" <<<"$LOCAL_IPV4" && HAS_NODE0=1
grep -Fxq "$NODE1_IP" <<<"$LOCAL_IPV4" && HAS_NODE1=1

if [[ "$NODE_RANK" == "auto" ]]; then
    if ((HAS_NODE0 == 1 && HAS_NODE1 == 0)); then
        NODE_RANK=0
    elif ((HAS_NODE0 == 0 && HAS_NODE1 == 1)); then
        NODE_RANK=1
    else
        die "cannot detect node rank: local IPv4 addresses are [$LOCAL_IPV4]"
    fi
fi

if [[ "$NODE_RANK" == "0" ]]; then
    LOCAL_IP=$NODE0_IP
    PEER_IP=$NODE1_IP
    ((HAS_NODE0 == 1)) || die "$NODE0_IP is not assigned to this host"
else
    LOCAL_IP=$NODE1_IP
    PEER_IP=$NODE0_IP
    ((HAS_NODE1 == 1)) || die "$NODE1_IP is not assigned to this host"
fi

if [[ -z "$IFACE" ]]; then
    IFACE=$(
        ip -o -4 addr show \
            | awk -v wanted="$LOCAL_IP" '
                {split($4, address, "/")}
                address[1] == wanted {print $2; exit}
            '
    )
fi
[[ -n "$IFACE" ]] || die "cannot find the interface for local IP $LOCAL_IP"
ip link show dev "$IFACE" >/dev/null 2>&1 \
    || die "interface does not exist: $IFACE"

mkdir -p -- "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/dp16_ep16_node${NODE_RANK}_$(hostname)_${TIMESTAMP}.log"
exec > >(tee "$LOG_FILE") 2>&1

section "CONFIG"
printf '[INFO] timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf '[INFO] hostname=%s node_rank=%s/%s\n' "$(hostname)" "$NODE_RANK" "$NNODES"
printf '[INFO] node0_ip=%s node1_ip=%s local_ip=%s peer_ip=%s\n' \
    "$NODE0_IP" "$NODE1_IP" "$LOCAL_IP" "$PEER_IP"
printf '[INFO] iface=%s master=%s:%s world=%s local_world=%s\n' \
    "$IFACE" "$NODE0_IP" "$MASTER_PORT" "$EXPECTED_WORLD_SIZE" "$LOCAL_WORLD_SIZE"
printf '[INFO] mc2=%s timeout_seconds=%s payload_mib=%s\n' \
    "$RUN_MC2" "$TIMEOUT_SECONDS" "$PAYLOAD_MIB"
printf '[INFO] ASCEND_RT_VISIBLE_DEVICES=%s ASCEND_VISIBLE_DEVICES=%s\n' \
    "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}" "${ASCEND_VISIBLE_DEVICES:-<unset>}"
printf '[INFO] log_file=%s\n' "$LOG_FILE"

section "HOST NETWORK"
printf '[INFO] local IPv4 addresses:\n%s\n' "$LOCAL_IPV4"
printf '[INFO] route to peer:\n'
ip route get "$PEER_IP" || true
if command -v ping >/dev/null 2>&1; then
    if ping -c 3 -W 2 "$PEER_IP"; then
        printf '[PASS] host ping to %s\n' "$PEER_IP"
    else
        warn "host ping failed; ICMP may be filtered, HCCL tests will still run"
    fi
else
    warn "ping is unavailable"
fi

if [[ "$NODE_RANK" == "0" ]] && command -v ss >/dev/null 2>&1; then
    if ss -ltn | awk -v suffix=":$MASTER_PORT" '$4 ~ suffix "$" {found=1} END {exit !found}'; then
        die "master port $MASTER_PORT is already listening on node 0"
    fi
fi

section "HOST AND ASCEND INVENTORY"
uname -a || true
printf '[INFO] memlock_kib=%s\n' "$(ulimit -l)"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E '^(Architecture|CPU\(s\)|NUMA node\(s\)|Model name):' || true
fi
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
else
    warn "npu-smi is unavailable"
fi

HCCN_TOOL=$(command -v hccn_tool 2>/dev/null || true)
if [[ -z "$HCCN_TOOL" && -x /usr/local/Ascend/driver/tools/hccn_tool ]]; then
    HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
fi
if [[ -n "$HCCN_TOOL" ]]; then
    for ((device_id = 0; device_id < LOCAL_WORLD_SIZE; device_id++)); do
        printf '[INFO] hccn device=%s IP configuration:\n' "$device_id"
        "$HCCN_TOOL" -i "$device_id" -ip -g || true
        printf '[INFO] hccn device=%s link state:\n' "$device_id"
        "$HCCN_TOOL" -i "$device_id" -link -g || true
    done
else
    warn "hccn_tool is unavailable; device IP/link inventory was not collected"
fi

for version_file in \
    /usr/local/Ascend/driver/version.info \
    /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
    /usr/local/Ascend/ascend-toolkit/latest/version.info; do
    if [[ -r "$version_file" ]]; then
        printf '[INFO] %s:\n' "$version_file"
        sed -n '1,80p' "$version_file"
    fi
done

section "PYTORCH NPU RUNTIME"
"$PYTHON_BIN" - "$LOCAL_WORLD_SIZE" <<'PY'
import importlib.metadata
import sys

required_devices = int(sys.argv[1])

import torch
import torch_npu


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


print(f"[INFO] python={sys.version.split()[0]} executable={sys.executable}")
print(f"[INFO] torch={torch.__version__} path={torch.__file__}")
print(f"[INFO] torch_npu={torch_npu.__version__} path={torch_npu.__file__}")
print(f"[INFO] vllm={package_version('vllm')}")
print(f"[INFO] vllm-ascend={package_version('vllm-ascend')}")
print(f"[INFO] npu_available={torch.npu.is_available()}")
device_count = torch.npu.device_count()
print(f"[INFO] visible_npu_count={device_count}")
print(
    "[INFO] mc2_dispatch_v2="
    f"{hasattr(torch_npu, 'npu_moe_distribute_dispatch_v2')} "
    "mc2_combine_v2="
    f"{hasattr(torch_npu, 'npu_moe_distribute_combine_v2')}"
)
if not torch.npu.is_available() or device_count < required_devices:
    raise SystemExit(
        f"[FATAL] need at least {required_devices} visible NPUs, got {device_count}"
    )
print(f"[PASS] at least {required_devices} NPUs are visible")
PY

export HCCL_IF_IP=$LOCAL_IP
export HCCL_SOCKET_IFNAME=$IFACE
export GLOO_SOCKET_IFNAME=$IFACE
export TP_SOCKET_IFNAME=$IFACE
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-120}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-200}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

section "DISTRIBUTED ENVIRONMENT"
printf '[INFO] HCCL_IF_IP=%s\n' "$HCCL_IF_IP"
printf '[INFO] HCCL_SOCKET_IFNAME=%s\n' "$HCCL_SOCKET_IFNAME"
printf '[INFO] GLOO_SOCKET_IFNAME=%s\n' "$GLOO_SOCKET_IFNAME"
printf '[INFO] TP_SOCKET_IFNAME=%s\n' "$TP_SOCKET_IFNAME"
printf '[INFO] HCCL_OP_EXPANSION_MODE=%s\n' "$HCCL_OP_EXPANSION_MODE"
printf '[INFO] HCCL_CONNECT_TIMEOUT=%s HCCL_EXEC_TIMEOUT=%s HCCL_BUFFSIZE=%s\n' \
    "$HCCL_CONNECT_TIMEOUT" "$HCCL_EXEC_TIMEOUT" "$HCCL_BUFFSIZE"

if command -v torchrun >/dev/null 2>&1; then
    LAUNCHER=(torchrun)
else
    LAUNCHER=("$PYTHON_BIN" -m torch.distributed.run)
fi

TEST_FILE=$(mktemp "${TMPDIR:-/tmp}/ascend950_dp16_ep16_XXXXXX.py")
cleanup() {
    rm -f -- "$TEST_FILE"
}
trap cleanup EXIT

cat >"$TEST_FILE" <<'PY'
import argparse
import os
import socket
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--payload-mib", type=int, required=True)
    parser.add_argument("--run-mc2", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def global_check(
    condition: bool,
    label: str,
    device: torch.device,
    rank: int,
) -> None:
    status = torch.tensor(
        [1 if condition else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    passed = int(status.item()) == 1
    if rank == 0:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}", flush=True)
    if not passed:
        raise RuntimeError(f"global validation failed: {label}")


def test_all_reduce(
    device: torch.device,
    rank: int,
    world_size: int,
    payload_mib: int,
) -> None:
    value = torch.tensor([rank + 1.0], dtype=torch.float32, device=device)
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    global_check(
        abs(float(value.item()) - expected) < 1e-5,
        f"HCCL all_reduce scalar expected={expected:g}",
        device,
        rank,
    )

    element_count = payload_mib * 1024 * 1024 // 4
    payload = torch.ones(element_count, dtype=torch.float32, device=device)
    iterations = 3
    dist.barrier()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        payload.fill_(1.0)
        dist.all_reduce(payload)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    global_check(
        bool(torch.all(payload == world_size).item()),
        f"HCCL all_reduce {payload_mib} MiB payload correctness",
        device,
        rank,
    )
    max_elapsed = torch.tensor([elapsed], dtype=torch.float32, device=device)
    dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        seconds = float(max_elapsed.item())
        payload_rate = payload_mib * iterations / max(seconds, 1e-12)
        print(
            f"[METRIC] all_reduce payload={payload_mib}MiB iterations={iterations} "
            f"max_elapsed_s={seconds:.6f} payload_rate_MiB_s={payload_rate:.2f}",
            flush=True,
        )


def test_all_to_all(
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    items_per_peer = 1024
    destinations = torch.arange(
        world_size,
        dtype=torch.float32,
        device=device,
    ).repeat_interleave(items_per_peer)
    send = destinations + rank * 1000
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send)
    expected = torch.cat(
        [
            torch.full(
                (items_per_peer,),
                source * 1000 + rank,
                dtype=torch.float32,
                device=device,
            )
            for source in range(world_size)
        ]
    )
    global_check(
        bool(torch.equal(received, expected)),
        "HCCL all_to_all_single cross-rank routing",
        device,
        rank,
    )


def test_mc2(
    device: torch.device,
    rank: int,
    world_size: int,
    local_world_size: int,
) -> None:
    has_dispatch = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
    has_combine = hasattr(torch_npu, "npu_moe_distribute_combine_v2")
    global_check(
        has_dispatch and has_combine,
        "MC2 dispatch_v2/combine_v2 APIs available",
        device,
        rank,
    )

    ep_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="hccl",
    )
    group_rank = dist.get_rank(group=ep_group)
    backend = ep_group._get_backend(torch.device("npu"))
    group_name = backend.get_hccl_comm_name(group_rank)

    token_count = 2
    # The Ascend 950 MC2 tiling check requires H in [1024, 8192].
    hidden_size = 1024
    top_k = 1
    expert_count = world_size
    peer_rank = (rank + local_world_size) % world_size
    x = torch.full(
        (token_count, hidden_size),
        rank + 1.0,
        dtype=torch.bfloat16,
        device=device,
    )
    expert_ids = torch.full(
        (token_count, top_k),
        peer_rank,
        dtype=torch.int32,
        device=device,
    )
    expert_scales = torch.ones(
        (token_count, top_k),
        dtype=torch.float32,
        device=device,
    )

    # Match simple_prompt_test_dp16.py: do not force hierarchy, which the
    # Ascend 950 communication-context path rejects during tiling.
    dispatch_outputs = torch_npu.npu_moe_distribute_dispatch_v2(
        x=x,
        expert_ids=expert_ids,
        expert_scales=expert_scales,
        scales=None,
        group_ep=group_name,
        ep_world_size=world_size,
        ep_rank_id=group_rank,
        moe_expert_num=expert_count,
        group_tp=group_name,
        tp_world_size=1,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=token_count * world_size,
        expert_token_nums_type=1,
    )
    (
        expand_x,
        _,
        assist_info_for_combine,
        _,
        ep_recv_counts,
        tp_recv_counts,
        expand_scales,
    ) = dispatch_outputs[:7]

    combined = torch_npu.npu_moe_distribute_combine_v2(
        expand_x=expand_x,
        expert_ids=expert_ids,
        expert_scales=expert_scales,
        assist_info_for_combine=assist_info_for_combine,
        ep_send_counts=ep_recv_counts,
        group_ep=group_name,
        ep_world_size=world_size,
        ep_rank_id=group_rank,
        moe_expert_num=expert_count,
        tp_send_counts=tp_recv_counts,
        expand_scales=expand_scales,
        group_tp=group_name,
        tp_world_size=1,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_rank_num=0,
        global_bs=token_count * world_size,
        comm_quant_mode=0,
    )
    torch.npu.synchronize()
    max_error = float((combined.float() - x.float()).abs().max().item())
    global_check(
        tuple(combined.shape) == tuple(x.shape) and max_error <= 1e-3,
        "MC2 default cross-node dispatch/combine identity",
        device,
        rank,
    )
    if rank == 0:
        print(
            "[INFO] MC2 routing forced every rank to an expert on the other node",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group(ep_group)


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    initialized = False
    try:
        if world_size != args.expected_world_size:
            raise RuntimeError(
                f"expected world_size={args.expected_world_size}, got {world_size}"
            )
        if local_world_size != args.local_world_size:
            raise RuntimeError(
                f"expected local_world_size={args.local_world_size}, "
                f"got {local_world_size}"
            )
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
        dist.init_process_group(
            backend="hccl",
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        initialized = True
        print(
            f"[RANK] global={rank} local={local_rank} "
            f"host={socket.gethostname()} device={device}",
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            print(
                f"[PASS] initialized HCCL world_size={world_size} "
                f"local_world_size={local_world_size}",
                flush=True,
            )

        test_all_reduce(device, rank, world_size, args.payload_mib)
        dist.barrier()
        test_all_to_all(device, rank, world_size)
        dist.barrier()
        if args.run_mc2:
            test_mc2(
                device,
                rank,
                world_size,
                local_world_size,
            )
        elif rank == 0:
            print("[SKIP] MC2 default test", flush=True)
        dist.barrier()
        if rank == 0:
            print("[PASS] ALL REQUESTED DISTRIBUTED TESTS PASSED", flush=True)
    except BaseException:
        print(
            f"[FAIL][rank={rank} local_rank={local_rank}]\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

section "TWO-NODE HCCL AND MC2 TEST"
printf '[INFO] launcher='
printf '%q ' "${LAUNCHER[@]}"
printf '\n'
printf '[INFO] start this script on both nodes with the same IPs and master port\n'

set +e
"${LAUNCHER[@]}" \
    --nnodes="$NNODES" \
    --nproc-per-node="$LOCAL_WORLD_SIZE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$NODE0_IP" \
    --master-port="$MASTER_PORT" \
    "$TEST_FILE" \
    --expected-world-size "$EXPECTED_WORLD_SIZE" \
    --local-world-size "$LOCAL_WORLD_SIZE" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --payload-mib "$PAYLOAD_MIB" \
    --run-mc2 "$RUN_MC2"
TEST_STATUS=$?
set -e

section "FINAL RESULT"
if ((TEST_STATUS == 0)); then
    printf '[PASS] node=%s completed DP16/EP16 communication preflight\n' "$NODE_RANK"
    printf '[PASS] HCCL all_reduce + all_to_all + requested MC2 checks passed\n'
else
    printf '[FAIL] node=%s distributed preflight exited with status=%s\n' \
        "$NODE_RANK" "$TEST_STATUS"
    printf '[INFO] preserve both node logs; the first [FAIL] is usually the root cause\n'
fi
printf '[INFO] log_file=%s\n' "$LOG_FILE"
exit "$TEST_STATUS"
