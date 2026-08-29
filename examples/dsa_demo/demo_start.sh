#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run on EACH Ascend950DT host: bash demo_start.sh IMAGE CONTAINER_NAME
set -Eeuo pipefail

die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }
warn() { printf '[WARN] %s\n' "$*" >&2; }

[[ $# -eq 2 ]] || die "Usage: bash $0 <image_id_or_name> <container_name>"
IMAGE=$1
NAME=$2
[[ "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "Invalid container name: $NAME"
command -v docker >/dev/null || die "Run this script on the Docker host"
docker image inspect "$IMAGE" >/dev/null || die "Image is not available locally: $IMAGE"
if docker container inspect "$NAME" >/dev/null 2>&1; then
    die "Container $NAME already exists. Use a new name; existing containers are kept."
fi

# Preserve the image's Python/CANN paths and make diagnostic tools visible.
IMAGE_PATH=$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" \
    | sed -n 's/^PATH=//p')
TOOLS_DIR=/opt/dsa-host-tools
CONTAINER_PATH="$TOOLS_DIR:${IMAGE_PATH}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RUN_ARGS=(
    --name "$NAME" --hostname dsa-host
    --network host --pid host --shm-size 256g --ulimit memlock=-1:-1
    --privileged --security-opt seccomp=unconfined --security-opt apparmor=unconfined
    --runtime runc --user root --workdir /home
    --env "PATH=$CONTAINER_PATH"
)

bind_path() {
    local source=$1 destination=$2 mode=${3:-ro}
    [[ -e "$source" ]] || die "Host path is missing: $source"
    [[ "$source" != *,* && "$destination" != *,* ]] || die "Comma in mount path: $source"
    local spec="type=bind,src=$source,dst=$destination"
    [[ "$mode" == ro ]] && spec+=",readonly"
    RUN_ARGS+=(--mount "$spec")
}

find_host_tool() {
    local name=$1 candidate
    candidate=$(type -P "$name" || true)
    for candidate in "$candidate" \
        "/usr/bin/$name" "/usr/sbin/$name" "/bin/$name" "/sbin/$name" \
        "/usr/local/bin/$name" "/usr/local/sbin/$name" \
        "/usr/local/Ascend/driver/tools/$name"; do
        if [[ -f "$candidate" && -x "$candidate" ]]; then
            readlink -f -- "$candidate"
            return 0
        fi
    done
    return 1
}

mount_host_tool() {
    local tool=$1 source
    local destination=${2:-$TOOLS_DIR/$tool}
    source=$(find_host_tool "$tool") || die "Tool is missing on the host: $tool"
    bind_path "$source" "$destination"
    printf '[INFO] host tool: %s -> %s\n' "$source" "$destination"
}

for device in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc /dev/ummu /dev/uburma; do
    [[ -e "$device" ]] || die "Host NPU device is missing: $device"
    RUN_ARGS+=(--device "$device:$device")
done

# These must be the CURRENT host's 950 communication configuration.
for config in /lib/route.conf /etc/hccl_rootinfo.json; do
    [[ -f "$config" && -r "$config" && -s "$config" ]] \
        || die "Need a readable, nonempty host config file: $config"
    bind_path "$config" "$config"
done
for directory in /etc/hixlep /usr/local/Ascend/driver /usr/local/dcmi /usr/lib64 /usr/local/sbin; do
    [[ -d "$directory" ]] || die "Host directory is missing: $directory"
    bind_path "$directory" "$directory"
done
# Retain the original A5 host-library setup, read-only. No yum/pip installs here.
for config in /etc/ascend_install.info /etc/hccn.conf; do
    if [[ -f "$config" ]]; then
        bind_path "$config" "$config"
    else
        warn "Optional host config not found: $config"
    fi
done
bind_path /home /home rw
for directory in /mnt /kvcache /models /root/host; do
    if [[ -d "$directory" ]]; then
        bind_path "$directory" "$directory" rw
    else
        warn "Optional data directory not found: $directory"
    fi
done
if [[ -d /var/log/npu ]]; then
    bind_path /var/log/npu /usr/slog rw
fi

# Inspect the image in a temporary container, with no host files or NPUs mounted.
# Only missing generic commands are supplied from the host; keep image Python.
TOOLS=(ip hostname ifconfig route ping ss ps lscpu
       awk grep sed tee mktemp date uname mkdir rm cat ls df find)
MISSING_TOOLS=$(docker run --rm --network none --runtime runc \
    --env "PATH=$CONTAINER_PATH" --entrypoint /bin/bash "$IMAGE" -c '
        command -v python >/dev/null || { echo "Image has no python command" >&2; exit 1; }
        for tool in "$@"; do
            command -v "$tool" >/dev/null || printf "%s\n" "$tool"
        done
        exit 0
    ' -- "${TOOLS[@]}") || die "Cannot inspect commands in image $IMAGE"
while IFS= read -r tool; do
    [[ -z "$tool" ]] || mount_host_tool "$tool"
done <<<"$MISSING_TOOLS"

# Driver utilities should match the mounted host driver.
mount_host_tool npu-smi /usr/local/bin/npu-smi
mount_host_tool hccn_tool /usr/bin/hccn_tool
if find_host_tool urma_admin >/dev/null; then
    mount_host_tool urma_admin /usr/bin/urma_admin
else
    warn "urma_admin is unavailable on this host (optional diagnostic tool)"
fi

printf '[INFO] Creating %s from %s; host libraries are mounted read-only.\n' "$NAME" "$IMAGE"
docker run -dit "${RUN_ARGS[@]}" --entrypoint /bin/bash "$IMAGE" -i

# Execute tools to catch missing shared libraries, not only missing command names.
if ! docker exec -i "$NAME" /bin/bash -s <<'CONTAINER_CHECK'
set -eo pipefail
for env_file in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [[ -f "$env_file" ]]; then source "$env_file"; fi
done
export PATH="/opt/dsa-host-tools:$PATH"
for tool in ip hostname ifconfig route ping ss ps lscpu awk grep sed tee mktemp \
    date uname mkdir rm cat ls df find python npu-smi hccn_tool; do
    command -v "$tool" || { printf '[FAIL] missing command: %s\n' "$tool" >&2; exit 1; }
done
for tool in awk grep sed tee mktemp date uname mkdir rm cat ls df find lscpu ps; do
    "$tool" --version >/dev/null
done
ip -V
ip -o -4 addr show
hostname
ifconfig -a >/dev/null
route -n >/dev/null
ping -V
ss -ltn >/dev/null
ps -p 1 -o comm=
df -h /dev/shm
printf '[INFO] memlock_kib=%s\n' "$(ulimit -l)"
npu-smi info
python -c 'import sys, torch, torch_npu; print("[INFO] python=", sys.executable, "torch=", torch.__version__, "torch_npu=", torch_npu.__version__); count = torch.npu.device_count(); print("[INFO] visible_npus=", count); assert torch.npu.is_available() and count >= 8, "Need 8 visible NPUs"'
printf '[PASS] Container tools and basic NPU runtime are ready; cross-node HCCL/MC2 still needs testing.\n'
CONTAINER_CHECK
then
    die "Container $NAME was created, but self-check failed. It is kept for inspection; paste the errors above."
fi
printf '[INFO] Enter the container: docker exec -it %q bash\n' "$NAME"
