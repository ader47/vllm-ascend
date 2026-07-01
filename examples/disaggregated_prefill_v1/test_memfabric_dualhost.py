#!/usr/bin/env python
# Tests cross-host memfabric batch_transfer_sync_read (the PD pull path).
# Based on memfabric_hybrid/examples/transfer/test_transfer_engine.py
#
# Architecture: P (Prefill) is the store server (store_server_role="Prefill").
# D (Decode) connects to P's store. P starts FIRST.
#
# USAGE (2 servers):
#   # Server A (Prefill/P — start FIRST, runs the config store):
#   python test_memfabric_dualhost.py --role Prefill \
#       --store-url tcp://<P_IP>:31001 \
#       --src-unique-id <P_IP>:31001
#
#   # Server B (Decode/D — start SECOND, connects to P's store):
#   python test_memfabric_dualhost.py --role Decode \
#       --store-url tcp://<P_IP>:31001 \
#       --src-unique-id <D_IP>:31000 \
#       --peer-unique-id <P_IP>:31001
#
# Flow: P allocates a random tensor + registers. D reads it via
# batch_transfer_sync_read and checks the sum is non-zero.

import argparse
import time
import torch
import torch_npu
from memfabric_hybrid import TransferEngine, set_log_level, set_conf_store_tls

SHAPE = (10, 50, 40, 20, 60)


def main():
    parser = argparse.ArgumentParser(description="Dual-host memfabric read test")
    parser.add_argument("--role", type=str, required=True, choices=["Decode", "Prefill"])
    parser.add_argument("--src-unique-id", type=str, required=True,
                        help="This node's unique_id (IP:PORT)")
    parser.add_argument("--store-url", type=str, required=True,
                        help="Config store URL (ALWAYS P's address — P is the store server)")
    parser.add_argument("--peer-unique-id", type=str, default=None,
                        help="Decode only: P's unique_id to read from")
    parser.add_argument("--npu-id", type=int, default=0)
    parser.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    torch.npu.set_device(device=args.npu_id)
    set_log_level(args.log_level)
    set_conf_store_tls(False, "")

    engine = TransferEngine()

    if args.role == "Prefill":
        # P is the store server (store_server_role="Prefill").
        ret = engine.initialize(
            args.store_url, args.src_unique_id, args.role, args.npu_id,
            store_server_role="Prefill",
        )
    else:
        # D connects to P's store.
        ret = engine.initialize(
            args.store_url, args.src_unique_id, args.role, args.npu_id,
        )
    if ret != 0:
        raise RuntimeError(f"engine.initialize failed ret={ret}")
    print(f"[{args.role}] Engine init OK: store={args.store_url} uid={args.src_unique_id}")

    buf = torch.zeros(SHAPE, dtype=torch.float16, device="npu")
    buf_bytes = buf.element_size() * buf.numel()
    engine.register_memory(buf.data_ptr(), buf_bytes)
    print(f"[{args.role}] Registered: ptr={hex(buf.data_ptr())} len={buf_bytes}")

    if args.role == "Decode":
        if not args.peer_unique_id:
            raise ValueError("Decode needs --peer-unique-id (P's unique_id)")
        run_decode(engine, buf, args)
    else:
        run_prefill(engine, buf, args)


def run_decode(engine, buf, args):
    """D: wait for P to register, then read P's buffer."""
    print(f"[Decode] Waiting 10s for P ({args.peer_unique_id}) to register...")
    time.sleep(10)

    total_bytes = buf.element_size() * buf.numel()
    print(f"[Decode] batch_transfer_sync_read: peer={args.peer_unique_id} "
          f"local={hex(buf.data_ptr())} len={total_bytes}")
    ret = engine.batch_transfer_sync_read(
        args.peer_unique_id,
        [buf.data_ptr()],
        [0],  # P's registered buffer starts at offset 0
        [total_bytes],
    )
    checksum = torch.sum(buf).item()
    if ret != 0:
        print(f"[Decode] FAIL: ret={ret}  sum={checksum:.2f}")
    elif abs(checksum) > 0.01:
        print(f"[Decode] SUCCESS: ret=0  sum={checksum:.2f}  DATA RECEIVED from P!")
    else:
        print(f"[Decode] WARNING: ret=0 but sum={checksum:.2f} (still zeros)")

    while True:
        time.sleep(10)
        print(f"[Decode] alive, sum={torch.sum(buf).item():.2f}")


def run_prefill(engine, buf, args):
    """P: fill buffer with random data, wait for D to read."""
    buf.copy_(torch.randn(SHAPE, dtype=torch.float16, device="npu"))
    checksum = torch.sum(buf).item()
    print(f"[Prefill] Filled buffer with random data, sum={checksum:.2f}")
    print(f"[Prefill] Waiting for D to read... (Ctrl+C to stop)")

    while True:
        time.sleep(10)
        print(f"[Prefill] alive, sum={torch.sum(buf).item():.2f}")


if __name__ == "__main__":
    main()
