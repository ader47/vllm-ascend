"""Benchmark the stateful token_to_slot cache-miss topk path vs hash path.

Usage:
    python benchmarks/ops/ben_cache_miss_topk_state.py

Select specific cases by editing the CONFIGS dict or setting env vars:
    USE_ATOMIC=1 python benchmarks/ops/ben_cache_miss_topk_state.py
"""

import os
import time
import numpy as np
import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401
import vllm_ascend.platform  # noqa: F401

from vllm_ascend.ops.triton.get_topk_indices import (
    CacheMissTopKScratch,
    CacheMissTopKState,
    get_cache_miss_topk_indices_triton,
    get_cache_miss_topk_indices_triton_state,
)


def benchmark_fn(fn, num_warmup=10, num_iter=100):
    """Benchmark a function using NPU events. Returns min time in ms."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_warmup + num_iter)

    for i in range(num_warmup + num_iter):
        with torch.no_grad():
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    return float(np.amin(times[num_warmup:]))


def generate_topk_with_hit_rate(
    old_cache_slots, topk, num_reqs, hit_rate, token_limit, device
):
    """Generate new_topk with a target hit_rate against old_cache_slots.

    old_cache_slots: [num_reqs, topk] — each row contains token ids currently
                     in the cache slots, or -1 for empty.
    Returns: new_topk [num_reqs, topk]
    """
    new_topk = torch.full((num_reqs, topk), -1, dtype=torch.int64, device=device)
    for r in range(num_reqs):
        old_slot = old_cache_slots[r]
        valid_old = old_slot[old_slot >= 0]
        n_valid = valid_old.numel()
        n_hit = max(0, min(int(topk * hit_rate), n_valid))
        n_miss = topk - n_hit

        hits = valid_old[torch.randperm(n_valid, device=device)[:n_hit]]

        # Generate miss tokens that are NOT in old_cache
        miss_candidates = set()
        max_attempts = topk * 10
        attempts = 0
        while len(miss_candidates) < n_miss and attempts < max_attempts:
            cand = torch.randint(0, token_limit, (n_miss * 2,), device=device)
            # Filter out tokens already in old_slot
            old_set = set(valid_old.tolist())
            for c in cand.tolist():
                if c not in old_set and c not in miss_candidates:
                    miss_candidates.add(c)
                if len(miss_candidates) >= n_miss:
                    break
            attempts += 1
        while len(miss_candidates) < n_miss:
            # Fallback: use high token values unlikely to collide
            miss_candidates.add(token_limit - 1 - len(miss_candidates))

        miss_list = list(miss_candidates)[:n_miss]
        misses = torch.tensor(miss_list, dtype=torch.int64, device=device)
        new_topk[r] = torch.cat([hits, misses])[:topk]

    return new_topk


def build_full_cache_state(topk, num_reqs, token_limit, device):
    """Build an initial cache state where all slots are filled with distinct tokens."""
    slot_to_token = torch.full((num_reqs, topk), -1, dtype=torch.int64, device=device)
    token_to_slot = torch.full((num_reqs, token_limit), -1, dtype=torch.int32, device=device)
    slot_stamp = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)

    for r in range(num_reqs):
        tokens = torch.randperm(token_limit, device=device)[:topk].to(torch.int64)
        slot_to_token[r] = tokens
        token_to_slot[r, tokens] = torch.arange(topk, dtype=torch.int32, device=device)

    return slot_to_token, token_to_slot, slot_stamp


def run_benchmark_case(num_reqs, topk, token_limit, hit_rate, use_atomic=False, num_warmup=10, num_iter=100):
    """Run one benchmark case comparing hash path and state path."""
    device = torch.device("npu:0")
    req_ids = torch.arange(num_reqs, dtype=torch.int64, device=device)

    # Build state for stateful path
    slot_to_token, token_to_slot, slot_stamp = build_full_cache_state(
        topk, num_reqs, token_limit, device
    )
    old_topk = slot_to_token.clone()

    # Generate new_topk
    new_topk = generate_topk_with_hit_rate(
        slot_to_token, topk, num_reqs, hit_rate, token_limit, device
    )

    # --- Hash path setup ---
    scratch = CacheMissTopKScratch()
    scratch_kwargs = scratch.prepare(
        num_reqs, topk, device,
        token_limit=token_limit,
        history_dtype=torch.int64,
    )

    def hash_path_fn():
        return get_cache_miss_topk_indices_triton(
            req_ids,
            old_topk,
            new_topk,
            **scratch_kwargs,
        )

    # --- State path setup ---
    state = CacheMissTopKState()
    state_kwargs = state.prepare(
        num_reqs, topk, device,
        req_dtype=torch.int64,
        token_limit=token_limit,
    )

    def state_path_fn():
        return get_cache_miss_topk_indices_triton_state(
            req_ids,
            new_topk,
            **state_kwargs,
            use_atomic=use_atomic,
        )

    # Warmup both paths once to initialize kernels
    with torch.no_grad():
        _ = hash_path_fn()
        _ = state_path_fn()

    # Benchmark
    hash_ms = benchmark_fn(hash_path_fn, num_warmup=num_warmup, num_iter=num_iter)
    state_ms = benchmark_fn(state_path_fn, num_warmup=num_warmup, num_iter=num_iter)

    # Correctness: verify outputs match on the last run
    with torch.no_grad():
        hash_out = hash_path_fn()
        state_out = state_path_fn()

    hash_plan = hash_out.clone()
    state_plan = state_out.clone()

    hash_tokens = set(hash_plan[hash_plan >= 0].tolist())
    state_tokens = set(state_plan[state_plan >= 0].tolist())

    # The slot-to-token mapping may differ, but the set of loaded tokens should match
    tokens_match = hash_tokens == state_tokens

    return {
        "num_reqs": num_reqs,
        "topk": topk,
        "token_limit": token_limit,
        "hit_rate": hit_rate,
        "use_atomic": use_atomic,
        "hash_ms": hash_ms,
        "state_ms": state_ms,
        "speedup": hash_ms / state_ms if state_ms > 0 else float("inf"),
        "correct": tokens_match,
    }


def main():
    use_atomic = os.environ.get("USE_ATOMIC", "0") == "1"

    CONFIGS = [
        (1, 512, 32768),
        (1, 1024, 32768),
        (1, 2048, 32768),
        (4, 512, 32768),
        (4, 1024, 32768),
        (4, 2048, 32768),
        (4, 2048, 65536),
        (8, 2048, 65536),
    ]

    HIT_RATES = [0.0, 0.5, 0.75, 0.90, 0.95, 1.0]

    print("=" * 100)
    print(f"Cache Miss TopK Benchmark: stateful (non-atomic, atomic={use_atomic}) vs hash")
    print("=" * 100)
    print(f"{'B':>3} {'K':>5} {'S':>6} {'hit%':>6} {'hash_ms':>10} {'state_ms':>10} {'speedup':>8} {'ok':>5}")
    print("-" * 100)

    results = []
    errors = []
    for num_reqs, topk, token_limit in CONFIGS:
        for hit_rate in HIT_RATES:
            try:
                r = run_benchmark_case(
                    num_reqs, topk, token_limit, hit_rate,
                    use_atomic=use_atomic,
                    num_warmup=5,
                    num_iter=20,
                )
                results.append(r)
                print(
                    f"{r['num_reqs']:>3} {r['topk']:>5} {r['token_limit']:>6} "
                    f"{r['hit_rate']:>6.2f} {r['hash_ms']:>10.4f} {r['state_ms']:>10.4f} "
                    f"{r['speedup']:>8.2f}x {'OK' if r['correct'] else 'FAIL':>5}"
                )
            except Exception as e:
                print(
                    f"{num_reqs:>3} {topk:>5} {token_limit:>6} "
                    f"{hit_rate:>6.2f} {'ERROR':>10} {'ERROR':>10} {'-':>8} {'ERR':>5}"
                )
                errors.append(f"B={num_reqs}, K={topk}, S={token_limit}, hit={hit_rate}: {e}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")

    if results:
        correct = sum(1 for r in results if r["correct"])
        avg_speedup = np.mean([r["speedup"] for r in results if r["correct"]])
        avg_hash = np.mean([r["hash_ms"] for r in results])
        avg_state = np.mean([r["state_ms"] for r in results])
        print(f"\nSummary: {correct}/{len(results)} correct, "
              f"avg hash={avg_hash:.4f}ms, avg state={avg_state:.4f}ms, "
              f"avg speedup={avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
