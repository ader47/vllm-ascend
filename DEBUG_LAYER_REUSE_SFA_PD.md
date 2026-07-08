# Debug Notes: P-side layer-reuse + sfa_pd_cpu_offload PD transfer accuracy corruption

**Branch:** `feat/sfa-offload-layerwise-reuse` (remote `ader47`). Push to this branch.
**Date:** 2026-07-09. This is a living handoff doc for continuing the investigation on another machine with Claude Code.

---

## 1. Problem statement (the accuracy matrix)

P (prefill/producer) node. Model is **SFA (sparse)** on Ascend.

| # | P-side config | PD transfer | Accuracy |
|---|---|---|---|
| 1 | AscendStore layerwise pool **WITH layer reuse** (`num_shared_buffers=2`) | MooncakeLayerwiseConnector (**push**) | **CORRECT** |
| 2 | no layer reuse (`num_shared_buffers=num_layers`) | sfa_pd_cpu_offload (**pull**) | **CORRECT** |
| 3 | AscendStore layerwise pool **WITH layer reuse** (`num_shared_buffers=2`) | sfa_pd_cpu_offload (**pull**) | **WRONG** |

Only the **combination** (layer reuse + sfa_pd pull) breaks. Either alone is fine.

P-node MultiConnector config (the broken one):
```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "layerwise_num_shared_buffers": "2",
    "layerwise_prefetch_layers": "2",
    "connectors": [
      { "kv_connector": "SFAPDCpuOffloadConnector", "kv_role": "kv_producer",
        "kv_connector_extra_config": {"use_layerwise": "true"}, ... },
      { "kv_connector": "AscendStoreConnector", "kv_role": "kv_producer",
        "kv_connector_extra_config": {"backend": "memcache", "use_layerwise": true,
          "layerwise_num_shared_buffers":"2", "layerwise_prefetch_layers":"2"} }
    ]
  }
}
```
AscendStore `backend=memcache` ⇒ `use_gva_layerwise=True` (GVA path).

---

## 2. Key commits on this branch (deployed state)

- `c7e71b11` — `fix(kv_transfer): drop PD D-side in-place decode free (race), keep null-pad layout`. Partial revert of `004ec1d8` (PD D-side MLA HBM reduction): keeps null-pad prefix (block-table layout for SFA), removes the in-place decode-block free (it raced the connector's async HBM→CPU copy under high concurrency). Separate issue, already fixed.
- `4f73e11e` — `debug(kv_pool): [ASC-WFLL]` probe (does AscendStore reuse gate fire on P?).
- `030b0a62` — `debug(kv_pool): uncomment PD-wait + [PDWAIT]/[RECVGATE]` probes. **The GVA save-thread PD-wait (`kv_transfer.py:~1188`) is now UNCOMMENTED on this branch.**
- `9beeca17` — `[KVSUM-P]` (superseded by `d459c953`).
- `d459c953` — `[KVVAL-PSAVE]` / `[KVVAL-DLOAD]` / `[MFPULL-D]` probes (current value-based checksum, not sum).

All probes gated by `VLLM_ASCEND_PD_REUSE_DEBUG=1` (off by default).

---

## 3. How to enable the probes / what each prints

```bash
export VLLM_ASCEND_PD_REUSE_DEBUG=1
```

- `[ASC-WFLL] role=… current=L needs_wait=… mate=M` — AscendStore `wait_for_layer_load` top (`pool_worker.py:~898`). Confirms the reuse gate fires and which mate it waits on.
- `[RECVGATE] layer=L waiting_for_save=M` — AscendStore recv thread gate-only branch (`kv_transfer.py:~1297`). The recv thread waits `layer_save_finished_events[mate]`.
- `[PDWAIT] L=L events_none=False … pd_done` — GVA save-thread PD-wait (`kv_transfer.py:~1197/1203`). Confirms `layer_transfer_finished_events` is non-None and the wait completed (sfa_pd set it on READ_DONE).
- `[KVVAL-PSAVE] layer=… head8=… mid4=… tail8=…` — P-side, at `save_kv_layer` (`sfa_v1.py`, `_pd_reuse_kvval_probe("PSAVE", …)`). The KV P just computed.
- `[KVVAL-DLOAD] layer=… head8=… mid4=… tail8=…` — D-side, right after `wait_for_kv_layer_from_connector` (`sfa_v1.py`, two call sites ~1586/1617, `_pd_reuse_kvval_probe("DLOAD", …)`). The KV D received/loaded. *(Also fires on P but stale there — ignore P's DLOAD, use P's PSAVE.)*
- `[MFPULL-D] layer=L transfers=N peer0=0x… local0=0x… len0=…` — D-side memfabric pull (`worker.py _do_read_batch`, after `batch_transfer_sync_read`). `peer0` = P's source GVA (slot ptr), `local0` = D dest HBM ptr.

---

## 4. CONFIRMED facts (do not re-litigate)

1. **P uses shared slots (layer reuse active on producer).** `[ASC-WFLL]` shows `role=kv_producer current=3 needs_wait=True mate=1` — the mate mapping only exists when `has_layer_reuse=True`, so P's model_runner merged tensors (`_merge_kv_cache_tensors_for_layer_reuse`) on P too.
2. **AscendStore reuse gate fires on P and gates the model forward.** `[ASC-WFLL] needs_wait=True` for reusing layers; MultiConnector fans `wait_for_layer_load` to both connectors, and it is called BEFORE the KV scatter (`sfa_v1.py:1586/1617`). So layer L+nsb's overwrite (scatter) is gated.
3. **The full gate chain fires and holds:**
   `[ASC-WFLL(L+nsb) waits layer_load_finished_events[L+nsb]` → recv `[RECVGATE] waits layer_save_finished_events[mate=L]` → save thread `[PDWAIT] batch_copy(L) + PD-wait(L), events_none=False, pd_done`.
   ⇒ **L+nsb's overwrite waits until D has read layer L** (sfa_pd sets the event on READ_DONE).
4. **PD-wait is active** (`events_none=False`, `pd_done`) — sfa_pd DOES set `layer_transfer_finished_events` (via MooncakeLayerwiseConnectorWorker super().__init__, `mooncake_layerwise_connector.py:1219-1220`; set on READ_DONE at `worker.py:_signal_layer_done`).
5. **Disabling reuse (`num_shared_buffers=num_layers`) makes accuracy CORRECT.** ⇒ corruption is specific to shared slots.
6. mooncake push (config 1) correct; sfa_pd pull (config 3) wrong; no-reuse + sfa_pd (config 2) correct.

---

## 5. RULED OUT (with evidence)

- **"sfa_pd doesn't set `layer_transfer_finished_events` → PD-wait is a no-op"** — REFUTED: `[PDWAIT] events_none=False`. sfa_pd inherits the set via mooncake super.
- **"AscendStore reuse gate doesn't fire on producer"** — REFUTED: `[ASC-WFLL] needs_wait=True mate=1` on `role=kv_producer`.
- **"sfa_pd gate mis-keyed (waits on current layer, not mate) is the root cause"** — REFUTED by adversarial verify (workflow `wzibtd7oj`): the AscendStore gate DOES fire on P and gates the forward, so the sfa_pd self-gate being mis-keyed is redundant/harmless. (The mis-key is real — `connector.py:188-189` waits `layer_send_done_events[absolute_current]` not the mate — but it is NOT the cause here.)
- **"memfabric conflict (AscendStore L2G + sfa_pd G2L both on memfabric)"** — RULED OUT: disabling reuse keeps both on memfabric yet fixes accuracy, so a memfabric-level conflict can't be it.
- **`transfer_tasks.clear()` race / save-side pool corruption** — fixed earlier (producer-side list reset in `process_layer_data`) and not on the PD-transfer path anyway.

---

## 6. The crux / open question

**With the AscendStore reuse gate confirmed holding (D reads layer L before L+nsb overwrites the shared slot), accuracy is STILL wrong under reuse.** So the corruption is **NOT the overwrite race**. It is in the **data**: either D receives the wrong bytes for some layer, or D writes/uses them wrong — and it only manifests when P's buffers are shared slots.

Two workflows (`wbsc5fw27`, `wzibtd7oj`, both synth failed on API 502 but maps+verify ran) and extensive reading could NOT pinpoint a P-side send/addressing bug: the maps conclude sfa_pd reads `kv_caches[layer_name]` (the shared slot tensor) per layer, gated by `reshape_cache_event`, and the addressing uses the slot's own data_ptr (not a per-layer flat offset). So by analysis the send should be correct — yet empirically it is not.

**Leading hypothesis to verify:** sfa_pd's memfabric pull addresses P's HBM via a scheme that assumes N separate per-layer buffers, but under reuse there are only `num_shared_buffers` slots — so for some layer the pull reads the wrong GVA/HBM region (or a stale/overwritten slot). The gate only ensures the slot isn't overwritten by L+nsb; it does NOT verify sfa_pd reads the correct GVA for layer L. mooncake push avoids it (P reads its own slot tensor and ships immediately; no deferred D-side GVA read). Disabling reuse gives N separate buffers so any per-layer addressing is correct.

---

## 7. NEXT STEP — P-vs-D value comparison (probes already in place, `d459c953`)

Run config 3 (reuse ON, `num_shared_buffers=2`), single prompt, `VLLM_ASCEND_PD_REUSE_DEBUG=1`. Collect from **both** P and D nodes:

**(a) Per-layer value compare** — P node `[KVVAL-PSAVE]` vs D node `[KVVAL-DLOAD]` for the SAME layer:
- all of head8/mid4/tail8 equal ⇒ transfer+load lossless ⇒ corruption is in **D-side usage** (D writes/reads wrong under reuse). Next: instrument D-side write addressing.
- some layer differs ⇒ transfer/load corrupts that layer's KV. Go to (b).

**(b) `[MFPULL-D] peer0` for slot-mates** — layers L and L+nsb (e.g., layer 1 and layer 3 with nsb=2) share one slot, so their `peer0` (P's source GVA) **must be identical**:
- identical ⇒ sfa_pd reads the shared slot GVA correctly (addressing OK) ⇒ look elsewhere (D-side, or content of the slot at read time).
- **different ⇒ sfa_pd computes a per-layer GVA/offset and reads the wrong region under reuse — ROOT CAUSE found.** Fix: make sfa_pd's pull addressing use the slot's GVA (respect `get_layerwise_storage_indices` / `num_shared_buffers`), not a per-layer flat offset.

Prefill KV is deterministic (the probe is taken during prefill, before sampling), so a single-prompt run is a valid comparison.

---

## 8. Key code locations

- Layer-reuse config / slot mapping: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_config.py` (`get_layerwise_storage_indices`, `prefetch_layer_map`, `has_layer_reuse`).
- model_runner tensor merging (shared slots): `vllm_ascend/worker/model_runner_v1.py` `_merge_kv_cache_tensors_for_layer_reuse` (~3952), `initialize_kv_cache` (~3987). Runs on ALL nodes (not role-gated).
- AscendStore GVA save thread (PD-wait): `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` `KVCacheStoreLayerSendingThread._handle_request` (~1150-1210). `layer_transfer_finished_events` passed via `get_shared_layer_transfer_events()` at `pool_worker.py:~324`.
- AscendStore reuse gate (recv-based): `pool_worker.py` `wait_for_layer_load` (~885), `_submit_ready_layer_loads` (~855) builds gate-only `LayerLoadTask(wait_for_save_layer=reuse_mate)`; recv `KVCacheStoreLayerRecvingThread._handle_request` gate-only branch (`kv_transfer.py:~1283`) waits `layer_save_finished_events[mate]`.
- sfa_pd producer send: `vllm_ascend/distributed/kv_transfer/sfa_pd_cpu_offload/worker.py` `SFAPDCpuOffloadProducerWorker` (extends `MooncakeLayerwiseConnectorWorker`), `_MembPullSendingThread._process_send_task` (~1012). Sets `layer_transfer_finished_events` in `_signal_layer_done` (~1135).
- sfa_pd consumer pull: `worker.py` `MembPullReadThread._do_read_batch` (~844), `batch_transfer_sync_read` (~904). Layer buffer dict built in `_resolve_read_layer` (~600-645): `p_k_base` (P GVA), `k_cpu_ptr` (D HBM), per-block via `_build_req_descriptors` (~647) / `_coalesce_desc` (~705).
- mooncake push (works): `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` `_transfer_kv_cache` (~447, batch_transfer_sync_write), sets `layer_transfer_finished_events` (~532).
- attention save/load hook: `vllm_ascend/attention/sfa_v1.py` `wait_for_kv_layer_from_connector` (~1586/1617), `maybe_save_kv_layer_to_connector` (~1887). Probe helper `_pd_reuse_kvval_probe` (~73).
- shared events module-global: `pool_worker.py` `_shared_layer_transfer_events`, `get/set_shared_layer_transfer_events` (~72-81).

---

## 9. Things to NOT waste time on (already tried / ruled out)

- Uncommenting the GVA PD-wait — already done (`030b0a62`), confirmed firing (`pd_done`), did NOT fix config 3.
- Fixing the sfa_pd self-gate mis-key (`connector.py:188`) — NOT the cause (AscendStore gate covers it); don't bother unless (b) above points to addressing.
- memfabric L2G/G2L conflict theory — disproven by the disable-reuse test.
- The PD D-side in-place free (`004ec1d8`) — already reverted in `c7e71b11`.

## 10. Cleanup TODO after root cause is found

Remove all `VLLM_ASCEND_PD_REUSE_DEBUG` probes (`[ASC-WFLL]`, `[RECVGATE]`, `[PDWAIT]`, `[KVVAL-PSAVE]`, `[KVVAL-DLOAD]`, `[MFPULL-D]`) and the `import os` they added in `pool_worker.py`, `kv_transfer.py`, `sfa_v1.py`, `worker.py`. Decide whether to keep the GVA PD-wait uncommented (it is correct coupling; keep) and whether to also fix the sfa_pd self-gate mis-key for robustness.
