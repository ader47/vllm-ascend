# Design Documents

This section provides an overview of the features implemented in vLLM Ascend. Developers can refer to this guide to understand how vLLM Ascend works.

- [KVPP: KV Cache Layer Parallelism](kvpp.md) — Physical cache placement, full-layer broadcast, and test design.
- [Delayed Nano Tail D2H Adaptation](delayed_nano_tail_d2h_adaptation.md) — One-round-delayed out-of-graph D2H with two-tail Attention and MTP/ACL graph correctness.
