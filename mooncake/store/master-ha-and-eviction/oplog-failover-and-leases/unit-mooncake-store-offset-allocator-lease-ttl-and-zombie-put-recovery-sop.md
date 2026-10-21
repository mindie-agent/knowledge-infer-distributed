---
category: sop
conditions:
  domain: mooncake
  hardware:
  - Ascend 910B
  - Ascend 910C
  - Distributed CPU/DRAM/RDMA Store Cluster
domain_path: mooncake/store/master-ha-and-eviction/oplog-failover-and-leases
evidence_strength: software_contract
id: unit-mooncake-store-offset-allocator-lease-ttl-and-zombie-put-recovery-sop
inherits_from:
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/store/mooncake-store.md#L1-L260
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/deployment/mooncake-store-deployment-guide.md#L1-L162
  source_locator: kvcache-ai/Mooncake@1f65cf13c9f886f8db8b88b7733367b18340ebf0:docs/source/design/store/mooncake-store.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: Mooncake Store Two-Phase Put Protocol, Offset Allocator & Lease TTL Eviction
  SOP
triggers:
  error_signatures:
  - NO_AVAILABLE_HANDLE
  - PutStart failed
  - put_start_discard_timeout_sec
  keywords:
  - mooncake-store
  - offset-allocator
  - cachelib
  - put-start
  - put-end
  - put-revoke
  - kv-lease-ttl
  - eviction
---
## Preconditions & Trigger

Apply this SOP when sizing memory pools, selecting space-allocation strategies (`OffsetAllocator` vs. `CachelibAllocator`), or troubleshooting `NoSpace` allocation failures, zombie uncommitted objects, and premature KV eviction in Mooncake Store (`mooncake_master` + client `MasterClient` / `TransferEngine`).

## Two-Phase `PutStart` / `PutEnd` / `PutRevoke` Lifecycle Contract

Mooncake Store decouples control-plane space reservation from client-driven RDMA data movement using a strict two-phase commit protocol:
1. **Phase 1 — Space Reservation (`PutStart`)**:
   - The writer calls `MasterClient::PutStart(key, slice_lengths, config)` on `mooncake_master`.
   - The master allocates contiguous slice ranges across registered storage segments (`MountSegment`) and returns replica `Replica::Descriptor` handles containing the target `segment_name`, virtual address/offset, and `length`, while marking the object metadata state as `PENDING`.
2. **Phase 2 — RDMA Data Transfer & Commit (`PutEnd` or `PutRevoke`)**:
   - The writer invokes `TransferEngine::submitTransfer(TransferRequest::WRITE)` directly to the allocated remote segment offsets.
   - If all slice transfers succeed, the writer calls `MasterClient::PutEnd(key)` to transition the object to `COMPLETE` and arm its initial `kv_lease_ttl`.
   - If any RDMA write fails or times out, the writer MUST immediately call `MasterClient::PutRevoke(key)` so the master frees the reserved segment slices.
3. **Zombie `PutStart` Reclamation (`put_start_discard_timeout_sec`)**:
   - When a writer crashes between `PutStart` and `PutEnd`, the object remains stuck in `PENDING` state and its slices cannot be read or evicted by normal LRU sweeps.
   - Configure `--put_start_discard_timeout_sec` (default `30` seconds) on `mooncake_master` so the background reaper automatically revokes uncommitted `PENDING` allocations whose age exceeds the discard timeout, ensuring `put_start_discard_timeout_sec >= client_rdma_write_timeout + 5s`.

## Lease-Based Read Protection (`kv_lease_ttl`) vs. Soft Eviction

1. **Zero-Copy Read Safety via Leases**:
   - Because `GetReplicaList(key)` returns raw RDMA buffer addresses to the reader and the reader pulls KV blocks asynchronously over RoCE/HCCS without notifying the master upon completion, deleting or reusing that memory while an RDMA `READ` is in flight causes silent tensor corruption.
   - Every `GetReplicaList(key)` (and initial `PutEnd(key)`) grants or extends a read lease for `--default_kv_lease_ttl` milliseconds (default `5000` ms = 5 s). While `now < lease_expire_time`, the object is **pinned** and immune to both `Remove(key)` and capacity-driven LRU eviction.
   - Sizing rule: Set `--default_kv_lease_ttl` strictly greater than the maximum P99 Decode KV pull latency plus queueing delay (e.g., `5000`–`10000` ms for 128K context prefill blocks), and set `--default_kv_soft_pin_ttl` to preserve hot prefix blocks across multi-turn agent turns.
2. **Watermark-Driven Eviction (`eviction_ratio` & `eviction_high_watermark_ratio`)**:
   - When cluster memory utilization crosses `--eviction_high_watermark_ratio` (default `0.95`), the background eviction thread reclaims unleased (`lease_expire_time < now`) objects in LRU order by `--eviction_ratio` (default `0.05`).
   - If all objects are actively leased when utilization hits `1.0`, `PutStart` fails fast with `ErrorCode::NO_AVAILABLE_HANDLE`; clients must fall back to local recomputation rather than blocking the inference loop.

## Allocator Selection (`offset` vs. `cachelib`)

- Always set `--memory_allocator=offset` (`OffsetAllocator`, the default O(1) segregated-fit bitmapped range allocator) for variable-length KV cache slices and high-concurrency P/D disaggregation. Unlike `cachelib` slab classes, `OffsetAllocator` supports exact byte-granularity coalescing with minimal internal fragmentation and native serialization for HA snapshots.

## Verification

- Monitor `mooncake_master` HTTP metrics (`http://<master_ip>:9003/metrics`) for `allocated_size_bytes / total_capacity_bytes`, `pending_put_count`, and `evicted_key_total`, confirming `pending_put_count` returns to `0` within `put_start_discard_timeout_sec` after an injected worker kill.
