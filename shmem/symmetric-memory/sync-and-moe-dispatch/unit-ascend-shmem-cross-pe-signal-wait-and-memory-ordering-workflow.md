---
schema: mindie-graded-unit/1
id: unit-ascend-shmem-cross-pe-signal-wait-and-memory-ordering-workflow
title: Ascend SHMEM Cross-PE Signal-Wait Synchronization, Intra-Device AIC-AIV Sync,
  and Hang Diagnosis Workflow
domain_path: shmem/symmetric-memory/sync-and-moe-dispatch
tier: L3
category: workflow
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: shmem
  synchronization_scope: cross-pe-and-intra-device
inherits_from:
- unit-ascend-shmem-symmetric-heap-addressing-and-team-axioms
- unit-ascend-shmem-device-rma-mte-and-sdma-data-movement-sop
supersedes: []
triggers:
  error_signatures:
  - aclshmem_signal_wait_until hang on remote PE
  - aclshmem_barrier_all deadlock across AI cores
  - aclshmem_sync_all uninitialized pSync symmetric array
  - stale payload read after aclshmem_signal_wait_until
  keywords:
  - aclshmem_signal_wait_until
  - aclshmem_fence
  - aclshmem_quiet
  - aclshmem_sync_all
  - aclshmem_barrier_all
  - ACLSHMEM_CMP_EQ
  - ACLSHMEM_CMP_GE
  - cross-core synchronization
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/synchronization_and_ordering.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/synchronization_and_ordering.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/memory_ordering_semantics.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/atomic_operations.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/runbooks/sync_hang_debugging.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/runbooks/memory_corruption_debugging.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/runbooks/runtime_error_troubleshooting.md
  source_digest: 493ddde9aa61aec3dd7ac07709923778c8052fbd15d25eaf22a2b70bb89f01a5
---

## Cross-PE Synchronization, Memory Ordering, and Hang Diagnosis Workflow

### Stage 1: Enforce Ordering Semantics (`fence` vs `quiet` vs `barrier_all` vs `sync_all`)

1. **`aclshmem_fence()` (Per-Destination Ordering)**:
   - Guarantees that all preceding RMA `put` and atomic operations issued by the calling thread toward a specific destination PE arrive before any subsequent `put` or atomic operation toward that **same** destination PE. It does **not** block for completion and does not order transfers to different PEs.
2. **`aclshmem_quiet()` (Global Remote Completion)**:
   - Blocks the calling core until all preceding outbound RMA `put`, atomic, and signaling operations toward **all** destination PEs have completed at the remote HBM controller.
3. **`aclshmem_sync_all()` / `aclshmem_team_sync(team)` vs `aclshmem_barrier_all()`**:
   - `aclshmem_sync_all()` synchronizes control flow across PEs **without** flushing all pending outbound RMA writes; use it when PEs have already ordered data via `aclshmem_quiet()` or `putmem_signal`.
   - `aclshmem_barrier_all()` executes `aclshmem_quiet()` followed by a full cross-PE control barrier.
4. **Intra-Device Multi-Core Barrier Before Cross-PE Collective Sync**:
   - Inside an AscendC kernel launched across `blockDim` AI Cores (AIV/AIC), only a designated leader core (`GetBlockIdx() == 0`)—or a coordinated core group—should issue cross-PE barriers, **after** all local AI Cores synchronize via `SyncAll()` and flush their GM caches (`dcci`). If core 0 signals a remote PE before sibling cores finish writing their GM slices, the remote PE reads torn data.

### Stage 2: Point-to-Point Signal-Wait Protocol & Cache Coherency

1. **Monotonic Signal Counters (`ACLSHMEM_SIGNAL_ADD` + `ACLSHMEM_CMP_GE`)**:
   - When multiple producers or pipelined steps signal the same symmetric `uint64_t* sig_addr`, use `ACLSHMEM_SIGNAL_ADD` with delta `1` and wait with `aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_GE, expected_step_count)`.
   - Never use `ACLSHMEM_SIGNAL_SET` with `ACLSHMEM_CMP_EQ` across multiple producers or back-to-back iterations without resetting `sig_addr` behind a barrier; otherwise a fast producer overwrites the flag or a consumer wakes early on a stale iteration value.
2. **D-Cache Invalidation After Signal Wakeup**:
   - On Ascend AI Cores with scalar Data Cache (`__gm__` scalar loads), `aclshmem_signal_wait_until` polls `sig_addr` bypassing the scalar cache, but subsequent scalar reads of metadata headers in `__gm__` must execute `DataCacheCleanAndInvalid` (`dcci`) before reading the newly arrived payload metadata.

### Stage 3: Triage Synchronization Hangs & Memory Corruption

1. **Symptom: Kernel Hangs in `aclshmem_signal_wait_until`**:
   - **Check 1 (Symmetric Offset Mismatch)**: Verify `sig_addr` was allocated via `aclshmem_calloc` (zero-initialized across all PEs) and not `aclrtMalloc`. Passing a non-symmetric local GM pointer causes the remote `putmem_signal` to fail or write to an unmapped offset.
   - **Check 2 (Asymmetric Participation / Deadlock)**: Log `(my_pe, target_pe, expected_count)` on the consumer and count outbound `putmem_signal` calls per `target_pe` on producers. In dynamic MoE routing, if `num_tokens_to_pe == 0`, the producer must still send a zero-byte header or signal `ACLSHMEM_SIGNAL_ADD` so the consumer's `wait_until` reaches `n_pes`.
   - **Check 3 (Missing `aclshmem_quiet` on Non-Fused Path)**: If using separate `put_nbi` + `signal_op`, ensure `aclshmem_fence()` is called between `put_nbi` and `signal_op`, and `aclshmem_quiet()` is called before kernel exit so the DMA queue is drained.
