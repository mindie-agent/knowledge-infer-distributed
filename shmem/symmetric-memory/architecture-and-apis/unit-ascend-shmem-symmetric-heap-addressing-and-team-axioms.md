---
schema: mindie-graded-unit/1
id: unit-ascend-shmem-symmetric-heap-addressing-and-team-axioms
title: Ascend SHMEM PGAS Symmetric Heap Addressing, UID Bootstrap, and Team Hierarchy
  Axioms
domain_path: shmem/symmetric-memory/architecture-and-apis
tier: L1
category: axiom
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: shmem
  memory_model: pgas-symmetric-heap
  interconnect:
  - hccs
  - roce
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
supersedes: []
triggers:
  error_signatures:
  - aclshmemx_init_attr_t initialization failure
  - aclshmem_ptr returned null for remote PE
  - aclshmem_malloc collective size mismatch across PEs
  - aclshmem_team_split_strided invalid stride or size
  keywords:
  - aclshmemx_get_uniqueid
  - aclshmemx_set_attr_uniqueid_args
  - aclshmem_ptr symmetric address
  - aclshmem_malloc collective heap
  - ACLSHMEM_TEAM_WORLD
  - ACLSHMEM_TEAM_SHARED
  - aclshmem_my_pe
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/guides/system_overview.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/guides/system_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/aclshmem_core_concepts.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/initialization_and_bootstrap.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/symmetric_memory_model.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/team_management.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/guides/host_api_guide.md
  source_digest: 8e20403476db508732626be65b7da9c6d8ec6e680385e477fd768c9832e8ec30
---

## Core Invariants & PGAS Symmetric Memory Axioms

Ascend SHMEM (`aclshmem`) implements a Partitioned Global Address Space (PGAS) over NPU High-Bandwidth Memory (HBM), exposing peer NPU memory across HCCS (intra-node / superpod direct load-store) and RoCE RDMA (inter-node one-sided DMA).

### 1. Symmetric Heap Allocation & Identical Offset Invariant

1. **Collective Symmetric Allocation**:
   - Every allocation on the symmetric heap (`aclshmem_malloc(size)`, `aclshmem_calloc(count, size)`, `aclshmem_align(alignment, size)`, `aclshmem_free(ptr)`) is a **collective operation** across all Processing Elements (`PE`s) in `ACLSHMEM_TEAM_WORLD`.
   - Every participating PE must invoke `aclshmem_malloc` in the exact same order with identical `size` arguments; diverging sizes or conditional single-PE calls deadlock the internal barrier or corrupt the symmetric offset table.
2. **Symmetric Virtual Address Translation (`aclshmem_ptr`)**:
   - A symmetric pointer `sym_ptr` returned on the local PE maps to a remote PE `target_pe` via constant-offset translation over the peer's mapped HBM aperture:
     $$\text{remote\_va}(\text{sym\_ptr}, \text{target\_pe}) = \text{peer\_base}[\text{target\_pe}] + (\text{sym\_ptr} - \text{local\_base})$$
   - `aclshmem_ptr(sym_ptr, target_pe)` returns a valid raw `__gm__` device pointer only when `target_pe` is directly addressable over P2P/HCCS (`ACLSHMEM_TEAM_SHARED`). When `target_pe` is only reachable over RoCE without direct virtual address mapping, `aclshmem_ptr` returns `nullptr`, and callers must use explicit RMA routines (`aclshmem_put*` / `aclshmem_get*`).
3. **Symmetric Heap Size Budget**:
   - Total symmetric heap capacity is fixed at initialization via `aclshmemx_init_attr_t` (`HBM` symmetric pool size). Over-allocating the symmetric heap starves standard `aclrtMalloc` / `NPUCachingAllocator` workspace pools.

### 2. UID Bootstrap & Initialization Lifecycle

1. **Rank-Zero UniqueID Broadcast**:
   - Rank 0 calls `aclshmemx_get_uniqueid(&uid)` to allocate a cluster-wide `aclshmemx_uniqueid_t` token (containing the root bootstrap socket endpoint).
   - Rank 0 broadcasts `uid` out-of-band (via `torch.distributed.broadcast_object_list`, TCPStore, or MPI) to all `n_pes` ranks.
2. **Attribute Binding & Collective Init**:
   - Every PE binds its rank and world size via `aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, &uid, &attr)` after binding its local NPU device (`aclrtSetDevice(local_rank)`), then calls `aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr)`.
   - Shutdown requires a collective `aclshmem_finalize()` after all in-flight kernels and RMA transfers complete.

### 3. Predefined & Split Team Hierarchy

1. **Predefined Teams**:
   - `ACLSHMEM_TEAM_WORLD`: All `0 .. n_pes - 1` PEs initialized in the job.
   - `ACLSHMEM_TEAM_SHARED`: Subset of PEs sharing a direct coherent/HCCS load-store domain with the calling PE.
2. **Sub-Team Creation (`aclshmem_team_split_strided` / `aclshmem_team_split_2d`)**:
   - `aclshmem_team_split_strided(parent_team, start, stride, size, config, config_mask, &new_team)` partitions `parent_team` into strided sub-teams (e.g., intra-node TP team with `stride=1, size=local_world_size` and inter-node EP/DP team with `stride=local_world_size`).
   - Teams must be destroyed via `aclshmem_team_destroy(new_team)` before `aclshmem_finalize()`.
