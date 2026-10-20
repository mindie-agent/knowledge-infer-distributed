---
schema: mindie-graded-unit/1
id: unit-ascend-shmem-device-rma-mte-and-sdma-data-movement-sop
title: Ascend SHMEM Device-Side RMA (MTE vs SDMA) and Put-With-Signal Data Movement
  SOP
domain_path: shmem/symmetric-memory/architecture-and-apis
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: shmem
  execution_scope: ascendc-device-kernel
  transport_engine:
  - mte
  - sdma
inherits_from:
- unit-ascend-shmem-symmetric-heap-addressing-and-team-axioms
supersedes: []
triggers:
  error_signatures:
  - aclshmem_put_nbi UB buffer reused before aclshmem_quiet
  - aclshmemx_putmem_signal_nbi UB alignment error
  - aclshmem_get_nbi stale data read in UB
  - SDMA RMA bandwidth degradation on small payload
  keywords:
  - aclshmem_put_nbi
  - aclshmem_get_nbi
  - aclshmem_ub_put_nbi
  - aclshmemx_mte_put_nbi
  - aclshmemx_sdma_put_nbi
  - aclshmem_putmem_signal
  - UB vs GM RMA
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/guides/device_api_guide.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/guides/device_api_guide.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/concepts/rma_communication.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/optimizations/mte_vs_sdma_selection.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/optimizations/ub_vs_gm_data_path.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/optimizations/put_with_signal_fusion.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/optimizations/nonblocking_overlap_pipeline.md
  source_digest: 714180212609118b090319900d8c50ebd3054408f647144be6e74d74b7e7a032
---

## Device-Side RMA Engine Selection & Data Movement SOP

Inside AscendC kernels, `aclshmem` provides blocking and non-blocking (`_nbi`) one-sided Remote Memory Access (RMA) over two hardware engines: the AI Core Memory Transfer Engine (**MTE**) and the System DMA Engine (**SDMA**).

### Step 1: Select MTE vs SDMA Transport Engine

1. **AI Core MTE Path (`aclshmem_ub_put*` / `aclshmem_ub_get*` / `aclshmemx_mte_*`)**:
   - Moves data directly between local AI Core Unified Buffer (`__ubuf__`, 32-byte aligned) and remote symmetric Global Memory (`__gm__`) over HCCS without staging through local GM.
   - **When to use**: Low-latency, fine-grained transfers (token dispatch payloads $\le 64\text{ KB}$ per message) or when data is already resident in `LocalTensor` (UB) after vector/cube compute.
   - **Constraint**: Consumes AI Core MTE2/MTE3 issue slots; requires source/destination sizes and UB addresses to be 32-byte aligned.
2. **System DMA Path (`aclshmem_gm_put*` / `aclshmemx_sdma_*`)**:
   - Moves data from local `__gm__` to remote `__gm__` using off-core SDMA channels.
   - **When to use**: Bulk transfers ($> 64\text{ KB}$) where freeing AI Core MTE bandwidth for concurrent Cube/Vector compute outweighs SDMA descriptor launch overhead.
   - **Constraint**: Source buffer must reside in Global Memory (`__gm__`) and any preceding UB writes must be flushed to GM with `DataCacheCleanAndInvalid` / `PipeBarrier<PIPE_ALL>()` before issuing SDMA RMA.

### Step 2: Enforce Blocking vs Non-Blocking (`_nbi`) Buffer Lifetime Rules

1. **Blocking RMA (`aclshmem_put`, `aclshmem_get`, `aclshmem_ub_put`)**:
   - Returns only after the local source buffer is safe to overwrite (for `put`) or the local destination buffer has received the remote payload (for `get`). Note: Blocking `put` does **not** guarantee remote visibility until `aclshmem_quiet()` or a fused signal arrives.
2. **Non-Blocking RMA (`aclshmem_put_nbi`, `aclshmem_get_nbi`, `aclshmem_ub_put_nbi`)**:
   - Enqueues the transfer descriptor and returns immediately so the AI Core can double-buffer compute and communication.
   - **Invariant**: Never overwrite a local UB/GM source buffer passed to `*_put_nbi` or read a local destination buffer passed to `*_get_nbi` until `aclshmem_quiet()` (or `aclshmem_WaitUntil` on the corresponding completion counter) completes.

### Step 3: Fuse Payload Transfer and Remote Notification with `put_signal`

1. **Never Issue Separate `put_nbi` + `quiet` + `signal_op` for Producer-Consumer Handoff**:
   - Issuing a standalone `aclshmem_put_nbi` followed by `aclshmem_quiet()` and a separate signal write incurs two round-trip flushes.
2. **Use Fused Put-With-Signal (`aclshmem_putmem_signal` / `aclshmem_putmem_signal_nbi` / `aclshmemx_ub_putmem_signal_nbi`)**:
   - ```cpp
     aclshmemx_ub_putmem_signal_nbi(
         remote_dst_gm, local_src_ub, nbytes,
         remote_sig_addr_gm, /*sig_val=*/1ULL,
         ACLSHMEM_SIGNAL_ADD, target_pe);
     ```
   - Hardware guarantees that the remote signal update (`ACLSHMEM_SIGNAL_SET` or `ACLSHMEM_SIGNAL_ADD` on `uint64_t* remote_sig_addr_gm`) becomes visible to `target_pe` **only after** all bytes of `remote_dst_gm` have landed in remote HBM.
