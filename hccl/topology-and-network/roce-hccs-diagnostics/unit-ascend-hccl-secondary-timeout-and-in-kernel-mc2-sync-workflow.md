---
schema: mindie-graded-unit/1
id: unit-ascend-hccl-secondary-timeout-and-in-kernel-mc2-sync-workflow
title: Ascend HCCL Secondary Timeout Root-Rank Triage, AIV Op-Expansion Starvation,
  and In-Kernel HcclComm Workflow
domain_path: hccl/topology-and-network/roce-hccs-diagnostics
tier: L3
category: workflow
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: hccl
  diagnosis_domain: secondary-timeout-and-in-kernel-mc2
  runtime:
  - torch_npu
  - ascendc
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-ascend-hccl-notify-wait-timeout-and-hccn-link-triage-sop
- unit-ascend-hccl-comm-group-construction-and-buffsize-rdma-sop
supersedes: []
triggers:
  error_signatures:
  - 'EI0002: Notify wait timeout secondary to single-rank exception'
  - HCCL_OP_EXPANSION_MODE=AIV prefill micro-batch starvation
  - HcclAicpuWait or HcommWait unfinished handle in custom MC2 kernel
  - Commit flag missing in device-side Hccl Send/Recv
  keywords:
  - hccl_timeout_secondary_to_rank_failure_oom
  - hccl_aiv_expansion_vs_prefill_micro_batch
  - hccl_channels_and_hcomm_completion
  - hccl_commit_flag_and_peerwise_counts
  - rank_mock_hccl_aic_aiv_syncall
  - Device-side Hccl client
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/hccl_timeout_secondary_to_rank_failure_oom.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/hccl_timeout_secondary_to_rank_failure_oom.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/hccl_aiv_expansion_vs_prefill_micro_batch.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/kernel/comm_hccl.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/accuracy/hccl_channels_and_hcomm_completion.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/runbooks/accuracy/hccl_commit_flag_and_peerwise_counts.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/rank_mock_hccl_aic_aiv_syncall.md
  source_digest: 585f77cfb47c4c5c19911ce0bfa3e340e25ce63ba79611834b20f889eeee2c13
---

## HCCL Secondary Timeout Triage, AIV Expansion Contention, and In-Kernel `Hccl` Workflow

### Stage 1: Distinguish Primary Root-Rank Failure from Secondary `EI0002` Notify Timeout

1. **Why `EI0002: Notify wait timeout` Is Usually a Secondary Symptom**:
   - When Rank $k$ crashes with an NPU HBM OOM, AICore exception (`EZ9999`), Python `KeyError`, or hangs in a host dataloader/tokenization call, all healthy peers $j \neq k$ enter the next collective (`AllReduce` / `AllToAll`) and block waiting for Rank $k$'s hardware `Notify` signal until `HCCL_EXEC_TIMEOUT` expires.
2. **Deterministic First-Failure Triage Protocol**:
   - Never start debugging network cables on the first rank that prints `EI0002` in console stdout.
   - Collect the earliest timestamp across all nodes' plogs (`~/ascend/log/debug/plog/plog-<pid>_*.log`):
     - Ranks printing `EI0002: Notify wait timeout` with `remote rank = k` are **healthy victims** waiting on Rank $k$.
     - Inspect Rank $k$'s plog and Python traceback **prior to** $t_{\text{timeout}} - \text{HCCL\_EXEC\_TIMEOUT}$: look for `alloc device memory failed`, `AICORE exception`, `Watchdog`, or diverging conditional control flow (e.g., only subset of ranks entering dynamic-batch dummy forward).

### Stage 2: Mitigate `HCCL_OP_EXPANSION_MODE=AIV` Core Contention in Prefill Micro-Batching

1. **Mechanism of AIV Expansion Contention**:
   - Setting `export HCCL_OP_EXPANSION_MODE=AIV` dispatches collective communication scheduling and reduction onto AI Vector (AIV) cores instead of the off-core AI CPU.
   - In multi-stream overlapped pipelines (e.g., dual micro-batch Prefill overlap or shared-expert + routed-expert stream concurrency), AIV-expanded collectives compete with compute streams for AIV core slots, causing head-of-line blocking or deadlocks when a compute kernel holds all AIV cores while waiting on a cross-stream event.
2. **Resolution**:
   - For compute-saturated multi-stream Prefill overlap, switch to `export HCCL_OP_EXPANSION_MODE=AICPU` (or bound AIV core allocation per stream) so collective orchestration runs on dedicated AI CPU threads without starving AIV vector kernels.

### Stage 3: In-Kernel AscendC `Hccl` (`MC2`) Client Invariants

When writing fused compute-and-communication (`MC2`) AscendC kernels using the device-side `Hccl<HCCL_SERVER_TYPE_AICPU>` client:

1. **Explicit `Commit()` and `Wait(handleId)` Lifecycle**:
   - Preparing a collective task (`hccl.AllReduce<commit=false>(...)`, `hccl.AlltoAllV<commit=false>(...)`) only writes the task descriptor into the AICPU command queue; the transfer does **not** start until `hccl.Commit(handleId)` is called (or `commit=true` is passed).
   - Every prepared `handleId` **must** be awaited via `hccl.Wait(handleId)` before reading the destination GM buffer or exiting the kernel, followed by `hccl.Finalize()` at kernel epilogue.
2. **Dual-Queue / Channel Separation (`queueId`) & AIC-AIV `SyncAll`**:
   - When overlapping two communication directions (e.g., `AllGather` + `ReduceScatter` or double-buffered tiles), bind them to distinct `queueId`s and synchronize AIC (Cube) and AIV (Vector) sub-blocks via `SyncAll<true>()` / `CrossCoreSetFlag` before calling `hccl.Commit(handleId)` so AICPU never reads partially computed GM tiles.
