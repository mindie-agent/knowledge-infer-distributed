---
schema: mindie-graded-unit/1
id: unit-hccl-aicpu-checker-mismatch-and-algo-selection-debug-sop
title: HCCL AICPU Functional Checker Mismatch, Wrong-Result Bisect, and Hang Debug
  SOP
domain_path: hccl/aicpu-collectives/debug-and-perf-tuning
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: hccl
  subsystem: aicpu-collective-algorithm
  stage: functional-debug-and-checker
inherits_from:
- unit-hccl-aicpu-dataflow-spec-and-five-stride-addressing-sop
- unit-hccl-aicpu-executor-binding-and-selector-registration-workflow
supersedes: []
triggers:
  error_signatures:
  - HVM Checker data verification failed at rank
  - AICPU OpExpansion Notify wait hang in simulation
  - HCCL AICPU output mismatch on non-divisible count
  - HCCL AICPU fallback to default algorithm unexpectedly
  keywords:
  - hccl-aicpu-debug
  - HVM Checker mismatch
  - Notify wait deadlock
  - OpExpansion DAG dump
  - slice offset bisect
  - CheckParams fallback
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-debug/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-test-tool-hvm/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-env-debug/SKILL.md
  source_digest: eca1f2f89da5cbe45c14fc5597d833189c56816f876b8732a88de8fab598a9f6
---

## HCCL AICPU Functional Checker Mismatch & Hang Debug SOP

When a custom or modified HCCL AICPU collective fails functional verification in `HVM` (`hccl-test-tool-hvm`) or on NPU hardware, isolate the root cause across four deterministic fault classes:

### Class 1: Algorithm Not Selected (Silent Fallback to Baseline)

1. **Symptom**: Performance or trace matches the legacy algorithm despite setting `HCCL_OP_EXPANSION_MODE=AICPU`.
2. **Diagnostic Action**:
   - Enable `export ASCEND_GLOBAL_LOG_LEVEL=1` and grep plog for `SelectAlgo` / `CheckParams`.
   - Verify whether `CheckParams()` rejected the test's `(rankSize, count, dataType, topoType)` tuple (e.g., `count < rankSize` or non-power-of-two `rankSize`), or whether `HCCL_ALGO_CONFIG` overrode the selector.

### Class 2: `OpExpansion` Hang / Notify Deadlock

1. **Symptom**: HVM simulation or NPU execution hangs inside a communication round.
2. **Diagnostic Action**:
   - Dump the expanded `OpExpansion` primitive graph per rank and verify **peerwise symmetry**:
     - Every `Send(dst_rank=q, tag=t, channel=c)` on rank $p$ must have a matching `Recv(src_rank=p, tag=t, channel=c)` on rank $q$.
     - Check for cross-channel `LinkWith` cycles where Rank $p$ waits for Rank $q$ on channel 0 before issuing on channel 1, while Rank $q$ waits on channel 1 before issuing on channel 0.
     - Verify loop bounds when `count` is small: if a rank skips zero-sized tail slices without skipping or matching the peer's `Notify` handshake, the peer hangs forever.

### Class 3: HVM `Checker` Numerical / Data Mismatch

1. **Symptom**: `HVM Checker` reports mismatch at specific `(rank, element_offset)` indices.
2. **Diagnostic Action (Offset-Based Bisect)**:
   - **Case A (Mismatch only in the last slice $[ (S-1) \cdot \text{base\_cnt}, N )$)**: Tail-slice count or byte offset bug. Check whether `tail_cnt` was computed as `N % S` instead of `N - (S - 1) * (N / S)`, or whether a reduction step used `base_cnt` instead of `tail_cnt` on chunk $S - 1$.
   - **Case B (Whole chunk $k$ is zero or stale on specific ranks)**: Missing `LinkRight` edge between `Recv(ScratchPad[s])` and `Reduce/Copy(ScratchPad[s])`, or premature reuse of `ScratchPad[s]` before the previous round's `Send` completed (`WAR` hazard).
   - **Case C (Periodic strided corruption every `srcStride` bytes)**: Five-stride descriptor unit mismatch (passing element count instead of byte stride to `srcStride`/`dstStride`, or forgetting `* sizeof(dtype)`).
