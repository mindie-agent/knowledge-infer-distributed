---
category: workflow
conditions:
  collective_transport:
  - hccs
  - roce-v2
  repository: cann/cannbot-skills
  revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  vendor: huawei-ascend
domain_path: hccl/verification
evidence_strength: software_contract
id: unit-ascend-hccl-vm-checker-and-env-readiness-workflow
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-ascend-hccl-aicpu-template-dataflow-and-dual-cmake-sop
provenance:
  evidence_refs:
  - docs/evidence/cannbot-hccl-and-parallel-provenance.json
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-env-check/SKILL.md#L6-L80
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-test-tool-hvm/SKILL.md#L6-L135
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-debug/SKILL.md#L6-L21
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-perf-optimize/SKILL.md#L6-L18
  source_digest: 4fab0be11bf360f3bccb3619f2ccd3c8bac0d3da1f6d4d2b7ab960607cc6cbdc
  source_locator: docs/evidence/cannbot-hccl-and-parallel-provenance.json
  source_type: cannbot_skill_and_source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: HCCL-VM collective verification, comm-domain topology matrix, and AICPU fault isolation workflow
triggers:
  error_signatures:
  - CHECKER_RUN_SUMMARY missing or failed
  - HCCL_ALGO directed algorithm not hit during hccl-vm verification
  - HCOMM_CODE_HOME or HCCL_ROOT NOT_READY
  keywords:
  - hccl-vm
  - hccl-test-tool-hvm
  - hccl-env-check
  - CHECKER_RUN_SUMMARY
  - CCU_SCHED
  - CCU_MS
  - AI_CPU
  - HCCL_ALGO
  - mock-comm
---
When verifying or debugging HCCL collective algorithms (`allgather`, `allreduce`, `alltoall`, `alltoallv`, `reduce`, `reduce_scatter`, `scatter`, `broadcast`) via `hccl-vm` (`${HCOMM_CODE_HOME}/test/hccl_vm`) and `hccl-env-check`, enforce the four-stage verification and fault-isolation workflow below (`communication/hccl-env-check/SKILL.md:6-80`, `communication/hccl-test-tool-hvm/SKILL.md:6-135`, `communication/hccl-aicpu-debug/SKILL.md:6-21`, `communication/hccl-aicpu-perf-optimize/SKILL.md:6-18`).

## 1. Stage 1 — Read-Only Environment Readiness Gate (`hccl-env-check`)

Before building or launching `hccl-vm`, verify five mandatory paths without mutating the workspace or guessing fallback directories:
- `HCCL_ROOT`: Must contain `.git`, `src/ops`, and `build.sh`; record named branch, `HEAD` commit SHA, and working-tree status (flag detached `HEAD` explicitly).
- `HCOMM_CODE_HOME` and `HCOMM_DOCS`: Verify `${HCOMM_CODE_HOME}/test/hccl_vm` and `${HCOMM_DOCS}/{comm_op_dev_guide,api_ref/comm_opdev}`.
- `CANN_HOME` and `ASCEND_HOME_PATH`: Verify `${CANN_HOME}/set_env.sh` is readable. Output `NOT_READY` immediately if any required variable or path is missing.

## 2. Stage 2 — `HCCL-VM` Execution Modes, Comm-Domain Matrix, and Evidence Tracks (`hccl-test-tool-hvm`)

1. **Execution Modes (`-m`)**:
   - `CCU_SCHED` (CCU scheduling), `CCU_MS` (CCU multi-stream), and `AI_CPU` (AICPU expansion mode, requiring device-side `libscatter_aicpu_kernel.so` symbols and QEMU runtime).
2. **Rank-to-Communication-Domain (`-t`) Mapping (`hccl-test-tool-hvm/SKILL.md:66-76`)**:
   - **2 ranks**: primary `112`, cross-server `121`
   - **4 ranks**: primary `114`, cross-server `122`
   - **8 ranks**: primary `118`, cross-server `124` / `141` (8-rank algorithms must **not** rely solely on `112`/`114` for primary acceptance)
   - **16 ranks**: primary `128`, cross-server `144`
   - **6 ranks (asymmetric)**: primary `12_2_4`
3. **Four Evidence Tracks (`--role`)**:
   - `directed`: Must provide both non-empty `HCCL_ALGO` (configuration DSL) and `--expect-algo` (registered class/algorithm full name) to prove the target template was actually selected and executed.
   - `baseline` and `regression`: Must leave `HCCL_ALGO` unset (testing default selector routing) and bind to each package's `artifact.json` over an identical test matrix.
   - `ordinary`: General smoke checks. Never treat `--dry-run` or `--no-checker` runs as Checker `PASS`.

## 3. Stage 3 — Strict `PASS` Criteria and Post-Verification Performance Estimation

- A test case is `PASS` if and only if: (1) `mpirun` exits 0 exactly once, (2) a complete and successful `CHECKER_RUN_SUMMARY` is present with zero embedded errors, (3) the executed algorithm set matches `--expect-algo` (when required), and (4) full untruncated logs and package hashes are preserved in a unique per-run directory.
- Only after functional `PASS` is established, run `hccl-vm` `estimate` mode (default `hccl-vm mock-comm 112` or user-specified topology) for analytical latency estimation. Never claim `hccl-vm` Checker passes or uncalibrated `CalcCostCoeff` coefficients as real hardware bandwidth gains (`hccl-aicpu-perf-optimize/SKILL.md:14-18`).

## 4. Stage 4 — Four-Boundary Fault Isolation (`hccl-aicpu-debug`)

| Failure Symptom | Primary Verification Target |
|---|---|
| **Host/Device Build or Symbol Load Failure** | First compiler error line, both CMake registrations (`src/ops/<op>/.../CMakeLists.txt` and `src/scatter_aicpu_kernel.cmake`), `AICPU_COMPILE` guards, and installed `.so` identity |
| **Target Algorithm Not Selected (`Selector Miss`)** | Comm-domain priority, `HCCL_ALGO` DSL vs registered name in `executor-selector`, `CalcCostCoeff` candidate list, `TopoMatch` attributes, and loaded package hash |
| **Checker Data Mismatch** | Compare Dataflow Spec vs `KernelRun` on source/dest direction, rank peer mapping, slice `offset`/`length`, `repeat`/`stride`, `CalcScratchMultiple`, and 64-bit/`HCCL_REDUCE_PROD` fallback |
| **Hang, Timeout, or Resource Error** | First failing case's `slaveThreadNum`, `notifyNumPerThread` / `notifyNumOnMainThread` vs `GetNotifyIdxMainToSub`/`SubToMain`, Pre/Post barrier pairing, and last completed channel action in plog |
