---
schema: mindie-graded-unit/1
id: unit-hccl-aicpu-executor-binding-and-selector-registration-workflow
title: HCCL AICPU Executor Subclass Binding, AlgConfig Selector Registration, and
  Dual-CMake Build Workflow
domain_path: hccl/aicpu-collectives/design-and-implementation
tier: L3
category: workflow
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: hccl
  subsystem: aicpu-collective-algorithm
  stage: executor-and-selector-registration
inherits_from:
- unit-ascend-hccl-aicpu-template-dataflow-and-dual-cmake-sop
- unit-hccl-aicpu-dataflow-spec-and-five-stride-addressing-sop
supersedes: []
triggers:
  error_signatures:
  - HCCL_OP_EXPANSION_MODE=AICPU algorithm not selected
  - AlgConfig selector registration missing for collective
  - undefined symbol in libhccl_alg_template.so
  - AICPU executor Prepare or Run returned HCCL_E_PARA
  keywords:
  - hccl-aicpu-implement
  - HCCL_OP_EXPANSION_MODE
  - AlgConfig selector
  - AICPU Executor subclass
  - libhccl_alg_template.so
  - HcclAlgTemplateRegister
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-implement/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-implement/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-env-config/SKILL.md
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/hccl-op-generator/doc/design.md
  source_digest: 2c8efbf1cc3e8bdb6d966de7be7ec43e71aff1f849854613127fa5c5d375f617
---

## HCCL AICPU Executor Subclass Binding & Selector Registration Workflow

### Stage 1: Implement the Algorithm Template & Executor Subclass

1. **Algorithm Template Class (`<Collective><AlgoName>Template`)**:
   - Inherit from the base collective template (`AllReduceTemplate`, `AllGatherTemplate`, `ReduceScatterTemplate`, or `AlltoAllTemplate`).
   - Implement:
     - `CheckParams(const OpParam& param)`: Validate `rankSize >= 2`, supported `HcclDataType`, and topology constraints (e.g., power-of-two `rankSize` for Recursive Halving-Doubling, or hierarchical `intra_rank_size` divisibility). Return `HCCL_E_NOT_SUPPORT` (never crash) when preconditions are not met so the selector falls back cleanly.
     - `CalcScratchPadSize(const OpParam& param)`: Return exact staging buffer bytes required in device HBM.
     - `BuildDataflow(OpExpansionBuilder& builder)`: Emit the primitive `Send`/`Recv`/`Reduce`/`Copy` DAG.
2. **Executor Subclass Binding**:
   - Bind the template class into the AICPU `Executor` registry for the target collective (`HCCL_ALG_REGISTER(<CollectiveType>, <AlgoEnum>, <TemplateClass>)`).

### Stage 2: Register in the `AlgConfig` Selector & Cost Model Table

1. **Extend the Algorithm Enum & Name Mapping**:
   - Add the new algorithm identifier to the collective's `AlgType` enum and string parser so `HCCL_ALGO_CONFIG` can force-select the algorithm by name during testing.
2. **Wire Selector Dispatch (`Selector::SelectAlgo`)**:
   - In the collective's selector (`AllReduceSelector`, `AllGatherSelector`, etc.):
     - Register the algorithm's cost-model estimator (`EstimateTime(param, topoAttr)`).
     - Ensure that when `HCCL_OP_EXPANSION_MODE=AICPU` is active, the selector evaluates all registered candidate algorithms passing `CheckParams()` and picks the minimum-cost candidate (or honors explicit `HCCL_ALGO_CONFIG` overrides).

### Stage 3: Dual-CMake Compilation (Host Sim vs Device Cross-Compile)

1. **Host Simulation Build (`x86_64` / `aarch64` Host Unit & HVM Checker)**:
   - Compile against the host simulation CMake target (`cmake -DBUILD_SIM=ON`) to produce the host-executable shared library for functional `Checker` verification and HVM trace generation.
2. **Device Cross-Compile Build (`aarch64-linux-gnu` AICPU Target)**:
   - Cross-compile the device AICPU plugin (`libhccl_alg_template.so`) using the CANN AICPU toolchain and deploy to `$ASCEND_OPP_PATH/aicpu/` (or `LD_LIBRARY_PATH` for runtime loading under `export HCCL_OP_EXPANSION_MODE=AICPU`).
