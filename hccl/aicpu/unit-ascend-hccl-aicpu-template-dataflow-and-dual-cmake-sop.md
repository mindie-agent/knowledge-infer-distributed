---
category: sop
conditions:
  collective_transport:
  - hccs
  - roce-v2
  repository: cann/cannbot-skills
  revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  vendor: huawei-ascend
domain_path: hccl/aicpu
evidence_strength: software_contract
id: unit-ascend-hccl-aicpu-template-dataflow-and-dual-cmake-sop
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
provenance:
  evidence_refs:
  - docs/evidence/cannbot-hccl-and-parallel-provenance.json
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-best-practice/SKILL.md#L11-L207
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/hccl-op-generator/AGENTS.md#L1-L180
  source_digest: 31bbef4be6929d5c86afd6ee5049d01e7f6227b66cec6559afc2dd9658cbc0f1
  source_locator: docs/evidence/cannbot-hccl-and-parallel-provenance.json
  source_type: cannbot_skill_and_source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: HCCL AICPU InsAlgTemplateBase dataflow contract, single-repeat scratch budget, and host/device dual CMake registration SOP
triggers:
  error_signatures:
  - undefined symbol InsTemp in libscatter_aicpu_kernel.so
  - CalcScratchMultiple multiplied by repeatNum causes excessive executor loop splitting
  - GetNotifyIdxMainToSub or GetNotifyIdxSubToMain length mismatch with CalcRes
  - INT64 UINT64 FP64 or HCCL_REDUCE_PROD fails on hardware reduction path
  keywords:
  - hccl
  - aicpu
  - InsAlgTemplateBase
  - CalcScratchMultiple
  - CalcRes
  - KernelRun
  - scatter_aicpu_kernel
  - Dataflow Spec
---
In HCCL's layered collective execution architecture (`hccl.h API -> <op>.cc -> Selector -> ExecutorRegistry -> Executor -> Template`), an AICPU `template` (`ins_temp_*` inheriting from `InsAlgTemplateBase` in `src/ops/op_common/algorithm/template/alg_v2_template_base.h`) is the bottom-level execution orchestration unit that defines which rank sends which byte slice to which peer, over which channel and thread, and when notify synchronization fires (`communication/hccl-aicpu-best-practice/SKILL.md:11-207`).

## 1. Mode A/B/C Invocation Boundary and Dataflow Spec Gate

1. **Classification Criterion (`hccl-aicpu-best-practice/SKILL.md:23-40`)**:
   - **Mode A (Read-Only)**: Inspecting existing templates, explaining APIs, or auditing CMake wiring.
   - **Mode B (Local Non-Dataflow Edit)**: Bounds checks, logging, CMake wiring, or early-exit guards that do **not** alter byte offsets, slice partitioning, stream assignment, or notify synchronization.
   - **Mode C (New Algorithm / Dataflow Refactor)**: Adding a new `ins_temp_*` class or modifying slice partitioning, scratch buffer layout, notify sync points, or send/recv topology. Mode C requires a verified **Dataflow Spec** (`check_spec.py` and `check_layout.py` PASS with zero semantic TBDs) before writing C++ template code.

## 2. `InsAlgTemplateBase` Method, Scratch, and Resource Contracts

1. **Mandatory Pure-Virtual and Core Methods (`hccl-aicpu-best-practice/SKILL.md:108-128`)**:
   - `std::string Describe() const override`: Single-line descriptor including `templateRankSize_`.
   - `void GetNotifyIdxMainToSub(std::vector<u32>&) override` and `void GetNotifyIdxSubToMain(std::vector<u32>&) override`: Must match the exact `slaveThreadNum`, `notifyNumPerThread`, and `notifyNumOnMainThread` requested in `CalcRes`.
   - `HcclResult CalcRes(HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, AlgResourceRequest&) override`: Requests master/slave threads (`threads[0]` is master, `threads[1..N]` are slaves), notifies, and channels.
   - `HcclResult KernelRun(const OpParam&, const TemplateDataParams&, TemplateResource&) override`: Implements the slice offset, length, repeat/stride, channel send/recv, and pre/post notify barriers.
2. **Single-Repeat Scratch Multiplier Invariant (`CalcScratchMultiple`)**:
   - `u64 CalcScratchMultiple(BufferType, BufferType) override` reports the CCL scratch buffer magnification factor **for a single repeat** (never multiply by `repeatNum` inside `CalcScratchMultiple`), because the outer `Executor` uses this return value to compute how many loops to split the collective payload into. Always override `CalcScratchMultiple` explicitly (return `0` when no scratch buffer is used).
3. **Static Cost-Model Dispatch and 64-Bit / Product Reduction Fallback (`hccl-aicpu-best-practice/SKILL.md:124-205`)**:
   - Templates participating in automatic selector routing must define `static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam)` (non-virtual, dispatched via executor template parameter) and `static constexpr TemplateProp props = {.algoType = AlgoType::X};` (`TemplateProp` has only `algoType`; never write a non-existent `.isNhr` member).
   - 64-bit integer/float data types (`INT64`, `UINT64`, `FP64`) and `HCCL_REDUCE_PROD` must route through the AICPU software reduction fallback path.
   - Guard all division/modulo operations on `templateRankSize_`, verify `channels.count()` before calling `channels.at()`, and wrap every `HcclResult` call in `CHK_RET` / `CHK_PRT_RET`.

## 3. Mandatory Dual CMake Registration (Host + Device)

An AICPU template is compiled into **both** host-side `libhccl.so` and device-side `libscatter_aicpu_kernel.so` (`hccl-aicpu-best-practice/SKILL.md:146-174`). Every new `ins_temp_<snake_name>.{h,cc}` under `$HCCL_REPO/src/ops/<op>/algorithm/template/aicpu/` must be registered in **both** places:

1. **Host CMake**: `$HCCL_REPO/src/ops/<op>/algorithm/template/aicpu/CMakeLists.txt` (`set(src_list ...)`).
2. **Device AICPU Kernel CMake**: `$HCCL_REPO/src/scatter_aicpu_kernel.cmake` (inside the `add_library(scatter_aicpu_kernel SHARED ...)` block for `<op>`).

Registering only the first file allows host compilation to succeed while failing at runtime when the AICPU device kernel loader cannot resolve the `InsTemp*` symbol. Verify both registrations and static rules via `check_template.py --strict-new <file>`.
