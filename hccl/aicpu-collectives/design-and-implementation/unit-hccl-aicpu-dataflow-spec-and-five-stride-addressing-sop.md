---
schema: mindie-graded-unit/1
id: unit-hccl-aicpu-dataflow-spec-and-five-stride-addressing-sop
title: HCCL AICPU Collective Algorithm Design, Five-Stride Buffer Addressing, and
  OpExpansion SOP
domain_path: hccl/aicpu-collectives/design-and-implementation
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: hccl
  subsystem: aicpu-collective-algorithm
  stage: dataflow-and-stride-design
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-ascend-hccl-aicpu-template-dataflow-and-dual-cmake-sop
supersedes: []
triggers:
  error_signatures:
  - AICPU collective slice address out of bounds
  - srcStride dstStride five-stride calculation mismatch
  - OpExpansion LinkWith LinkRight dependency cycle
  - tail slice truncation in ring or halving-doubling
  keywords:
  - hccl-aicpu-design
  - five-stride addressing
  - OpExpansion DAG
  - LinkWith LinkRight
  - HCCL slice partitioning
  - ScratchPad memory layout
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-design/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-design/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-implement/SKILL.md
  - https://gitcode.com/cann/cannbot/blob/d0f8c40ebc208fa17323781c89e4ec6ea16e4c52/plugins-official/hccl-op-generator/doc/design.md
  source_digest: 8c09eba0ce34cde238537da0962c566b7d9f6f58d702e0b8e2316180f3cc0fd6
---

## HCCL AICPU Collective Dataflow Specification & Five-Stride Addressing SOP

When designing a custom or optimized HCCL collective algorithm on the AICPU template framework (`hccl-op-generator`), every step in the algorithm must be lowered into an explicit `OpExpansion` DAG with exact five-stride memory addressing and tail-slice arithmetic.

### Step 1: Partition Tensor Slices with Exact Remainder Handling

1. **Base Slice & Tail Slice Invariant**:
   - Given total element count $N$, element size $E$, and $R$ participating ranks (or $S = k \cdot R$ pipeline chunks):
     $$\text{base\_cnt} = \lfloor N / S \rfloor, \quad \text{tail\_cnt} = N - (S - 1) \cdot \text{base\_cnt}$$
   - Every rank/chunk index $i \in [0, S-2]$ processes `base_cnt` elements at byte offset $i \cdot \text{base\_cnt} \cdot E$, while chunk $S - 1$ processes `tail_cnt` elements. Never assume $N$ is evenly divisible by `rank_size`.
2. **ScratchPad Staging Allocation**:
   - Declare `ScratchPad` workspace size in `CalcScratchPadSize(count, dataType, rankSize)` to cover all concurrently live intermediate chunks (e.g., `2 * max_chunk_bytes` for double-buffered Ring or Halving-Doubling reduction staging). Never overwrite the user's `inputMem` if `inputMem != outputMem` (out-of-place contract).

### Step 2: Construct Five-Stride Transfer Descriptors

1. **Five-Stride Addressing Contract**:
   - AICPU DMA / RoCE / SDMA descriptors express multi-dimensional non-contiguous slice gathers and scatters via five stride parameters:
     - `count`: Elements per innermost contiguous block.
     - `srcStride` / `dstStride`: Byte stride between consecutive blocks along dimension 1.
     - `repeat`: Number of blocks in dimension 1.
     - `srcOuterStride` / `dstOuterStride`: Byte stride across outer batches/ranks when batching multiple peer slices into a single descriptor.
2. **Contiguity Collapse Rule**:
   - Whenever `srcStride == count * elem_bytes` and `dstStride == count * elem_bytes`, collapse `count <- count * repeat` and `repeat <- 1` to maximize single-burst SDMA/RDMA payload size.

### Step 3: Build the `OpExpansion` Dependency DAG (`LinkWith` / `LinkRight`)

1. **Primitive Node Types**:
   - Decompose each communication round into `Send`, `Recv`, `Reduce` (local AIV/AICPU reduction), and `Copy` primitive nodes inside the `BuildDataflow()` method.
2. **Explicit Producer-Consumer Edges**:
   - Connect dependent primitives using `LinkRight` (sequential execution on the same stream/channel: `recv_node.LinkRight(reduce_node)`) and `LinkWith` (cross-channel synchronization barrier via hardware Notify registers).
   - **Invariant**: Every `Recv` into `ScratchPad[slot]` must precede its consumer `Reduce` or `Send` via a `LinkRight`/`LinkWith` edge, and the next communication round's `Recv` into the **same** `ScratchPad[slot]` must depend on the completion of the previous round's consumer (`WAR` hazard protection).
