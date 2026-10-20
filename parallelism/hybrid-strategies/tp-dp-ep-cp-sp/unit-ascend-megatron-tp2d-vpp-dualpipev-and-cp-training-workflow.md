---
schema: mindie-graded-unit/1
id: unit-ascend-megatron-tp2d-vpp-dualpipev-and-cp-training-workflow
title: Ascend MindSpeed/Megatron TP-2D, Virtual Pipeline (VPP/DualPipeV), and Hybrid
  CP Configuration Workflow
domain_path: parallelism/hybrid-strategies/tp-dp-ep-cp-sp
tier: L3
category: workflow
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  domain: multi-dimensional-parallelism
  framework:
  - mindspeed
  - megatron-lm
  workload: distributed-training-and-long-context
inherits_from:
- unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
supersedes: []
triggers:
  error_signatures:
  - tp_x * tp_y must equal tensor_model_parallel_size
  - num_layers must be divisible by pipeline_model_parallel_size * vpp_size
  - DualPipeV micro-batch count less than 2 * pp_size
  - hybrid context parallel ulysses_degree * ring_degree != cp_size
  keywords:
  - Megatron TP-2D tp_x tp_y
  - Virtual Pipeline Parallelism VPP
  - DualPipeV bidirectional schedule
  - hybrid_Adaptive Context Parallel
  - Ulysses + Ring CP
  - MoE EP + ETP Megatron
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/megatron_parallelism_overview.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/megatron_parallelism_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/megatron_tensor_and_sequence_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/megatron_pipeline_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/megatron_context_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/megatron_moe_expert_parallel.md
  source_digest: eece983adf97239783d545acb0dc668605f25dfd6472519b6a3030979257f33a
---

## MindSpeed / Megatron TP-2D, VPP/DualPipeV, and Hybrid Context Parallelism Workflow

### Stage 1: Configure 2D Tensor Parallelism (`TP-2D`) for High-Degree Intra-Node Scaling

1. **Why 1D TP Degrades at $P_{\text{TP}} \ge 16$**:
   - Standard 1D Megatron TP executes `AllGather` / `ReduceScatter` across all $P_{\text{TP}}$ ranks in a single collective domain, increasing ring hops and activation staging overhead.
2. **`TP-2D` Grid Decomposition (`--tp-2d --tp-x X --tp-y Y`)**:
   - Factor $P_{\text{TP}} = X \times Y$ (e.g., $8 = 4 \times 2$ or $16 = 4 \times 4$).
   - Both input activations $A$ and weights $W$ are partitioned across the 2D mesh `(tp_x, tp_y)` using SUMMA-style 2D matrix multiplication (`AllGather` along `tp_x`, local GEMM, `ReduceScatter` along `tp_y`), restricting each collective to a smaller subgroup of size $X$ or $Y$ that aligns with Ascend dual-Die / HCCS sub-rings and overlaps per-tile communication with Cube GEMM.

### Stage 2: Select Pipeline Schedule (1F1B vs Interleaved VPP vs `DualPipeV`)

1. **Interleaved Virtual Pipeline (`--num-layers-per-virtual-pipeline-stage`)**:
   - Assigns $V$ non-contiguous model chunks per PP rank ($L \bmod (P_{\text{PP}} \cdot V) = 0$), reducing the pipeline bubble fraction from $\frac{P_{\text{PP}} - 1}{M}$ to $\frac{P_{\text{PP}} - 1}{M \cdot V}$ at the cost of $V\times$ more P2P `Send`/`Recv` transitions.
2. **Bidirectional V-Shape Pipeline (`DualPipeV`) for MoE Compute-Comm Overlap**:
   - Assigns two symmetric model chunks (one ascending stage $s$, one descending stage $2P_{\text{PP}} - 1 - s$) to each PP rank and feeds micro-batches from Episode ends ($M \ge 2 P_{\text{PP}}$).
   - Overlaps Forward `AlltoAll` dispatch/combine of micro-batch $m_a$ with Backward weight gradient (`WGrad`) / input gradient (`DGrad`) compute of micro-batch $m_b$, hiding cross-node MoE EP RoCE latency behind Cube compute.

### Stage 3: Configure Hybrid Context Parallelism (`Ulysses + Ring` / `hybrid_Adaptive`)

1. **2D Context Parallel Mesh (`P_CP = P_Ulysses × P_Ring`)**:
   - Set `--context-parallel-size CP --context-parallel-algo hybrid_cp_algo --ulysses-degree-in-cp U`:
     - Place the inner `Ulysses` group ($P_{\text{Ulysses}} \le 8$, requiring $H_{kv} \bmod P_{\text{Ulysses}} = 0$) inside the high-bisection intra-node HCCS domain for fast `AllToAll` head/sequence transposition.
     - Place the outer `Ring` group ($P_{\text{Ring}} = P_{\text{CP}} / P_{\text{Ulysses}}$) across nodes using causal zig-zag token striping and asynchronous P2P $(K, V)$ ring shifts overlapped with `npu_fusion_attention`.
