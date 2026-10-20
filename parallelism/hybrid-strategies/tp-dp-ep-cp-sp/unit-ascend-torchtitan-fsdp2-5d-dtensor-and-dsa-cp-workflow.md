---
schema: mindie-graded-unit/1
id: unit-ascend-torchtitan-fsdp2-5d-dtensor-and-dsa-cp-workflow
title: Ascend TorchTitan 5D DeviceMesh, FSDP2 DTensor Sharding, and DeepSeek Sparse
  Attention (DSA) CP Workflow
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
  - torchtitan
  - pytorch-dtensor
  workload: fsdp2-5d-and-dsa-sparse-attention
inherits_from:
- unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
supersedes: []
triggers:
  error_signatures:
  - DeviceMesh 5D dimension order mismatch in TorchTitan
  - fully_shard FSDP2 DTensor placement conflict with EP
  - DSA sparse index kv gather load imbalance across CP ranks
  keywords:
  - TorchTitan 5D DeviceMesh
  - FSDP2 fully_shard DTensor
  - Async TP DTensor
  - DeepSeek Sparse Attention DSA CP
  - ep_tp vs dp_shard_cp mesh
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/torchtitan_parallelism_overview.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/torchtitan_parallelism_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/torchtitan_data_and_tensor_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/torchtitan_pipeline_context_and_moe_parallel.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/training/concepts/parallelism/dsa_sparse_attention_parallel.md
  source_digest: ba2a802b8805f1ab75c7d1079fb0eda07bd81ac3a46c0d390e5c28c9a7169dab
---

## TorchTitan 5D `DeviceMesh`, FSDP2 `DTensor`, and DeepSeek Sparse Attention (DSA) CP Workflow

### Stage 1: Construct the Dual Dense & MoE `DeviceMesh` Hierarchy

1. **5D Dense Process Mesh (`[pp, dp_replicate, dp_shard, cp, tp]`)**:
   - Initialize the root `DeviceMesh("npu", mesh_shape=(pp, dp_rep, dp_shard, cp, tp))` with innermost `tp` mapped to intra-node HCCS ranks.
   - Flatten `(dp_shard, cp)` into `dp_shard_cp` so `FSDP2` (`fully_shard`) shards parameter states across both data-parallel and context-parallel ranks without duplicating optimizer states across CP replicas.
2. **Decoupled MoE Sub-Mesh (`[pp, dp_replicate, ef_dp, ep, etp]`)**:
   - Slice the `(dp_shard, cp, tp)` axes and reshape into `(ef_dp, ep, etp)` for MoE experts:
     - `ep` shards the expert dimension `dim=0` of 3D expert weight tensors `[num_local_experts, hidden, ffn_dim]` via `Shard(0)`.
     - `ef_dp` applies FSDP2 `fully_shard` across the remaining non-EP data-parallel ranks when $P_{\text{DP}} \cdot P_{\text{CP}} > P_{\text{EP}}$.

### Stage 2: Apply FSDP2 (`fully_shard`) & DTensor Sequence Parallelism

1. **Bottom-Up `fully_shard` Application Order**:
   - Apply ` parallelize_module` (`ColwiseParallel` / `RowwiseParallel` with `PrepareModuleInput` and `SequenceParallel` `Shard(1)` placements) **before** calling `fully_shard(transformer_block, mesh=dp_shard_cp_mesh, reshard_after_forward=True)`.
   - Calling `fully_shard` on each `TransformerBlock` and finally on the root `Model` enables automatic `AllGather` prefetching of Layer $l+1$'s parameters on a secondary HCCL communication stream while Layer $l$ executes on the compute stream.

### Stage 3: Parallelize DeepSeek Sparse Attention (DSA) Across Context Parallel (`CP`) Ranks

1. **DSA Indexer & Top-$K$ Sparse KV Gather bottleneck**:
   - In DeepSeek Sparse Attention (DSA), a lightweight Lightning Indexer scores all historical tokens and selects top-$k$ KV blocks per query token. Under standard Ring CP, top-$k$ selected indices cluster on early/sink sequence chunks, causing severe compute and communication skew.
2. **Balanced DSA Context Parallelism**:
   - Stripe query and KV sequence positions across `cp` ranks using zig-zag permutation so both causal prefix density and high-score sink tokens are uniformly distributed across all $P_{\text{CP}}$ ranks.
   - Exchange only the compressed MLA latent KV cache ($c_t^{KV} \in \mathbb{R}^{d_c}$ and RoPE key $k_t^R \in \mathbb{R}^{d_r}$, where $d_c + d_r \ll H_q d_h$) via `AllGather` across `cp_mesh` before running local top-$k$ index selection and sparse FlashAttention (`npu_sparse_flash_attention`).
