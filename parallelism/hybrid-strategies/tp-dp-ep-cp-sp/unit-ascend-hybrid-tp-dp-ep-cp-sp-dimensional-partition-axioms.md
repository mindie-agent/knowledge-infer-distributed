---
schema: mindie-graded-unit/1
id: unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
title: Ascend Multi-Dimensional Parallelism (TP, DP, EP, CP, SP, PP) Tensor Partitioning
  and Communication Axioms
domain_path: parallelism/hybrid-strategies/tp-dp-ep-cp-sp
tier: L1
category: axiom
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  domain: multi-dimensional-parallelism
  dimensions:
  - tp
  - dp
  - ep
  - cp
  - sp
  - pp
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
supersedes: []
triggers:
  error_signatures:
  - world_size not divisible by tp_size * dp_size * pp_size
  - expert_parallel_size must divide num_routed_experts
  - num_key_value_heads must be divisible by tensor_parallel_size
  - sequence length not divisible by context_parallel_size
  keywords:
  - hybrid parallelism axioms
  - TP DP EP CP SP PP
  - ColumnParallelLinear RowParallelLinear
  - Ulysses All-to-All CP
  - Ring Attention CP
  - communication volume per layer
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-analysis/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/parallel_strategy_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/tp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/dp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/ep.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/cp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/sp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/pp.md
  source_digest: 0e948c87ddf1e1e47bc23b0ffa3562206428df8090bb543c31eb5cad49b5f647
---

## Multi-Dimensional Parallelism Partitioning & Communication Complexity Axioms

### 1. World-Size Decomposition & Divisibility Invariants

1. **Attention / Dense Process Grid**:
   $$W = P_{\text{PP}} \times P_{\text{DP}} \times P_{\text{CP}} \times P_{\text{TP}}$$
   - For GQA/MQA attention with $H_q$ query heads and $H_{kv}$ key-value heads:
     - When $P_{\text{TP}} \le H_{kv}$: $H_q \bmod P_{\text{TP}} = 0$ and $H_{kv} \bmod P_{\text{TP}} = 0$ (each TP rank holds $H_{kv} / P_{\text{TP}}$ KV heads).
     - When $P_{\text{TP}} > H_{kv}$ (wide TP): $P_{\text{TP}} \bmod H_{kv} = 0$, and KV heads must be replicated across $P_{\text{TP}} / H_{kv}$ ranks (or KV-Cache Parallelism `KVP` / MLA latent replication is applied).
2. **MoE Routed-Expert Process Grid (Decoupled Attention-MoE Mapping)**:
   - The total rank pool across Attention `DP × TP × CP` is re-indexed for MoE layers as:
     $$P_{\text{DP}} \times P_{\text{CP}} \times P_{\text{TP}} = P_{\text{EP}} \times P_{\text{MoE-TP}} \times P_{\text{MoE-DP}}$$
   - Total routed experts $E_{\text{routed}}$ must satisfy $E_{\text{routed}} \bmod P_{\text{EP}} = 0$, placing $E_{\text{local}} = E_{\text{routed}} / P_{\text{EP}}$ experts per EP rank.

### 2. Per-Dimension Sharding & Collective Communication Contracts

| Dimension | Partitioned Axis | Forward / Backward Collectives | Per-Layer Activation Communication Volume (Bytes, Element Size $e$) |Optimal Physical Link |
| :--- | :--- | :--- | :--- | :--- |
| **TP (Megatron 1D)** | Column-split $W_{\text{up/qkv}}$ ($[H, D/P_{\text{TP}}]$), Row-split $W_{\text{down/o}}$ ($[D/P_{\text{TP}}, H]$) | $2 \times \text{AllReduce}$ per Transformer block (Attention + MLP) | $4 \frac{P_{\text{TP}} - 1}{P_{\text{TP}}} B S H e$ | Intra-node / SuperPod HCCS |
| **SP + TP** | Sequence axis $S / P_{\text{TP}}$ on LayerNorm / Dropout / Residual | $2 \times (\text{AllGather} + \text{ReduceScatter})$ replacing TP `AllReduce` | $4 \frac{P_{\text{TP}} - 1}{P_{\text{TP}}} B S H e$ (identical volume to TP, $1/P_{\text{TP}}$ activation memory) | Intra-node / SuperPod HCCS |
| **EP (MoE)** | Expert axis $E_{\text{routed}} / P_{\text{EP}}$ | Dispatch `AlltoAllV` + Combine `AlltoAllV` | $2 \frac{P_{\text{EP}} - 1}{P_{\text{EP}}} B S K_{\text{top}} H e$ | Intra-SuperPod HCCS or RoCE |
| **CP (Ulysses)** | Sequence axis $S / P_{\text{CP}} \leftrightarrow$ Head axis $H_q / P_{\text{CP}}$ | $4 \times \text{AllToAll}$ ($Q, K, V$ scatter; $O$ gather) | $4 \frac{P_{\text{CP}} - 1}{P_{\text{CP}}} B S H e$ | High-bisection HCCS |
| **CP (Ring)** | Sequence axis $S / P_{\text{CP}}$ (causal striped/zig-zag load balance) | $P_{\text{CP}} - 1$ P2P Ring `SendRecv` steps of $(K, V)$ blocks | $2 (P_{\text{CP}} - 1) B (S / P_{\text{CP}}) H_{kv} d_h e$ | Overlap with FlashAttention on HCCS/RoCE |
| **PP (1F1B / VPP)** | Layer axis $L / P_{\text{PP}}$ | P2P `Send`/`Recv` of boundary activations $[B_{\text{micro}}, S, H]$ | $2 B_{\text{micro}} S H e$ per stage boundary | Inter-node RoCE |
