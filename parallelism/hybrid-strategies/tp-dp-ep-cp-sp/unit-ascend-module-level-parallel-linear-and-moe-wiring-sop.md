---
schema: mindie-graded-unit/1
id: unit-ascend-module-level-parallel-linear-and-moe-wiring-sop
title: Ascend Inference Module-Level Parallel Linear, Vocab Embedding, and MoE EP/TP
  Wiring SOP
domain_path: parallelism/hybrid-strategies/tp-dp-ep-cp-sp
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  domain: multi-dimensional-parallelism
  workload: llm-inference
  stage: module-level-parallel-wiring
inherits_from:
- unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
- unit-ascend-module-decoupled-tp-ep-cp-kvp-parallel-workflow
supersedes: []
triggers:
  error_signatures:
  - RowParallelLinear bias added before AllReduce
  - VocabParallelEmbedding out-of-range token mask missing
  - fused QKV ColumnParallelLinear uneven head split
  - shared expert output duplicated across EP ranks
  keywords:
  - model-infer-parallel-impl
  - VocabParallelEmbedding
  - MergedColumnParallelLinear
  - QKVParallelLinear
  - RowParallelLinear reduce_results
  - shared expert TP vs routed expert EP
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-impl/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-impl/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/tp.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/parallelism/ep.md
  source_digest: 48948c2944f7a4e190f00cef6d5447c729495891d65ea656d325618103247d72
---

## Inference Module-Level Parallel Linear, Embedding, and MoE Wiring SOP

When parallelizing a single-device PyTorch / `torch_npu` LLM into a multi-NPU inference model, apply strict per-module transformation rules to preserve numerical equivalence with single-rank execution:

### Step 1: Parallelize Embeddings & LM Head (`VocabParallelEmbedding` / `ParallelLMHead`)

1. **Vocabulary Partitioning (`[vocab_start_index, vocab_end_index)`)**:
   - Pad `vocab_size` to a multiple of `tp_size` ($V_{\text{padded}} = \lceil V / P_{\text{TP}} \rceil \cdot P_{\text{TP}}$) so each TP rank owns $V_{\text{local}} = V_{\text{padded}} / P_{\text{TP}}$ rows.
   - **Mask & Local Index Shift**:
     ```python
     masked_mask = (input_ids < vocab_start) | (input_ids >= vocab_end)
     local_ids = torch.where(masked_mask, 0, input_ids - vocab_start)
     local_out = F.embedding(local_ids, local_weight)
     local_out.masked_fill_(masked_mask.unsqueeze(-1), 0.0)
     dist.all_reduce(local_out, op=dist.ReduceOp.SUM, group=tp_group)
     ```
2. **LM Head Logits Gather**:
   - Shard `lm_head.weight` along the output vocabulary axis (`ColumnParallelLinear(gather_output=True)` during greedy/sampling logits computation, or keep sharded for distributed cross-entropy / speculative verification).

### Step 2: Wire Attention QKV & Output Projections (`QKVParallelLinear` + `RowParallelLinear`)

1. **Fused QKV Projection (`QKVParallelLinear`)**:
   - Do **not** naively slice a concatenated $[Q \|\, K \|\, V]$ weight tensor into `tp_size` contiguous chunks; that mixes $Q$, $K$, and $V$ across ranks.
   - Partition $Q$ ($H_q / P_{\text{TP}}$ heads), $K$ ($H_{kv} / P_{\text{TP}}$ heads), and $V$ ($H_{kv} / P_{\text{TP}}$ heads) independently per rank, then concatenate each rank's local $[Q_{\text{local}}, K_{\text{local}}, V_{\text{local}}]$ along dim 0.
2. **Output & Down Projections (`RowParallelLinear`)**:
   - Shard weight along the input feature dimension (`dim=1`, shape $[H_{\text{out}}, H_{\text{in}} / P_{\text{TP}}]$).
   - **Bias Addition Invariant**: Compute local matmul $Y_{\text{partial}} = X_{\text{local}} W_{\text{local}}^T$, execute `dist.all_reduce(Y_partial, group=tp_group)` (or `reduce_scatter` when Sequence Parallel is active), and add `bias` **only once after** the collective reduction. Adding `bias` before `all_reduce` multiplies the bias by $P_{\text{TP}}$.

### Step 3: Wire Hybrid MoE Blocks (Shared Experts + Routed Experts)

1. **Routed Experts (`EP` AllToAllV or MC2 Dispatch/Combine)**:
   - Each rank holds $E_{\text{local}} = E_{\text{routed}} / P_{\text{EP}}$ full experts (`gate_up_proj` + `SwiGLU` + `down_proj` without inner TP `AllReduce`).
   - Top-$K$ router runs on local tokens $\to$ `AlltoAllV` (or `npu_moe_distribute_dispatch`) scatters tokens to expert-owning EP ranks $\to$ `npu_grouped_matmul` $\to$ `AlltoAllV` (or `npu_moe_distribute_combine`) gathers and scales by router weights.
2. **Shared Experts Anti-Duplication Rule**:
   - When the model includes always-active Shared Experts alongside EP-routed experts:
     - Either shard the Shared Expert weights with TP across the `tp_group` (if `AllReduce` follows the MoE block) **or** compute the Shared Expert purely locally on each rank's DP/SP token slice and add it **after** the routed-expert combine reduction. Never add an unsharded Shared Expert output **before** an `AllReduce` across `tp_group`, which scales the shared expert contribution by $P_{\text{TP}}$.
