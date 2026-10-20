---
category: workflow
conditions:
  collective_transport:
  - hccs
  - roce-v2
  repository: cann/cannbot-skills
  revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
  vendor: huawei-ascend
domain_path: serving/parallel-strategy
evidence_strength: software_contract
id: unit-ascend-module-decoupled-tp-ep-cp-kvp-parallel-workflow
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
provenance:
  evidence_refs:
  - docs/evidence/cannbot-hccl-and-parallel-provenance.json
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-analysis/SKILL.md#L12-L450
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-parallel-impl/SKILL.md#L11-L220
  source_digest: e8d1c8af698eb1115a1c0989e760b37bc54e4bedc914c852413ad05fa9dc262c
  source_locator: docs/evidence/cannbot-hccl-and-parallel-provenance.json
  source_type: cannbot_skill_and_source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Module-decoupled TP/EP/DP/CP/KVP parallel strategy sizing, CommManager physical group reuse, and packed TND padding workflow
triggers:
  error_signatures:
  - world_size is not divisible by tp_size
  - N_h * v_head_dim is not divisible by oproj_tp_size
  - oproj_tp_size must equal kvp_size when KVP is enabled
  - moe_ep_group_mc2 physical group reuse conflict
  keywords:
  - parallel_config
  - attn_tp_size
  - dense_tp_size
  - moe_tp_size
  - o_proj_tp_size
  - cp_size
  - kvp_size
  - CommManager
  - allow_physical_reuse
---
When designing and implementing distributed inference parallelism for Dense, GQA, MoE, or MLA Transformers on Ascend Atlas A2/A3 clusters, decompose parallelism per module rather than applying a monolithic global TP degree (`model/model-infer-parallel-analysis/SKILL.md:12-450`, `model/model-infer-parallel-impl/SKILL.md:11-220`).

## 1. Module-Decoupled Parallel Degrees & Quantitative Memory Sizing

Each module's DP or EP degree is derived independently from `world_size`:
$$\text{attn\_dp\_size} = \frac{W}{\text{attn\_tp\_size}}, \quad \text{dense\_dp\_size} = \frac{W}{\text{dense\_tp\_size}}, \quad \text{moe\_ep\_size} = \frac{W}{\text{moe\_tp\_size}}$$

1. **Per-Card Weight Memory (`BF16 = 2` bytes/param)**:
   $$P_{\text{card}} = \frac{P_{\text{attn}}}{\text{attn\_tp\_size}} + \frac{P_{\text{ffn}}}{\text{dense\_tp\_size}} + \frac{P_{\text{moe}}}{\text{moe\_ep\_size}} + \frac{P_{\text{embed}}}{\text{embed\_tp\_size}} + \frac{P_{\text{lmhead}}}{\text{lmhead\_tp\_size}}$$
2. **Per-Card KV Cache Memory ($B_{\text{local}} = B / \text{attn\_dp\_size}$)**:
   - **Standard GQA**: $B_{\text{local}} \times S \times L \times 2 \times N_{kv} \times D_h \times \text{bytes\_per\_kv}$.
   - **MLA Absorb**: $B_{\text{local}} \times S \times L \times (\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}) \times \text{bytes\_per\_kv}$. Never omit `qk_rope_head_dim` (sizing by `kv_lora_rank` alone underestimates MLA KV cache HBM by ~12%).
   - Reject any candidate where parameter + KV cache memory exceeds $0.80 \times M_{\text{card}}$ (hard ceiling $0.95 \times M_{\text{card}}$) or where `tp_size` exceeds the single-node NPU count (crossing from intra-node HCCS ~56 GB/s on A2 to inter-node RoCE/RDMA).
3. **Hard Divisibility & KVP/MLA Constraints**:
   - `world_size % *_tp_size == 0`, `num_attention_heads % attn_tp_size == 0`, `num_key_value_heads % attn_tp_size == 0`, and `num_experts % moe_ep_size == 0`.
   - For MLA models with independent `o_proj_tp_size`: `(num_attention_heads * v_head_dim) % o_proj_tp_size == 0`.
   - When Key-Value Parallelism (`kvp_size > 1`) is enabled for ultra-long decode: enforce `o_proj_tp_size == kvp_size`.

## 2. `CommManager` Physical Group Reuse vs Dedicated HCCL Groups (`model-infer-parallel-impl/SKILL.md:112-161`)

- In `init_parallel_comm_group()`, register module groups via `comm_manager.register_group(name, group_num, group_size, ...)`. By default, `CommManager` reuses physical HCCL communicators across logical groups that share identical `(group_num, group_size, group_stride)` geometry.
- **Mandatory `allow_physical_reuse=False` Exceptions**:
  - `moe_ep_group_mc2` (used by `npu_moe_distribute_dispatch_v2` / `combine_v2` fullmesh MC2 communication, sized via `calc_moe_hccl_buffer_size`) and `*_stream1` secondary-stream micro-batch groups **must** pass `allow_physical_reuse=False` so they allocate dedicated physical HCCL communicators and avoid cross-stream or MC2 buffer collisions.
- Wrap all linear/embedding modules (`QKVParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear`, `VocabParallelEmbedding`, `ColumnParallelLinear`) even when their `tp_size == 1` to preserve unified `weight_loader` and quantization dispatch.

## 3. Packed `TND` `attn_tp + attn_dp` AllGather/ReduceScatter Padding Contract (`model-infer-parallel-impl/SKILL.md:176-188`)

When both `attn_tp_size > 1` and `attn_dp_size > 1` are active under packed `TND` execution:
1. Update paged KV cache metadata `num_head` to `max(num_kv_heads // attn_tp_size, 1)`.
2. Keep inter-layer hidden states partitioned as `[T_padded / attn_tp_size, H]` to save `attn_tp_size`x activation memory:
   - Pad total packed token count $T_{\text{real}}$ up to $T_{\text{padded}}$ (a multiple of `attn_tp_size`) at model entry and slice `input_ids` per TP rank, while keeping `position_ids` and RoPE `cos`/`sin` at $T_{\text{real}}$.
   - Inside Attention: `all_gather_into_tensor` reconstructs `[T_padded, H]`, slice back to `[:T_real]` before QKV projection, KV cache write (`slot_mapping`), and FlashAttention (`actual_seq_lengths_kv`), then zero-pad the O-projection output back to `[T_padded, H]` before `reduce_scatter_tensor`.
   - At Prefill LMHead exit: `AllGather` hidden states first, then `index_select(cu_seq_q - 1)` on $T_{\text{real}}$ before `lm_head`.
