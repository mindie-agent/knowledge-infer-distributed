---
schema: mindie-graded-unit/1
id: unit-ascend-shmem-low-latency-moe-cross-rank-dispatch-combine-mutator
title: Ascend SHMEM Low-Latency MoE Cross-Rank Token Dispatch/Combine and Double-Buffer
  Mutator
domain_path: shmem/symmetric-memory/sync-and-moe-dispatch
tier: L4
category: mutator
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: shmem
  workload: moe-ep-token-dispatch-combine
  optimization_lever: one-sided-put-signal-double-buffer
inherits_from:
- unit-ascend-shmem-symmetric-heap-addressing-and-team-axioms
- unit-ascend-shmem-device-rma-mte-and-sdma-data-movement-sop
- unit-ascend-shmem-cross-pe-signal-wait-and-memory-ordering-workflow
supersedes: []
triggers:
  error_signatures:
  - MoE AllToAll dispatch latency bottleneck in decode
  - hcclAlltoAllV synchronization overhead small batch MoE
  - moe_cross_rank_dispatch_with_putmem_signal
  keywords:
  - shmem-ops-generator
  - MoE SHMEM dispatch combine
  - one-sided token dispatch
  - double-buffered symmetric slot
  - core-to-PE partitioning
  - putmem_signal MoE
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/shmem-ops-generator/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/shmem-ops-generator/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/moe_cross_rank_dispatch_with_putmem_signal.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/optimizations/moe_cross_rank_dispatch_sync_delay.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/patterns/alltoall_exchange.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/patterns/ring_shift_and_halo_exchange.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/optimizations/multi_core_parallel_partitioning.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/communication/shmem/guides/pytorch_extension_integration.md
  source_digest: c31aa0921e86cc9581e5111194e3ff3cfd43f14d243f1f87cdc713d3cb6f5627
---

## SHMEM Low-Latency MoE Cross-Rank Token Dispatch & Combine Mutator

For latency-sensitive MoE Expert Parallelism (`EP`) decode steps ($B \le 64$ tokens per rank), two-sided `HcclAlltoAllV` incurs collective handshake and count-exchange latency. Replacing two-sided AllToAllV with an AscendC + `aclshmem` one-sided `putmem_signal` kernel eliminates the global count barrier and overlaps token packing with HCCS/RoCE RMA.

### Mutation 1: Pre-Allocate Double-Buffered Symmetric Dispatch Slots

1. **Symmetric Buffer Layout (Allocated Once at Model Init via `aclshmem_calloc`)**:
   - Allocate two alternating phase slots (`phase = step & 1`) so iteration $t+1$ can issue `putmem_signal_nbi` into slot $(t+1) \bmod 2$ without waiting for a global barrier to clear slot $t \bmod 2$:
     $$\text{slot\_bytes} = 2 \times N_{\text{PE}} \times B_{\max} \times K_{\text{top}} \times H_{\text{token\_bytes}}$$
   - Allocate a symmetric signal matrix `uint64_t sig_table[2][N_PE]` initialized to `0`.

### Mutation 2: Partition AI Vector Cores Across Destination PEs (`Core-to-PE` Sharding)

1. **Avoid Contention on the Same Remote PE**:
   - Partition `num_aiv_cores` across destination PEs (`target_pe = (my_pe + core_id) % n_pes` in strided waves) so all cores do not hammer PE 0 simultaneously (preventing HCCS/RoCE ingress congestion):
   - ```cpp
     // Inside AscendC AIV kernel for MoE Dispatch:
     int32_t phase = step_idx & 1;
     uint64_t expected_epoch = (step_idx >> 1) + 1;
     for (int32_t wave = core_idx; wave < n_pes; wave += num_cores) {
         int32_t dst_pe = (my_pe + wave) % n_pes;
         uint32_t token_cnt = local_expert_counts[dst_pe];
         __gm__ uint8_t* remote_slot = sym_recv_buf
             + ((phase * n_pes + my_pe) * max_tokens_per_pe) * hidden_bytes;
         if (token_cnt > 0) {
             // Pack tokens into UB, then issue fused UB -> remote GM put + signal:
             aclshmemx_ub_putmem_signal_nbi(
                 remote_slot, packed_tokens_ub, token_cnt * hidden_bytes,
                 &sym_sig_table[phase * n_pes + my_pe],
                 expected_epoch, ACLSHMEM_SIGNAL_SET, dst_pe);
         } else {
             // Zero tokens routed to dst_pe: still advance remote signal to prevent hang!
             aclshmem_uint64_atomic_set(
                 &sym_sig_table[phase * n_pes + my_pe], expected_epoch, dst_pe);
         }
     }
     ```

### Mutation 3: Defer `aclshmem_signal_wait_until` to the Consumer Boundary

1. **Decouple Issue from Wait (Sync-Delay Optimization)**:
   - Do **not** call `aclshmem_signal_wait_until` immediately inside the dispatch sender loop. Issue all outbound `putmem_signal_nbi` transfers first, overlap any local shared-expert compute or quantization metadata preparation, and only spin on `aclshmem_signal_wait_until(&sym_sig_table[phase * n_pes + src_pe], ACLSHMEM_CMP_EQ, expected_epoch)` immediately before each source PE's tokens are dequantized/consumed by GroupMatmul.
