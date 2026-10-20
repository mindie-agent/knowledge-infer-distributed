---
category: workflow
conditions:
  parallelism:
  - EP
  - MC2
  - All2All
  platform_family:
  - a2
  - a3
  - a5
  ranktable_schema: '1.0'
content_digest: sha256:98562d2d5e6aef9899f7a6181ff63193f74ce4ef587d0bd45691d8a0d2bb51c0
domain_path: serving/moe-ep
evidence_strength: reproduced_empirical
id: unit-multi-node-moe-ep-mc2-all2all-convergence-workflow
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-ascend-hccl-notify-wait-timeout-and-hccn-link-triage-sop
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - docs/evidence/cannbot-distributed-provenance.json#unit-multi-node-moe-ep-mc2-all2all-convergence-workflow
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/concepts/techniques/moe_dispatch_combine_v2.md#L24-L117
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/optimizations/experience/rules/mc2_dedicated_comm_domain.md#L22-L47
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/optimizations/experience/rules/moe_ep_dispatch_path_by_stage.md#L26-L64
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/runbooks/runtime/moe_dispatch_combine_hccl_buffsize.md#L26-L70
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/guides/experience/interventions/multimachine_dp_ep_moe_collective_alignment.md#L24-L56
  source_digest: fd7433c8c9b2de6d6d629e4d77ffb5dc09c57788a3f7d6d6c19dff678589f4d6
  source_locator: docs/evidence/cannbot-distributed-provenance.json#unit-multi-node-moe-ep-mc2-all2all-convergence-workflow
  source_type: reproduced_run+source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Bring up and converge multi-node MoE EP with MC2 dispatch/combine and All2All
triggers:
  error_signatures:
  - HCCL_BUFFSIZE is too SMALL
  - MoeDistributeDispatchV2 do tiling failed
  - batch_size is not divisible by moe_ep_size
  keywords:
  - moe
  - expert parallel
  - mc2
  - dispatch
  - combine
  - all2all
  - hccl
  - ep
  - fullmesh_v2
  - copy_expert_num
  - eplb
---
# Multi-node MoE expert parallelism with MC2 and All2All: bring-up and convergence

This workflow brings up a multi-node MoE expert-parallel (EP) deployment on Ascend and
converges it before scaling. It covers two All2All paths: the fused MC2 pair
`npu_moe_distribute_dispatch` / `npu_moe_distribute_combine` for small-batch decode,
and the explicit All2All / double-routing path for large prefill. The stages are
ordered so that a topology or participation error is caught before a performance
experiment can hide it.

## Stage 1 - Admit the topology and the parallel arithmetic

Do not start the service until all of these are true:

| Check | Rule | Failure if skipped |
| --- | --- | --- |
| Device enumeration | A2: 8 logic devices; A3: 16 dies (8 dual-die cards); A5: read `npu-smi` | wrong world size, duplicate or missing ranks |
| Rank table | `device_id` local, `rank_id` global, `device_ip` per-die RoCE IP | wrong peer, `EI0006` |
| Host NIC | `HCCL_SOCKET_IFNAME` routes to every peer and is identical on all nodes | `EI0006` / connect timeout |
| EP arithmetic | `world_size % moe_tp_size == 0`; `total_experts % ep_size == 0` | expert map imbalance or startup assert |
| Batch arithmetic | global batch divisible by `moe_ep_size` | startup `ValueError` or uneven dispatch |
| SP arithmetic | `input_max_len` aligned to `attn_tp_size`; `max_position_embeddings` includes `next_n + 1` | tail truncation or position overflow |
| A2 MC2 ceiling | `experts_per_rank <= 24` for MC2; otherwise double-routing/eager | operator rejection or wrong path |
| Buffer | `HCCL_BUFFSIZE` equal on all ranks and sized for the largest collective | `HCCL_BUFFSIZE is too SMALL`, MC2 tiling failure |
| Transport | `HCCL_INTRA_ROCE_ENABLE=0` unless intra-node RDMA is intentional | transport mismatch, wrong counters |

Write the expert map down before launch:

```text
ep_size = world_size // moe_tp_size
experts_per_rank = total_routed_experts // ep_size
local_expert_num = experts_per_rank + (shared or zero experts placed locally)
```

On A3, a 16-die node with `moe_tp_size=1` gives `ep_size=16`; on A2, an 8-die node
gives `ep_size=8`. A5 must be enumerated; do not reuse either number. If the model has
zero/identity experts, include them in `expert_num` but keep them distinguishable from
routed experts for the MC2 `copy_expert_num` path.

Admission also includes a link check: all RoCE links UP, optical modules present, LLDP
peer port correct, and each NPU's `device_ip` pingable from its peers. This is the same
physical baseline used by the HCCL triage SOP; do it once, before the service starts.

## Stage 2 - Select the All2All path and give MC2 its own communicator

Use the phase to choose the path, not a single global switch:

| Phase | Typical shape | Path | Reason |
| --- | --- | --- | --- |
| Prefill | long sequence, many tokens | double-routing: `npu_moe_init_routing_v2` + two `all_to_all_single` + `npu_moe_re_routing` + `npu_moe_finalize_routing` | variable-length All2All sends only valid tokens; lower peak memory |
| Decode | small batch, latency-sensitive | MC2 `npu_moe_distribute_dispatch` + `npu_moe_distribute_combine` | fused dispatch/combine removes multiple All2All round trips and supports graph capture |
| TP-MoE (`moe_tp_size > 1`) | experts sharded by TP | `init_routing_v2` + GroupedMatmul + `finalize_routing`, then all-reduce on the TP group | EP path is not selected |

For MC2:

- Create a dedicated HCCL communicator, conventionally `moe_ep_group_mc2`, with the
  same topology as `moe_ep_group`. MC2 dispatch/combine must not share a communicator
  with EP, TP, CP or prefill collectives.
- Bind `dispatch_kwargs["group_ep"]` and `group_tp` to the dedicated group. Prebuild
  the kwargs in `process_weights_after_loading` / before graph capture so the trace
  does not see Python-side construction.
- Use `comm_alg="fullmesh_v2"` only when the model and operator support it and the
  model has no zero/identity experts. `fullmesh_v2` does not support zero experts; a
  model with `copy_expert_num > 0` keeps the dedicated group but omits `fullmesh_v2`.
- Size the dedicated group's buffer from the operator's `NEEDED_HCCL_BUFFSIZE` (or
  `calc_moe_hccl_buffer_size(..., is_full_mesh_v2=True)`), and let the ordinary
  `moe_ep_group` fall back to the default buffer.
- `HCCL_OP_EXPANSION_MODE=AIV` can reduce communication latency on supported
  platforms, but it can interact with prefill micro-batching and add run-to-run
  variation. Treat it as a measured tuning knob, not a correctness requirement.

For explicit All2All / double-routing:

- the first All2All exchanges per-expert token counts, the second exchanges tokens;
- the reverse path uses `npu_moe_re_routing` and `finalize_routing` to restore the
  original token order and apply routing weights;
- do not mix a variable-length `all_gather_v` / `reduce_scatter_v` path across
  multiple machines. Pad to a global maximum with `all_reduce(MAX)` and use fixed-size
  collectives, or use the supported All2All-v path.
- for All2All-v, the `input_splits` / `output_splits` arrays are produced by a
  non-blocking D2H copy. They must be synchronized before the host-side async
  collective reads them. In the single-local-expert branch, add an explicit
  `torch.npu.synchronize()` (or a blocking D2H) rather than relying on an implicit
  sync from `repeat_interleave`.

## Stage 3 - Converge correctness before tuning throughput

Run these checks in order; each one is a gate.

1. **Single-layer golden.** Compare the EP layer against a non-EP / single-rank
   reference with the same weights, routing logits and token IDs. Use a fixed seed and
   a small batch first.
2. **Routing-weight preservation.** Confirm the top-k weights are not cast, normalized
   or reordered between the router and the expert input. A weight that is silently
   converted changes the output even when the expert math is correct.
3. **Dispatch/combine inverse.** Check that `assist_info_for_combine` maps every
   received token back to its source rank, that `expert_token_num` sums to the
   dispatched token count, and that `ep_send_counts` / `ep_recv_counts` agree. A
   duplicate or dropped token is a correctness bug, not a performance issue.
4. **Zero-expert semantics.** If the model has zero/identity experts, verify
   `MoE(ori_x) = ori_x` for a token routed only to a zero expert, and verify
   `copy_expert_num` is set to the zero-expert count. Do not enable `fullmesh_v2` in
   this case.
5. **Graph/eager parity.** Run the same batch eager and in the target graph mode.
   Compare logits and the collective ledger. A graph that captures a different number
   of collectives will hang or diverge later.
6. **Collective participation.** Every DP rank that shares the MoE collective must
   enter it the same number of times. An idle DP rank must run a dummy/placeholder
   forward; a local-state gate must not decide whether the collective is entered.
7. **Cross-rank error check.** Search all rank logs for `EI0002`, `EI0006`,
   `HCCL_E_WAIT_TIMEOUT`, `HCCL_BUFFSIZE is too SMALL` and
   `MoeDistributeDispatchV2 do tiling failed`. A clean client response is not a clean
   cluster.

Only after all seven pass should throughput tuning begin. If a check fails, fix the
path or the arithmetic; do not compensate with a larger `HCCL_EXEC_TIMEOUT`.

## Stage 4 - Scale, tune and soak

Scale one dimension at a time:

- increase global batch before increasing EP;
- keep `experts_per_rank` within the A2 MC2 ceiling (24) or switch that deployment to
  double-routing/eager;
- keep the batch divisible by `moe_ep_size` and the SP length aligned to `attn_tp_size`;
- re-run the Stage 3 gates at every new EP size; a larger EP changes the communicator,
  the buffer requirement and the expert map.

Tune only what the evidence supports:

| Knob | When to change | Do not |
| --- | --- | --- |
| `HCCL_BUFFSIZE` | operator prints `NEEDED_HCCL_BUFFSIZE` or MC2 tiling fails | raise it to fix a timeout |
| `comm_alg` | fullmesh_v2 is supported and no zero experts exist | force it on a zero-expert model |
| dedicated `moe_ep_group_mc2` | MC2 is used at all | share the EP/TP/CP communicator |
| `HCCL_OP_EXPANSION_MODE=AIV` | measured latency benefit, no prefill micro-batch conflict | treat it as a correctness switch |
| number of communicators | buffer accounting shows HBM pressure | leave orphan groups per experiment |

Soak and fault injection:

- run a sustained mixed prefill/decode load, not only single requests;
- kill one non-critical rank or flap a link and verify the job fails cleanly rather
  than deadlocking silently;
- verify recovery does not reuse stale communicator state or stale expert maps;
- compare the tuned run to the pre-tuning baseline on the same ranks, batch and
  sequence distribution.

Promote the configuration only after repeated soak runs and a clean Stage 3 ledger. A
single successful request, or a single benchmark point, is not convergence.

## Stop conditions and rollback

Stop and roll back to the last verified configuration if:

- any Stage 3 gate fails at the new scale;
- a collective timeout appears without a matching participation or link explanation;
- `HCCL_BUFFSIZE` must grow on every batch increase without a stable operator-printed
  need;
- the expert map or routing weights change between runs with a fixed seed.

Rollback order: restore the previous EP/TP plan, then the previous dispatch path
(MC2 vs double-routing), then the previous buffer and communicator layout. Do not
leave a half-migrated `moe_ep_group_mc2` or a stale `fullmesh_v2` flag in the launch
script.
