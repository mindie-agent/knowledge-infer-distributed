---
category: sop
conditions:
  accelerator: ascend
  communication_stack: hccl
  diagnosis_domain: comm-group-and-buffsize-rdma
  runtime:
  - torch_npu
  - mindie
domain_path: hccl/topology-and-network/roce-hccs-diagnostics
evidence_strength: reproduced_empirical
id: unit-ascend-hccl-comm-group-construction-and-buffsize-rdma-sop
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-ascend-hccl-notify-wait-timeout-and-hccn-link-triage-sop
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/hccl_comm_group_construction_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/hccl_buffsize_too_small.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/moe_dispatch_combine_hccl_buffsize.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/pd_kv_transfer_timeout_from_hccl_rdma_budget.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/tensor_prep_before_hccl_collective.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-env-config/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-env-debug/SKILL.md
  source_digest: d6c40aa1ba022f59bcc69f94a43215e9f01f4bdb22fe4627a493db48c4376def
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/hccl_comm_group_construction_pitfalls.md
  source_type: curated_repo
revision: 2
schema: mindie-graded-unit/1
status: active
supersedes: []
tier: L2
title: Ascend HCCL Communicator Group Construction, HCCL_BUFFSIZE Sizing, and PD KV-Transfer
  RDMA Budget SOP
triggers:
  error_signatures:
  - 'EI0006: HCCL buffer size is too small'
  - Notify wait timeout during first collective on sub-group
  - PD disaggregated KV transfer RDMA timeout under burst load
  - hcclCommInitRootInfo rank order mismatch across sub-groups
  keywords:
  - HCCL_BUFFSIZE
  - hccl_buffsize_too_small
  - new_group global creation order
  - tensor_prep_before_hccl_collective
  - pd_kv_transfer_timeout_from_hccl_rdma_budget
  - HCCL_RDMA_TIMEOUT
  - HCCL_RDMA_RETRY_CNT
---
## HCCL Communicator Group Construction, Buffer Sizing, and RDMA Budget SOP

### Step 1: Enforce Deterministic Communicator Group Construction (`torch.distributed.new_group`)

1. **Identical Global Call Order for `new_group`**:
   - Every rank in `WORLD` must invoke `torch.distributed.new_group(ranks=...)` in the **exact same global sequence**, even for sub-groups to which the calling rank does not belong.
   - **Forbidden Anti-Pattern**: Calling `new_group` conditionally inside `if my_rank in tp_group:` causes rank-to-communicator root handshakes to cross-wire or deadlock at the first collective launch.
2. **Monotonically Sorted Rank Lists & Explicit Warmup**:
   - Always sort `ranks` in ascending global rank order (`sorted(sub_ranks)`) so Rank 0 of the sub-group is deterministic across all nodes.
   - Because `torch_npu` lazily initializes the underlying `HcclComm` handle on the **first collective call** on that group, run an explicit 1-element `dist.all_reduce(dummy_tensor, group=g)` followed by `torch.npu.synchronize()` during model startup to separate communicator creation errors from inference execution.
3. **Contiguous Device Tensor Preparation Before Any HCCL Collective**:
   - Ensure every tensor passed to `dist.all_reduce`, `dist.all_gather_into_tensor`, `dist.reduce_scatter_tensor`, or `dist.all_to_all_single` is on the bound NPU device (`tensor.is_npu`), non-empty (`tensor.numel() > 0`), and memory-contiguous (`tensor = tensor.contiguous()`). Non-contiguous views (e.g., transposed attention heads before projection) trigger hidden host-dispatched copies or misaligned DMA reads.

### Step 2: Size `HCCL_BUFFSIZE` for MoE AllToAll & Multi-Communicator Workloads

1. **Prevent `EI0006` (Buffer Too Small) and Silent Multi-Chunk Serialization**:
   - Each active `HcclComm` allocates $2 \times \text{HCCL\_BUFFSIZE}$ (default `200 MB` per communicator) of NPU HBM for RDMA/SDMA staging.
   - For MoE models using `npu_moe_distribute_dispatch` / `npu_moe_distribute_combine` or large-batch `AlltoAllV`, a single communication buffer must hold worst-case token routing imbalance:
     $$\text{HCCL\_BUFFSIZE (MB)} \ge \left\lceil \frac{2 \times B_{\text{micro}} \times S_{\text{seq}} \times K_{\text{top}} \times H \times \text{sizeof(dtype)} \times \text{EP\_world\_size}}{1024^2} \right\rceil$$
   - Set `export HCCL_BUFFSIZE=512` (or `768` for wide EP $\ge 32$) while auditing total HBM footprint across TP + DP + EP + PP communicators:
     $$\text{Total HCCL HBM} \approx N_{\text{active\_comms}} \times 2 \times \text{HCCL\_BUFFSIZE}$$

### Step 3: Tune RoCE RDMA Retransmission Budget for PD-Disaggregated KV Transfer

1. **Why PD KV-Cache Transfers Timeout Under Micro-Burst Congestion**:
   - In Prefill-Decode (PD) disaggregated serving (`LLMDataDist` / `Mooncake` / `NIXL` over HCCL RoCE), concurrent multi-layer KV block pulls across P-instances and D-instances create RoCE PFC pause frames and transient packet drops.
   - Hardware RoCE retransmission timeout per attempt is $4.096\,\mu\text{s} \times 2^{\text{HCCL\_RDMA\_TIMEOUT}}$, repeated `HCCL_RDMA_RETRY_CNT` times. If the budget is too small, RoCE tears down the QP with `link error / cqe error` during transient congestion.
2. **Production PD RoCE Tuning**:
   - Set `export HCCL_RDMA_TIMEOUT=20` ($\approx 4.29\text{ s}$ base backoff ceiling) and `export HCCL_RDMA_RETRY_CNT=7` across both Prefill and Decode nodes, and verify DSCP/PFC lossless queue priority (`HCCL_RDMA_TC` / `HCCL_RDMA_SL`) matches the ToR switch QoS profile.
