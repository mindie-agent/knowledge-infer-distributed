---
schema: mindie-graded-unit/1
id: unit-hccl-aicpu-cost-model-calibration-and-hvm-estimate-mutator
title: HCCL AICPU Cost-Model α-β-γ Calibration, HVM Trace Profiling, and Selector
  Threshold Mutator
domain_path: hccl/aicpu-collectives/debug-and-perf-tuning
tier: L4
category: mutator
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  communication_stack: hccl
  subsystem: aicpu-collective-algorithm
  optimization_lever: cost-model-and-pipeline-chunking
inherits_from:
- unit-hccl-aicpu-dataflow-spec-and-five-stride-addressing-sop
- unit-hccl-aicpu-executor-binding-and-selector-registration-workflow
- unit-hccl-aicpu-checker-mismatch-and-algo-selection-debug-sop
supersedes: []
triggers:
  error_signatures:
  - HCCL selector chose suboptimal Ring over HalvingDoubling
  - HVM Estimate latency diverges from hardware profile
  - high startup overhead α in multi-round collective
  - low link utilization β in single-channel AICPU collective
  keywords:
  - hccl-aicpu-perf-optimize
  - HVM Estimate cost model
  - alpha beta gamma collective model
  - Ring vs Halving-Doubling crossover
  - Hierarchical Mesh AllReduce
  - multi-channel OpExpansion
provenance:
  source_type: curated_repo
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-perf-optimize/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-aicpu-perf-optimize/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-test-tool-hvm/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/communication/hccl-env-perf/SKILL.md
  source_digest: 8f5e55487bfe423b26399451221947d203897df9036a3d29a65a03fb2e254c48
---

## HCCL AICPU Cost-Model Calibration & Pipeline Optimization Mutator

### Mutation 1: Calibrate the $\alpha$-$\beta$-$\gamma$ Analytical Cost Model in `EstimateTime`

1. **Analytical Latency Decomposition**:
   - Every AICPU collective algorithm estimates its execution time $T(N, R)$ as the critical path across $K$ synchronous rounds of per-hop startup latency $\alpha$, per-byte inverse bandwidth $\beta$ (distinguishing intra-node HCCS $\beta_{\text{intra}}$ vs inter-node RoCE $\beta_{\text{inter}}$), and per-byte reduction compute cost $\gamma$:
     - **Ring AllReduce ($2(R-1)$ steps)**:
       $$T_{\text{ring}}(N, R) = 2(R - 1)\alpha + 2\frac{R - 1}{R} N \beta + \frac{R - 1}{R} N \gamma$$
     - **Recursive Halving-Doubling ($2 \log_2 R$ steps)**:
       $$T_{\text{rhd}}(N, R) = 2(\log_2 R)\alpha + 2\frac{R - 1}{R} N \beta + \frac{R - 1}{R} N \gamma$$
     - **Hierarchical 2D ($R = R_{\text{intra}} \times R_{\text{inter}}$)**:
       Reduces inter-node RoCE rounds from $2(R-1)$ to $2(R_{\text{inter}}-1)$ (or $2\log_2 R_{\text{inter}}$) by executing intra-node ReduceScatter $\to$ inter-node AllReduce $\to$ intra-node AllGather.
2. **Calibrate Crossover Thresholds via `HVM Estimate`**:
   - Run `hccl-test-tool-hvm` in `Estimate` mode across geometric message sizes ($4\text{ KB} \dots 512\text{ MB}$) and export the Chrome trace (`trace.json`) to inspect per-channel idle gaps.
   - Update `EstimateTime` coefficients $(\alpha_{\text{intra}}, \alpha_{\text{inter}}, \beta_{\text{intra}}, \beta_{\text{inter}}, \gamma)$ so the selector picks Recursive Halving-Doubling / Mesh for latency-bound small/medium payloads ($N < N_{\text{cross}}$) and multi-channel Ring / Hierarchical Ring for bandwidth-bound payloads ($N \ge N_{\text{cross}}$).

### Mutation 2: Multi-Channel Splitting & Pipeline Chunking inside `BuildDataflow`

1. **Saturate Bidirectional Link Bandwidth with Multi-Channel Splitting**:
   - When single-channel DMA fails to saturate HCCS/RoCE bandwidth on medium-to-large tensors ($N \ge 2\text{ MB}$), split the tensor into $C \in \{2, 4\}$ independent sub-slices and assign each sub-slice to a distinct hardware transport channel (`channel_id = 0 .. C-1`) in `BuildDataflow`.
2. **Overlap Inter-Node RoCE with Intra-Node HCCS Pipelining**:
   - In Hierarchical collectives, chunk the outer dimension into $M$ pipeline stages so Stage $m+1$'s intra-node ReduceScatter over HCCS executes concurrently with Stage $m$'s inter-node AllReduce over RoCE RDMA.
