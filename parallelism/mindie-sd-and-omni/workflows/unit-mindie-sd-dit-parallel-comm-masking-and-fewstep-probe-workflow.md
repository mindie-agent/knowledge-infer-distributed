---
schema: mindie-graded-unit/1
id: unit-mindie-sd-dit-parallel-comm-masking-and-fewstep-probe-workflow
title: MindIE-SD dit-parallel-opt topology selection, CP x sparse seam, comm-masking c/f bound, and DiT-only few-step multirank workflow
tier: L3
category: workflow
evidence_strength: reproduced_empirical
domain_path: parallelism/mindie-sd-and-omni/workflows
inherits_from:
  - unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
  - unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
  - unit-mindie-sd-usp-cfg-fa-power-cap-eplb-and-yuanrong-sop
conditions:
  cann_versions:
    - ">=8.5.0"
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
  workloads:
    - Multi-rank DiT parallelism selection, comm-stream masking optimization, and CP x sparse attention integration
triggers:
  error_signatures:
    - "comm-stream masking achieves only ~50% overlap instead of expected 1 - 1/n"
    - "CP + sparse attention output is byte-identical to dense baseline or degrades quality"
    - "per-step latency across card group uniformly elevated by ~10x despite npu-smi Health=OK"
  keywords:
    - dit-parallel-opt
    - mask_bound_calc
    - bubble_attribution
    - collective_attribution
    - fewstep_multirank_probe
    - cp-sparse-combination
    - parallel-form-selection
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
  source_digest: a48ef062ddc8a6354655aafe27336e4ee7ccf2e45cc7e0c494e4f6e9fb73256c
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
  supersedes: []
revision: 1
---
## DiT Parallelism & Communication Masking Workflow

### Step 1: Enforce Same-Domain Topology Card Selection & 4-Step Smoke Gate
1. Inspect `npu-smi info -t topo` and `npu-smi info` before launching any multi-rank job:
   - **2 cards**: Must be an adjacent pair within the same `UB`/`HCCS` interconnect island.
   - **4 cards**: Must be a complete same-island `UB`/`HCCS` group (never stitch across `SYS`/PCIe boundaries unless explicitly benchmarking cross-island behavior).
   - **8 cards**: Full node (`2 x 4` islands).
2. **4-Step Smoke Gate for Residual Driver Faults**: Run a 4-step smoke test and inspect per-step DiT latency. If per-step latency across the card group is uniformly elevated by ~10$\times$ despite `npu-smi Health=OK`, a prior `SIGKILL` left HCCL/driver state wedged—reset the devices or switch to a clean card group before collecting data.

### Step 2: Select Parallel Form with the DiT-Only Few-Step Multirank Protocol (`fewstep_multirank_probe.py`)
1. **DiT-Only Metric Scope**: Measure **only** the `<Pipeline>.diffuse` denoising stage wall clock (minimum across ranks in $\mu\text{s}$), strictly excluding VAE decode, text encoding, weight loading, warmup, and HTTP serialization (whose variance is an order of magnitude larger than DiT step variance on few-step runs).
2. **Extrapolation Gate ($r = \text{DiT\_step}(\text{anchor}) \div \text{DiT\_step}(\text{few-step})$)**:
   - Run $\ge 3$ hot repetitions per cell in the same window.
   - Few-step ranking can extrapolate to full steps **only if** $|r - 1| \le 5\%$, per-cell dispersion $\le 3\%$, and rank ordering agrees between few-step and anchor runs.
   - Re-verify at full step count whenever features change single-step structure (quantization shrinks compute and raises comm share; sparse swaps attention kernels; DLO unlocks previously OOM parallel forms).
3. **Pure Ulysses vs Composite AllGather-KV $\times$ Ulysses (`--usp k --allgather-degree m`)**:
   - **GQA Decisive Rule**: Composite CP (`AllGather-KV`) reduces cross-island traffic **only when the model uses GQA** (KV heads $<$ Q heads). When $H_Q = H_K = H_V$, composite CP transfers the same cross-island volume as pure Ulysses while adding extra local plumbing compute.
   - **Topology Reversal**: Within a single `UB` island (4 cards), **bulk** `USP4` is fastest; across `SYS`/PCIe islands, **head-parallel** asynchronous small `AlltoAll` can outperform bulk `AlltoAll` on wall clock (note: `torch.compile` is incompatible with head-parallel due to Dynamo `recompile_limit` fallback).

### Step 3: Enforce the 4-Part Seam Contract When Stacking CP $\times$ Sparse Attention
1. **Gather-Before-Sparse**: Sparse block selection requires the **full K/V sequence**; use `AllGather-KV x Ulysses` (never Ring attention).
2. **Global Window Offset**: Compute causal/window offset as `full_kv_len - local_logical_q_len` (0 when no joint text-video concat tensor is present, **not** `k_len - q_len`).
3. **Block-Size Divisibility**: Every rank's KV shard length must be an exact integer multiple of the sparse/quantization block size (e.g., 64 or 128).
4. **Per-Head Mask Isolation**: Block-sparse masks are per-head and must never be reused across head shards.

### Step 4: Bound, Implement, and Verify Comm-Stream Masking (`mask_bound_calc.py` & `bubble_attribution.py`)
1. **Calculate the Theoretical Masking Bound First**:
   - Let $C$ be the per-layer communication time of the target collective family, $F$ be the independent compute time available to overlap (e.g., sparse FA kernel duration), $n$ be the number of head chunks, $c = C/n$, and $f = F/n$:
     $$\text{Max Masking Ratio} = \begin{cases} 1 - \frac{1}{n} & \text{if } f \ge c \ (F \ge C) \\ 1 - \frac{c + (n-1)(c-f) + \text{drain}}{C} & \text{if } f < c \ (F < C, \text{independent of } n) \end{cases}$$
   - When $c/f > 1$, increasing chunk count $n$ yields **zero** additional overlap; you must either compress communication bytes $C$ (FP8/MXFP8 comm quantization) or move more independent compute into $F$.
2. **Implementation Invariants**:
   - Permute the chunk dimension to the outermost axis **once** before the loop so each chunk is a contiguous slice (never materialize `as_strided` copies per chunk).
   - Place **only HCCL collectives** on the side `comm_stream`; keep **all AI Core compute/reshape/cast kernels on the main compute stream** (`side-stream AI Core kernel count == 0`, otherwise side-stream vector ops steal Vector cores from FA).
   - Synchronize via **per-chunk `torch.npu.Event`** (not `wait_stream`), keep at most one communicator active at a time, and use **software-pipelined dispatch** (enqueue chunk $g+1$ forward `AlltoAll` **before** chunk $g$ reverse output `AlltoAll` so reverse `AlltoAll` overlaps with chunk $g+1$ FA instead of serializing after the entire layer).
3. **Three-Criterion Engagement & Bubble Verification**:
   - Verify collective invocation count increases by $\approx n\times$, side-stream AI Core kernels $= 0$, `Communication(Not Overlapped)` drops in `step_trace_time.csv`, and output `md5` is **byte-identical** to unmasked USP (Ulysses `AlltoAll` is a pure permutation with no reduction).
