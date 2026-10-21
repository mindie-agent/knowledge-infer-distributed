---
category: sop
conditions:
  domain: vllm-ascend
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: vllm-ascend/pipeline-parallel/dynamic-chunked-pp/quadratic-latency-and-calibration
evidence_strength: software_contract
id: unit-vllm-ascend-dynamic-chunked-pp-quadratic-predictor-and-calibration-sop
inherits_from:
- unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md#L1-L260
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md#L1-L210
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: vLLM-Ascend Dynamic Chunked Pipeline Parallelism (DCPP) Quadratic Latency Predictor
  & Calibration SOP
triggers:
  error_signatures:
  - enable_dynamic_chunked
  - DynamicChunkScheduler
  - pipeline bubble
  keywords:
  - vllm-ascend
  - pipeline-parallelism
  - dynamic-chunked-pp
  - dcpp
  - quadratic-latency-model
  - token-budget
  - pipeline-bubble
---
## Preconditions & Trigger

Apply this SOP when enabling or tuning **Dynamic Chunked Pipeline Parallelism (DCPP)** in `vllm-ascend` (`pipeline_parallel_size > 1`) for long-sequence prefill or mixed prefill/decode workloads on Ascend NPUs to minimize pipeline bubbles caused by quadratic causal-attention latency growth.

## Why Fixed-Token Chunking Causes Pipeline Bubbles in PP

1. **Quadratic Attention Scaling Asymmetry**:
   - In standard chunked prefill with Pipeline Parallelism (`PP`), splitting a long prompt of length $L$ into $K$ equal token chunks ($c_1 = c_2 = \dots = c_K = L/K$) results in monotonically increasing execution time per chunk because later chunks attend to all preceding cached KV tokens:
     $$T(c_i, s_i) = a \cdot c_i^2 + b \cdot c_i \cdot s_i + d \cdot c_i + e$$
     where $c_i$ is the number of scheduled tokens in chunk $i$ and $s_i = \sum_{j < i} c_j$ is the already-computed prefix context length.
   - Because later chunks ($i \to K$) have much larger $s_i$, their stage forward time $T(c_i, s_i)$ exceeds earlier chunks, creating stage stalls (pipeline bubbles) across the PP ring.
2. **Equal-Duration Dynamic Chunking Objective**:
   - DCPP replaces fixed token-count chunking with **latency-equalized chunking** (`DynamicChunkScheduler`), solving the quadratic equation at each scheduling step so every micro-batch across the pipeline stages targets the same per-stage execution duration $T_{\text{target}}$:
     $$a \cdot c_i^2 + (b \cdot s_i + d) \cdot c_i + (e - T_{\text{target}}) = 0$$
   - Consequently, early chunks with $s_i = 0$ receive a larger token budget $c_1$, while later chunks with large prefix context $s_i$ receive progressively smaller token budgets $c_i$, equalizing stage execution times and collapsing PP bubble overhead.

## Configuration & Offline/Online Coefficient Calibration SOP

1. **Calibrate Hardware/Model Quadratic Coefficients $(a, b, d, e)$**:
   - The predictor requires hardware-specific and model-specific coefficients reflecting AI Core Cube/Vector throughput on the target TP/PP slice:
     - $a$: Self-attention quadratic coefficient ($\text{ms} / \text{token}^2$).
     - $b$: Cross-context attention coefficient ($\text{ms} / (\text{token} \times \text{cached\_token})$).
     - $d$: Linear FFN / projection coefficient ($\text{ms} / \text{token}$).
     - $e$: Per-stage constant launch & P2P communication overhead ($\text{ms}$).
   - Run the DCPP profiling/calibration benchmark across representative `(chunk_size, context_len)` grid points (`c_i \in [256, 512, 1024, 2048, 4096]`, `s_i \in [0, 4096, 8192, 16384, 32768]`) and fit $(a, b, d, e)$ via non-negative least squares (`scipy.optimize.nnls`).
2. **Enable DCPP in `--additional-config`**:
   - Pass the calibrated predictor parameters and token bounds to `vllm serve` via `--additional-config`:
     ```json
     {
       "dynamic_chunked_config": {
         "enable_dynamic_chunked": true,
         "Reference_latency_ms": 50.0,
         "min_chunk_tokens": 128,
         "max_chunk_tokens": 8192,
         "quad_coeff_a": 1.2e-6,
         "cross_coeff_b": 2.4e-6,
         "linear_coeff_d": 0.015,
         "const_coeff_e": 1.8
       }
     }
     ```
   - **Boundary Clamping Invariant**: Always set `min_chunk_tokens >= 128` (aligned to Ascend Cube fractal tile size `16` or `64`) and `max_chunk_tokens <= max_num_batched_tokens` so extreme quadratic roots at very long contexts ($s_i > 64\text{K}$) never degenerate into tiny unaligned micro-chunks that starve Cube utilization.

## Verification

- Compare `vllm serve` TTFT and total throughput on 16K–64K input prompts with `enable_dynamic_chunked: true` vs. static `--max-num-batched-tokens`, and verify stage forward durations in Ascend profiler traces remain uniform across chunks $c_1 \dots c_K$.
