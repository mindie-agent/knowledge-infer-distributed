---
schema: mindie-graded-unit/1
id: unit-mindie-sd-usp-cfg-fa-power-cap-eplb-and-yuanrong-sop
title: MindIE-SD and vLLM-Omni USP Ulysses/KV-AllGather, HcclAlltoAllV workaround, FA_Power_Cap, Dynamic EPLB, and YuanrongTransferEngineConnector SOP
tier: L2
category: sop
evidence_strength: software_contract
domain_path: parallelism/mindie-sd-and-omni/collectives-and-connectors
inherits_from:
  - unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
  - unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
conditions:
  cann_versions:
    - ">=8.5.0"
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
  workloads:
    - Multi-NPU DiT sequence/context parallelism, MoE expert load balancing, thermal power-cap mitigation, and cross-stage Omni NPU memory transfer
triggers:
  error_signatures:
    - "HcclAlltoAllV SIGSEGV on split path"
    - "PCIe TDP PMC thermal throttling during ultra-long video FlashAttention"
    - "YuanrongTransferEngineConnector requires an NPU memory pool"
    - "profile shows only allGather and no allToAll when sequence sharding is enabled"
  keywords:
    - usp_attention
    - kv_gather_group
    - HcclAlltoAllV
    - FA_Power_Cap
    - insertcomm
    - blockattn
    - Dynamic-EPLB
    - EPLB_AUTH_KEY
    - YuanrongTransferEngineConnector
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/parallelism.md
  source_digest: 15c2118d6d627dd36b500626f40662a1361ad03bf55aeadfa50eb7368da07f72
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/parallelism.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/dit-parallel-opt/SKILL.md
    - vllm-project/vllm-omni@20778540fd04eb3753632d0ea1c966f8415cc5f3:vllm_omni/platforms/npu/omni_connectors/yuanrong_transfer_engine_connector.py
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Configure `usp_attention` (Ulysses AlltoAll + Composite KV-AllGather)**:
   - **Explicit `ParallelConfig` Wrapper Requirement**: Setting sequence sharding alone on a diffusion attention processor does **not** activate Ulysses head-sharded `AlltoAll` (profiling will show only `AllGather` and zero `AlltoAll`). You must assign `ParallelConfig(context_parallel_config=cp_cfg)` (wrapped in `ParallelConfig`, not bare `ContextParallelConfig`) to each `attn.processor._parallel_config`.
   - **Composite `ulysses_group` + `kv_gather_group` Semantics**: In `mindiesd`, `kv_gather_group` performs **KV-AllGather** across the outer context-parallel group after Ulysses `AlltoAll` (it is **not** Ring Attention; `ring_degree > 1` is unsupported with `attn_mask` and bypasses sparse attention backends).
   - **Low-Precision Communication & Dual-Stream Overlap**: `usp_attention` supports optional `exposed` or `all` FP8 / MXFP8 communication quantization with dual-stream compute-communication overlap.

2. **Bypass `HcclAlltoAllV` Split-Path `SIGSEGV` via Uniform `S_PAD` Padding**:
   - On CANN versions where uneven `HcclAlltoAllV` (split path) triggers `SIGSEGV` while uniform `HcclAlltoAll` succeeds:
     1. Pad each rank's input chunk along the sequence dimension to `S_PAD` (rounded up to a multiple of `128` from the global `max(out_sizes)` across all ranks).
     2. Execute uniform `HcclAlltoAll(count = S_PAD * row_elems)`.
     3. Slice the first `out_sizes[j]` valid rows per chunk.
   - Verify engagement by checking that `kernel_details.csv` replaces `hcom_alltoallv` with `hcom_alltoall` and output matches the reference bitwise.

3. **Mitigate Ultra-Long Video PCIe Power-Cap Throttling (`FA_Power_Cap`)**:
   - During ultra-long sequence video generation on PCIe cards (`> 100K` tokens), sustained full-sequence FlashAttention Cube bursts can exceed the instantaneous TDP envelope and trigger hardware PMC frequency throttling (`aicore_freq` drop).
   - Enable `FA_Power_Cap` in `mindiesd` via one of two modes:
     - `--comm_type 1` (`insertcomm`): Splits attention heads into chunks and interleaves `AlltoAll -> FA` per head chunk so communication intervals smooth peak Cube power draw.
     - `--comm_type 2` (`blockattn`): Splits Query ($Q$) into 2 sequence blocks and inserts a brief synchronization interval between the two FA calls.

4. **Configure `Dynamic EPLB` for MoE Diffusion Models (`mindiesd.eplb`)**:
   - For MoE DiT models (`MiniMax-H3`, `HunyuanImage-3.0`), enable dynamic expert load balancing (`mindiesd.eplb`) in `A2A` (AlltoAll), `AG` (AllGather), or `EX` (expert-parallel) mode with bounded per-step expert migration (`max_move`) and mandatory IPC authentication (`export EPLB_AUTH_KEY=<secret>`).

5. **Configure `YuanrongTransferEngineConnector` for Cross-Stage NPU Disaggregation**:
   - In multi-stage `vllm-omni` deployments across NPU workers, configure `YuanrongTransferEngineConnector` with `protocol: "ascend"` and `memory_pool_device: "npu"` (`_resolve_pool_device` rejects non-NPU pool devices).
   - The connector allocates a contiguous `torch.uint8` NPU memory pool registered via `engine.register_memory(base_ptr, pool_size)` managed by a 4096-byte aligned `BufferAllocator`, uses a ZMQ control listener (`QUERY_INFO` / `CLEANUP_KEY`), and calls `_sync_tensor_device` (`current_stream(device).synchronize()`) before exposing sender buffer addresses so downstream stages never pull incomplete device writes.
