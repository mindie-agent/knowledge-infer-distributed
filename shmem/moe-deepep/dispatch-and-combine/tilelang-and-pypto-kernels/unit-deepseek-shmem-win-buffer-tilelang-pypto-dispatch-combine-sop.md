---
category: sop
conditions:
  domain: shmem
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: shmem/moe-deepep/dispatch-and-combine/tilelang-and-pypto-kernels
evidence_strength: software_contract
id: unit-deepseek-shmem-win-buffer-tilelang-pypto-dispatch-combine-sop
inherits_from:
- unit-ascend-shmem-symmetric-heap-addressing-and-team-axioms
- unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_shmem_guide.md#L266-L561
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_shmem_guide.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: DeepSeek-V4 SHMEM Symmetric Window Buffer & Tilelang/pyPTO One-Sided MoE Dispatch/Combine
  SOP
triggers:
  error_signatures:
  - MoeDistributeCppBuffer
  - shmem_ub_put_nbi_new
  - WaitUntil timeout
  keywords:
  - shmem
  - deepep
  - moe-distribute
  - tilelang
  - pypto
  - shmem-ub-put-nbi
  - shmem-signal
  - wait-until
  - dispatch-combine
---
## Preconditions & Trigger

Apply this SOP when replacing legacy HCCL IPC handle exchange in `CamMoeDistributeDispatchV2` / `CamMoeDistributeCombineV2` with SHMEM symmetric window buffers (`_pyshmem.MoeDistributeCppBuffer`), or when authoring custom fused MoE Dispatch/Combine kernels in **Tilelang-Ascend** or **pyPTO** using one-sided SHMEM RMA (`put` / `get` / `signal` / `wait_until`).

## Pattern 1: Injecting SHMEM Symmetric Window Buffers into `MoeDistribute` Operators

1. **Eliminate Per-Rank IPC Handle Gathering**:
   - Standard `CamMoeDistributeDispatchV2` requires allocating a communication window in HBM and exchanging IPC handles across all EP ranks during setup.
   - Instead, initialize `_pyshmem.MoeDistributeCppBuffer(rank, num_ranks, win_size, local_server_Socket)` which allocates a single symmetric heap window (`win_size`, typically `>= 512 MB` depending on `max_bs * hidden_size * topk`) mapped identically across all EP peers.
2. **Pass `shmem_addr` to HCCL `GetWinContext`**:
   - Extract the raw symmetric base pointer via `win_buff.get_base_ptr()` (`uint64_t`) and pass it to the HCCL window context initializer (`ep_hcomm_info` / `GetWinContext(shmem_addr, win_size)`) before invoking `torch_npu.npu_moe_distribute_dispatch_v2` and `torch_npu.npu_moe_distribute_combine_v2`.
   - Ensure `win_buff.finalize()` is called strictly after `torch.npu.synchronize()` during engine teardown so in-flight MTE/RMA writes do not fault on unmapped symmetric pages.

## Pattern 2: Tilelang-Ascend One-Sided UB-to-Remote-GM RMA (`T.shmem_ub_put_nbi_new`)

1. **Direct Unified Buffer (UB) to Remote Global Memory (GM) Put**:
   - In Tilelang-Ascend kernels (`@tilelang.jit`), allocate symmetric GM tensors on the host via `aclshmem_malloc`.
   - Inside the kernel, copy or compute expert token tiles in local Unified Buffer (`T.alloc_ub`), synchronize Vector/Cube pipelines (`T.set_flag("v", "mte3", 0)` / `T.wait_flag("v", "mte3", 0)`), and issue non-blocking one-sided RMA directly from local UB to the target rank's symmetric GM address:
     ```python
     T.shmem_ub_put_nbi_new(
         ub_buf,
         remote_gm_symm_ptr,
         transfer_elems,
         target_pe_rank,
     )
     ```
   - This bypasses staging through local GM, cutting local HBM bandwidth consumption in half during MoE token routing.

## Pattern 3: pyPTO One-Sided Put + Signal + WaitUntil Synchronization Contract

1. **Ordering Data Payload (`ShmemPut`) Before Completion Flag (`ShmemSignal`)**:
   - When implementing custom Dispatch/Combine in `pyPTO`, allocate two symmetric tensors via `aclshmem_malloc`: a data buffer (`symm_data_buf`) and a zero-initialized `int64` / `uint64` signal flag buffer (`symm_sig_buf`).
   - In the producer kernel on Rank $i$ sending tokens to Rank $j$:
     1. Issue `pypto.ir.ShmemPut(dst_symm_data_ptr, src_ub_tile, size_bytes, pe=target_rank)` to write the token payload into Rank $j$'s symmetric GM.
     2. Issue a memory fence / barrier (`ShmemQuiet` or sequential MTE3 ordering) followed by `pypto.ir.ShmemSignal(dst_symm_sig_ptr, signal_val=1, sig_op=SHMEM_SIG_SET, pe=target_rank)` (or atomic `SHMEM_SIG_ADD`).
2. **Bounded Receiver Spin-Wait (`WaitUntil`) & Reset**:
   - On the consumer rank, invoke `pypto.ir.WaitUntil(local_symm_sig_ptr, cmp_op=SHMEM_CMP_EQ, cmp_val=expected_peers)` before loading the received tokens from `local_symm_data_ptr` into UB/L1 for grouped-matmul computation.
   - **Deadlock Prevention Invariant**: Every rank MUST clear its consumed `local_symm_sig_ptr` slots back to `0` (or use monotonically increasing step counters `step_id` with `SHMEM_CMP_GE`) before the next layer's Dispatch phase begins; otherwise, stale non-zero flags cause the next layer's `WaitUntil` to return prematurely on uninitialized token buffers.

## Verification

- Benchmark `CamMoeDistributeDispatchV2` + `CombineV2` backed by `MoeDistributeCppBuffer` against standard IPC mode and verify identical output tensors (`torch.allclose(..., atol=1e-3, rtol=1e-3)`).
- Validate custom Tilelang/pyPTO SHMEM kernels across 8/16 NPU ranks over 100 consecutive iterations to confirm zero `WaitUntil` hangs or stale-flag races.
