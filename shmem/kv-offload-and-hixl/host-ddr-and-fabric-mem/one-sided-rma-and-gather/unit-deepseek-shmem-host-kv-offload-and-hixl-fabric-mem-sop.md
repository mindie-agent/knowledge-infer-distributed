---
category: sop
conditions:
  domain: shmem
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: shmem/kv-offload-and-hixl/host-ddr-and-fabric-mem/one-sided-rma-and-gather
evidence_strength: software_contract
id: unit-deepseek-shmem-host-kv-offload-and-hixl-fabric-mem-sop
inherits_from:
- unit-ascend-shmem-symmetric-heap-addressing-and-team-axioms
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_shmem_guide.md#L1-L265
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_shmem_guide.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: DeepSeek-V4 SHMEM Symmetric Host DDR KVCache Offload & HIXL Fabric-Memory One-Sided
  RMA SOP
triggers:
  error_signatures:
  - aclshmem_malloc failed
  - OPTION_ENABLE_USE_FABRIC_MEM
  - npu_gather_selection_kv_cache
  keywords:
  - shmem
  - aclshmem-malloc
  - host-side
  - npu-gather-selection-kv-cache
  - hixl
  - fabric-mem
  - d2rh
  - rh2d
  - deepseek-v4
---
## Preconditions & Trigger

Apply this SOP when implementing or troubleshooting KVCache Host DDR offloading on Ascend 910B/910C for DeepSeek-V4 / MLA long-context inference using either:
1. **Intra-Node / SuperPod Symmetric Host Memory (`SHMEM` `MemType.HOST_SIDE`)** with fused `torch_npu.npu_gather_selection_kv_cache` operators, or
2. **Cross-Node One-Sided RDMA/HCCS (`HIXL` `TransferSync` / `TransferAsync`)** with symmetric fabric memory (`OPTION_ENABLE_USE_FABRIC_MEM: "1"`).

## Approach 1: Symmetric Host DDR Pool via `aclshmem_malloc` & `npu_gather_selection_kv_cache`

1. **Initialize Hybrid Device + Host Symmetric Heap (`aclshmemx_init_attr_t`)**:
   - Standard `SHMEM` initialization only allocates NPU HBM (`local_mem_size`). To enable direct NPU operator access to Host DDR without CPU staging copies, configure `aclshmemx_host_Attr_t` inside `aclshmemx_init_attr_t` before calling `aclshmemx_set_attr`:
     - Set `attributes.local_mem_size = <device_pool_bytes>` (e.g., `1024 * 1024 * 1024` for 1 GB Device HBM).
     - Set `attributes.host_attr.size = <host_pool_bytes>` (e.g., `16 * 1024 * 1024 * 1024` for 16 GB Host DDR).
   - Allocate the Host KVCache tensor via `aclshmem_malloc(kv_shape, dtype=torch.bfloat16, device=f"npu:{rank}", mem_type=MemType.HOST_SIDE)`.
   - **Hardware Addressing Invariant**: Even though `mem_type=MemType.HOST_SIDE` physically resides in Host DDR, `aclshmem_malloc` maps it into the NPU's unified symmetric virtual address space and returns a tensor with `device="npu:<rank>"`, allowing AI Core / DMA engines to read and write Host DDR cache blocks directly inside NPU compute streams.
2. **Execute Fused Host Gather/Scatter (`torch_npu.npu_gather_selection_kv_cache`)**:
   - Avoid launching per-block `aclrtMemcpyAsync` host-to-device calls from Python. Invoke the fused operator `torch_npu.npu_gather_selection_kv_cache` passing the `MemType.HOST_SIDE` tensor directly alongside the Device HBM KVCache tensor and page selection indices so the NPU DMA engine pipelines sparse KVCache prefetching and eviction asynchronously.

## Approach 2: Cross-Node HIXL One-Sided RMA (`D2rH` / `rH2D`) with Fabric Memory

1. **Enable Symmetric Fabric Memory Registration (`OPTION_ENABLE_USE_FABRIC_MEM`)**:
   - When using CANN `HIXL` (`hixl.Engine`) to transfer KVCache between local NPU HBM (`MEM_DEVICE`) and remote Host DDR (`MEM_HOST`) across nodes (`D2rH` Write and `rH2D` Read), pass `{"OPTION_ENABLE_USE_FABRIC_MEM": "1"}` during `engine.initialize(local_engine_id, options)`.
   - **Registration Ordering Invariant**: With `OPTION_ENABLE_USE_FABRIC_MEM="1"`, every rank in the HIXL communication group MUST register `MEM_DEVICE` and `MEM_HOST` buffers of **identical byte sizes and in the exact same call order** (`engine.register_mem(addr, size, MEM_DEVICE)` followed by `engine.register_mem(host_addr, host_size, MEM_HOST)`).
   - Why this invariant exists: Fabric-memory mode exchanges a compact base descriptor once during connection setup and computes remote rkeys/addresses via deterministic symmetric offsets; asymmetric buffer sizes or out-of-order `register_mem` calls across ranks cause remote RMA reads/writes to hit unmapped pages or corrupt adjacent tensors.
2. **One-Sided Transfer Execution**:
   - Connect peers via `engine.connect(remote_engine_id, timeout_ms)`.
   - Offload KVCache from local NPU HBM to remote Host DDR (`D2rH`) using `TransferOp.WRITE`, and prefetch KVCache from remote Host DDR back into local NPU HBM (`rH2D`) using `TransferOp.READ` via `engine.transfer_sync` or `engine.transfer_async`.

## Verification

- Verify `host_tensor.device` reports `npu:<rank>` after `aclshmem_malloc(..., mem_type=MemType.HOST_SIDE)` while `npu-smi info` HBM usage remains unchanged by `host_attr.size`.
- Run a round-trip `D2rH` (`WRITE`) followed by `rH2D` (`READ`) via HIXL with `OPTION_ENABLE_USE_FABRIC_MEM="1"` and assert bitwise equality (`torch.equal(local_npu_src, local_npu_dst)`).
