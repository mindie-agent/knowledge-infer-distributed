---
category: sop
conditions:
  domain: mooncake
domain_path: mooncake/transfer-engine/rdma-and-ascend-direct/transport-and-zero-copy
evidence_strength: software_contract
id: unit-mooncake-transfer-engine-segment-buffer-and-p2p-handshake-sop
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/transfer-engine/index.md#L1-L162
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/architecture.md#L1-L95
  source_locator: kvcache-ai/Mooncake@1f65cf13c9f886f8db8b88b7733367b18340ebf0:docs/source/design/transfer-engine/index.md
  source_type: source_contract
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
status: active
tier: L2
title: Mooncake Transfer Engine Segment Memory Registration, Zero-Copy Batch Submit
  & P2P Handshake SOP
triggers:
  error_signatures:
  - ERR_INVALID_ARGUMENT
  - openSegment failed
  - registerLocalMemory failed
  keywords:
  - mooncake
  - transfer-engine
  - register-local-memory
  - open-segment
  - p2p-handshake
  - batch-transfer
  - zero-copy
  - kv-cache
---
## Applicability boundary for offline analysis

Segment identity, buffer bounds, handshake metadata and batch completion are Transfer Engine API contracts. They can be inspected or tested using the supported host/TCP path without an Ascend SoC or an asymmetric-TP PD deployment. The pinned source below defines the described API. Before selecting RDMA or Ascend Direct for a live transfer, establish that transport's device registration, memory and endpoint requirements; a host-side check does not establish device-transfer support or performance.

## Preconditions & Trigger

Apply this SOP when configuring or troubleshooting Mooncake Transfer Engine (`TransferEngine`) for zero-copy KV cache block transfers across DRAM, GPU/NPU HBM, RDMA NICs (`rdma`), Ascend Direct (`ascend`), or TCP fallback (`tcp`) in P2P handshake (`P2PHANDSHAKE`) or centralized metadata store (`etcd` / `redis` / `http`) topologies.

## Segment Abstraction & Memory Registration Contract

1. **Address Space Unification via Segments**:
   - Mooncake Transfer Engine abstracts contiguous or non-contiguous memory buffers (DRAM, VRAM/HBM, or NVMe files) into a globally addressable **Segment** identified by a `SegmentID` (`int64_t`) and `SegmentName` (`std::string`).
   - Before any remote `READ` or `WRITE` transfer can reference a local buffer, the buffer MUST be pinned and registered with the underlying transport (e.g., `ibv_reg_mr` on RoCE/InfiniBand or `AdxlEngine::RegisterMem` on Ascend NPU) via `registerLocalMemory(addr, length, location, remote_accessible, update_metadata)`.
   - Never issue `registerLocalMemory` on the per-request hot path; register the pre-allocated KV cache block pool or arena once at engine initialization and reuse sub-slice offsets (`source` / `target_offset`) within the registered range.
2. **Metadata Backend vs. `P2PHANDSHAKE` Mode**:
   - **Centralized Metadata Mode (`etcd` / `redis` / `http`)**: Each node publishes its NIC/RoCE GIDs, Queue Pair (QP) endpoints, and registered memory descriptors (`rkey`, virtual address ranges, device affinity) to the metadata server during `init()` and `registerLocalMemory(..., update_metadata=true)`.
   - **Serverless P2P Handshake Mode (`P2PHANDSHAKE`)**: Eliminates the external metadata store by embedding a dynamic RPC listener inside each `TransferEngine` instance (`local_server_name` formatted as `<host_ip>:<rpc_port>`). Leave `rpc_port` as `0` or auto-assigned across multi-process ranks to bind an ephemeral port in `12001..15000` and avoid `EADDRINUSE` collisions on shared hosts.
   - When resolving a remote peer via `openSegment(segment_name)`, the initiator caches the peer's segment descriptor locally; subsequent transfers to the same peer bypass metadata lookups unless `syncSegmentCache()` is explicitly invoked after dynamic memory expansion.

## Asynchronous Batch Submission & Completion Polling SOP

1. **Allocate Reusable Batch Descriptors**:
   - Create a batch context with `allocateBatchID(max_batch_size)` to amortize work-request (WR) doorbell overhead across fragmented KV cache blocks (e.g., multi-layer PagedAttention blocks).
2. **Submit Vectorized `TransferRequest` Slice Lists**:
   - Populate `std::vector<TransferRequest>` where each entry specifies:
     - `opcode`: `TransferRequest::READ` (Decode pulls KV from Prefill) or `TransferRequest::WRITE` (Prefill pushes KV to Decode).
     - `source`: Local registered virtual address pointer.
     - `target_id`: Remote `SegmentID` returned by `openSegment()`.
     - `target_offset`: 64-bit byte offset within the remote segment.
     - `length`: Exact byte length of the KV block or layer slice.
   - Submit the batch via `submitTransfer(batch_id, requests)` and verify return code `0` (`ERR_INVALID_ARGUMENT` or negative codes indicate unregistered address ranges or out-of-bounds segment offsets).
3. **Bounded Completion Polling & Batch Reclamation**:
   - Poll `getTransferStatus(batch_id, task_id, &status)` until all tasks reach `TransferStatusEnum::COMPLETED` or a task transitions to `FAILED` / `TIMEOUT`.
   - Enforce a hard caller-side polling budget rather than spinning indefinitely on `WAITING`. Once all tasks in `batch_id` reach a terminal state, call `freeBatchID(batch_id)` to release the internal descriptor slot.
   - Multi-rail path selection and failover across NICs are handled inside the transport layer using the pre-configured topology matrix (`auto-discovery` or explicit NIC matrix mapping CPU NUMA / PCIe switch affinity to preferred RNICs).

## Verification

- Confirm startup logs report successful transport registration (`Transport rdma initialized` or `Transport ascend initialized`) and `registerLocalMemory` returns `0` for all KV cache arenas.
- Validate P2P or metadata connectivity by running `openSegment("<peer_ip>:<rpc_port>")` and verifying a non-negative `SegmentID` is returned before serving inference traffic.
