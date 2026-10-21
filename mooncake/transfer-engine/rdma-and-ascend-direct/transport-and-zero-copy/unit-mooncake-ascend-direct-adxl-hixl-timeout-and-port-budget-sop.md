---
category: sop
conditions:
  domain: mooncake
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: mooncake/transfer-engine/rdma-and-ascend-direct/transport-and-zero-copy
evidence_strength: software_contract
id: unit-mooncake-ascend-direct-adxl-hixl-timeout-and-port-budget-sop
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/transfer-engine/transport/ascend_direct_transport.md#L1-L266
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/getting_started/supported-protocols.md#L1-L165
  source_locator: kvcache-ai/Mooncake@1f65cf13c9f886f8db8b88b7733367b18340ebf0:docs/source/design/transfer-engine/transport/ascend_direct_transport.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: Mooncake Ascend Direct (ADXL/HIXL) Transport Port Budget, Auto-Connect & Timeout
  Arithmetic SOP
triggers:
  error_signatures:
  - ASCEND_ERR_TIMEOUT
  - ERR_KERNELS_FAIL
  - AdxlEngine Connect failed
  keywords:
  - mooncake
  - ascend-direct
  - adxl
  - hixl
  - hccs
  - roce
  - hccl-port
  - timeout-arithmetic
---
## Preconditions & Trigger

Apply this SOP when deploying Mooncake Transfer Engine with `"protocol": "ascend"` (`AscendDirectTransport` backed by CANN `AdxlEngine` / HIXL) on Ascend 910B (A2) or Ascend 910C (A3) clusters for Host-to-NPU (`MEM_HOST`) or NPU-to-NPU (`MEM_DEVICE`) KV cache transfers over HCCS and RoCE.

## Build & Memory Registration Invariants

1. **Compilation & Runtime Flag Alignment**:
   - Build Mooncake with `-DUSE_ASCEND_DIRECT=ON` (requires CANN >= 8.5 with `AdxlEngine` headers and libraries). Do NOT mix `-DUSE_ASCEND=ON` (legacy HCCL transport) or `-DUSE_ASCEND_HETEROGENEOUS=ON` in the same binary; those CMake flags are mutually exclusive.
   - `AscendDirectTransport` inspects `buffer_desc.location`: buffers prefixed with `"cpu"` are registered as `adxl::MEM_HOST`, `"npu"` as `adxl::MEM_DEVICE`, and wildcard `"*"` dynamically probes pointer attributes via `aclrtPointerGetAttributes`.
   - For Host DRAM buffers (`MEM_HOST`) transferred to/from NPU HBM without a pre-registered device arena, set `ASCEND_BUFFER_POOL="4:8"` (`<num_blocks>:<block_size_mb>`, e.g., 4 blocks of 8 MB) so the transport stages chunks through a pinned intermediate device buffer pool (`CopyWithTeBufferPool`) instead of failing registration.

## Port Allocation Formula & Ephemeral Port Avoidance

1. **Deterministic Base Port Calculation**:
   - When `AscendDirectTransport::install()` initializes `AdxlEngine`, it selects a listen port for the local NPU device (`logic_device_id`):
     - If `ASCEND_BASE_PORT` is exported, the engine binds to `ASCEND_BASE_PORT + logic_device_id`.
     - Otherwise, if `/etc/hccn.conf` defines `hccl_port_<device_id>`, that port is used; if absent, it defaults to `60000 + physical_device_id`.
   - If the target port is already occupied, `FindAvailableTcpPort` falls back to scanning a random port in `10000..30000`, which can collide with `vLLM` ZMQ/NIXL side channels or Ray worker ports. Always export an explicit non-overlapping `ASCEND_BASE_PORT` per container instance on multi-instance hosts.

## Connection Pool & Timeout Arithmetic

1. **Enforce Connect vs. Transfer Timeout Budgets**:
   - `ASCEND_CONNECT_TIMEOUT` (milliseconds, default `10000` ms = 10 s) bounds `adxl_->Connect()`, while `ASCEND_TRANSFER_TIMEOUT` (milliseconds, default `10000` ms = 10 s) bounds `TransferSync` / `TransferAsync` operations.
   - Because `checkAndConnect()` automatically disconnects and reconnects once on transfer timeout (`ASCEND_ERR_TIMEOUT`) before returning `ERR_KERNELS_FAIL`, caller RPC deadlines (`outer_timeout`) MUST satisfy:
     $$\text{outer\_timeout} \ge 2 \times \text{ASCEND\_CONNECT\_TIMEOUT} + \text{ASCEND\_TRANSFER\_TIMEOUT} + 3000\,\text{ms}$$
2. **Global Resource Pool vs. Short-Connection Mode**:
   - By default, ADXL maintains persistent RDMA/HCCS queue pairs across peers (`AutoConnect=1`). In large-scale multi-node P/D clusters where all-to-all QP footprints exhaust RoCE NIC context memory:
     - Set `ASCEND_GLOBAL_RESOURCE_CONFIG_PATH=/path/to/global_resource.json` so `AdxlEngine` shares a global RDMA resource pool across instances (`UseGlobalResource=1`).
     - Set `ASCEND_USE_SHORT_CONNECTION=1` to disconnect idle peer sessions immediately after batch completion when peer fan-out exceeds hardware QP limits.
   - Enable `ASCEND_AUTO_DISCONNECT=1` (`EnableAutoDisconnect=1`) so `AdxlEngine` automatically tears down stale connections after `ASCEND_CONNECT_TIMEOUT` of inactivity, preventing half-open RoCE QP leaks when a peer pod restarts.
3. **RoCE vs. HCCS Transport Selection**:
   - Keep `ASCEND_RDMA_TC` and `ASCEND_RDMA_SL` aligned with the data-center RoCE DSCP/PFC lossless queue configuration when crossing ToR switches. On intra-node or superpod topologies, set `ASCEND_TRANS_MODE=ROCE` only when isolating HCCS bandwidth for tensor-parallel `AllReduce` collectives.

## Verification

- Inspect startup logs for `AscendDirectTransport initialized on <host_ip>:<adxl_port>` and verify no fallback to random ports `10000..30000` occurred.
- Run a bidirectional `READ`/`WRITE` smoke batch between two NPU ranks and confirm zero `ASCEND_ERR_TIMEOUT` or `reconnect` warnings in the transfer worker thread logs.
