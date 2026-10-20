---
category: sop
conditions:
  connector:
  - MooncakeConnector
  - LlmDataDist
  kv_transport:
  - roce
  - rdma
  supports_asymmetric_tp: true
content_digest: sha256:df6099eb4552994e03db39b0e0cccd91955e22fa0bd0f216647831ab30a48a44
domain_path: serving/pd-disaggregation
evidence_strength: reproduced_empirical
id: unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
provenance:
  evidence_refs:
  - docs/evidence/cannbot-distributed-provenance.json#unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
  - https://github.com/vllm-project/vllm-ascend/pull/1691
  - https://github.com/vllm-project/vllm-ascend/issues/4842
  - https://github.com/vllm-project/vllm-ascend/pull/2947
  - https://github.com/vllm-project/vllm-ascend/issues/5044
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/runbooks/runtime/pd_prefill_kv_block_release_vs_decode_pull.md#L29-L105
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/runbooks/runtime/mooncake_ascend_hixl_kv_transfer_bringup.md#L20-L76
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/transfer-engine/transport/ascend_direct_transport.md#L55-L100
  source_digest: fd7433c8c9b2de6d6d629e4d77ffb5dc09c57788a3f7d6d6c19dff678589f4d6
  source_locator: docs/evidence/cannbot-distributed-provenance.json#unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
  source_type: reproduced_run+source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: PD-disaggregated KV connector and asymmetric P-TP/D-TP KV-head slicing
triggers:
  error_signatures:
  - '103901'
  - mem type:1
  - prefill_tp_size must be greater than or equal to the decode_tp_size
  - Force freed expired request
  - ASCEND_TRANSFER_TIMEOUT
  keywords:
  - mooncake
  - llm datadist
  - kv connector
  - pd disagg
  - asymmetric tp
  - kv head
  - rdma
  - roce
  - register_buffer
  - delayed free
---
# PD-disaggregated KV connector and asymmetric P-TP / D-TP KV-head slicing

In prefill-decode disaggregation the P instance computes KV and the D instance pulls
it. Two connector families are common on Ascend: `MooncakeConnector` (vLLM-Ascend's
`mooncake_connector.py`, backed by a Mooncake/Ascend direct transfer engine) and
`LlmDataDist` (CANN LLM-DataDist, wrapped by vLLM-Ascend's data-dist connector). Both
are pull-based by default: P must not release a KV block until every D rank that
pulls it has acknowledged. Both also fail silently if the KV-head slice is wrong, so
the procedure checks transport, layout and lifecycle in that order.

## Step 1 - Pin the P/D parallel plan and prove the asymmetric-TP arithmetic

Record before starting:

- P: `tp_size`, `dp_size`, `pp_size`, KV heads, head dim, block size, dtype, layers;
- D: `tp_size`, `dp_size`, `pp_size`, the same KV geometry;
- the connector (`MooncakeConnector`, `MooncakeConnectorStoreV1`, or `LlmDataDist`);
- the model's attention type (MLA/SFA vs GQA/MHA vs linear attention/Mamba).

The Mooncake connector requires the prefill TP to be at least the decode TP:

```text
prefill_tp_size >= decode_tp_size
prefill_tp_size % decode_tp_size == 0        # for integer pulling
num_d_block_heads = max(1, num_kv_heads // decode_tp_size)
num_p_block_heads = max(1, num_kv_heads // prefill_tp_size)
tp_num_need_pulls = num_d_block_heads // num_p_block_heads
require tp_num_need_pulls >= 1
```

For a 64-KV-head model with P-TP=8 and D-TP=4: each P rank holds 8 heads, each D rank
needs 16 heads, so each D rank pulls two P shards. For MLA/SFA, the connector uses
`tp_num_need_pulls = 1` because the absorbed KV layout is not sliced per head; do not
apply the GQA formula blindly. For Mamba/GDN linear attention, the state is sliced by
head and the same `P-TP >= D-TP` and divisibility rule applies, but the slice is a
conv/SSM state slice, not a KV-head slice.

If the arithmetic does not divide, change the parallel plan. Do not rely on a runtime
`max(1, ...)` to hide a non-divisible head count; it turns a clean failure into a
silent wrong-head transfer.

Also verify the rank table on both sides: `rank_id -> server_id -> device_id ->
device_ip`, with `device_ip` the NPU RoCE IP and `HCCL_SOCKET_IFNAME` the host NIC that
routes to the peer. The KV path does not use `HCCL_SOCKET_IFNAME` for data, but HCCL
bootstrap and the connector handshake still depend on it.

## Step 2 - Bring up the connector and register device KV memory

### MooncakeConnector / Mooncake Ascend direct

- Build Mooncake with the Ascend direct backend (`-DUSE_ASCEND_DIRECT=ON`); a missing
  backend makes `TransferEngine.initialize()` return non-zero.
- Call `torch.npu.set_device(<device>)` before constructing the engine. The engine
  reads the physical NPU id from the calling thread's Ascend context; constructing it
  first binds the wrong card.
- Initialize with positional arguments, for example
  `initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)`. The pybind signature
  is positional; keyword arguments are rejected.
- For the store connector, set `register_buffer=true` in both P and D
  `kv_connector_extra_config`. Without it the host half may register (`mem type:0`)
  while the NPU HBM half never does; the transfer then fails with an engine error such
  as `103901`. A successful device registration appears as `mem type:1` in the plog.
- Device memory registered on the HCCS IPC path is 2 MiB aligned. Over-allocate and
  narrow to the aligned start; do not hand an unaligned slice to the engine.
- Set `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT` explicitly. If an outer
  `MC_TRANSFER_TIMEOUT` exists, keep it strictly greater than twice the inner transfer
  timeout, because a timed-out transfer may be retried once.

### LlmDataDist

- Construct `LLMDataDist(role, cluster_id)`, set `LLMConfig.device_id`, call `init`,
  then `link(name, cluster_rank_info, rank_table_json)`.
- Poll `query_register_mem_status(comm_id)` until `RegisterMemStatus.OK`; treat
  `FAILED` as a hard stop, not a retry-forever state.
- Register the actual paged KV tensors for every layer/group. A link that is up but
  has no registered device memory is not a usable KV path.

Common checks for both connectors:

- the KV cache is on NPU device memory, not host memory;
- each layer/group is registered, including hybrid/Mamba groups;
- the P and D sides agree on block size and KV geometry;
- `HCCL_BUFFSIZE` is large enough for the HCCL control path and equal on all ranks;
- `HCCL_INTRA_ROCE_ENABLE` is `1` only if intra-node RDMA is intentional.

## Step 3 - Validate the KV-head slice and the layout before sending traffic

Same TP on both sides: a block can be transferred directly, but the layout must still
match the consumer.

Asymmetric TP (P-TP > D-TP): the D worker pulls `tp_num_need_pulls` P shards per KV
head group. The transfer buffer is laid out as:

```text
[block, split, token, head_per_split, dim]
```

Before attention reads it, restore it to the consumer layout:

```text
[block, token, split, head_per_split, dim]
```

If this transpose is skipped, the block length and head count may look plausible
while heads are interleaved incorrectly; the result is silent wrong output, not a
crash. If a fused transpose op is available, verify it produces the same result as the
reference transpose on one block before enabling it for the whole cache.

Execution-mode layout:

- eager KV is typically 4-D `[num_block, block_size, num_kv_head, head_dim]`;
- torchair graph KV is typically 3-D
  `[num_block, block_size, num_kv_head * head_dim]`.

Compute the block rank from the actual tensor, not a hard-coded 3:

```python
block_rank = len(first_kv_cache.shape) - 1
```

A hard-coded rank includes `num_block` in the block length under the 3-D layout and
corrupts the transfer sizing. When concatenating a 3-D graph cache, view it back to
`[num_block, block_size, num_kv_head, head_dim]` before the normal path.

For Mamba/GDN state, verify the state slice offsets against the remote TP size. A
mismatch transfers the wrong conv/SSM slice even though the KV blocks themselves are
correct.

## Step 4 - Complete the handshake and release only after all decode ranks ACK

P-side `request_finished` for a remote-decode request should:

- keep the computed blocks and return `delay_free_blocks=True`;
- export `remote_block_ids`, `remote_ptp_size`, `remote_host`, `remote_port`,
  `remote_request_id`, `last_token_id` and the prompt/block metadata needed by D.

D-side pull completion should send at least
`[request_id, decode_tp_rank, decode_tp_size]`. P must count the set of distinct
decode ranks, not increment a blind counter:

```text
done = set()
if rank in done: ignore duplicate
done.add(rank)
if len(done) == decode_tp_size: release blocks
```

Why this matters:

- one D rank finishing is not completion; the other ranks may still be pulling the
  same P blocks;
- a duplicate ACK from one rank must not satisfy another rank's slot, or P releases
  early and the still-pulling rank reads overwritten memory;
- fire-and-forget completion can be lost. Prefer a request-reply ACK, or make the
  sender wait for a receive before treating the handshake as complete.

Handle the pre-admission rejection path explicitly. If a remote-prefill request is
rejected before `update_state_after_alloc()` schedules a receive (for example,
prompt + max tokens exceeds the model limit), enqueue an empty receive for that
request. The worker then skips the data copy but still sends the completion signal, so
P can release the stranded blocks.

## Step 5 - Abort, timeout and reclaim: prove the blocks return

Define the budgets before the run:

- `ASCEND_CONNECT_TIMEOUT` for link establishment;
- `ASCEND_TRANSFER_TIMEOUT` for one transfer;
- the connector's abort timeout, for example
  `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`, for a delayed-free request that never
  completes.

On abort or timeout:

1. Remove the request from pending send and receive queues (`_reqs_need_send`,
   `_reqs_need_recv`, delayed-free maps) so a late completion cannot resurrect it.
2. If P already returned remote block IDs, make sure D sends the empty/completion
   signal; otherwise P holds the blocks until the timeout.
3. Let the timeout path force-release the delayed request and log it
   (`Force freed expired request`). A force release is a safety net, not a normal
   completion; count it and investigate the cause.
4. Verify the free-block count returns to the pre-request baseline. A successful API
   call is not proof of reclamation; the allocator's free list is.
5. Check that no request ID remains in `finished_requests`, `reqs_to_process`, or the
   delayed-free map, and that no late ACK produces a `KeyError` / "request not found".

Fault-injection checks:

- kill D during a multi-block transfer, then verify P releases after the ACK/timeout
  path and a new request can reuse the blocks;
- cancel a queued request before any block is sent and verify the empty completion
  path;
- pause one D rank while the others finish and verify P does **not** release early;
- duplicate one D rank's completion and verify P does not over-count.

Close with evidence: P/D TP sizes, KV heads, `tp_num_need_pulls`, connector and
registration mode (`mem type:1`), timeout values, before/after free-block counts, and
the fault-injection result. Do not claim physical HBM release from a block-table
count alone; that requires an allocator-level check.

## Connector boundaries

| Connector | Strength | Main trap |
| --- | --- | --- |
| `MooncakeConnector` | RDMA/Ascend direct, layerwise/push variants exist | device memory registration, asymmetric head slicing, delayed-free accounting |
| `LlmDataDist` | CANN-native link and cache manager | pull completion must cover every decode TP rank; duplicate ACK handling |

They are different implementations with different config keys. A fix proven for one
is not automatically valid for the other.
