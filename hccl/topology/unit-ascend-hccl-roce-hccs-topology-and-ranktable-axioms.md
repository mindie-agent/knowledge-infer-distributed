---
category: axiom
conditions:
  ranktable_schema: '1.0'
  transport:
  - hccs
  - roce-v2
  vendor: huawei-ascend
content_digest: sha256:055b5f9e20d091ba1d8a424f0ef5c3a2ce26acfbd2237c395f5bc375e579f496
domain_path: hccl/topology
evidence_strength: hardware_spec
id: unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/cannbot-distributed-provenance.json#unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/user_guide/cluster_info_config/rank_table_config_a3.md
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_BUFFSIZE.md
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_SOCKET_IFNAME.md
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/docs/zh/user_guide/hccl_env/HCCL_INTRA_ROCE_ENABLE.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/runbooks/runtime/hccl_buffsize_too_small.md#L20-L51
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/transfer-engine/transport/ascend_direct_transport.md#L55-L100
  source_digest: fd7433c8c9b2de6d6d629e4d77ffb5dc09c57788a3f7d6d6c19dff678589f4d6
  source_locator: docs/evidence/cannbot-distributed-provenance.json#unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
  source_type: hardware_spec+official_docs+ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L1
title: Ascend HCCL transport and rank-table axioms across A2, A3 and A5
triggers:
  error_signatures:
  - EI0006
  - HCCL_BUFFSIZE is too SMALL
  - rank_table_file
  keywords:
  - hccl
  - hccs
  - roce
  - ranktable
  - device_id
  - rank_id
  - device_ip
  - a2
  - a3
  - a5
  - superpod
---
# Ascend HCCL transport and rank-table axioms

These axioms describe the control-plane contract that a distributed Ascend inference
launcher must satisfy. They apply to Atlas A2 (`910B`), Atlas A3 (`910_93`, dual-die)
and Ascend 950 (A5) once the platform is correctly enumerated. The number of dies per
card and the fabric label change by generation; the identity and transport rules do
not.

## Axiom 1 - Intra-node and inter-node transports are separate domains

| Generation | SoC family | Node-local logic devices | Intra-node default | Inter-node default |
| --- | --- | --- | --- | --- |
| A2 | `910B` | 8, one die per card | HCCS | RoCE v2 |
| A3 | `910_93` | 16, eight dual-die cards (`Phy-ID` 0..15) | HCCS / SuperPod fabric | RoCE v2 |
| A5 | `950` | generation-specific; read `npu-smi info -t board` | high-speed node fabric; verify the label | RoCE v2 |

Consequences:

- A collective that stays inside one node uses HCCS by default. A collective that
  crosses nodes uses RoCE. The two paths have different latency, packet size,
  congestion behavior and counters; a green intra-node link does not prove that the
  cross-node path is healthy.
- A3 is not "eight cards with eight ranks". It is eight physical cards with two dies
  each, exposed as sixteen logic devices per node. Any world size, rank table or EP
  plan that assumes eight dies on A3 is wrong even if the process starts.
- A5 is a new generation. Do not copy A2/A3 device counts, die numbering or EP
  ceilings into a 950 deployment. Enumerate the devices, then build the rank table.
- The inter-node transport is RoCE v2. `HCCL_INTRA_ROCE_ENABLE` changes the
  intra-node choice only; it cannot turn an inter-node HCCS path into RoCE or vice
  versa.

## Axiom 2 - `rank_table_file` binds three distinct identities

A version 1.0 rank table has `server_list[].server_id`, and each server has a
`device` list. Four fields are routinely confused:

| Field | Scope | Must mean | Must not mean |
| --- | --- | --- | --- |
| `server_id` | server | key used by the launcher / `cluster_rank_info` | an NPU address |
| `device_id` | node | logic device visible to this process | a global rank, or the physical `Phy-ID` |
| `rank_id` | cluster | global collective rank | local device order |
| `device_ip` | device | RoCE IP of that NPU die | host management NIC IP |

Minimal two-server shape (placeholders only):

```json
{
  "version": "1.0",
  "status": "completed",
  "server_count": "2",
  "server_list": [
    {
      "server_id": "0",
      "device": [
        {"device_id": "0", "device_ip": "<npu-roce-ip-a>", "rank_id": "0"},
        {"device_id": "1", "device_ip": "<npu-roce-ip-b>", "rank_id": "1"}
      ]
    },
    {
      "server_id": "1",
      "device": [
        {"device_id": "0", "device_ip": "<npu-roce-ip-c>", "rank_id": "2"},
        {"device_id": "1", "device_ip": "<npu-roce-ip-d>", "rank_id": "3"}
      ]
    }
  ]
}
```

Invariants:

1. `device_id` is node-local and may repeat on different servers. Do not deduplicate
   it globally. On A2 it is normally 0..7; on A3 it is normally 0..15 for the sixteen
   dies of eight dual-die cards. On A5 take the value from the live enumeration.
2. `rank_id` is global and unique. The union of all `device` entries must cover the
   launcher's `0..world_size-1` rank set. Gaps, duplicates, or a `rank_id` that does
   not equal the launcher's rank produce a wrong peer or a connect timeout.
3. `device_ip` is the NPU's RoCE IP, not the host NIC IP. Read it with
   `hccn_tool -i <device_id> -ip -g` or from the mounted `/etc/hccn.conf`. It must be
   unique per die and reachable from every peer that will exchange KV or collectives.
4. `server_count` must equal the number of `server_list` entries, `version` must be
   `1.0`, and `status` must be `completed`. A half-written table is not a degraded
   table; it is an invalid one.
5. `server_id` may be a logical string or a host address, but the launcher's
   `cluster_rank_info` mapping must use the same key. The field is not an NPU address.
6. `Chip Phy-ID` reported by `npu-smi` is a physical identifier. In a container the
   process may see a remapped logic device. The rank table must use the logic device
   that the process actually opens, not the physical `Phy-ID` copied from a board
   listing.
7. A3's sixteen dies are sixteen independent RoCE endpoints. A single `device_ip`
   reused by two dies, or a rank table that lists only eight A3 entries, silently
   collapses the world size.

## Axiom 3 - `HCCL_BUFFSIZE` is a per-communicator capacity contract

- Unit is MB. The default is 200 MB. Each communicator reserves an input and an output
  buffer, so the rough HBM cost is about `2 x HCCL_BUFFSIZE` MB per communicator, not
  per process.
- The value must be identical on every rank that joins the communicator. A rank with a
  smaller buffer either fails admission or stalls the collective.
- It must cover the largest single collective in the workload, not the average. MoE
  dispatch/combine, large EP all-to-all and long-sequence CP are the usual drivers.
- Too small produces `HCCL_BUFFSIZE is too SMALL` or, for MC2, a tiling failure such
  as `MoeDistributeDispatchV2 do tiling failed, ret is -1`. The operator normally
  prints `NEEDED_HCCL_BUFFSIZE`; size from that print, not from a copied magic number.
- For MC2 dispatch/combine the working estimate is:

```text
NEEDED_HCCL_BUFFSIZE ~= (
    maxBs * bytesPerTokenDispatch * epWorldSize * localExpertNum
  + maxBs * bytesPerTokenCombine * (topK + sharedExpertNum)
) * 2
```

  The trailing `* 2` covers dispatch plus combine. Add headroom for alignment and for
  the operator's own metadata. With `comm_alg="fullmesh_v2"` use the operator's
  full-mesh buffer branch; do not reuse the non-full-mesh number.
- `HCCL_BUFFSIZE` is not a timeout. Raising it does not fix `EI0002` or `EI0006`, and
  raising it without bounding the number of communicators can cause an HBM OOM.

## Axiom 4 - `HCCL_SOCKET_IFNAME` selects the host bootstrap NIC

- It names the host network interface used for HCCL's socket/control bootstrap. It is
  not the NPU `device_ip`, and it is not the KV data path.
- A comma-separated list is allowed; the first matching interface becomes the root
  NIC. Use the interface that routes to every peer. Never use `lo`, a down interface,
  or an interface on an isolated management VLAN.
- Set the same value on all nodes of the job. A node that selects a different NIC
  reaches a different address family or subnet and exposes itself as
  `EI0006 Get_Socket_Timeout` on its peers.
- Keep the three names distinct: `HCCL_SOCKET_IFNAME` is the HCCL host NIC,
  `HCCL_IF_IP` is the host IP bound for HCCL, and `GLOO_SOCKET_IFNAME` is the Gloo
  control-plane NIC. A correct value for one does not imply a correct value for the
  others.
- For Mooncake / HIXL KV transfer, `HCCL_SOCKET_IFNAME` still governs HCCL bootstrap;
  the KV data path uses the registered device memory and the transfer engine's
  `device_ip` / session address. Fixing the NIC name does not register device memory.

## Axiom 5 - `HCCL_INTRA_ROCE_ENABLE` changes intra-node transport only

- Default `0`: intra-node traffic uses HCCS. On A2 servers and A3 supernodes this is
  the normal high-bandwidth path.
- `1`: intra-node traffic is forced to RDMA/RoCE. Use it only when the HCCS path is
  intentionally unavailable, when a direct-transfer backend requires it, or when
  isolating an HCCS-specific fault. It is not a general performance switch.
- It does not change inter-node transport, which is already RoCE, and it does not
  create a SuperPod or extend an HCCS domain across nodes.
- The value must be consistent across the communicator. Mixing HCCS and RDMA ranks in
  one collective is a transport mismatch, not a valid heterogeneous deployment.
- In MC2 code the legacy pair `HCCL_INTRA_PCIE_ENABLE=1` and
  `HCCL_INTRA_ROCE_ENABLE=0` selected the hierarchy algorithm. Newer operators prefer
  an explicit `comm_alg`; do not rely on the environment pair when the operator
  exposes `comm_alg`.

## Failure signatures that violate these axioms

| Signature | First axiom to check | Typical cause |
| --- | --- | --- |
| `EI0006 Get_Socket_Timeout` | 2, 4 | rank absent / wrong `device_ip` / wrong `HCCL_SOCKET_IFNAME` |
| `EI0002` notify wait timeout | 1, 3 | cross-node link degradation, undersized buffer, or a late rank |
| `HCCL_BUFFSIZE is too SMALL` | 3 | buffer below the largest dispatch/combine |
| `MoeDistributeDispatchV2 do tiling failed` | 3 | MC2 buffer below the operator's `NEEDED_HCCL_BUFFSIZE` |
| wrong output with no communication error | 2, 5 | duplicated `device_id`, wrong `rank_id`, or mixed transport |

## Evidence boundary

The transport separation, rank-table identities, A3 dual-die enumeration and the
three environment contracts are hardware/official-document facts. The A5 row is
deliberately enumeration-neutral: it asserts the invariant, not a die count. This
unit does not report a measured bandwidth, latency or successful NPU run; those
belong to the L2/L3 procedures.
