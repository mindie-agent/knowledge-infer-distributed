---
category: workflow
conditions:
  domain: mooncake
  hardware:
  - Ascend 910B
  - Ascend 910C
  - Distributed CPU/DRAM/RDMA Store Cluster
domain_path: mooncake/store/master-ha-and-eviction/oplog-failover-and-leases
evidence_strength: software_contract
id: unit-mooncake-store-etcd-oplog-snapshot-fail-closed-promotion-workflow
inherits_from:
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/design/store/mooncake-store.md#L1-L260
  - https://github.com/kvcache-ai/Mooncake/blob/1f65cf13c9f886f8db8b88b7733367b18340ebf0/docs/source/deployment/mooncake-store-deployment-guide.md#L1-L162
  source_locator: kvcache-ai/Mooncake@1f65cf13c9f886f8db8b88b7733367b18340ebf0:docs/source/design/store/mooncake-store.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L3
title: Mooncake Store Master High Availability, Etcd OpLog Replay & Fail-Closed Standby
  Promotion Workflow
triggers:
  error_signatures:
  - Failed to catch up op_log
  - CatchUpOpLogs failed
  - MAX_GAP_WAIT_TIME
  keywords:
  - mooncake-store
  - high-availability
  - etcd
  - oplog
  - snapshot
  - standby-promotion
  - fail-closed
---
## Preconditions & Trigger

Use this workflow when deploying or diagnosing Mooncake Store (`mooncake_master`) in High Availability (`HA`) mode with Primary-Standby replication (`--enable_ha=true`, `--enable_offload=true`) backed by `etcd` and shared/local snapshot storage, or when investigating standby promotion aborts after a primary master crash.

## Architecture & State Synchronization Invariants

Mooncake Store separates control-plane metadata (`mooncake_master`) from zero-copy data-plane transfers (`TransferEngine`). In HA mode, multiple `mooncake_master` processes contend for an `etcd` leader lease while Standby instances maintain a warm metadata shadow via two complementary channels:
1. **Real-Time Incremental Stream (`EtcdOpLogChangeNotifier`)**: Every metadata mutation (`PutEnd`, `Remove`, `Revoke`) committed by the Primary is appended as a sequenced OpLog entry (`sequence_id`, payload checksum, key, object descriptor) and watched by Standby instances via `etcd`.
2. **Periodic Baseline Checkpoints (`Snapshot` Catalog)**: The Primary periodically serializes its full metadata tree into a timestamped snapshot (`snapshot_id` formatted as `YYYYMMDD_HHMMSS_fff`, `manifest.txt`, and `descriptors` shards) and updates ` mooncake-master-snapshot/latest.txt`.

## Step-by-Step HA Deployment & Fail-Closed Promotion Workflow

### Step 1: Provision `etcd` Quorum & Launch Primary/Standby `mooncake_master` Instances
- Start an odd-numbered `etcd` quorum (minimum 3 endpoints) and pass the comma-separated endpoint list via `--etcd_endpoints="<etcd_ip_a>:2379,<etcd_ip_b>:2379,<etcd_ip_c>:2379"`.
- Launch at least two `mooncake_master` instances on distinct fault domains with explicit snapshot persistence enabled:
  ```bash
  mooncake_master \
    --enable_ha=true \
    --etcd_endpoints="<etcd_ip_a>:2379,<etcd_ip_b>:2379,<etcd_ip_c>:2379" \
    --enable_snapshot=true \
    --enable_snapshot_restore=true \
    --snapshot_backup_dir="/opt/mooncake/snapshot_backup" \
    --rpc_port=50051 \
    --http_metrics_port=9003
  ```
- Verify one instance acquires leadership and logs `Promoted to Primary Master`, while peer instances log `Running in Standby mode, watching OpLog stream`.

### Step 2: Diagnose Standby OpLog Gap Resolution (`CatchUpOpLogs`)
- When the Primary master fails and its `etcd` lease expires, the winning Standby executes `PromoteToPrimary()` before serving client RPCs:
  1. It reads the cluster's latest committed `last_sequence_id` from `etcd` and compares it against its local `applied_sequence_id`.
  2. If `applied_sequence_id < last_sequence_id`, the Standby enters `CatchUpOpLogs()`, performing bounded point-reads on `etcd` (up to 3 retries per missing sequence entry) and verifying every entry's CRC checksum.
  3. **Sequence Wait Rule**: If an out-of-order OpLog entry arrives before its predecessor, `OpLogApplier` buffers the entry and waits up to **3 seconds** (`MAX_GAP_WAIT_TIME`) for the missing `sequence_id`.

### Step 3: Handle Fail-Closed Promotion Aborts (`Failed to catch up op_log`)
- **Critical Invariant**: Mooncake Store enforces a **fail-closed promotion policy**. If any OpLog entry between `applied_sequence_id` and `last_sequence_id` was compacted/deleted in `etcd` before the Standby applied it, or fails checksum validation after 3 retries, the Standby **refuses promotion and aborts** (`LOG(FATAL) << "Failed to catch up op_log, refusing promotion to prevent split-brain metadata corruption"`).
- Why fail-closed is mandatory: Promoting a Standby with missing `PutEnd` or `Remove` records would cause the allocator to hand out already-occupied RDMA segment offsets to new `PutStart` requests, silently corrupting in-flight KV cache tensors across workers.
- **Recovery Action**:
  1. Do not disable checksum checks. Instead, ensure `etcd` auto-compaction retention exceeds the Primary snapshot interval (`snapshot_interval_seconds`).
  2. Restart `mooncake_master` with `--enable_snapshot_restore=true` so it loads the latest intact snapshot from `mooncake-master-snapshot/latest.txt` and replays only the post-snapshot OpLog tail.

## Verification

- Query `http://<master_ip>:9003/metrics` and confirm `mooncake_master_is_leader == 1` on the active Primary and `0` on Standby instances.
- Kill the Primary process (`kill -9 <pid>`) during an active benchmark and verify the Standby logs `CatchUpOpLogs succeeded, missing_entries=0, promoting to Primary` and resumes client RPCs within one `etcd` lease TTL window without offset collisions.
