---
category: workflow
conditions:
  domain: hccl
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: hccl/benchmark/hccl-test-suite/roce-pfc-and-retcode-triage
evidence_strength: software_contract
id: unit-hccl-test-retcode-zombie-cleanup-and-roce-pfc-ecn-triage-workflow
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
provenance:
  evidence_refs:
  - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/hccl-test/references/common-issues.md#L1-L265
  - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/hccl-test/SKILL.md#L150-L260
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/hccl-test/references/common-issues.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L3
title: Ascend hccl-test Retcode Diagnosis, Zombie IPC Cleanup & RoCE PFC/ECN Packet-Loss
  Triage Workflow
triggers:
  error_signatures:
  - ret=107002
  - ret=507033
  - Notify wait execute failed
  - roce_out_of_seq
  keywords:
  - hccl-test
  - hccn-tool
  - roce
  - pfc
  - ecn
  - zombie-process
  - ipcs
  - aclrtsetdevice
  - notify-wait-timeout
---
## Preconditions & Trigger

Use this workflow when `hccl_test` (`all_reduce_test`, `alltoallv_test`, etc.) or multi-node HCCL initialization fails with `aclrtSetDevice failed`, `HcclCommInitRootInfo failed`, `Notify wait execute failed`, `fftsplus` task errors, or degraded inter-node RoCE bus bandwidth (`busbw`).

## Step-by-Step Diagnostic & Remediation Workflow

### Step 1: Clear Zombie Processes & Leaked System V Shared Memory (`ret=107002` / `507033`)
- **Symptom**: `aclrtSetDevice failed, ret=107002` / `507033`, `device is busy`, or `HCCL_E_INTERNAL` immediately after an aborted `mpirun` run.
- **Root Cause**: Killed MPI workers leave behind orphaned `all_*_test` processes holding NPU device contexts and leaked System V shared memory segments (`ipcs -m`), preventing subsequent HCCL intra-node ring initialization.
- **Remediation**:
  1. Terminate residual benchmark processes across all participating hosts:
     ```bash
     pkill -9 -f "_test" || true
     ```
  2. Inspect and release orphaned shared-memory segments owned by the test user (`nattch == 0`):
     ```bash
     ipcs -m | awk '$6 == 0 {print $2}' | xargs -r -n 1 ipcrm -m
     ```
  3. Confirm `npu-smi info` shows `0 MB` HBM usage and `"No running processes found"` on all target NPU IDs before re-running.

### Step 2: Diagnose Host-Side NIC Socket vs. NPU RoCE Link Health (`ret=6` / `ret=15`)
- **Symptom**: `HcclCommInitRootInfo failed, ret=6` (`HCCL_E_PARA`) or `ret=15` (`HCCL_E_RUNTIME`), or `Notify wait execute failed` timeout after `HCCL_CONNECT_TIMEOUT` (default `120s`).
- **Triage Procedure**:
  1. **Host TCP Control Plane (`HCCL_IF_IP`)**: When a host has multiple bond/Docker/BMC interfaces, HCCL may bind its root-info socket to an unreachable subnet. Explicitly export `HCCL_IF_IP=<host_roce_or_mgmt_ip>` (or `HCCL_SOCKET_IFNAME=<iface>`) on each node.
  2. **NPU RoCE Physical Link & IP/Gateway (`hccn_tool`)**: Loop over all local NPU IDs (`0..7`) and verify link status, IP assignment, and ARP/ICMP reachability:
     ```bash
     for i in $(seq 0 7); do
       hccn_tool -i "$i" -link -g
       hccn_tool -i "$i" -ip -g
       hccn_tool -i "$i" -net_health -g
     done
     ```
  3. Verify cross-node NPU-to-NPU RoCE reachability using `hccn_tool -i <dev_id> -ping -g address <remote_npu_roce_ip> pkt 1500`. If standard MTU (`1500`) succeeds but jumbo frames (`pkt 4096`) drop, fix the ToR switch MTU mismatch before running collectives.

### Step 3: Isolate RoCE Packet Loss, PFC Pause Storms & ECN Congestion
- **Symptom**: Single-node `all_reduce_test` achieves expected HCCS bandwidth, but multi-node runs stall on message sizes `>= 16M` or exhibit severe bandwidth collapse.
- **Triage Procedure**:
  1. Query RoCE error and pause counters on every NPU port before and after a multi-node `alltoallv_test` run:
     ```bash
     for i in $(seq 0 7); do
       hccn_tool -i "$i" -stat -g | grep -E "roce_rx_err|roce_tx_err|roce_out_of_seq|mac_rx_pfc_pkt|mac_tx_pfc_pkt|roce_cnp"
     done
     ```
  2. **Interpret Hardware Counters**:
     - Non-zero delta in `roce_out_of_seq` (out-of-sequence packets) proves packet drops on the RoCE fabric; verify lossless Priority Flow Control (`PFC`) is enabled on the matching DSCP/Traffic Class (`ASCEND_RDMA_TC` / `HCCL_RDMA_TC`) and ECN (`roce_cnp`) marking thresholds are active on the ToR switch.
     - Rapidly climbing `mac_rx_pfc_pkt` or `mac_tx_pfc_pkt` alongside zero throughput indicates a PFC pause storm or head-of-line blocking from a degraded optical transceiver (`hccn_tool -i <id> -optical -g`).
  3. **A/B Transport Isolation**: Set `export HCCL_INTRA_ROCE_ENABLE=1` on a single node to force intra-node communication over RoCE instead of HCCS; if single-node RoCE also fails on a specific NPU ID, isolate that NPU's optical port without blaming inter-node spine switches.

## Verification

- Re-run `mpirun -f hostfile -n 16 ./bin/all_reduce_test -b 8K -e 1G -f 2 -d bf16 -o sum -p 8 -c 1` with `ASCEND_SLOG_PRINT_TO_STDOUT=0` and confirm `roce_out_of_seq` delta remains `0` across all 16 ranks.
