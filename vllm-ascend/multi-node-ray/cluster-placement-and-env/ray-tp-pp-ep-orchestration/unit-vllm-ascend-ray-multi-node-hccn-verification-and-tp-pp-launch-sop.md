---
category: sop
conditions:
  domain: vllm-ascend
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: vllm-ascend/multi-node-ray/cluster-placement-and-env/ray-tp-pp-ep-orchestration
evidence_strength: software_contract
id: unit-vllm-ascend-ray-multi-node-hccn-verification-and-tp-pp-launch-sop
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
- unit-ascend-hybrid-tp-dp-ep-cp-sp-dimensional-partition-axioms
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/tutorials/features/ray.md#L1-L165
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/developer_guide/testing/e2e/multi_node/nightly_ci_test.md#L1-L160
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/features/ray.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: vLLM-Ascend Multi-Node Ray Cluster NPU Resource Registration, HCCN Verification
  & TP/PP Launch SOP
triggers:
  error_signatures:
  - HCCL_IF_IP
  - GLOO_SOCKET_IFNAME
  - net health status
  keywords:
  - vllm-ascend
  - ray
  - multi-node
  - hccn-conf
  - hccl-if-ip
  - gloo-socket-ifname
  - distributed-executor-backend
---
## Preconditions & Trigger

Apply this SOP when deploying multi-node `vllm-ascend` inference across Ascend 910B (A2) or Ascend 910C (A3) hosts using `--distributed-executor-backend ray` (e.g., `TP=16` or `TP=8, PP=2`), or when diagnosing Ray placement-group hangs, `HCCL_E_PARA` interface binding mismatches, and container `/etc/hccn.conf` mount omissions.

## Pre-Flight Container & HCCN RoCE Verification

1. **Mandatory Host Mounts for Multi-Node Containers**:
   - Every container participating in a multi-node Ray cluster MUST run with `--net=host` and mount `/etc/hccn.conf:/etc/hccn.conf:ro` alongside `/usr/local/Ascend/driver`, `/usr/local/sbin/npu-smi`, and `/dev/davinci*`. Without `/etc/hccn.conf` inside the container, HCCL cannot resolve physical NPU RoCE IP addresses and fails during cross-node ring setup.
2. **Verify All 8/16 NPU RoCE Links & Net Health**:
   - Before starting `ray`, verify on every host that all local NPUs have active RoCE links, valid IP/mask assignments, and healthy gateways:
     ```bash
     for i in $(seq 0 7); do
       hccn_tool -i "$i" -link -g
       hccn_tool -i "$i" -net_health -g
     done
     ```
   - Ensure `hccn_tool -i <id> -net_health -g` reports `net health status: Success` across all ranks.

## Deterministic Network Interface & Ray Cluster Startup

1. **Export Unified Communication Environment Variables Before `ray start`**:
   - Ray worker actors inherit environment variables from the `ray start` daemon on each host. You MUST export the host's routable IP and NIC interface name on **every node** *before* executing `ray start`:
     ```bash
     export HCCL_IF_IP=<local_node_ip>
     export GLOO_SOCKET_IFNAME=<net_iface>
     export TP_SOCKET_IFNAME=<net_iface>
     export HCCL_SOCKET_IFNAME=<net_iface>
     export ASCEND_PROCESS_LOG_PATH=/workspace/plog_ray
     export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
     ```
   - **Critical Invariant**: Always set `HCCL_IF_IP` to each node's own local IP (`<head_ip>` on the head node, `<worker_ip>` on each worker node)—never copy the head node's `HCCL_IF_IP` onto worker nodes, or HCCL root-info socket binding will fail with `EADDRNOTAVAIL`.
2. **Start Ray Head and Worker Daemons with Explicit `--resources='{"NPU": N}'`**:
   - If the installed Ray version does not auto-detect `NPU` resources or if restricting a container to a subset of cards, pass `--resources='{"NPU": 8}'` (or `16` on A3 16-Die nodes):
     - **On Head Node**:
       ```bash
       ray start --head --num-gpus=0 --resources='{"NPU": 8}' --port=6379 --node-ip-address=<head_ip>
       ```
     - **On Each Worker Node**:
       ```bash
       ray start --address='<head_ip>:6379' --num-gpus=0 --resources='{"NPU": 8}' --node-ip-address=<worker_ip>
       ```
   - Run `ray status` on the head node and confirm the cluster reports the aggregate NPU capacity (e.g., `0.0/16.0 NPU` across 2 nodes) before launching `vllm serve`.

## Launching Multi-Node `vllm serve` (`TP` vs. `PP` Placement)

1. **Intra-Node `TP` + Inter-Node `PP` (Recommended for Standard RoCE Bandwidth)**:
   - Keep Tensor Parallel (`--tensor-parallel-size 8`) within each 8-NPU node's high-bandwidth HCCS domain and span Pipeline Parallel (`--pipeline-parallel-size 2`) across nodes:
     ```bash
     vllm serve <model_path> \
       --distributed-executor-backend ray \
       --tensor-parallel-size 8 \
       --pipeline-parallel-size 2 \
       --host 0.0.0.0 --port 8000
     ```
2. **Cross-Node `TP` / `EP` (`--tensor-parallel-size 16`)**:
   - When model weights or MoE experts require `TP=16` (or `DP` + `--enable-expert-parallel`), verify `HCCL_CONNECT_TIMEOUT=1200` and `HCCL_EXEC_TIMEOUT=1200` are exported on all nodes so large weight-loading phases do not trigger premature HCCL watchdog aborts.

## Verification

- Confirm `ray status` displays `16.0/16.0 NPU` allocated once `vllm serve` finishes placement-group creation, and verify `/v1/completions` responds with valid generated tokens.
