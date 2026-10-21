---
category: workflow
conditions:
  domain: vllm-ascend
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: vllm-ascend/pd-disaggregation/mooncake-connectors/multi-node-and-colocated
evidence_strength: software_contract
id: unit-vllm-ascend-mooncake-layerwise-and-store-pd-disaggregation-workflow
inherits_from:
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md#L1-L260
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md#L1-L215
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md#L1-L220
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L3
title: vLLM-Ascend PD Disaggregation via MooncakeConnectorLayerwise & MooncakeConnectorStore
  Workflow
triggers:
  error_signatures:
  - MooncakeConnectorLayerwise
  - MooncakeConnectorStore
  - kv_port collision
  keywords:
  - vllm-ascend
  - pd-disaggregation
  - mooncake-connector-layerwise
  - mooncake-connector-store
  - kv-transfer-config
  - load-balance-proxy
  - metaserver
---
## Preconditions & Trigger

Use this workflow when deploying Prefill/Decode (PD) disaggregation or multi-instance KV sharing in `vllm-ascend` using Mooncake's three KV connector architectures:
1. **`MooncakeConnector` (`kv_producer` / `kv_consumer`)**: Request-boundary P2P pull/push over RoCE/HCCS with an HTTP `metaserver`.
2. **`MooncakeConnectorLayerwise`**: Overlaps layer-by-layer KV cache push from Prefill to Decode during the forward pass to hide KV transfer latency.
3. **`MooncakeConnectorStore`**: Pool-based KV cache sharing via `mooncake_master` (`--kv-role kv_both`) for multi-node/colocated PD disaggregation or prefix sharing across instances.

## Step-by-Step Deployment & Port Calculation Workflow

### Step 1: Select Connector Variant & Verify RoCE/HCCS Readiness
- Before starting vLLM instances on Ascend 910B (A2) or 910C (A3), verify every NPU card reports `link status: UP` via `hccn_tool -i <id> -link -g` and export deterministic communication interfaces (`HCCL_IF_IP=<host_ip>`, `GLOO_SOCKET_IFNAME=<iface>`, `TP_SOCKET_IFNAME=<iface>`).
- **Single-Node Partitioning**: On an 8-NPU node (or 16-Die A3 node), pin Prefill and Decode to disjoint physical devices via `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` (Prefill) and `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` (Decode).

### Step 2: Configure P2P `MooncakeConnector` or `MooncakeConnectorLayerwise`
- Launch the lightweight proxy / metaserver (`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` or `load_balance_proxy_layerwise_server_example.py` for layerwise mode):
  ```bash
  python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --host <proxy_ip> --port 8000 \
    --prefiller-hosts <prefill_ip> --prefiller-ports 8100 \
    --decoder-hosts <decode_ip> --decoder-ports 8200
  ```
- **Prefill Instance (`kv_role: "kv_producer"`)** and **Decode Instance (`kv_role: "kv_consumer"`)**:
  Pass `--kv-transfer-config` with unique `engine_id` and non-overlapping `kv_port` bases:
  ```json
  {
    "kv_connector": "MooncakeConnectorLayerwise",
    "kv_buffer_device": "npu",
    "kv_role": "kv_producer",
    "kv_parallel_size": 1,
    "kv_port": "20001",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {"dp_size": 2, "tp_size": 2},
      "decode": {"dp_size": 2, "tp_size": 2}
    }
  }
  ```
- **Critical Port & Asymmetric TP/DP Invariant**:
  - Each DP rank + TP rank worker binds its own ZMQ/Mooncake handshake port offset from `kv_port`. When co-locating Prefill and Decode on the same host (or running `dp_size > 1`), ensure `kv_port_decode >= kv_port_prefill + (dp_size * tp_size) + 100` (e.g., `20001` for Prefill and `21001` for Decode) so worker initialization never collides on local ZMQ sockets.
  - Always match the proxy script to the connector: `MooncakeConnectorLayerwise` sends Decode requests *first* to register target layer buffers before Prefill streams each layer's KV blocks (`load_balance_proxy_layerwise_server_example.py`), whereas standard `MooncakeConnector` runs Prefill first and pulls KV on Decode (`load_balance_proxy_server_example.py`).

### Step 3: Configure Pool-Based `MooncakeConnectorStore` (`kv_both` / Colocated Sharing)
- When sharing a centralized DRAM/HBM KV pool across multiple vLLM-Ascend instances (either disaggregated PD or colocated multi-instance serving):
  1. Launch `mooncake_master --port 50051 --max_threads 8` and create a `mooncake.json` configuration file specifying `"local_hostname": "<node_ip>"`, `"metadata_server": "P2PHANDSHAKE"`, `"protocol": "ascend"`, `"master_server_address": "<master_ip>:50051"`, and `"global_segment_size"`.
  2. Export `MOONCAKE_CONFIG_PATH=/path/to/mooncake.json` and launch each vLLM-Ascend instance with `"kv_connector": "MooncakeConnectorStore"` and `"kv_role": "kv_both"` (for colocated prefix sharing) or `"kv_producer"` / `"kv_consumer"` (for store-backed PD disaggregation).

## Verification

- Send a multi-turn completion request through the proxy (`http://<proxy_ip>:8000/v1/completions`) and verify the Prefill worker logs layerwise/store KV send completions while the Decode worker produces tokens without recomputing the prompt prefix.
