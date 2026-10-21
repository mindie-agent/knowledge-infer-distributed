---
category: workflow
conditions:
  domain: dynamo
  hardware:
  - Ascend 910B
  - Ascend 910C
  - Disaggregated Prefill/Decode Cluster
domain_path: dynamo/disaggregated-serving/worker-discovery-and-prefill/typed-pool-and-bootstrap
evidence_strength: software_contract
id: unit-dynamo-disaggregated-prefill-decode-discovery-and-bootstrap-workflow
inherits_from:
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/lib/llm/src/kv_router/prefill_router/mod.rs#L51-L261
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/docs/fern/pages/cli/disaggregated-serving/overview.mdx#L9-L67
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md#L35-L39
  source_locator: ai-dynamo/dynamo@7bc6263706afdecdbdd316751cc0b72e35cee970:lib/llm/src/kv_router/prefill_router/mod.rs
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L3
title: Dynamo Disaggregated Prefill/Decode Discovery, Bootstrap Handoff & Conditional
  Bypass Workflow
triggers:
  error_signatures:
  - Prefill router not yet activated
  - No disaggregated params in prefill response
  - x-bypass-remote-prefill
  keywords:
  - dynamo
  - prefill-router
  - disaggregated-serving
  - bootstrap-info
  - conditional-disagg
  - staged-kv-cleanup
  - nixl
  - mooncake
---
## Preconditions & Trigger

Use this workflow when deploying or diagnosing Dynamo disaggregated serving (`PrefillRouter` in `lib/llm/src/kv_router/prefill_router/mod.rs`) across independent Prefill and Decode worker pools connected by NIXL or Mooncake KV transport, or when configuring conditional disaggregation bypass (`--router-conditional-disagg`).

## Architecture & Lifecycle State Machine

`PrefillRouter` is a forward-only pipeline operator positioned between Migration and the Decode `RoutingHost`. Its atomic lifecycle (`PrefillLifecycleState`) transitions among:
- `Pending (0)`: Constructed before compatible typed prefill services are discovered.
- `Active (1)`: Activated once discovery resolves a compatible prefill `WorkerSetTarget` matching the decode pool's `model_name` and `namespace`, binding an dedicated `PrefillBinding` (`target_id`, `endpoint_id`, `router: Arc<RoutingHost>`, `prefill_router_mode`).
- `Unavailable (2)`: Marked unavailable when all prefill workers in the bound cohort withdraw.

## Step-by-Step Disaggregated Handoff & Conditional Bypass Workflow

### Step 1: Isolate Side-Channel Transport Ports Across Workers
- Launch independent Prefill and Decode workers advertising the same `namespace` and `model_name`.
- Ensure every worker on a shared host binds a unique NIXL or Mooncake side-channel port (e.g., `VLLM_NIXL_SIDE_CHANNEL_PORT` or `ASCEND_BASE_PORT`) and unique ZMQ KV-event publisher ports so multi-worker registration does not fail with port conflicts.

### Step 2: Execute Prefill Leg & Extract `PrefillOutcome` (`Bootstrap` vs. `Completed`)
- Depending on the backend engine's KV transfer protocol, `PrefillRouter` transforms the prefill worker's response into one of two handoff modes via `into_decode_request(req, outcome)`:
  1. **Concurrent Bootstrap Handoff (`PrefillOutcome::Bootstrap`)**:
     - Extracts `BootstrapInfo` (`bootstrap_host`, `bootstrap_port`, `bootstrap_room`, and a freshly generated `handoff_id: Uuid::new_v4()`) via `extract_bootstrap_info(&params)` alongside `prefill_worker_id` and `prefill_dp_rank`.
     - **DP Rank Invariant**: `prefill_dp_rank` MUST be propagated up front to the decode worker; without it, SGLang/DP engines park the decode request and issue a blocking HTTP query back to the prefill bootstrap server, stalling the prefill forward pass.
     - Prefill and decode run concurrently: waiting for prefill completion before starting the decode leg deadlocks rendezvous-based KV transfers.
  2. **Completed Prefill Handoff (`PrefillOutcome::Completed`)**:
     - Attaches `req.prefill_result = Some(result)`, `req.migration_link = worker_link`, and `req.routing_mut().prefill_worker_id = Some(worker_id)` after the prefill worker stages KV blocks.
- In both surviving handoff outcomes, `into_decode_request` sets `req.staged_kv_cleanup = true` to record that KV blocks are staged on the prefill worker for one specific decode target and can only be freed by that decode worker's KV-transfer-complete path.

### Step 3: Strip Internal Handoff Metadata on Terminal Output
- When a request completes directly on the prefill leg (`PrefillOutcome::Terminal`, e.g., `max_tokens=1`), `strip_terminal_disaggregated_params()` removes `data.disaggregated_params = None` before returning the stream chunk to the external client, because `disaggregated_params` is an internal engine-to-engine KV handoff contract.

### Step 4: Configure Conditional Disaggregation Bypass (`x-bypass-remote-prefill`)
- When `--router-conditional-disagg` is enabled (`conditional_disagg_prefill_busy_threshold` and `conditional_disagg_decode_busy_threshold`), the router evaluates whether a cache-hot Decode worker already holds sufficient prefix overlap (`cost = max(0, potential_decode_blocks - overlap_credit_blocks) + active_request_blocks`) or whether the Prefill pool exceeds its busy threshold.
- When the bypass policy triggers, `PrefillRouter` stamps the annotation `BYPASS_REMOTE_PREFILL_ANNOTATION` (`"x-bypass-remote-prefill"`), skips remote prefill entirely (`staged_kv_cleanup` remains `false`), and dispatches the request directly to the selected Decode worker to run prefill+decode locally.

## Verification

- Confirm the frontend logs `Activating prefill router` once both prefill and decode worker sets register.
- Inspect request traces to verify `bootstrap_room` / `prefill_dp_rank` populate on disaggregated requests while short/cache-hit requests stamped with `x-bypass-remote-prefill` execute directly on the decode pool.
