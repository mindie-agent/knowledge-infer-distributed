---
category: sop
conditions:
  domain: dynamo
  hardware:
  - Ascend 910B
  - Ascend 910C
  - Distributed LLM Router Runtime
domain_path: dynamo/runtime/backpressure-and-cancellation/queue-threshold-and-abort
evidence_strength: software_contract
id: unit-dynamo-router-queue-backpressure-and-inflight-cancellation-sop
inherits_from:
- unit-tgi-async-rpc-cancellation-boundary
- unit-pd-disaggregated-kv-connector-and-asymmetric-tp-sop
provenance:
  evidence_refs:
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/lib/llm/src/kv_router/routing_host/cancellation.rs#L17-L248
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md#L249-L299
  source_locator: ai-dynamo/dynamo@7bc6263706afdecdbdd316751cc0b72e35cee970:lib/llm/src/kv_router/routing_host/cancellation.rs
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: Dynamo Router Policy-Class Queue Backpressure (HTTP 529) & Staged-KV Decode
  Cancellation Budget SOP
triggers:
  error_signatures:
  - decode cleanup budget exhausted
  - HTTP 529
  - Request was cancelled
  keywords:
  - dynamo
  - cancellation
  - cleanup-budget
  - staged-kv
  - backpressure
  - http-529
  - wspt
  - fcfs
---
## Preconditions & Trigger

Apply this SOP when configuring policy-class admission queues, Deficit Round Robin (DRR) scheduling, or diagnosing leaked remote-prefill KV blocks after client disconnects in Dynamo's `RoutingHost` (`lib/llm/src/kv_router/routing_host/cancellation.rs`).

## Policy-Class Queueing, FCFS/WSPT Ordering & HTTP 529 Admission Control

1. **Cache-Bucket Classification & Queue Gating**:
   - The router maps `(x-dynamo-meta-policy-class, uncached_isl)`—where `uncached_isl = raw_isl - best_cached_tokens_across_eligible_workers`—to a physical policy-class queue using strictly increasing `uncached_isl_buckets` floors.
   - Each class holds requests in a pending heap (`fcfs` or `wspt`) when all eligible workers exceed `prefill_busy_threshold_frac * max_num_batched_tokens`:
     - **`fcfs` (Tail TTFT Optimization)**: Orders by adjusted arrival time (`priority_jump - arrival_offset`).
     - **`wspt` (Average TTFT Optimization)**: Orders by `(1 + priority_jump) / scheduling_cost_tokens`, where `scheduling_cost_tokens = max(1, raw_isl_tokens - cached_tokens)`.
   - Within each class, the complete pending-queue key is `(strict_priority, due_at, policy_key)`, and cross-class dispatch is arbitrated via Deficit Round Robin (DRR) charging the uncached-token snapshot captured at enqueue.
2. **Per-Worker Scaled Queue Limits & HTTP 529 Rejection**:
   - Configure `request_queue_limit_per_worker`, `raw_isl_token_queue_limit_per_worker`, and `cached_token_queue_limit_per_worker` in the policy YAML. The effective class limit equals `configured_limit_per_worker * discovered_endpoint_count`.
   - Limits are checked **before** enqueueing: the request that crosses the threshold is admitted, and subsequent arrivals are rejected immediately with **HTTP 529** and the effective limit. Worker removal does NOT evict already-queued requests; new arrivals reject with HTTP 529 until queued backlog drains.

## Client Disconnect Cancellation vs. Staged-KV Decode Cleanup Budget (`120s`)

1. **Immediate Cancellation for Aggregated, Prefill & Bypassed Decode (`CancelWhenStopped`)**:
   - In `cancellation.rs`, `DispatchCancellation::for_request(phase, staged_kv)` evaluates whether an aborted client context (`context.stopped()`) should immediately drop the routing operation:
     - For `RequestPhase::Aggregated`, `RequestPhase::Prefill`, and `RequestPhase::Decode` with `StagedKv::Absent` (when conditional disaggregation bypassed remote prefill via `x-bypass-remote-prefill`), the policy is `CancelWhenStopped`.
     - `cancel_on_stop()` uses `tokio::select! { biased; result = &mut operation => Ok(result), _ = context.stopped() => Err(cancelled_error(context.id())) }` so a simultaneously completed ownership-bearing result is preserved for normal cleanup, while any pending operation is dropped immediately with `ErrorType::Cancelled`.
2. **Carve-Out for Disconnected Decode with Staged Remote KV (`DispatchWhenStopped`)**:
   - **Critical Invariant**: When remote prefill has completed (`RequestPhase::Decode` with `StagedKv::Present`, i.e., `request.staged_kv_cleanup == true`), KV blocks are already staged on the prefill worker for one specific decode worker, and **only** that decode worker's KV-transfer-complete path can free them. Abandoning the decode leg in the router would strand those KV blocks on the prefill worker until hardware/store lease expiration.
   - Therefore, `(RequestPhase::Decode, StagedKv::Present)` selects `DispatchWhenStopped`, continuing worker selection, routing-decision recording, and dispatch even after `context.stopped()` fires.
3. **Shared `CleanupBudget` (`CLEANUP_DISPATCH_TIMEOUT = 120s`) & Non-Migratable Error Mapping**:
   - Because the decode leg traverses multiple sequential routing stages (and conditional disaggregation may re-select), assigning an independent timeout per stage would multiply retention time. Instead, all stages share a single `CleanupBudget` (`120s`), whose clock starts the first time any stage observes `context.stopped()` (`budget.remaining()` saturates at `Duration::ZERO`).
   - If the `120s` budget expires before reaching the decode worker, `cleanup_budget_exhausted(context_id, stage)` logs a structured warning (`"decode cleanup budget exhausted before the worker was reached; staged KV blocks will not be released until they expire"`) and returns `ErrorType::Cancelled`.
   - **Why `ErrorType::Cancelled` is mandatory**: `ErrorType::Cancelled` is registered in `migration::NON_MIGRATABLE`. Returning any other error type would trigger the migration operator to retry the disconnected client's request on a *different* decode worker, wasting a fresh dispatch without freeing the KV blocks staged for the original decode worker.

## Verification

- Verify `cleanup_budget_decays_and_saturates_at_zero` and `drops_pending_operation_when_context_stops` pass in `cargo test -p dynamo-llm kv_router::routing_host::cancellation`.
- Confirm router logs distinguish ordinary client disconnects (`Request <id> was cancelled`) from staged-KV cleanup timeouts (`exhausted its decode cleanup budget at <stage>`).
