---
category: sop
conditions:
  domain: dynamo
domain_path: dynamo/kv-router/radix-and-cost-model/prefix-overlap-and-selection
evidence_strength: software_contract
id: unit-dynamo-kv-router-radix-overlap-and-worker-cost-selection-sop
inherits_from: []
provenance:
  evidence_refs:
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md#L10-L89
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md#L14-L248
  - https://github.com/ai-dynamo/dynamo/blob/7bc6263706afdecdbdd316751cc0b72e35cee970/lib/llm/src/kv_router/request_guard.rs#L115-L274
  source_locator: ai-dynamo/dynamo@7bc6263706afdecdbdd316751cc0b72e35cee970:docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md
  source_type: source_contract
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
status: active
tier: L2
title: Dynamo KV-Aware Router Radix Prefix Overlap, Multi-Tier Cache Credits & Worker
  Cost Selection SOP
triggers:
  error_signatures:
  - Rejected cohort
  - PYTHONHASHSEED
  - dynamo-two-tier-cost-fn
  keywords:
  - dynamo
  - kv-router
  - radix-tree
  - overlap-score-credit
  - worker-selection
  - two-tier-cost-fn
  - pythonhashseed
---
## Applicability boundary for offline analysis

Block hashing, worker-set membership and the routing cost formula are host-side software contracts. Source inspection and local arithmetic tests do not require an Ascend SoC, HCCL rank table or a particular PD connector. The pinned Dynamo source below defines these formulas; compare the actual implementation when using another revision. Live throughput or cache-hit improvements still require measurements on the actual serving fleet.

## Preconditions & Trigger

Apply this SOP when configuring or tuning the Dynamo KV-aware router (`python3 -m dynamo.frontend --router-mode kv`) to maximize prefix cache hit rates across multi-worker vLLM / SGLang / MindIE pools without overloading cache-hot replicas.

## Canonical Block Hashing & Worker-Set Checksum Invariants

1. **Deterministic Cross-Worker Block Hashing (`PYTHONHASHSEED=0`)**:
   - The Dynamo KV router (`lib/llm/src/kv_router/indexer/`) indexes block hashes published over ZMQ (`publisher/zmq_listener.rs`) and matches them against canonical sequence hash chains (`compute_block_hash_for_seq` + `compute_next_seq_hash`, with Eagle lookahead window `stride + 1` when `is_eagle=true`).
   - For all Python-based inference workers (vLLM / vLLM-Ascend), you MUST export `PYTHONHASHSEED=0` across every worker process so KV cache block hashes emitted over ZMQ match across replicas and match the router's indexer lookups.
2. **Model Deployment Card (MDC) Checksum Admission**:
   - Every replica joining a worker set (`namespace`, `component`, `endpoint`, `model`, `worker_type`) must advertise an identical MDC checksum. The first valid card reserves the set's router configuration; newcomers with mismatched `--router-mode`, `kv_block_size`, or `source_path` are placed into rejected cohorts (`ERROR` log) and receive zero traffic until the incumbent cohort fully drains.

## Cost Model Arithmetic & Multi-Tier Overlap Credits

1. **Default Additive Cost Formula (`dynamo-default-cost-fn`)**:
   For each eligible candidate worker, the router evaluates:
   ```text
   raw_prefill_blocks      = active_prefill_blocks + incoming_prompt_blocks
   adjusted_prefill_blocks = max(0, raw_prefill_blocks - overlap_credit_blocks)
   potential_decode_blocks = active_decode_blocks + incoming_active_blocks
   active_request_blocks   = decode_active_request_weight * active_requests
   cost = prefill_load_scale * adjusted_prefill_blocks + potential_decode_blocks + active_request_blocks
   ```
   - `overlap_credit_blocks` sums weighted prefix matches across device HBM (`overlap_score_credit`, default `1.0`), host-pinned DRAM offload (`host_cache_hit_weight`, default `0.75`), disk/NVMe tier (`disk_cache_hit_weight`, default `0.25`), and shared cache (`shared_cache_multiplier`, default `0.5`).
   - To prevent a single cache-hot worker from suffering TTFT spikes under bursty shared-prefix traffic, configure `overlap_score_credit_decay > 0.0` (decays device-local overlap credit when `active_prefill_blocks` is elevated) or enable AIS prefill load decay (`router_prefill_load_model: ais`), which anchors elapsed time to the oldest active prefill on each worker.
2. **Two-Tier Cache/Load Policy (`dynamo-two-tier-cost-fn`)**:
   - When strict cache locality is preferred until load imbalance exceeds a hard envelope, configure `--router-policy-config worker-selection.yaml` with `type: dynamo-two-tier-cost-fn`:
     ```yaml
     worker_selection:
       aggregated: dynamo-two-tier-cost-fn
       prefill: dynamo-two-tier-cost-fn
       decode: dynamo-two-tier-cost-fn
       instances:
         - name: dynamo-two-tier-cost-fn
           type: dynamo-two-tier-cost-fn
           parameters:
             cache_threshold: 0.5
             balance_abs_threshold: 32
             balance_rel_threshold: 1.1
     ```
   - **Two-Tier Selection Rule**: The router routes to the least-loaded worker ONLY when both load gates hold simultaneously (`max_active_requests - min_active_requests > balance_abs_threshold` AND `max_active_requests > balance_rel_threshold * min_active_requests`); otherwise, if the best worker's device-KV overlap strictly exceeds `cache_threshold` (`> 50%` of request blocks), it selects the highest-overlap worker.

## Verification

- Inspect frontend startup logs for `Activating prefill router` and verify no MDC checksum rejection errors appear when workers register.
- Compare `RouteObservation` metrics (`best_router_tokens`, `selected_router_tokens`, and worker-reported `engine_data.kv_cache_hit.reused_tokens` in `request_guard.rs`) to confirm router-predicted prefix hits align with engine-executed KV reuse.
