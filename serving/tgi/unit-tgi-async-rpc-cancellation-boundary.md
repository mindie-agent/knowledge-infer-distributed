---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:8614cefe8d0f339ea490a68c80875b5bf5ba03651759bb6a18e49bac455817ec
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-async-rpc-cancellation-boundary
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-async-rpc-cancellation-boundary-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L41-L53
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L151-L226
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L307-L324
  source_digest: 060dd148ba2dba840558a2541e80c240432b3776d948a2544d9fde3c3c809a0d
  source_locator: docs/evidence/unit-tgi-async-rpc-cancellation-boundary-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: An async RPC wrapper does not supply inference cancellation or transactional
  cache rollback
triggers:
  error_signatures: []
  keywords:
  - tgi
  - an
  - async
  - rpc
  - wrapper
  - does
  - not
  - supply
  - inference
  - cancellation
  - or
  - transactional
  - cache
  - rollback
---
At this TGI revision, Prefill and Decode call model.generate_token synchronously inside async RPC methods. Decode calls cache.pop for each input batch before concatenation/generation and caches the successor only afterward; the wrapper contains no rollback across those operations. SIGINT/SIGTERM change a cooperative flag polled by serve_inner, which has no explicit server.stop or drain call in this unit.

Scope and caveats:

- A missing later batch causes the handler to raise after earlier pop calls. Actual cache removal and recovery semantics depend on the Cache implementation, which was not included; do not assert a measured lost batch.
- Event-loop blocking and transport-level cancellation remain structural inferences, not observed runtime traces.
- Cache-delete/clear API calls alone do not prove physical device-memory release.
- Use distinct Prefill:151-191 and Decode:193-226 references when stating absence of a handler-local cancellation check.

Evidence scope: All 11 corrected references read from pinned Git; no server process or grpc runtime test run.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [server/text_generation_server/server.py:41-53](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L41-L53)
- [server/text_generation_server/server.py:151-226](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L151-L226)
- [server/text_generation_server/server.py:307-324](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/server/text_generation_server/server.py#L307-L324)
