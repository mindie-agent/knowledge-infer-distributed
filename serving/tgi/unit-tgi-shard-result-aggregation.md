---
category: sop
conditions:
  repository: huggingface/text-generation-inference
  revision: b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed
content_digest: sha256:f5a559afeaa88e11460d7ce7b1edd660b4633d088cd4deb9bb948c9cdd3159b5
domain_path: serving/tgi
evidence_strength: single_observation
id: unit-tgi-shard-result-aggregation
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-tgi-shard-result-aggregation-review.json
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v2/sharded_client.rs#L51-L96
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v2/sharded_client.rs#L122-L159
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v2/sharded_client.rs#L168-L185
  - https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v3/sharded_client.rs#L122-L168
  source_digest: ba0c870091e0dd93631900dcfba6b3c2b1ddf3f9d48bb15a4d6c98967734c169
  source_locator: docs/evidence/unit-tgi-shard-result-aggregation-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Shard fan-out must aggregate control errors explicitly
triggers:
  error_signatures: []
  keywords:
  - tgi
  - shard
  - fan-out
  - must
  - aggregate
  - control
  - errors
  - explicitly
---
At the pinned TGI revision, sharded info, health and filter_batch await every shard but return only the last Result; prefill, decode and clear_cache collect all Results. Generation lists are merged last shard first and the retained next_batch comes from that last shard. v2 warmup drops None values before choosing a minimum; v3 chooses a lexicographic minimum of a tuple. These are distinct aggregation contracts.

Reviewed limits and corrections:

- Use build.rs:16-20 for the v2 protoc error branch.
- Treat underlying tonic deadlines, transport cancellation and server cache lifetime as unverified.
- The warmup image's PNG bytes and JPEG MIME mismatch are source facts; acceptance by a deployed decoder was not tested.
- Do not infer all-shard health or component-wise v3 token minima from the returned control result.

Evidence scope: All 21 corrected source-reference ranges read from pinned Git; actual execution not performed.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [backends/client/src/v2/sharded_client.rs:51-96](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v2/sharded_client.rs#L51-L96)
- [backends/client/src/v2/sharded_client.rs:122-159](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v2/sharded_client.rs#L122-L159)
- [backends/client/src/v2/sharded_client.rs:168-185](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v2/sharded_client.rs#L168-L185)
- [backends/client/src/v3/sharded_client.rs:122-168](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/backends/client/src/v3/sharded_client.rs#L122-L168)
