# Distributed Inference Domain Boundaries (`domain: "infer-distributed"`)

This repository (`knowledge-infer-distributed`) stores
distributed inference knowledge units and source-verification evidence under the
canonical domain `infer-distributed`. Runtime engine code lives exclusively in
`mindie-agent/knowledge`.

## Operating Rules

1. **Domain Scope & Isolation**: Keep distributed parallelism, multi-shard
   coordination, RPC/stream cancellation across process boundaries, collective
   communication, and PD state-transfer mechanisms in `infer-distributed`. Do
   not mix single-engine local block bookkeeping (`infer-core`), Ascend kernel
   implementations (`operators`), `vllm-ascend` adapter corpus files
   (`vllm-ascend`), or unclosed private histories into this repository. Reject
   foreign-domain `mindie://<other-domain>/...` URIs on `explain` and `feedback`.
2. **Unit Protocol & Index Rebuild**: Every unit must validate as
   `mindie-graded-unit/1` with a path-independent `id`, monotonic `revision`,
   exact `content_digest`, explicit `conditions`, and line-level
   `provenance.source_locator` / `evidence_refs`. Rebuild
   `index/routing-manifest.json` and `index/shard-topology.json` via the graded
   indexer after any unit change.
3. **Evidence Honesty & Three-Valued Conditions**: Bind source-derived units to
   the exact inspected `repository` and `revision` (e.g.,
   `huggingface/text-generation-inference` @
   `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed`). Never promote `single_observation`
   source-inspection SOPs to `reproduced_empirical` or `L4` without real
   execution/ablation receipts, and never treat `condition_unknown` as a match.
4. **Publication Gate**: [`./publication-contract.json`](./publication-contract.json)
   (`mindie-publication-contract/2`) and thin maintenance entry points under
   `scripts/` define the validator pin and local checks; run
   `python3 scripts/build_routing_index.py --root . --check` after any unit or
   index edit.
