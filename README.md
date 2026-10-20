# Distributed Inference & State Transfer Knowledge (`infer-distributed`)

This independent local repository (`/Volumes/data/code/knowledge-infer-distributed`,
canonical domain `infer-distributed`) owns source-verified diagnostic
procedures and mechanisms for distributed inference execution, shard result
aggregation, cross-process RPC cancellation boundaries, collective
communication, and prefill/decode (PD) state transfer. Runtime indexing, query,
distillation, and synchronization code remains in `mindie-agent/knowledge`.
Authoritative validator pinning and program execution status are recorded in
[`./publication-contract.json`](./publication-contract.json) and
`/Volumes/data/temp/mindie-003-execution/execution-status.json`.

## Directory Structure & Admitted Units

- `serving/tgi/` — Pinned `text-generation-inference` distributed router-to-shard
  and shard-aggregation `L2` units:
  - `serving/tgi/unit-tgi-shard-result-aggregation.md` (`L2`, `category: sop`,
    `evidence_strength: single_observation`, `revision: 1`,
    `content_digest: sha256:f5a559afeaa88e11460d7ce7b1edd660b4633d088cd4deb9bb948c9cdd3159b5`)
  - `serving/tgi/unit-tgi-async-rpc-cancellation-boundary.md` (`L2`,
    `category: sop`, `evidence_strength: single_observation`, `revision: 1`,
    `content_digest: sha256:8614cefe8d0f339ea490a68c80875b5bf5ba03651759bb6a18e49bac455817ec`)
- `docs/evidence/reviewed-source-admission-v1.json` — Admission provenance from
  `/Volumes/data/temp/mindie-003-execution/wave3/research/editorial-review-packets/root-tgi-editorial-review.json`.
- `index/routing-manifest.json` & `index/shard-topology.json` — Derived routing
  and shard indexes rebuilt from the Markdown units.
- `publication-contract.json`, `.github/workflows/check.yml`, & `scripts/`
  (`validate_units.py`, `rebuild_index.py`, `check_publication_contract.py`,
  `apply_feedback.py`, `run_all_checks.py`) — Local `mindie-publication-contract/2`
  declaration and thin CLI wrappers delegating to `mindie_knowledge.graded`.

## Unit Protocol, Evidence & Revision Rules

- **Schema**: Every unit conforms to `mindie-graded-unit/1` (`id`, `revision`,
  `content_digest`, `tier`, `category`, `evidence_strength`, `title`,
  `domain_path`, `inherits_from`, `conditions`, `triggers`, `provenance`).
- **Exact Source Conditions**: Both initial units are bound to
  `conditions.repository: "huggingface/text-generation-inference"` and
  `conditions.revision: "b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed"`. Default
  `Store` queries match only when the exact `repository` and `revision` are
  supplied; mismatched revisions are rejected and omitted revisions remain
  `condition_unknown` (visible only when `include_unknown=True`).
- **Evidence Scope & Revision Discipline**: These `L2` units capture static
  source mechanisms (rank-0 vs. non-zero shard output handling and tonic gRPC
  cancellation boundaries at `client/src/client.rs` and `server/text_generation_server/server.py`)
  and make no live NPU execution or throughput claim. `id` remains stable across
  directory moves; any substantive update must increment `revision`, recompute
  `content_digest`, and rebuild `index/*.json`.

## Publication & Strict `Feed` Boundary

Validator revision and contract metadata are tracked in
[`./publication-contract.json`](./publication-contract.json), and live release
readiness is tracked in `/Volumes/data/temp/mindie-003-execution/execution-status.json`.
Although the public GitHub remote
`https://github.com/mindie-agent/knowledge-infer-distributed` has been created,
new content has not yet been pushed (`new_content_pushed: false`) and the paired
formal runtime/plugin release cutover remains pending.
