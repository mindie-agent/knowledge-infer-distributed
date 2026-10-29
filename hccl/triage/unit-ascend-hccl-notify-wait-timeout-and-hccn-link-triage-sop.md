---
category: sop
conditions:
  vendor: huawei-ascend
content_digest: sha256:c684ee76dba31f22be0f0caf63899ace6ab5494c8ff411784cdf675390990f1d
domain_path: hccl/triage
evidence_strength: software_contract
id: unit-ascend-hccl-notify-wait-timeout-and-hccn-link-triage-sop
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/cannbot-distributed-provenance.json#unit-ascend-hccl-notify-wait-timeout-and-hccn-link-triage-sop
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/maintenref/troubleshooting/docs/zh/error_codes/EI0002-Communication_Error_Timeout.md
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/maintenref/troubleshooting/docs/zh/error_codes/EI0006-Communication_Error_Get_Socket.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/runbooks/runtime/hccl_timeout_secondary_to_rank_failure_oom.md#L20-L63
  - https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/model/inference/runbooks/runtime/collective_participation_per_rank_consistency.md#L22-L84
  - https://gitcode.com/Ascend/msmodeling/blob/1adde03a04264fa976893fdf50495270175043e6/contrib/optix/multihost_infer/multihost_inference_optimization/README.md#L145-L215
  source_digest: fd7433c8c9b2de6d6d629e4d77ffb5dc09c57788a3f7d6d6c19dff678589f4d6
  source_locator: docs/evidence/cannbot-distributed-provenance.json#unit-ascend-hccl-notify-wait-timeout-and-hccn-link-triage-sop
  source_type: software_contract+runbook
  supersedes: []
revision: 2
schema: mindie-graded-unit/1
tier: L2
title: 'Triage EI0006/EI0002 notify-wait timeouts: separate rank skew from HCCN link
  degradation'
triggers:
  error_signatures:
  - EI0006
  - EI0002
  - HCCL_E_WAIT_TIMEOUT
  - Get_Socket_Timeout
  - An timeout occurs when the Notify register waits for execution
  keywords:
  - hccl
  - timeout
  - notify wait
  - rank skew
  - hccn_tool
  - optical
  - lldp
  - roce
  - link
  - participation
---
# Triage `EI0006` / `EI0002` / `HCCL_E_WAIT_TIMEOUT`: rank skew or link degradation

## Applicability boundary for offline analysis

Steps 1 and 2 analyze logs, rank participation and host control flow; they also apply to an offline reproduction without a known SoC, transport or rank table. Stop when the first failure or participation divergence explains the timeout and the affected behavior is verified. Steps 3 and 4 are conditional live-device investigations: enter them only after establishing the actual rank-to-device mapping, transport and rank-table schema, then read and match `unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms`. Never synthesize those facts from this SOP or run physical checks merely to complete the checklist.

`EI0006` is `Get_Socket_Timeout`. `EI0002` is a notify-wait timeout: a rank is waiting
for a peer's notify register to advance, and the default collective budget is 1800 s.
`HCCL_E_WAIT_TIMEOUT` is the same wait-timeout family. None of these is a root cause.
They say "a peer did not arrive"; the job is to decide whether the missing peer is a
rank that never entered the collective or a physical path that stopped delivering.

Follow the evidence-dependent branches below. Do not start by raising
`HCCL_EXEC_TIMEOUT`; that only moves the observation window.

## Step 1 - Freeze the scene and find the first failure on every rank

Collect the complete rank set, not rank 0's stack. The collective is asynchronous, so
the stack that reports the timeout is usually not the failing operator.

```bash
# one log directory per rank, or a merged log with rank prefixes
grep -RInE \
  'EI0006|EI0002|HCCL_E_WAIT_TIMEOUT|Get_Socket_Timeout|Notify register waits' \
  "$RANK_LOG_DIR"

# non-communication fatals that often precede the timeout
grep -RInE 'EL0004|Memory_Allocation_Failure|OOM|out of memory|Aborted|core dumped|exit code' \
  "$RANK_LOG_DIR"
```

Record for each rank: first timestamp, operator name, peer rank from the message,
notify id, task id, communicator/group, and whether the process is still alive when
the timeout is reported. Preserve `plog` and the launcher's exit status before
restarting anything.

Decision:

- If any rank has an earlier non-communication fatal error (OOM, illegal access,
  process exit, launcher kill), treat the timeout as a secondary symptom and fix that
  rank first. Do not spend time on optics yet.
- If all ranks are alive and the only errors are wait/socket timeouts, continue to
  Step 2.

If you need a sharper stack, rerun a minimal reproduction with
`ASCEND_LAUNCH_BLOCKING=1`. This serializes collective launch and makes the failing
call visible, but it changes timing and performance; remove it after localization.

## Step 2 - Prove or disprove per-rank collective participation

A wait timeout without a physical fault is usually a participation divergence: one
rank crashes, exits, skips a collective, or enters a different number of collectives
because its local batch is empty.

Build a collective ledger per rank for the failing step:

- expected sequence of collectives (all-reduce, all-gather, all-to-all, dispatch,
  combine, PP send/recv) and their order;
- actual sequence from the rank's trace or `ASCEND_LAUNCH_BLOCKING` log;
- per-rank token count, local batch size and DP/EP rank;
- whether an idle DP rank ran a dummy forward or returned before the collective.

Typical divergence signatures:

- an idle DP batch returns `None` or skips `forward_batch` while its TP peers enter
  the collective;
- a conditional collective is gated on a local cache hit, local sequence length or
  local queue state;
- a graph replay path captures a different number of collectives than the eager path;
- one rank exits on an exception before the collective, while the others wait.

If the ledger diverges, the fix is scheduler/control-plane alignment, not optics:
add a dummy/placeholder forward for the idle rank, align the prefill/decode phase
across DP ranks, or remove the local-state gate. Rerun the same step and confirm the
ledger matches before continuing.

If every rank entered the same collective and the peer still does not advance,
continue to Step 3. This is the boundary where physical triage becomes legitimate.

## Step 3 - Conditional live check: inspect the waiting peer's physical path

Run the checks on the device that the waiting rank actually uses. On A3 the device
range is normally 0..15; on A2 0..7; on A5 read the live enumeration. If a rank maps
to a die, check that die, not the card's first die.

```bash
DEV=<device-id>
PEER_IP=<peer-npu-roce-ip>

hccn_tool -i "$DEV" -link -g
hccn_tool -i "$DEV" -optical -g
hccn_tool -i "$DEV" -stat -g
hccn_tool -i "$DEV" -lldp -g
hccn_tool -i "$DEV" -ping -g address "$PEER_IP"
```

Interpretation:

| Observation | Meaning | Next action |
| --- | --- | --- |
| `-link -g` is DOWN or flapping | cable, optical module, or switch port | replace/drain; do not retry the job first |
| `-optical -g` shows no module or abnormal Rx power | optical degradation | replace the module/fiber |
| `-stat -g` shows retry, CRC, discard or error counters increasing | physical/link degradation | compare before/after; replace the suspect segment |
| `-lldp -g` is empty or names the wrong peer/port | wrong cable or switch port | correct the patching |
| `-ping -g address` fails | L3/RoCE IP/VLAN/routing problem | fix the NPU IP/route, not the collective |
| all five are clean | do not declare hardware failure | continue to Step 4 |

Take a `-stat -g` snapshot before and after the failing run. Absolute counters are
not meaningful without a baseline. A single clean snapshot does not prove the link
was clean during the timeout.

## Step 4 - Conditional live check: correlate the rank table, NIC and transport

Map the failing rank through the three identities:

```text
rank_id -> server_id -> device_id -> device_ip
```

Then check:

- the rank table's `device_id` is the logic device the process opened, not a physical
  `Phy-ID`;
- `device_ip` is the NPU RoCE IP, unique per die, and reachable from the peer;
- `HCCL_SOCKET_IFNAME` selects a host NIC that routes to every peer and is the same
  on all nodes;
- `HCCL_INTRA_ROCE_ENABLE` is consistent; if it is `1`, an intra-node collective may
  be exercising RoCE and a different set of counters than the default HCCS path;
- `HCCL_BUFFSIZE` is equal on all ranks and large enough for the failing collective;
- `HCCL_EXEC_TIMEOUT` is the intended budget, not an accidental override.

Classify the failing collective:

- intra-node (HCCS by default): a green RoCE link does not clear it; check the HCCS
  path and the die mapping;
- inter-node (RoCE): check `device_ip` reachability, switch port, optical module and
  RoCE counters;
- mixed/unknown: do not guess; record the operator's communicator and transport from
  the log before changing anything.

If the timeout is close to 1800 s and one rank is simply slower because of workload
skew, the fix is workload balance or a deliberate timeout budget change for that
experiment. A permanent timeout increase is not a fix.

## Step 5 - Remediate, rerun the same step, and close with evidence

Remediate only the class you proved:

- **Participation skew**: add the dummy/placeholder collective participation, align
  the phase, or remove the local-state gate. Rerun the same step; the ledger must
  match on every rank.
- **Physical link**: replace the cable/module/switch port or drain the node. Recheck
  `-link`, `-optical`, `-lldp` and a before/after `-stat` delta. Rejoin only when the
  path is clean.
- **Rank table / NIC**: fix `device_id`, `rank_id`, `device_ip`, or
  `HCCL_SOCKET_IFNAME`; restart the whole job so every rank reads the same table.
- **Buffer**: size `HCCL_BUFFSIZE` from the operator's printed need and keep it equal
  on all ranks.

For an offline code or parser repair, close when the identified failure and the
requested invariants pass the relevant local behavior regression. State that live
job health remains unverified; do not continue physical checks without new evidence.

For a live-job recovery objective, close only when:

- the same collective, on the same rank set, completes through the previously failing
  step;
- the all-rank logs show no new `EI0006`, `EI0002` or `HCCL_E_WAIT_TIMEOUT` for a
  sustained run;
- the link counters and optical/lldp state remain clean after the run;
- the exact commands, rank mapping, before/after counters and timeout values are
  recorded.

Rollback all diagnostic changes (`ASCEND_LAUNCH_BLOCKING`, temporary timeout increases,
dummy flags that are not part of the scheduler fix) before declaring the job healthy.

## Do not

- Do not diagnose from rank 0's asynchronous timeout stack alone.
- Do not replace optics because a timeout occurred; prove the link first.
- Do not treat a green `hccn_tool -tls -g` or a green intra-node link as proof that the
  cross-node RoCE path is healthy.
- Do not raise `HCCL_EXEC_TIMEOUT` before separating participation skew from link
  degradation.
- Do not declare a live job healthy on a single successful retry; require a
  sustained run and the counters relevant to the diagnosed transport.
