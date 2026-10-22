# Experiment Log

Use this reference to keep a local, append-only record of the contract,
baselines, hypotheses, gate results, and verdicts. The log is the memory of the
optimization effort: it lets a later session, or another agent, see what was
already measured, which ideas were rejected and why, and which results are still
inconclusive.

## Location

Write the log inside the workload's repository unless the user names another
location:

```text
.perf-log/
├── log.jsonl          # append-only event stream, one JSON object per line
└── artifacts/
    └── <event-id>/    # raw evidence: profiles, traces, benchmark output, diffs
```

Create the directory on first use. Ask the user once whether `.perf-log/`
should be committed or added to `.gitignore`; default to ignored, because traces
and environment details may contain hostnames, paths, or internal data.

## When to write

Write each event at the moment it happens, not at the end of the session. A run
that crashes, times out, or is interrupted must still leave its hypothesis and
its last completed gate in the log.

| Moment | Event `type` |
| --- | --- |
| Contract frozen or changed | `contract` |
| Baseline measured or re-measured | `baseline` |
| Stage map or profile completed | `profile` |
| Candidate proposed, including ones rejected before implementation | `hypothesis` |
| A validation gate finishes, passes, fails, or errors | `gate` |
| Final decision for a hypothesis | `verdict` |
| Free-form observation that affects later decisions | `note` |

Log hypotheses that are ranked out before implementation too. A recorded
"best-case gain below noise" is negative knowledge that prevents repeating the
same idea.

## Event schema

Every event carries:

```json
{
  "id": "H-003.gate-2",
  "type": "gate",
  "ts": "2026-10-22T14:05:11Z",
  "session": "2026-10-22-a",
  "rev": "a1b2c3d (dirty)",
  "setup": "8xH100 SXM, 1 node, FSDP, bs=32, seq=4096, bf16, warm",
  "summary": "one line a human can scan"
}
```

- `id`: stable identifier. Hypotheses use `H-<n>`; their gates and verdicts
  extend it (`H-003.gate-2`, `H-003.verdict`). Contracts, baselines, and
  profiles use `C-<n>`, `B-<n>`, `P-<n>`.
- `rev`: control or candidate revision, marked `dirty` when uncommitted changes
  exist.
- `setup`: the exact measured setup, or `proxy: ...` when it is not the target.

Type-specific fields:

- `contract`: `primary_metric`, `secondary_metrics`, `correctness_gate`,
  `promote_threshold`, `reject_threshold`, `budget`.
- `baseline`: `metrics` (absolute values with units), `variability`,
  `memory`, `artifacts`.
- `profile`: `stage_budget` (stage → share or time), `limit` (estimated
  physical or empirical bound), `artifacts`.
- `hypothesis`: `mechanism` (one causal change), `lever` (position in the
  lossless-first order), `attacked_budget`, `best_case_gain`,
  `confidence`, `cost`, `rank_score`, `based_on` (ids of the profile,
  baseline, or earlier verdict events that motivated it), `status`
  (`proposed`, `deprioritized`, `in_progress`).
- `gate`: `hypothesis`, `baseline` (id of the control it was compared
  against), `gate` (activation, invariance, finite-grad,
  component, paired-e2e, quality, scale), `outcome` (`pass`, `fail`,
  `error`), `control`, `candidate`, `delta`, `variability`, `artifacts`.
- `verdict`: `hypothesis`, `verdict` (`promote`, `reject`, `inconclusive`),
  `headroom_captured`, `reason`, `transfer_boundary`, `next_action`.

Record absolute control and candidate values, not only deltas. Put large raw
output under `artifacts/<event-id>/` and reference it by relative path.

## Rules

- **Append only.** Never edit or delete earlier lines. To correct an event,
  append a new one with `"supersedes": "<old id>"` and the reason.
- **Read before proposing.** At the start of a session, read `log.jsonl` and
  summarize: the active contract, latest baseline, promoted changes, rejected
  and inconclusive hypotheses, and open next actions. Do not re-propose a
  rejected hypothesis unless the setup, contract, or bottleneck has changed;
  when you do, cite the earlier id and what changed.
- **Invalidate on contract change.** A new `contract` or `baseline` event
  makes earlier performance comparisons non-attributable. Mark affected open
  hypotheses as needing re-measurement in the next `note`.
- **Separate harness failures.** Use gate `outcome: error` and verdict
  `inconclusive` when the candidate was never validly measured.
- **No secrets.** Do not log tokens, credentials, or private data samples.

## Reading the log

A quick human view:

```bash
jq -r '[.ts, .id, .type, .summary] | @tsv' .perf-log/log.jsonl
jq -c 'select(.type == "verdict")' .perf-log/log.jsonl
```
