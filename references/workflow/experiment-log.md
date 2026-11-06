# Experiment Log

[Home](../../README.md) / [Library](../README.md) / [Workflow](README.md)

Use this reference to keep a local, append-only record of the contract,
baselines, hypotheses, gate results, and verdicts. The log is the memory of the
optimization effort: it lets a later session, or another agent, see what was
already measured, which ideas were rejected and why, and which results are still
inconclusive.

## Location

Use the workload's existing experiment record and writer when available. Preserve
its append-only history, identifiers and artifact links; map the fields below to
that format rather than creating a parallel journal. Follow local workspace rules
for generated data. If no record exists, use this layout unless the user names
another location:

```text
.perf-log/
├── log.jsonl          # append-only event stream, one JSON object per line
└── artifacts/
    └── <event-id>/    # raw evidence: profiles, traces, benchmark output, diffs
```

Create the directory on first use. Reuse the repository's ignore policy and keep
generated traces and environment details out of commits by default; they may
contain hostnames, paths or internal data. Do not ask again when the user or the
workspace has already established how experiment artifacts are stored.

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
  "setup": "8x<GPU model>, 1 node, FSDP, bs=32, seq=4096, bf16, warm",
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
  low/central/high predicted savings with units and assumptions, cheapest
  falsifier, stop condition, `confidence`, `cost` (including setup and validation),
  `rank_score`, `based_on` (ids of the profile,
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
- **Read before proposing.** Start with the current summary or index, then retrieve
  relevant events and artifacts. Search the original log when needed. Summarize: the active contract, latest baseline, promoted changes, rejected
  and inconclusive hypotheses, and open next actions. Do not re-propose a
  rejected hypothesis unless the setup, contract, or bottleneck has changed;
  when you do, cite the earlier id and what changed.
- **Invalidate by dependency.** A changed contract, workload or environment
  invalidates affected comparisons. A new event alone does not invalidate
  identical immutable inputs or references. Mark affected hypotheses in the
  next `note`; use a fresh paired control for final timing.
- **Separate harness failures.** Use gate `outcome: error` and verdict
  `inconclusive` when the candidate was never validly measured.
- **No secrets.** Do not log tokens, credentials, or private data samples.

## Reading the log

A quick human view:

```bash
jq -r '[.ts, .id, .type, .summary] | @tsv' .perf-log/log.jsonl
jq -c 'select(.type == "verdict")' .perf-log/log.jsonl
```


## Memory review at stage boundaries

The event log and original artifacts remain authoritative. Memory review, sometimes
called dreaming, checks what future sessions should reuse. It does not rerun every
gate, prove a speedup, or create another journal.

During work, capture useful observations with the existing `note` event and writer.
Record the evidence ids and the setup where a conclusion applies. Separate tested
conclusions from guesses in the note text. Keep successes, failures, strategies,
and plans when useful; do not prescribe a quota or record only rejected levers.
Read current GPU occupancy, process state, and other cheap live facts from tools.
Do not preserve them as timeless rules.

At a completed stage, a handoff, or a contradiction that affects the next decision:

1. Retrieve relevant notes, verdicts, and their original artifacts. Check the measured
   boundary, revision, shapes, dtype, backend, hardware, and applicable gates.
2. Separate what the output proves from the agent's explanation. A transcript records
   a claim; parity reports and paired measurements determine its support.
3. Correct an unsupported conclusion with an appended `note` that cites and supersedes
   it. Preserve original events. Missing or changed evidence makes reuse uncertain;
   it does not turn a hypothesis into a verified rule.
4. Combine duplicate conclusions in the existing current summary. Keep a short index
   of useful findings, scope conditions, and event/artifact links. Put detail below
   the index or in linked evidence. Repair broken links before relying on them.
5. Check whether the next workload falls inside each conclusion's transfer boundary.
   Retrieve fresh evidence for changed dependencies. Do not widen a tolerance or
   inherit another task's permissions through a memory note.

Use the existing summary or state document. If repeated retrieval is expensive and
no summary exists, an optional `.perf-log/index.md` can serve as a derived index.
It is a navigation aid, not an additional source of truth or required artifact.
Keep only decision-relevant material in the startup context; move detail to links.
Omit stale entries from the index while retaining their historical evidence.
An empty index is valid. Search the original log when it cannot answer the question.

For an already authorized team, each worker writes through its assigned journal
writer or to its own evidence file. One owner reviews and updates the shared index.
Do not let concurrent workers overwrite one summary, and do not start a team merely
for memory maintenance. No new schema, daemon, or provider call is required.

Budget the review within the task. Focus on conclusions used by the next decision
and facts likely to drift. Stop when the handoff is sufficient; do not scan all
history or build a wiki larger than the evidence it summarizes. Public skill edits
receive only abstract, attributable lessons under the existing documentation workflow.
Private logs and environment details stay with their owning workload.

To claim that this process improves performance work, compare equivalent tasks and
budgets: time to a valid decision, repeated failed ideas, context/tool cost, and final
accepted results. A cleaner index alone does not establish an improvement.

Design source: [Anthropic, The components of agent memory](https://www.anthropic.com/research/components-of-agent-memory),
reviewed 2026-11-06. Its memory results motivate this procedure; they do not measure
local optimization quality or time savings.
