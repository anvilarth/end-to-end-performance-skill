# Fast experiment loop

[Home](../../README.md) / [Library](../README.md) / [Workflow](README.md)

Use for iterative performance work. Reuse existing lab files and tools rather than
introducing a second experiment framework. These are decision aids, not a mandatory
paperwork stage for a one-line fix.
Store decisions using [the experiment log](experiment-log.md) or the workload's
existing equivalent; the record below does not require another file or schema.

## Spend time on decisions

Track wall time to first valid baseline, first decisive rejection/promotion, and final
accepted delivery. Separate setup, control, profiling, candidate execution and idle
or repair time. These expose workflow waste; GPU utilization alone does not.

Reserve enough time for the final representative gate and packaging before screening.
Derive the reserve from measured baseline duration and required repetitions. Do not
start another arm when it cannot finish plus validate within the remaining budget.

A compact candidate record is enough:

| Field | Required information |
| --- | --- |
| Identity | Candidate flag/patch, baseline and input manifest identities |
| Evidence | Timed boundary, measured exposed cost and attribution confidence |
| Prediction | Low/central/high seconds saved, assumptions, expected useful rate |
| Falsifier | Cheapest test that can disprove the mechanism or parity |
| Budget | Allowed resources, command, deadline and rejection condition |
| Result | Activation, parity, paired timing, coverage, exit status, artifact paths |
| Decision | Promote/reject/inconclusive; next action or condition for reopening |

Reference stable contract fields rather than copying them into every report.
Unknown expected gain is a reason for an attribution probe, not invented precision.

## Two different experiment purposes

**Screen:** small but representative real inputs, including the limiting mode/shape.
Keep output length and cache semantics realistic. Establish activation and parity,
then determine whether the mechanism plausibly beats the minimum useful gain.
A positive screen means eligible for validation, never promoted.

**Promotion:** frozen representative manifest and final execution surface, relevant
scale/topology, all expected outputs, declared quality gates, paired uninstrumented
controls and candidates. Preserve failures and retries. A changed workload means a
new comparison, not a replacement denominator for the old result.

Validate expensive runs with a short A/A canary first. If control variability
exceeds the intended effect, repair that before running more candidates. Distinguish
performance noise from stochastic output disagreement; each needs its own gate.
Do not widen tolerances to rescue a candidate or discard an inconvenient control.

## Independent promotion after automated search

For agent-driven repeated optimization, keep acceptance outside the candidate's
editable experiment path. Reproduce the exact candidate revision in a fresh
process/environment using the protected workload manifest, evaluator and gates.
Candidate-side screening cannot substitute for this run. Separate reproducibility
from code review; neither alone proves the other.

Advance the accepted baseline only after this check passes, retaining the original
frozen reference and failed trials for cumulative attribution. An unavailable
independent run leaves the candidate screened, not promoted. Bound automated search
by the authorized resource/time budget; this procedure does not authorize new jobs.
See [the OpenRSI case and limits](../sources/openrsi-sglang-omni.md).

## Persistent measurement harness

When engine launch, weight loading, autotuning or graph capture dominate each
iteration (minutes of startup for seconds of measurement), load once and swap
the candidate in place: replace the forward entrypoint, recapture graphs, cache
kernel binaries across candidates. Reported ranges are two orders of magnitude
more experiments per hour and much less idle hardware.

Use two tiers. A cheap tier (a layer subset, a fixed cached request set) screens
candidates and absorbs crashes, illegal accesses and hangs; a full tier
promotes. Calibrate the cheap tier before trusting it: measure its variance and
its rank agreement with the full tier on a few known candidates, and tune the
smallest layer count that still ranks correctly. A cached request set stops
measuring prefill; say so in the record. Under a long budget with several
parallel searchers, keep the shared log flat and do not prescribe a lever order
or a curated knowledge base: reported ablations found both reduced the final
gain, while the experiment record itself remains required for verdicts.

## Profiling repair budget

Before full capture, prove one intended kernel from the real spawned worker is
visible. Check permissions, child targeting, NVTX range, warmup placement and that
profiler tools are not competing. A default repair budget of two targeted attempts
is reasonable unless the user/task specifies otherwise. Each retry must change a
named suspected cause. On exhaustion, use wall timers/timeline/backend counters if
sufficient, or report the blocked attribution as inconclusive. A missing required
capture remains missing; fallback evidence does not silently satisfy that gate.

## Concurrency search

Use a fixed manifest or realistic arrival trace. Increase workers/concurrency in
coarse steps within authorized resources, stopping at a measured plateau or a hard
constraint. Refine between the last useful and first failing point. Recheck output
parity as batching changes floating-point behavior or generation. Measure aggregate
server concurrency across clients, not just each client's semaphore. Validate the
selected operating point with tails and full drain included.

## Reuse and invalidate precisely

- Cache decoded inputs only if the target permits it; include preprocessing in the
  timed boundary otherwise. Record cache contents and state for both arms.
- Reuse references only for identical model, input, preprocessing and output contract.
- Reuse negative results until hardware, shape, backend or the measured cost changes.
- Reuse a loaded worker for warm-mode screens, reset mutated state between arms, and
  measure fresh processes separately when cold behavior is part of delivery.
- Keep worker orchestration minimal. Do not create manager/executor agents unless
  requested. If an existing authorized team is used, maintain one decision owner and
  distinct allocations; obey its plan/ownership protocol without inventing new gates.

## Closure

Before handoff, perform a bounded [memory review](experiment-log.md#memory-review-at-stage-boundaries)
of conclusions useful to the next decision. Update the existing summary with links
to source events, scope conditions, and unresolved evidence. Keep the flat event log
authoritative; a derived index must not prescribe the next lever or hide rejected arms.

Stop when the requested target is validated. For open-ended "maximum" work, record
the searched arms, binding constraint, remaining candidate headroom and untested
boundaries. Continue only while there is a plausible budgeted material improvement.
A well-supported rejection saves time; an unmeasured claim of maximum does not.
