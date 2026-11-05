# 7. Validation, composition and delivery

[Home](../../README.md) / [Library](../README.md) / [Handbook](README.md) / 07

**Open when:** a candidate has passed its cheap screen or a session needs an
explicit verdict. **Exit when:** evidence supports promote/reject/inconclusive,
and the delivered entrypoint plus journal let another session reproduce it.
This chapter expands [measurement and parity](../measurement-and-parity.md)
and the [experiment loop](../experiment-loop.md).

## In this chapter

- [1. Use a gate ladder proportional to the change](#1-use-a-gate-ladder-proportional-to-the-change)
- [2. Keep references independent and coverage explicit](#2-keep-references-independent-and-coverage-explicit)
- [3. Interpret numerical differences at the right level](#3-interpret-numerical-differences-at-the-right-level)
- [4. Run the final A/B under matched conditions](#4-run-the-final-ab-under-matched-conditions)
- [5. Assign an explicit verdict](#5-assign-an-explicit-verdict)
- [6. Compose wins with explicit invalidation](#6-compose-wins-with-explicit-invalidation)
- [7. Validate what will actually be delivered](#7-validate-what-will-actually-be-delivered)
- [8. Leave a resumable decision record](#8-leave-a-resumable-decision-record)

## 1. Use a gate ladder proportional to the change

Run the cheapest decisive check first. Reuse valid evidence; do not rerun every
gate after an unrelated documentation change or a harmless result-label fix.

| Gate | Purpose | Typical trigger |
| --- | --- | --- |
| Activation and completeness canary | Candidate ran; expected inputs produced required outputs | Every new execution path, worker/config or packaging change |
| Local semantic/numerical checks | Correct values, masks, aliases, state and gradients | Operator, precision, layout, packing or algorithm changes |
| Representative short screen | Predicted mechanism exists; rough savings justify more work | Every new performance hypothesis |
| Full-path paired measurement | Useful end-to-end gain meets threshold | Candidate proposed for promotion |
| Trajectory/task-quality/scale gate | Effects not covered by local agreement are acceptable | Training behavior, generated outputs, distributed paths or approximation |
| Delivered-entrypoint smoke | Committed/installed implementation behaves as tested | Packaging, integration or launch changes before delivery |

Define the required gates before inspecting final results. A lossless scheduling
change and a quantized model do not need identical evidence, but both need the
applicable contract. User-specified gates are not optional just because the
candidate appears low risk.

## 2. Keep references independent and coverage explicit

Load reference artifacts or execute the preserved control independently of the
candidate. A candidate and comparator that import the same modified function
can agree on the same bug. Store reference identity, input IDs and comparator
version with results. Verify the comparator on a known matching case and a
deliberate meaningful mismatch when it is new or materially changed.

Report expected, completed, missing, duplicate, extra and failed IDs. For each
field or tensor family, include its evaluated count and failures. Missing
reference data remains missing with a reason; do not convert it into success,
zero error or a smaller hidden denominator.

For generated text or structured outputs, distinguish:

- transport completion and finish reason;
- parse/schema validity and complete required fields;
- agreement with the accepted reference;
- task quality under the agreed independent evaluation.

A constant output can have excellent schema validity. A high agreement score
can be misleading on an imbalanced field. Include the relevant positive/negative
cases and denominators; use fresh labels or human review when the improvement
claim requires semantic judgment that existing references cannot provide.

## 3. Interpret numerical differences at the right level

Compare absolute and relative error, non-finite counts and worst cases in the
domain where each is meaningful. Near zero, relative error alone is unstable.
For factorizations or constrained outputs, use the appropriate residual or
invariant rather than elementwise equality alone. For training, compare updates
and state as described in [chapter 5](05-training-and-distributed.md).

Floating-point reduction order and backend changes can alter results without
being bugs. Diagnose against a trusted higher-precision implementation where
useful, but do not redefine the acceptance contract after seeing the candidate.
Check reduced-precision settings separately: accumulation dtype, TF32 policy,
autocast, quantized weights/activations/KV and approximate special functions.

If numerical gates fail, retain the smallest representative failing case and
its full original context. Determine whether the cause is an indexing/semantic
bug, permitted roundoff beyond an incorrectly chosen pre-existing threshold,
or a genuine change requiring user acceptance. Do not label the candidate
promoted while that decision is unresolved.

## 4. Run the final A/B under matched conditions

Use the frozen workload, state restoration, warmup/cache policy and measurement
rule from [chapter 1](01-contract-and-baseline.md). Balance order across repeated
blocks when drift matters; ABBA is one option, not a magic cure for interference.
Keep the real independent sampling unit when computing uncertainty.

Run on the authorized hardware without competing benchmark arms. Record relevant
co-tenancy, power/thermal constraints and versions. Do not silently raise clocks,
add GPUs or remove required work to improve the candidate. If resource layout
is the intended change, compare at the agreed cost or GPU budget and state it.

Report baseline and candidate absolute time/rate, useful counts, variability,
memory, correctness/quality and required tail metrics. Speedup is
`baseline_time / candidate_time` for matched work. Percent less time and percent
more throughput are different: 100 to 80 seconds is 20% less time and 25% more
throughput. Label the result accordingly.

Preserve raw per-run records. Do not discard slow candidate runs unless the
exclusion rule was defined independently and applied symmetrically. If an
infrastructure failure invalidates a run, retain the failure record and reason.
Stop according to the planned repetition/sequential rule rather than sampling
until a desired confidence interval appears.

## 5. Assign an explicit verdict

| Verdict | Evidence | Next action |
| --- | --- | --- |
| Promote | Required correctness/quality gates pass and representative performance meets the contract | Keep the implementation; record its supported scope and deliver |
| Reject | A valid test disproves the mechanism, violates the contract or shows insufficient improvement | Preserve evidence and remove/disable the candidate from the promoted path |
| Inconclusive | Missing reference, invalid capture, excessive uncertainty, incomplete run or untested required scope | Name the missing evidence and cheapest bounded way to obtain it |

A valid flat performance result rejects the proposed speedup; an empty profiler
report does not. A passed screen is a promising candidate, not a fourth promotion
category. An OOM on a required valid input is a candidate failure, not an excuse
to delete that input from the manifest.

Keep the reason precise. “No improvement” is less reusable than “packing added
3.2 ms and erased the 3.0 ms GEMM gain on the required small batches.” Record
what change in workload, hardware or evidence would justify trying it again.

## 6. Compose wins with explicit invalidation

After promoting A, treat A as the new performance baseline while preserving the
original semantic reference when needed. Re-measure the budget before choosing B.
For interacting candidates, test the combined implementation; individual gains
are not additive when they remove the same work or change resource pressure.

Examples of invalidation:

- A new bucket policy changes shape frequency: rerun relevant compile/kernel
  dispatch and weighted performance checks.
- A different microbatch changes optimizer/communication amortization: revisit
  full-update timings and the accumulation equivalence gate.
- A new cache policy changes hit rate and state: revisit reuse correctness and
  both cold/warm performance claims.
- A wrapper adds a layout conversion: include that cost in the operator and
  full-path comparison even if the raw kernel is unchanged.

Do not re-open unrelated checks without a cause. Keep a compact dependency map
in the journal: what evidence depends on which shape, state, hardware or API.

## 7. Validate what will actually be delivered

Record the final commit and any required external package revision. Execute the
real launcher from the installed or packaged path and check worker imports,
configuration, selected backend, output location and a small real-input result.
If the final state materially differs from the benchmarked one, rerun the affected
performance/quality gate; a smoke test alone cannot transfer the old claim.

Provide a reproducible command, environment identity, required inputs/reference
locations and artifact paths. Include fallback conditions and known unsupported
cases. Keep secrets and private data out of public patches. Follow the user's
requested repository workflow for PRs and merging.

## 8. Leave a resumable decision record

Update the established journal through its normal writer. Do not create a
competing tracker just because this skill offers a schema. A useful handoff is:

```text
Contract and target:
Current promoted commit/configuration:
Baseline -> candidate metric, counts and uncertainty:
Correctness/quality gates and artifact paths:
Hardware, workload identity and cache/state policy:
Rejected/inconclusive hypotheses and reasons:
Remaining measured headroom:
Next experiment, expected saving, falsifier and budget:
Exact reproduce/check command:
```

Separate observed results from estimates. A resumed session should know what
already passed, what is running, what failed, and what remains unverified without
reading the entire conversation. Stop when the target is met or remaining
affordable headroom is exhausted; further research needs a concrete open question.

---

[← Kernel engineering](06-kernel-engineering.md) · [Handbook](README.md) · [Sources →](sources.md)
