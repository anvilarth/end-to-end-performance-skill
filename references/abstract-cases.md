# Abstract performance cases

Use these generalized cases to recognize common decision patterns. They encode
reusable lessons, not results for any particular model or organization.

## A small-scale win that reverses at target scale

For example, a training optimization improves step time on one node with 8 GPUs.
At 32 GPUs across multiple nodes, the same change exposes more communication,
disrupts overlap, and regresses the complete step.

**Verdict:** reject for the 32-GPU target. Preserve the 8-GPU result only as a
setup-specific finding; do not generalize it.

**Principle:** ask for the real usage setup first and validate scale and topology
as part of the optimization.

## A fast kernel with no end-to-end benefit

A fused operation is `1.4x` faster in a production-shaped microbenchmark. It
also passes tensor and gradient parity. In an interleaved full-training test,
however, median step time is unchanged and peak memory increases.

**Verdict:** reject. Correctness and local speed are prerequisites, but the
deliverable did not improve. The added memory makes the system strictly worse.

**Principle:** optimize the deliverable, not the component.

## A bottleneck that moves after a successful change

Compiler-cache reuse removes most graph compilation misses and improves startup
time. A new stage breakdown shows that dataset planning and first-batch
construction now dominate the remaining startup interval.

**Verdict:** promote the cache change if it passes fallback and parity gates,
then stop tuning compilation and re-rank the new stage budgets.

**Principle:** re-profile after material wins.

## A partially lossless optimization

A candidate produces exact outputs for the default inference mode and reduces
latency by 10%. A secondary conditioning mode differs on half of matched inputs,
with deviations well outside serialization noise.

**Verdict:** promote only the verified default path if it can be isolated. Do not
describe the entire implementation as lossless.

**Principle:** lossless is a measured, mode-specific property.

## A short-shape win that fails the real workload

An operator wins on a short sequence used during development. On the longest
common production shape it is slower, although numerical parity remains clean.

**Verdict:** reject the general fast path. Consider shape-gated dispatch only if
the short-shape traffic share and total expected benefit justify its complexity.

**Principle:** benchmark real shapes and weight results by real workload mix.

## Two winners that conflict

Scheduling change A and memory change B each improve step time independently.
Their combination performs worse than A alone because both rely on the same
overlap window and B increases allocator pressure.

**Verdict:** promote the best validated stack, not the union of all positive
arms.

**Principle:** independently passed changes still require a combined gate.

## A precision candidate that is fast but below the quality floor

A reduced-precision path improves end-to-end throughput by 35% and improves
quality relative to an earlier quantized baseline. Its absolute quality remains
below the predeclared production threshold.

**Verdict:** retain it as research evidence or a Pareto candidate; do not
promote it and do not call it nearly lossless without a defined tolerance.

**Principle:** report absolute metrics and enforce the original gate.

## A benchmark too noisy for the claimed effect

An optimization appears 1% faster across different workers, while repeated
controls on those workers differ by 4%. Same-worker interleaving was not used.

**Verdict:** inconclusive. Repair the measurement design before changing the
implementation again.

**Principle:** make the measurement admissible before interpreting it.

## An invalid environment is not a candidate regression

A custom path fails before timing because a required runtime dependency is
missing. The control runs successfully.

**Verdict:** inconclusive for the optimization. Fix and verify the environment;
do not record the candidate as slower or incorrect.

**Principle:** separate implementation verdicts from harness and environment
failures.

## Removing work beats optimizing it

Profiling reveals an unused model call in every inference request. Removing it
preserves exact outputs and saves more total time than several proposed kernel
micro-optimizations combined.

**Verdict:** promote after the full mode matrix passes exact parity and paired
latency gates.

**Principle:** remove unnecessary work before accelerating necessary work.
