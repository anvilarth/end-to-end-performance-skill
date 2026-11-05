# Performance decision cases

[Home](../../README.md) / [Library](../README.md) / [Workflow](README.md)

Use these generalized cases to recognize common decision patterns. They encode
reusable lessons, not results for any particular model or organization.

## A small-scale win that reverses at target scale

For example, a training optimization improves step time on one node with 8 GPUs.
At 32 GPUs across multiple nodes, the same change exposes more communication,
disrupts overlap, and regresses the complete step.

**Verdict:** reject for the 32-GPU target. Preserve the 8-GPU result only as a
setup-specific finding; do not generalize it.

**Principle:** recover the real usage setup first and validate scale and topology
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

## A compile win that loses in the real entry point

Graph compilation makes a static-shape benchmark 10% faster with clean
parity. In the production CLI the input mix spans dozens of resolutions and
partial last batches, so recompilation per shape dominates and throughput
falls below the uncompiled baseline.

**Verdict:** reject for the production path. Keep the benchmark number as a
setup-specific finding.

**Principle:** benchmark the real entry point with the real shape mix, not a
convenient static proxy.

## GPU sharing that also helps a stage that looks saturated

Running three worker processes per GPU under a CUDA MPS daemon raises a
launch-bound encoder stage by about 15% with bitwise outputs. A frame-ranking
stage has dominant kernels that already keep the tensor pipe above 85% busy,
on a GPU at its power cap. With the same layout, typical inputs gain the few
percent that an idle-time estimate predicts, but the longest inputs gain about
30%. On typical inputs, the gain stops growing after two or three processes.

**Verdict:** promote on both stages. On the second, record the plateau after
two or three processes as the measured ceiling for typical inputs.

**Principle:** an idle-time estimate from one process does not see host phases
between requests, so it can underpredict the gain from GPU sharing. Measure
each process count on the real input mix, including the longest inputs.

## Removing work beats optimizing it

Profiling reveals an unused model call in every inference request. Removing it
preserves exact outputs and saves more total time than several proposed kernel
micro-optimizations combined.

**Verdict:** promote after the full mode matrix passes exact parity and paired
latency gates.

**Principle:** remove unnecessary work before accelerating necessary work.

## The output contract decides the verdict

More requests in flight on a dynamic-batching transcription server give about
2x throughput. The outputs then differ from the control on more rows than
repeated control runs do. A batch-invariant engine mode gives exact text but
runs about 30% slower.

**Verdict:** reject the lever under an exact-text contract, and promote it
under a word-level match gate. Use the batch-invariant mode only to test
host-side changes.

**Principle:** set the required output match, exact or tolerant, and measure
how much repeated control runs differ before a load sweep.

## A speed-up measured on failed requests

A higher memory fraction appears to raise the throughput of a serving engine.
However, about 15% of the requests stop at once with out-of-memory errors, and
these failures take almost no time.

**Verdict:** reject this memory fraction, because it fails requests. Measure
again at a fraction with no failures.

**Principle:** count failures and validate outputs before you read throughput.
Set a success floor before the run, and mark runs below it invalid.

## A steady window that measured a wave

All clients start together, and the benchmark replaces its in-flight requests
only once or twice. The middle half of completions shows a rate about 20% above
the sustained rate. A preprocessing starvation seen in that window is only the
tail of the first wave.

**Verdict:** inconclusive. Measure again with a ramped client start, or with an
engine-side rate over a long run, before you act on the diagnosis.

**Principle:** a serving rate supports a decision only after the benchmark
replaces its in-flight requests several times. Synchronized starts make waves
that show false bottlenecks.

## A layout that wins only under heavy load

A disaggregated prefill/decode layout gives the same throughput as one-GPU
replicas at moderate load. It gives about 50% more throughput only with
thousands of requests in flight per node. At twice that load, throughput stays
flat, while timeouts appear and latency doubles.

**Verdict:** promote the layout only where the workload sustains that load, and
run it at the smallest load on its throughput plateau.

**Principle:** sweep offered load; compare layouts, and run each at the
smallest load on its plateau.

## A precision rung that unlocks a layout

A 4-bit weight format makes a large model about 20% faster on its current
layout. Its larger effect is that one replica now fits one GPU, which adds
about 25% more throughput. A disaggregated layout loses while its prefill part
does not fit one GPU. After the rung, it gives the highest throughput at high
load.

**Verdict:** evaluate the rung with the layouts it enables, and test again each
layout that lost only on memory fit. Apply the quality gate to the rung alone
and lossless checks to each layout, then check the composed stack again.

**Principle:** a precision change can matter more for the layout it allows than
for its kernel speed.

## An embedded engine with its features switched off

A training framework runs its rollout engines with the prefix cache, CUDA
graphs, scheduler overlap and a fast attention backend switched off. A layout
change with these settings gives about 15%. Restoring the features, with a
higher memory fraction, gives more than 3x.

**Verdict:** restore each disabled feature whose original reason no longer
holds, with parity checks, before you tune the layout.

**Principle:** compare the settings of an embedded engine with its standalone
defaults before topology work. An integration can disable the features that
give most of the engine's speed.

## Utilization rises while throughput falls

A convolutional encoder keeps its GPU busy within each batch with thousands of
large kernels, but idles between batches while it waits for input. A second
process per GPU under CUDA MPS raises utilization from about 75% to nearly
100%. Throughput falls by about 40% from cache interference. CUDA graphs gain
nothing, because there are no launch gaps to remove.

**Verdict:** reject both levers for this stage, and look for headroom in
kernels, precision, or the host side.

**Principle:** utilization measures busy time, not useful work; judge every
lever by throughput at the production batch size.

## A cascade that costs more than it saves

A multi-model pipeline skips the later models for items that the first model
rejects. Only about one item in ten is rejected early, which saves a few
percent of GPU time. Throughput falls by about 20%, because each item now
waits for the first verdict.

**Verdict:** reject at this early-exit rate. Revisit the cascade only if early
exits reach about a third of the items.

**Principle:** at fixed work in flight, throughput is work in flight divided by
latency. A wait in a worker or client is useful only if it saves more time than
it adds.

## A host fix with no gain on one GPU

Workers forked after the framework starts its thread pools inherit large pools.
One thread per pool, set before the fork, reduces host CPU time per item to
about one eighth. Throughput on one GPU does not change, because the GPU was
the limit.

**Verdict:** promote. All GPUs of a node share its host CPU, and the old cost
per item caps the node far below target.

**Principle:** judge host-side changes by CPU-seconds per item at node scale.

## A compression plan with no room above the floor

A plan proposes to prune layers or to distill a student that sees a coarser
input. The unchanged model scores only slightly above the quality floor. An
oracle student, built from the teacher's labels on that input, stays far below
it. The smallest cut, one layer with its best repair, misses the floor too.

**Verdict:** reject the plan before training, and stop all compression trials.
Return to exact levers, or ask the owner to review the floor.

**Principle:** before a compression campaign, test an oracle student and the
smallest cut with its best repair. If the oracle misses the floor, no student on
that input reaches it; if the smallest cut misses it, deeper cuts miss it too.

## A one-ulp kernel that flips near-tied decisions

A ranking stage selects frames by score, and many scores are near ties. A norm
kernel that differs from the eager operation by one unit in the last place
gains about 2%. The difference grows through the layers and changes the
selection on about a fifth of the inputs.

**Verdict:** classify the kernel as an output change. Promote it only with the
owner's explicit acceptance, and set no numeric tolerance for it.

**Principle:** for discrete choices over near-tied scores, no tolerance makes a
numerical change lossless. It is bitwise, or it is an accepted output change.

## A mismatch that is the baseline's own error

A fused attention kernel nearly halves the training step, but it changes the
log-probabilities and the gradient norm. Against an fp32 reference, the
baseline's attention gradients have several times more error than the kernel's.

**Verdict:** classify the change as intended-lossless. A gradient-direction gate
admits it to the loss-curve gate; promote only after that gate passes.

**Principle:** when a candidate deviates from the baseline, compare both with a
higher-precision reference. The error of the baseline sets the tolerance.

## A quality metric that a trivial baseline passes

A frame-selection gate counts a hit when a selected frame is within a few
frames of a reference frame. A rerun of the reference scores almost perfectly,
and plain uniform sampling scores only a few points lower. The policy selects
about half of all frames, so almost any selection is within the tolerance.

**Verdict:** do not use this metric as a gate. Tighten the tolerance until the
trivial baseline clearly fails, or use exact selection agreement.

**Principle:** before you trust a quality metric, score a reference rerun and a
trivial baseline. A usable gate passes the rerun and clearly fails the baseline.

## Removed work that changes the training objective

A policy-gradient trainer removes two forward passes, which together take over
a third of the step time. One pass computes reference log-probabilities that a
zero KL coefficient leaves unused. The other pass computes behavior-policy
log-probabilities, and the change reuses the values that the rollout engine
returns. However, engine and trainer numerics differ, so the importance ratio
moves and about 2% of tokens are clipped.

**Verdict:** classify the dropped reference pass as removed work and the reuse
as an algorithm change. Report the divergence and the clip fraction, and let
the training owner decide.

**Principle:** work is free to remove only when nothing uses its result, or when
the remaining path reproduces that result bitwise.
