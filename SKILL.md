---
name: end-to-end-performance-skill
description: Optimize PyTorch and CUDA training or inference workloads end to end. Use when asked to profile, diagnose, benchmark, or accelerate model execution; reduce step time, startup time, latency, memory, or cost; validate a performance change; investigate distributed or compilation bottlenecks; or decide whether an optimization should be promoted. Start by establishing the real usage setup, apply a lossless-first strategy, require measured correctness and numerical parity, and consult KernelWiki only when profiling proves that a custom GPU kernel is warranted.
---

# End-to-End Performance

Optimize the delivered workload, not an isolated component. Treat a complete
training step, startup sequence, batch, generation, or served request as the
unit of success.

## Start with the real usage setup

Before proposing optimizations, ask how the code is actually used. Do not begin
from a convenient local setup and assume that its result transfers.

Establish the target environment:

- training or inference, and the exact end-to-end boundary;
- GPU model, GPU count, nodes, interconnect, and distributed topology;
- model, checkpoint, dtype, attention backend, and compiler settings;
- batch size, gradient accumulation, sequence lengths, resolutions, and input
  distribution;
- serving concurrency, batching, cache state, and latency or throughput target;
- important modes, adapters, conditioning paths, and tail cases;
- cold-start, warm steady-state, or both;
- memory, cost, quality, and operational constraints.

If the intended setup is unknown, ask for it. If it cannot be reproduced, label
the chosen setup as a proxy and state exactly what it cannot prove.

Treat scale and topology as part of the optimization. State exactly which setup
the evidence covers, and revalidate when the intended environment materially
changes. Communication, sharding, recomputation, memory pressure, stragglers,
and overlap can change or reverse the result. Use smaller setups for screening
when useful, then run the promotion gate on the intended setup.

## Core principles

1. **Optimize the real deliverable.** Name one primary end-to-end metric, the
   fixed workload, the correctness contract, and the promotion threshold before
   changing code. Treat throughput, latency, memory, quality, startup, and cost
   as distinct metrics rather than interchangeable claims.

2. **Measure before proposing.** Build a full stage map covering CPU work, input
   processing, transfers, synchronization, GPU execution, communication,
   serialization, and output handling. Do not infer the bottleneck from model
   architecture alone.

3. **Find the available headroom.** Estimate a physical, mathematical, or
   empirical limit. For every candidate, identify the measured budget it attacks
   and its maximum possible end-to-end contribution. Reject work whose best-case
   impact is below measurement noise or the promotion threshold.

4. **Remove work before accelerating work.** Prefer eliminating unused model
   calls, redundant recomputation, synchronization, copies, conversions,
   serialization, initialization, or repeated I/O before tuning an operation
   that need not run.

5. **Use the least behavior-changing lever first.** Explore candidates in this
   order unless evidence justifies a different order:

   1. remove unnecessary work;
   2. improve scheduling, overlap, caching, locality, or topology;
   3. select a better equivalent PyTorch implementation;
   4. use compilation, fusion, or layout changes;
   5. introduce a custom CUDA or Triton kernel;
   6. reduce precision;
   7. change the algorithm or accept approximation.

6. **Treat lossless as a measured property.** Distinguish exact equivalence from
   numerically different but parity-preserving behavior. Measure deviations even
   when a change is intended to be lossless.

7. **Run parity before performance.** Invalidate performance claims from an arm
   that fails representation, sample-order, RNG, loss, gradient, output, or
   quality gates. Correctness is a prerequisite, not a speed result.

8. **Benchmark real shapes, modes, scale, and topology.** Cover production batch
   sizes, sequence lengths, resolutions, conditioning modes, precision, gradient
   accumulation, concurrency, distributed layout, memory pressure, and important
   tails. A convenient subset can screen candidates but cannot promote them.

9. **Make measurements admissible.** Compare candidate and control on the same
   hardware allocation, interleave them when possible, separate cold and warm
   behavior, repeat enough times to resolve the target effect, and report both
   absolute values and deltas. Prefer paired distributions over isolated means.

10. **Change one causal mechanism at a time.** Keep screening arms attributable.
    Combine changes only after each component has independently passed its gate.

11. **Assume interactions.** Re-measure the complete stack because individually
    useful optimizations may overlap, compete for memory, alter scheduling, or
    become antagonistic when composed.

12. **Re-profile after material wins.** A successful optimization changes the
    bottleneck. Rebuild the end-to-end stage map before choosing the next target.

13. **Treat memory as performance.** Record allocated and reserved peaks,
    fragmentation, host memory, communication buffers, and OOM margin. Include
    any batch-size or topology opportunity enabled by a memory reduction.

14. **Separate verdicts from failures.** Distinguish an optimization regression
    from an invalid harness, dependency problem, compilation failure, unhealthy
    worker, or unsupported backend. Report `inconclusive`, not `reject`, when the
    candidate was never validly measured.

15. **Be willing to stop.** Reject elegant implementations, large local wins,
    and sunk work when they fail the end-to-end contract. A well-supported no-go
    decision is a successful performance result.

16. **Record negative knowledge.** Preserve the baseline, conditions, absolute
    metrics, deviation metrics, limit, verdict, and reason. Prevent future agents
    from repeating invalid or low-headroom ideas.

## Optimization workflow

### 1. Freeze the contract

Record:

- the real usage setup and complete deliverable being optimized;
- one primary performance metric and relevant secondary metrics;
- representative inputs, shapes, modes, seeds, scale, and topology;
- the control revision, environment, hardware, and software versions;
- exact-equivalence or numerical and quality tolerances;
- promotion, rejection, and stop conditions;
- the maximum authorized runtime or compute budget.

Ask for approval before expensive, long-running, or production-affecting runs.
Do not invent a quality trade-off that the user did not authorize.

### 2. Establish the baseline

Instrument the complete path with low-overhead stage timings and memory metrics.
Warm up compilation, allocators, caches, and kernels as required by the target
mode, while measuring cold behavior separately when it matters. Verify that the
baseline itself is stable enough to resolve the desired effect.

Read [references/measurement-and-parity.md](references/measurement-and-parity.md)
when designing the benchmark or correctness gates.

### 3. Rank bottlenecks and candidates

For each material stage, classify the likely bound: compute, memory bandwidth,
launch overhead, synchronization, communication, host work, I/O, or memory
capacity. Rank candidates by:

`expected end-to-end impact x confidence / implementation and validation cost`

State the best-case end-to-end gain before implementation. Prefer cheap,
reversible, behavior-preserving changes with a large measured budget.

### 4. Implement one candidate

Keep the change independently switchable when practical. Preserve a clean
control path. Add only the instrumentation needed to prove activation,
correctness, and performance. Do not stack speculative changes into one arm.

### 5. Validate in ascending cost

Run the cheapest decisive gates first:

1. static and activation checks;
2. exact representation or invariance checks;
3. finite loss and gradients;
4. isolated component timing when useful for diagnosis;
5. paired end-to-end comparison;
6. longer loss-curve or quality validation when numerical effects can
   accumulate;
7. production-shaped scale and tail validation.

Kill an arm immediately when it crosses a predeclared rejection boundary.

### 6. Decide and iterate

Return exactly one verdict: `promote`, `reject`, or `inconclusive`. If promoted,
re-profile before selecting the next candidate. If composing passed changes,
repeat parity and end-to-end gates on the combined stack.

Consider reduced precision or algorithmic approximation only after the
lossless-first search is documented, unless the user explicitly chooses a
quality-performance trade-off. Present approximate candidates as a Pareto
frontier, never as lossless speedups.

## Custom-kernel escalation

Use standard PyTorch, compiler, and CUDA facilities first. Consult KernelWiki
when all of the following are true:

- profiling identifies a concrete GPU-kernel bottleneck;
- its recoverable budget can materially affect the end-to-end metric;
- production shapes, dtypes, scale, topology, and hardware are known;
- existing operators or compiler output do not meet the target;
- the validation plan includes numerical parity, resource use, and full-path
  measurement.

Treat KernelWiki as a source of kernel techniques and implementation evidence,
not as proof that a technique benefits this workload.

When a timeline and production-shaped component benchmark justify deeper kernel
analysis, use the
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) to collect
and interpret Nsight Compute evidence. Use it to classify utilization, stalls,
memory traffic, instruction mix, occupancy, and tail effects. Do not start with
NCU merely because the workload uses CUDA: first prove that the selected kernel
has enough recoverable end-to-end budget to matter.

The escalation boundary is:

`end-to-end profile -> material kernel budget -> NCU diagnosis -> KernelWiki or
existing implementation search -> candidate kernel -> parity -> end-to-end gate`

## Required result format

Report:

1. **Verdict:** `promote`, `reject`, or `inconclusive`.
2. **Real setup:** target environment, workload, modes, scale, and topology.
3. **Contract:** primary metric, correctness gate, and fixed conditions.
4. **Baseline:** absolute end-to-end performance, memory, and quality values.
5. **Bottleneck:** measured stage budget and estimated limit.
6. **Change:** one causal mechanism and evidence that it activated.
7. **Parity:** exact or numerical deviations, with absolute candidate and control
   values.
8. **Performance:** paired absolute results, delta, variability, and tails.
9. **Trade-offs:** memory, quality, cold start, complexity, and mode coverage.
10. **Transfer boundary:** which setups were measured and which remain unproven.
11. **Decision:** fraction of available headroom captured and the reason for the
    verdict.
12. **Next action:** re-profile, validate at target scale, compose passed changes,
    investigate a new bottleneck, or stop.

Read [references/abstract-cases.md](references/abstract-cases.md) for generalized
examples of correct promotion and rejection decisions.

## Acknowledgements

Thank you to MIT HAN Lab and all contributors to
[KernelWiki](https://github.com/mit-han-lab/KernelWiki) and
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill). Their work
provides valuable kernel-optimization knowledge and a rigorous Nsight Compute
analysis workflow. This skill complements those projects by deciding when to
descend to kernel-level work and by requiring every local improvement to return
to a representative end-to-end correctness and performance gate.
