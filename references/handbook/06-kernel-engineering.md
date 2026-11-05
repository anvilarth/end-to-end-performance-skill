# 6. Kernel engineering

**Open when:** a measured operator family has material recoverable end-to-end
cost and cheaper changes do not address it. **Exit when:** a contract-compatible
implementation improves the complete path, or its best affordable improvement
is below the threshold. Sources: [kernel implementation](sources.md#kernel-implementation)
and [GPU MODE discovery](sources.md#gpu-mode-and-implementation-discovery).

## In this chapter

- [1. Write the operator contract before choosing a language](#1-write-the-operator-contract-before-choosing-a-language)
- [2. Search for an equivalent implementation first](#2-search-for-an-equivalent-implementation-first)
- [3. Establish a correct local harness](#3-establish-a-correct-local-harness)
- [4. Capture the smallest useful counter set](#4-capture-the-smallest-useful-counter-set)
- [5. Form a mechanism-specific optimization](#5-form-a-mechanism-specific-optimization)
- [6. Tune a bounded search space](#6-tune-a-bounded-search-space)
- [7. Debug memory and synchronization before integrating](#7-debug-memory-and-synchronization-before-integrating)
- [8. Integrate through the real framework contract](#8-integrate-through-the-real-framework-contract)

## 1. Write the operator contract before choosing a language

Extract actual calls from representative inputs. Record shape families, strides,
layout, dtype, accumulation precision, device architecture, alignment, masks,
causality, epsilon, scale/bias conventions and output structure. Identify aliasing,
allowed input mutation, required determinism and state carried between calls.
Include backward and higher-order gradients only when the workload requires them.

List supported tails and edge cases: non-divisible dimensions, short/empty
dimensions where legal, non-contiguous views, extreme values, masked rows and
recurrent chunk boundaries. Define a fast-path predicate and correct fallback
for cases outside the specialization. Never silently dispatch unsupported shapes
into a tutorial's unmasked loads.

Measure call frequency and total exposed operator cost. Include preparation,
packing, allocation, launches, output restoration and required synchronization.
A faster GEMM with a new per-call transpose can lose at the operator boundary.

## 2. Search for an equivalent implementation first

Inspect the current framework/engine backend and available vendor/library
implementation. Then search [triton-index](sources.md#gpu-mode-and-implementation-discovery),
FlashAttention/FlashInfer or the appropriate upstream library for the specific
operator. Read its contract, tests, limitations and license before adopting code.

Use [the DSL decision guide](../kernel-dsls.md) to choose implementation effort:

| Need shown by evidence | Likely starting point | Escalate when |
| --- | --- | --- |
| Supported standard operator | Existing framework/vendor backend | Its actual implementation lacks required behavior or leaves material headroom |
| Elementwise/reduction fusion, straightforward tiled operation | Triton or existing compiler fusion | Needed layout/synchronization control is unavailable or measured code generation is limiting |
| Tile-oriented attention/GEMM/recurrence composition | TileLang or an existing tiled library | Required architectural primitives/layout control justify lower-level work |
| Architecture-specific tensor-core/layout/pipeline control | CuTe DSL, CUTLASS or CUDA C++ | Only as far as the measured budget repays development and validation |

The language is a means, not a performance rank. A specialized implementation
may outperform one library on one shape and lose elsewhere. Keep the existing
implementation as a control and fallback during development.

## 3. Establish a correct local harness

Use an independent reference and representative shape distribution. Make input
generation reproducible and preserve baseline/candidate state separately.
Restore mutable inputs after warmup and autotuning. Check forward values,
required gradients and metadata before timing.

Report per-shape latency and the workload-weighted contribution, not just an
unweighted geometric mean that can hide expensive regressions. Include wrapper
cost separately when kernel-only timing is useful, then include it in operator
and application timing. Declare cache policy, stream behavior, warmup and
compilation. A cache-cleared microbenchmark and a repeatedly reused application
buffer answer different questions.

Use CUDA events or a suitable benchmark utility for asynchronous device work.
Avoid host timing of a bare launch. For extremely short kernels, batching calls
can improve timing resolution but can also change cache/state behavior; document
that trade-off. Do not benchmark reference and candidate concurrently on the
same device unless contention is itself the experiment.

## 4. Capture the smallest useful counter set

Start with one representative shape and one clearly labeled invocation. Warm up
outside capture; keep reference computation outside the selected range. Confirm
the report contains the intended kernel, device, launch geometry and counters.

For an application using a push/pop NVTX range named `candidate`, a template is:

```bash
ncu --target-processes all --nvtx --nvtx-include "candidate/" \
  --set basic --export "$PERF_ARTIFACT_DIR/kernel" \
  python kernel_bench.py --case "$CASE_ID"
```

Check the installed CLI's range syntax and options. Here `kernel_bench.py` and
its arguments represent the real local harness; the range must enclose the
actual worker's invocation. Add only the sections/metrics needed for the next
question. A full metric set can require many replay passes.

Nsight Compute replay, cache control and instrumentation can change execution.
Non-deterministic launches or stateful application replay may invalidate capture.
Keep normal-run latency separately. Counter collection must not be treated as
an end-to-end performance gate. Stop profiler repair when its expected value no
longer repays the remaining experiment budget.

## 5. Form a mechanism-specific optimization

| Counter/timeline evidence | Candidate action | Common reason it fails |
| --- | --- | --- |
| Too many bytes moved for little math | Fuse compatible operations, remove materialization, improve reuse/coalescing | Extra registers/spills, wrong aliases, required output still materialized |
| Low arithmetic intensity and bandwidth near the attainable limit | Reduce traffic or change allowed representation | More warps cannot remove the bandwidth bound |
| Tensor/matrix operation underuses compute | Better tiling/layout, use supported tensor-core path, improve tile scheduling | Tail waste, packing overhead, small problem lacks parallelism |
| Too few independent blocks or long tail waves | Smaller tiles, split reduction dimension, persistent/grouped scheduling | Added reductions, atomic contention, workspace or determinism changes |
| Register/shared-memory pressure limits active work | Reduce tile/stage count or live values | Lost reuse/overlap exceeds the occupancy benefit |
| Dependent memory/math stalls | Pipeline independent work or loads; improve locality | Incorrect dependencies, synchronization cost, resource pressure |
| Special-function instructions dominate | Reuse values or find an equivalent formulation | Approximation changes the numerical contract |
| Launch gaps dominate many tiny operators | Fuse, batch or graph replay | Device work is already hidden or fusion blocks efficient kernels |

Compute arithmetic intensity from the relevant traffic level: HBM, cache or
shared memory. Compare against hardware capabilities for the actual dtype and
instruction path. Peak vendor FLOPS, occupancy and percentage-of-peak counters
are context, not universal acceptance thresholds. Validate that a proposed
ceiling is attainable for the operation and shape before calling a gap recoverable.

## 6. Tune a bounded search space

Choose a few plausible tile sizes, warp counts, pipeline stages or scheduling
variants based on the bound. Record the search budget, representative shapes,
ranking metric and stopping rule. Correctness comes before timing each candidate.
Reject unsupported/resource-exceeding variants quickly and preserve failures.

Track compilation and tuning time. Cache dispatch decisions with sufficient
identity: architecture, software/compiler version, dtype, shape/layout and
relevant numerical settings. Do not reuse a best configuration across incompatible
GPUs or strides. Keep fallback coverage and check transitions between fast and
fallback paths. Repeatedly tuning against the final evaluation set weakens the
claim that performance generalizes.

Stop when the remaining theoretical gain cannot matter end to end, the bounded
search plateaus, or implementation plus validation exceeds the available budget.
An arbitrary 80% or 90% hardware-utilization target is not a stop rule.

## 7. Debug memory and synchronization before integrating

Use small edge cases, indexing inspection and visualization when needed. Run
the relevant Compute Sanitizer tool for risky low-level changes:

```bash
compute-sanitizer --tool memcheck python kernel_bench.py --case "$CASE_ID"
compute-sanitizer --tool racecheck python kernel_bench.py --case "$CASE_ID"
```

`memcheck` targets invalid/misaligned memory accesses; `racecheck` targets shared
memory hazards. Use `initcheck` or `synccheck` for the corresponding suspected
issues. Tool support and coverage depend on the implementation and environment.
A clean run is useful evidence, not proof of all global-memory race freedom or
mathematical correctness. Sanitized timings are not performance results.

## 8. Integrate through the real framework contract

For a PyTorch custom operator, declare schema, mutation/aliasing, output metadata
and needed autograd behavior accurately. Check fake/meta behavior used by
compilation. `torch.library.opcheck` tests registration integration; it does not
prove numerical correctness. Use independent value tests and appropriate gradient
checks as separate evidence. Do not introduce custom-op machinery for an ordinary
composition that works correctly through the existing framework path.

Confirm activation in the actual model worker, supported shape dispatch, package
installation and fallback behavior. Run repeated calls, shape changes and relevant
training/streaming state transitions. Then run the unprofiled operator benchmark
and [complete delivery gate](07-validation-and-delivery.md).

Example: a kernel family is 8 ms of a 100 ms request. Halving it has an ideal
4 ms gain; 3 ms of packing leaves about 1 ms before other effects. If the accepted
improvement threshold is 5 ms, stop this route even though the kernel is 2x faster.
