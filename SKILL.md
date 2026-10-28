---
name: end-to-end-performance-skill
description: Optimize PyTorch and CUDA training or inference workloads end to end. Use when asked to profile, diagnose, benchmark, or accelerate model execution; reduce step time, startup time, latency, memory, or cost; validate a performance change; investigate distributed or compilation bottlenecks; raise GPU utilization or share a GPU between processes (CUDA MPS); or decide whether an optimization should be promoted. Start by establishing the real usage setup, apply a lossless-first strategy, require measured correctness and numerical parity, and write or swap in custom kernels (Triton, TileLang, CuTe DSL) once a kernel profile shows recoverable budget.
---

# End-to-End Performance

Optimize the delivered workload, not an isolated component. Treat a complete
training step, startup sequence, batch, generation, or served request as the
unit of success.

Run the loop below in order. Each step has an output and an exit condition;
do not start a step before the previous step's output exists, and post the
step's output to the user when the step says so. When the user asks for
progress, answer from the latest outputs in the reporting format.

## The loop

### Step 0. Load the log

Read [references/experiment-log.md](references/experiment-log.md). Open
`.perf-log/log.jsonl` in the workload repository, or create it.

- **Output:** a short summary of the active contract, latest baseline,
  promoted changes, rejected and inconclusive hypotheses, and open next actions.
- **Exit:** summary written. Do not re-propose a rejected hypothesis unless the
  setup, contract, or bottleneck has changed.

### Step 1. Real setup and contract

Establish how the code is actually used, asking once for everything missing:
training or inference and the exact end-to-end boundary; GPU model and count,
nodes, interconnect, topology; model, checkpoint, dtype, attention backend,
compiler settings; batch, sequence lengths, resolutions, input distribution;
serving concurrency and caching; important modes and tails; cold or warm;
memory, cost, quality, and operational constraints. If the setup cannot be
reproduced, label the chosen one a proxy and state what it cannot prove.

Then freeze: one primary metric and its target, secondary metrics, fixed
representative inputs and seeds, control revision and versions, exact or
numerical and quality tolerances, promotion threshold, and the hardware and
time budget.

- **Output:** a `contract` event and one line to the user: `now ? -> target X,
  metric M, budget B, lossless unless you say otherwise`.
- **Exit:** contract logged.

### Step 2. Baseline

Instrument the full path with low-overhead stage timings and memory metrics.
Warm up as the target mode requires; measure cold separately when it matters.
Repeat until the variance resolves the promotion threshold. Read
[references/measurement-and-parity.md](references/measurement-and-parity.md)
for the benchmark and gate design.

- **Output:** `baseline` event with absolute value, variance, and memory peaks.
- **Exit:** baseline stable enough to resolve the threshold.

### Step 3. Time budget and ceilings

Profile the baseline: a timeline (CPU, input, transfers, sync, GPU, comms,
output) first, then a kernel profile of a production-shaped window as soon as
GPU-bound time is concentrated in a few kernel classes (see Custom-kernel
escalation). For every stage classify the bound (compute, bandwidth, launch,
sync, communication, host, I/O, capacity) and estimate its ceiling.

- **Output:** `profile` event and, to the user, the budget table:

  | Stage or kernel class | Share of e2e | Bound | Achieved vs ceiling | Recoverable e2e |
  | --- | --- | --- | --- | --- |

  plus one sentence: whether the target is reachable losslessly, from the sum
  of recoverable budgets.
- **Exit:** every stage above the promotion threshold has a bound and a
  recoverable estimate.

### Step 4. Lever table

Read [references/playbooks/README.md](references/playbooks/README.md), the
class playbook, and `common.md`; for kernel rows read
[references/kernel-dsls.md](references/kernel-dsls.md). For each budget row
list candidate levers, ordered by the least behavior-changing rung first:

1. remove unnecessary work;
2. scheduling, overlap, caching, locality, topology (several processes per GPU
   under CUDA MPS, more smaller replicas instead of wider tensor parallelism);
3. an equivalent faster PyTorch implementation;
4. compilation, fusion, layout;
5. swap in or write a kernel (library backend, Triton, TileLang, CuTe DSL);
6. reduced precision;
7. algorithm change or approximation.

Rank by `best-case e2e gain x confidence / cost`. Drop levers whose best case
is below noise or the threshold.

- **Output:** a `hypothesis` event per lever, and to the user the lever table:

  | Lever | Rung | Budget attacked | Best-case e2e | Cost | Status |
  | --- | --- | --- | --- | --- | --- |

- **Exit:** the top levers are assigned to hardware. Run independent arms in
  parallel; while a long benchmark runs, build or parity-test the next one.

### Step 5. Implement one lever

One causal mechanism per arm, switchable, with a clean control path and only
the instrumentation needed to prove activation, correctness, and speed.

- **Exit:** the arm runs and its activation is observable.

### Step 6. Gates in ascending cost

1. static and activation checks;
2. exact representation or invariance checks;
3. finite loss and gradients;
4. isolated component timing, for diagnosis only;
5. paired end-to-end comparison on the same allocation, interleaved;
6. longer loss-curve or quality validation when numerical effects accumulate;
7. production-shaped scale and tail validation.

Kill the arm at the first failed gate. A broken harness, dependency, compile,
or worker makes it `inconclusive`, not `reject`.

- **Output:** a `gate` event after every gate, before the next one.

### Step 7. Verdict and status

- **Output:** a `verdict` event (`promote`, `reject`, `inconclusive`) with
  reason, headroom captured, transfer boundary, and next action; an updated
  lever table; and a status message in the reporting format.

### Step 8. Next iteration

- `promote`: compose with the promoted stack, re-run parity and the paired gate
  on the composition, then return to Step 3: the bottleneck has moved.
- `reject` or `inconclusive`: return to Step 4 and take the next lever.
- Leave the loop only when the target is met, the remaining lossless
  recoverable budget is below the threshold, or the budget is spent. If the
  target is above the lossless ceiling, present the approximate levers (rungs
  6 and 7) as a measured Pareto with quality deltas for the user to choose, and
  keep working lossless levers meanwhile. Never end a turn with only
  "stopping" or "waiting for a decision".

## Rules that apply at every step

- **Real deliverable, real conditions.** Throughput, latency, memory,
  quality, startup, and cost are separate metrics. Production shapes, modes,
  precision, concurrency, scale, and topology decide promotion; smaller setups
  only screen. Revalidate when the environment changes materially.
- **Measure before proposing.** Do not infer the bottleneck from the
  architecture.
- **Parity before performance.** Lossless is measured, not assumed: report
  deviations even for intended no-op changes, with absolute values for control
  and candidate.
- **Memory is performance.** Record allocated and reserved peaks,
  fragmentation, host memory, and OOM margin, and any batch or topology
  opportunity a reduction opens.
- **Authority.** Inside the hardware and budget the user allocated, run
  profiles, benchmarks, kernels, and parity checks without asking. Ask before
  production-affecting changes, spending beyond the budget, or a quality
  trade-off the user has not authorized; never invent one.
- **Negative knowledge is a result.** Log rejected levers with conditions and
  reasons so later sessions do not repeat them.

## Custom-kernel escalation

Descend to kernels when the end-to-end profile shows GPU-bound time
concentrated in a few kernels or kernel classes whose recoverable budget can
move the primary metric past the promotion threshold. Typical signals:
memory-bound kernels far below HBM bandwidth, chains of elementwise casts and
copies, tiny-grid or one-warp kernels, an unfused quantize or normalization
between two GEMMs, an attention or recurrence shape with no fast library path.

1. **Profile kernels early.** Nsight Compute on a production-shaped window costs
   minutes; run it as soon as the timeline points at GPU-bound time, not as a
   last resort. Use
   [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) to
   classify throughput against the pipe and bandwidth ceilings, stalls,
   occupancy, and tails. Convert each kernel's gap to its ceiling into an
   end-to-end budget.
2. **Swap before writing.** Check whether the framework, engine, or a kernel
   library already ships a faster implementation behind a backend flag,
   dispatcher, or stale guard. A swap is a flag plus a parity gate.
3. **Pick the language from the evidence.** Follow
   [references/kernel-dsls.md](references/kernel-dsls.md): Triton for fusion of
   memory-bound chains, TileLang for tile-level GEMM, attention, and recurrence
   variants, CuTe DSL when the ceiling needs architecture-specific tensor-core,
   memory, and pipelining features. Search
   [KernelWiki](https://github.com/mit-han-lab/KernelWiki) for techniques and
   prior implementations; it is evidence of technique, not of benefit here.
4. **Gate like any other arm.** Production shapes and dtypes, a numerical
   parity test against the control op, resource use, then the full-path paired
   gate. A faster microbenchmark is not a speedup.

When kernel budget remains and the user asked for more speed, write the kernel;
do not wait to be told.

## Reporting

Status messages to the user stay short and lead with the answer:

1. now -> target -> gap, in the primary metric, and whether the target is
   reachable losslessly, with the evidence;
2. what changed since the last update: verdicts with paired numbers;
3. what is running, where, and when each result is due;
4. the top levers from the lever table with best-case gains;
5. a decision the user must make, only if one blocks progress.

Use plain words for mechanisms; put profiles, parity tables, and logs in files
and link them.

Each verdict, and the final write-up, records in the log or a results file:
verdict; real setup; contract; baseline; measured bottleneck and limit; the one
causal change and its activation evidence; parity with absolute control and
candidate values; paired performance with variability and tails; trade-offs in
memory, quality, cold start, complexity, and modes; transfer boundary; fraction
of headroom captured; next action; and the logged event ids.

Read [references/abstract-cases.md](references/abstract-cases.md) for generalized
examples of correct promotion and rejection decisions.
