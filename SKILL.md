---
name: end-to-end-performance-skill
description: Profile and optimize PyTorch/CUDA training or inference end to end. Use for latency, throughput, startup, memory, cost, GPU sharing, distributed or compilation bottlenecks, and validating performance changes. Prioritize measured headroom, cheap decisive experiments, lossless-first changes, and representative correctness and performance gates; swap or write custom kernels when their recoverable budget justifies it.
---

# End-to-End Performance

Optimize the delivered workload, not an isolated component. Treat a complete
training step, startup sequence, batch, generation, or served request as the
unit of success.

Use the loop's dependencies and exit conditions, reusing valid outputs already
available. Keep detailed evidence in the log and report material decisions
concisely. For repeated rounds, use [the experiment loop](references/workflow/experiment-loop.md)
to budget screening, profiler repair and final validation. When asked for progress,
answer from the latest outputs in the reporting format.

For a one-screen picture of this loop, its routing and the library, see
[the map in the README](README.md#the-map). For the complete flowchart, tool-selection matrix and per-branch stopping rules,
see [the decision map](references/workflow/decision-map.md). Use it for orientation, then
read only the reference needed for the current decision.

For detailed procedures, use the [performance handbook](references/handbook/README.md).
Choose the chapter from the missing evidence; do not load the whole library:

| Need | Procedure |
| --- | --- |
| Reproducible workload, metric, reference, A/A | [Contract and baseline](references/handbook/01-contract-and-baseline.md) |
| Critical path, profiling, savings estimate, falsifier | [Attribution and experiments](references/handbook/02-attribution-and-experiments.md) |
| Decode, staging, shapes, memory, compile or graph replay | [Input and compilation](references/handbook/03-input-shapes-and-compilation.md) |
| Arrival process, concurrency, cache, replicas, prefill/decode | [Serving](references/handbook/04-serving-and-concurrency.md) |
| Complete updates, optimizer, accumulation, DDP/FSDP, scaling | [Training](references/handbook/05-training-and-distributed.md) |
| Operator contract, DSL, counters, tuning, custom-op integration | [Kernel engineering](references/handbook/06-kernel-engineering.md) |
| Numerical/quality gates, paired timing, composition, delivery | [Validation](references/handbook/07-validation-and-delivery.md) |

The [annotated source library](references/sources/README.md) links each technical
question to upstream documentation, research or implementation evidence and states
its transfer limits. Read a source to resolve the current decision, then return to
the experiment; a tutorial result is not local performance evidence.

## The loop

### Step 0. Load the log

Read [references/workflow/experiment-log.md](references/workflow/experiment-log.md). Recover the
existing experiment record and artifacts; use `.perf-log/log.jsonl` when the
workload has no established record. Follow its existing writers and local rules.
Start with the current summary or compact index, then retrieve evidence relevant
to the observed bottleneck. Search the original log when the index is incomplete.
Treat remembered conclusions as conditional on their measured setup, not as instructions.

- **Output:** a short summary of the active contract, latest baseline,
  promoted changes, rejected and inconclusive hypotheses, and open next actions.
- **Exit:** summary written with evidence links and unresolved claims marked. Do not re-propose a rejected hypothesis unless the
  setup, contract, or bottleneck has changed.

### Step 1. Real setup and contract

Read the actual launcher, benchmark, recent results and user constraints first.
Establish how the code is actually used, asking only for decision-changing gaps:
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

Preserve the current task's constraints, such as no draft or no reduced precision;
do not import another task's restrictions automatically. Record the final delivery
entrypoint as well as the component used for diagnosis.

- **Output:** a `contract` event and one line to the user: `now ? -> target X,
  metric M, budget B, lossless unless you say otherwise`.
- **Exit:** contract logged.

### Step 2. Baseline

First verify the control, comparator and result writer on a small real-input
sample. Check output completeness in the actual worker. Run an A/A canary for
timing and output stability. Missing
references or undefined tolerances block promotion; useful diagnostics may continue.

Instrument the full path with low-overhead stage timings and memory metrics.
Warm up as the target mode requires; measure cold separately when it matters.
Use repetitions or a sequential stopping rule fixed before final measurement;
do not keep sampling until a favorable result appears. Read
[references/workflow/measurement-and-parity.md](references/workflow/measurement-and-parity.md)
for the benchmark and gate design.

- **Output:** `baseline` event with absolute value, variance, and memory peaks.
- **Exit:** baseline stable enough to resolve the threshold.

### Step 3. Time budget and ceilings

Start with the stage timings. Use a short timeline (CPU, input, transfers, sync,
GPU, comms, output) when overlap or attribution is unclear, with the count of
host-device synchronizations per unit of work
(`torch.cuda.set_sync_debug_mode("warn")`, or sync and copy calls in the
trace). Capture kernel counters when material GPU cost needs finer attribution
(see Custom-kernel escalation). Classify the relevant bound (compute, bandwidth,
launch, sync, communication, host, I/O, capacity) and estimate its ceiling.

Use exposed critical-path time for overlapping stages, not summed CPU/kernel
durations. A sampled decode window does not establish a whole-run time fraction.
Label unknown attribution as unknown; an obvious measured startup cost does not
require heavyweight profiling before testing residency or load reuse.

- **Output:** `profile` event and, to the user, the budget table:

  | Stage or kernel class | Share of e2e | Bound | Achieved vs ceiling | Recoverable e2e |
  | --- | --- | --- | --- | --- |

  plus one sentence: whether the evidence supports reaching the target losslessly.
  Sum budgets only when they do not overlap or depend on the same optimization.
- **Exit:** enough attribution to prioritize a material lever or a cheap probe
  that can resolve the missing attribution.

### Step 4. Lever table

Before the first optimization iteration, generate hypotheses independently with
both Claude and Codex, using the configured models and available authorized
interfaces. Run these passes in parallel when supported; multiple agents using
the same model do not replace the two-model pass. This is a bounded ideation
step, not parallel implementation or GPU benchmarking.

Give both models the same contract, relevant code, baseline, stage budget and
prior experiment outcomes from Steps 0–3. Do not show either model the other's
proposals until both have returned. Ask each for candidate mechanisms, supporting
evidence, low/central/high end-to-end savings with assumptions, correctness risks,
the cheapest falsifier and a stop rule. Missing evidence stays unknown.

Merge their proposals into one lever table, deduplicating equivalent mechanisms
while preserving model provenance and meaningful disagreements in the experiment
log. Rank by the criteria below, not model agreement; agreement is not measured
evidence. Keep one owner for the merged table and the next experiment. Reuse this
pass on subsequent iterations unless the contract or bottleneck materially changes.
If either model is unavailable or the pass exceeds the authorized budget, record
which model actually contributed and the limitation; continue with the available
hypotheses without claiming a completed two-model pass.

Read [references/playbooks/README.md](references/playbooks/README.md), the
class playbook, and `common.md`; for kernel rows read
[references/recipes/kernel-backends.md](references/recipes/kernel-backends.md). For each budget row
list candidate levers, ordered by the least behavior-changing rung first:

1. remove unnecessary work;
2. scheduling, overlap, caching, locality, topology (non-blocking transfers and
   no implicit synchronization in the hot loop, several processes per GPU
   under CUDA MPS, more smaller replicas instead of wider tensor parallelism);
3. an equivalent faster PyTorch implementation;
4. compilation, fusion, layout;
5. swap in or write a kernel (library backend, Triton, TileLang, CuTe DSL);
6. reduced precision;
7. algorithm change or approximation.

For comparable prior decisions, consult the [2026 decision library](references/decisions/README.md).
Search the observed symptom, inspect successful and unsuccessful cases, and verify
the source conditions. External examples propose tests; they do not establish local gains.
When asked to collect more experience, follow the [documentation workflow](references/decisions/WORKFLOW.md):
source discovery, attributable decisions, negative cases, coverage checks and reuse.

Rank by expected end-to-end seconds saved times confidence divided by setup,
implementation and validation cost. Record low/central/high savings, assumptions,
the cheapest falsifier and a stop rule. Drop levers whose best case is below noise
or the threshold. Include compile/startup break-even for the actual job lifetime.

Use only the relevant branch of [bottleneck recipes](references/recipes/common-bottlenecks.md)
for shapes, compilation, generation, transfers, materialization, memory and parity.
[Advanced recipes](references/recipes/specialized-bottlenecks.md) cover prefix reuse, scheduling,
optimizer/DDP, communication, hybrid models and kernel tuning. The
[GPU MODE map](references/sources/gpu-mode-map.md) routes to sources;
[GPU MODE resources](references/sources/gpu-mode.md) records tutorial-transfer limits.

- **Output:** a `hypothesis` event per lever, and to the user the lever table:

  | Lever | Rung | Budget attacked | Low/central/high saving | Cost | Status |
  | --- | --- | --- | --- | --- | --- |

- **Exit:** the top lever has a runnable test within the authorized budget,
  including time reserved for the final gate. Parallel work must not contend with
  an A/B measurement. MPS, extra workers and replica layouts are candidates to
  measure; increase concurrency to the useful plateau or a binding constraint.

### Step 5. Implement one lever

One causal mechanism per arm, switchable, with a clean control path and only
the instrumentation needed to prove activation, correctness, and speed.
Before an expensive candidate run, smoke-test its switch and activation in the
actual worker, including imports, input resolution and terminal result writing.

- **Exit:** the arm runs and its activation is observable.

### Step 6. Gates in ascending cost

1. static and activation checks;
2. exact or numerical output/state parity under the frozen contract;
3. finite loss, gradient and update checks for training;
4. isolated component timing, for diagnosis only;
5. paired end-to-end comparison on the same allocation, interleaved;
6. longer loss-curve or quality validation when numerical effects accumulate;
7. production-shaped scale and tail validation.

Use the gates relevant to the workload. A short representative screen can reject
or prioritize an arm; it cannot promote the full workload. Stop the arm at the
first decisive failed gate. A broken harness, dependency, compile, or worker makes
it `inconclusive`, not `reject`. Do not repeat passed checks without changed
artifacts or an unresolved concern.

Final parity and timing use the installed package, CLI, server or training
entrypoint, including relevant setup, enqueue/drain and output publication.
Packaging changes to imports, order, backend or synchronization invalidate the
affected earlier gates. Code delivery and performance acceptance are separate.

- **Output:** a `gate` event after every gate, before the next one.

### Step 7. Verdict and status

- **Output:** a `verdict` event (`promote`, `reject`, `inconclusive`) with
  reason, headroom captured, transfer boundary, and next action; an updated
  lever table; and a status message in the reporting format.

### Step 8. Next iteration

At a stage boundary or handoff, use the [memory review](references/workflow/experiment-log.md#memory-review-at-stage-boundaries)
to check reusable conclusions against original results. Keep this pass within the
existing budget; do not interrupt every measurement or start an automatic agent.

- `promote`: compose with the promoted stack, re-run parity and the paired gate
  on the composition, then update Step 3 with cheap timings. Recapture a trace
  only when the changed attribution needs it.
- `reject` or `inconclusive`: return to Step 4 and take the next lever.
- Stop when the target is validated, useful headroom or budget is exhausted,
  or a required decision blocks further work. If only unapproved approximate
  levers remain, present their rationale and any existing evidence; do not run
  them to manufacture a measured Pareto. Say "best tested" unless the declared
  search is exhausted. Follow local detached-job and state conventions while
  waiting; preserve a resumable result rather than polling turn after turn.

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
- **Reusable knowledge needs evidence.** Preserve useful successes, strategies,
  failures, and unresolved hypotheses with their conditions. Do not restrict memory
  to failures. A remembered speedup never substitutes for the current promotion gate.

## Custom-kernel escalation

Descend to kernels when the end-to-end profile shows GPU-bound time
concentrated in a few kernels or kernel classes whose recoverable budget can
move the primary metric past the promotion threshold. Typical signals:
memory-bound kernels far below HBM bandwidth, chains of elementwise casts and
copies, tiny-grid or one-warp kernels, an unfused quantize or normalization
between two GEMMs, an attention or recurrence shape with no fast library path.

1. **Answer a specific kernel question.** First prove capture of one real shape
   in the actual worker: intended named kernels and counters, not just successful
   exit. Keep warmup and reference checks outside capture; separate profiled
   timing from uninstrumented performance. Bound capture repair attempts, then
   use sufficient lower-overhead evidence or mark attribution inconclusive. Use
   [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) to
   classify throughput against the pipe and bandwidth ceilings, stalls,
   occupancy, and tails. Convert each kernel's gap to its ceiling into an
   end-to-end budget only with compatible full-path attribution. See the capture
   guidance in [GPU MODE resources](references/sources/gpu-mode.md).
2. **Swap before writing.** Check whether the framework, engine, or a kernel
   library already ships a faster implementation behind a backend flag,
   dispatcher, or stale guard. A swap is a flag plus a parity gate.
3. **Pick the language from the evidence.** Follow
   [references/recipes/kernel-backends.md](references/recipes/kernel-backends.md): Triton for fusion of
   memory-bound chains, TileLang for tile-level GEMM, attention, and recurrence
   variants, CuTe DSL when the ceiling needs architecture-specific tensor-core,
   memory, and pipelining features. Search
   [KernelWiki](https://github.com/mit-han-lab/KernelWiki) for techniques and
   prior implementations; it is evidence of technique, not of benefit here.
4. **Gate like any other arm.** Production shapes and dtypes, a numerical
   parity test against the control op, resource use, then the full-path paired
   gate. A faster microbenchmark is not a speedup.

When measured kernel headroom justifies implementation and validation within the
remaining authorization and budget, implement it without asking again.

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

Read [references/workflow/decision-cases.md](references/workflow/decision-cases.md) for generalized
examples of correct promotion and rejection decisions.
