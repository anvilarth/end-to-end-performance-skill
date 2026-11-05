# 1. Contract and baseline

[Home](../../README.md) / [Library](../README.md) / [Handbook](README.md) / 01

**Open when:** the workload, metric, reference or measurement boundary is unclear,
or an old baseline may not describe the current job. **Exit when:** another session
can execute the control, produce complete outputs, compare them and resolve the
required performance difference. Next: [attribution](02-attribution-and-experiments.md).

## In this chapter

- [1. Recover the executable truth](#1-recover-the-executable-truth)
- [2. Specify useful work and the clock](#2-specify-useful-work-and-the-clock)
- [3. Freeze a representative workload](#3-freeze-a-representative-workload)
- [4. Make the comparator runnable before optimizing](#4-make-the-comparator-runnable-before-optimizing)
- [5. Separate cold, warm and stateful conditions](#5-separate-cold-warm-and-stateful-conditions)
- [6. Choose clocks that answer the question](#6-choose-clocks-that-answer-the-question)
- [7. Run A/A, then choose the A/B budget](#7-run-aa-then-choose-the-ab-budget)
- [Baseline record and failure routes](#baseline-record-and-failure-routes)

## 1. Recover the executable truth

Read the current launcher, configuration, checkpoint identity, recent successful
result and existing journal before asking the user to reconstruct them. Follow
the launcher into the actual worker: a shell option may be ignored, overridden
or consumed only by the parent. Identify environment variables, imported package
locations, subprocess boundaries and the path that produces final outputs.

Record the control commit plus any working-tree patch, dependency versions,
model/config hashes, relevant hardware and topology, and input identity. A path
alone is insufficient when its contents can change. Preserve the control so the
candidate cannot silently become its own reference.

Ask only for a missing choice that affects the result: for example whether the
target is offline throughput or interactive latency, or whether a precision
change is allowed. A local artifact that already answers the question is enough.
If the actual environment is unavailable, explicitly define the proxy and its
unverified differences; continue useful diagnostics without promoting the proxy.

## 2. Specify useful work and the clock

Write the primary metric as a formula with units. Examples:

| Workload | Useful numerator | Clock boundary | Necessary companion checks |
| --- | --- | --- | --- |
| Offline captioning | Valid completed clips from the frozen manifest | Required input preparation through final output serialization, including queue drain | Missing/duplicate IDs, quality, generated length, failures |
| Online generation | Successful requests satisfying the agreed latency/quality conditions | Fixed observation window under a declared arrival process | Offered versus completed rate, queue growth, TTFT, inter-token and end-to-end tails |
| Training | Valid examples or unmasked training tokens | Complete optimizer updates, including accumulation and required synchronization | Loss normalization, effective batch, learning trajectory, peak memory |
| Startup | One ready-to-serve or ready-to-train instance | Process launch through readiness, with explicit loading/compile/cache policy | First real request/update succeeds |
| Kernel diagnosis | One operation with declared shape/layout/dtype | Device execution under a declared measurement method | Correctness, workspace and wrapper cost; application result remains separate |

State whether preprocessing is included. Starting from cached tensors may be a
valid product contract; it cannot support a claim about media-to-result latency
when decoding was required. For finite jobs include the initial fill and final
drain. For services separate steady-state capacity from process startup.

For tokens, distinguish prompt, generated and total tokens. Use actual IDs or
documented engine counts when required; retokenizing text can produce different
counts. Preserve stop conditions and maximum lengths. Faster completion caused
by truncation or an early error is not more useful work.

## 3. Freeze a representative workload

Use immutable IDs and hashes where feasible. Include the modes that affect the
execution path: short and long sequences, resolution/aspect-ratio families,
empty or minimal inputs when supported, ragged batches, recurring prefixes,
audio/video duration, and required dtypes/layouts. Retain difficult cases rather
than dropping them after a candidate fails.

Keep three scopes distinct:

- **Canary:** a few real inputs that expose broken loading, output writing or
  comparison quickly.
- **Screen:** a small stratified workload that can falsify an optimization at
  low cost. It must preserve the mechanism under investigation.
- **Promotion:** the full agreed workload and acceptance gate. Do not relabel
  a favorable screen as this gate after seeing its result.

Select the screen before tuning. Record what it underrepresents. A batch of long
prompts may isolate prefill but does not establish mixed-traffic throughput.
Repeated inputs may be appropriate for cache studies but distort a no-reuse
contract. Keep a held-out shape/load slice for heavily tuned candidates when
generalization beyond the tuning set matters.

## 4. Make the comparator runnable before optimizing

Run the control and writer on the canary. Check expected IDs, exactly one required
result per input, valid schema, stop/failure status and complete artifact files.
Then compare against an independent reference: a frozen implementation, accepted
outputs, trusted higher-precision computation where applicable, or the agreed
quality evaluation. Missing reference coverage is a missing gate, not zero error.

Define the applicable equivalence level:

- Exact IDs, masks, index mappings, bookkeeping and serialization constraints.
- Numerical tolerances by tensor/field and dtype, including non-finite handling.
- Gradient/update checks for changes touching training.
- Task-level quality for outputs where tensor closeness is insufficient.

Keep sample denominators and failed cases in reports. Schema validity does not
measure whether a caption is correct. Agreement with a reference is not human
semantic quality unless that reference and criterion were accepted for the task.
Do not loosen tolerances after inspecting a failure to manufacture acceptance.

## 5. Separate cold, warm and stateful conditions

List state that can influence performance: OS file cache, decoded inputs,
compiled artifacts, allocator pools, CUDA graphs, model residency, prefix/KV
caches, recurrent state and connection pools. Decide which is reused in the
real workflow and make both arms match that condition.

For a warm experiment, exercise the shape families and lazy initialization
actually expected, then start the measured interval. Record warmup work and
verify that timing stabilized; a fixed count alone is not evidence. For a cold
experiment, document the reset mechanism and what remains warm. A fresh Python
process does not necessarily clear disk compilation caches or the OS page cache.
Do not flush shared host caches or change shared clocks merely to imitate cold
conditions; use an isolated setup or disclose the limitation.

Restore matched inputs and state after warmup and autotuning if the function
mutates them. Separate model-state restoration from cache policy. Training arms
must start from equivalent weights, optimizer, scaler and RNG state.

## 6. Choose clocks that answer the question

For end-to-end latency, use a monotonic host clock around the required work and
wait for its actual completion before stopping. GPU launches are asynchronous:
timing only the Python call often measures enqueue time. CUDA events measure a
device interval with the dependencies you placed between events; they do not
automatically include preprocessing, other streams, remote ranks or writing.

Avoid synchronizing after every operation in a throughput benchmark. That can
destroy the pipeline being measured. Instrument stage ranges and use a separate
diagnostic trace when fine attribution would perturb the real path. For several
devices, define completion across all required work and use the slowest required
rank or actual job completion, not the average of rank-local times.

Report per-run absolute duration, useful count, rate and relevant memory peaks.
Do not average request rates blindly: total useful work divided by total elapsed
time has different weighting from the mean of per-request reciprocals.

## 7. Run A/A, then choose the A/B budget

Repeat the control under the intended reset policy. Check output stability and
variation across runs, input strata and host/GPU conditions. If expected savings
are 2% but control timings vary by 8%, first identify interference, compilation,
unbalanced samples or measurement overhead. Additional repetitions may help,
but they do not fix a changing workload.

Choose repetitions or a sequential stopping rule before the final comparison.
Pair arms on the same input blocks and balance order when drift matters. The
independent unit may be a complete run or traffic interval; thousands of tokens
from one request are not thousands of independent benchmark replicates.
Report uncertainty that matches those units and acknowledge small samples.

Example: a 100-second control with a 1-second practical threshold needs enough
stability to resolve about 1%, not merely a candidate that once finishes at 99.
If the final budget cannot resolve that difference, return inconclusive or
target a larger lever. Do not spend the delivery budget on endless A/A repair.

## Baseline record and failure routes

Adapt these fields to the established journal rather than creating parallel files:

```text
entrypoint / control revision / environment / hardware
manifest identity / expected IDs / workload strata
primary metric formula / clock start and completion / target
cache and warmup policy / state restoration
reference identity / tolerances / quality gate
per-run duration, useful count, failures, peak memory
variability / measurement budget / final stopping rule
artifact locations / unresolved limitations
```

If outputs are incomplete, fix the harness before timing candidates. If only the
quality reference is missing, label subsequent timing as diagnostic. If the
baseline is reproducible and a dominant cost is already obvious, proceed directly
to its cheapest causal experiment. Otherwise use the next chapter's attribution
ladder. Sources: [measurement and numerics](../sources/README.md#measurement-and-numerics).

---

[← Overview](README.md) · [Handbook](README.md) · [Attribution & experiments →](02-attribution-and-experiments.md)
