# 2. Attribution and experiment selection

**Open when:** a valid baseline exists but the next change lacks a measured
mechanism or savings estimate. **Exit when:** one affordable experiment has
enough evidence to run, a correctness gate and a stop rule. Use the relevant
[implementation chapter](README.md#choose-the-next-chapter), then
[validate](07-validation-and-delivery.md).

## In this chapter

- [1. Start with a cheap time and work map](#1-start-with-a-cheap-time-and-work-map)
- [2. Escalate only to the tool that can answer the question](#2-escalate-only-to-the-tool-that-can-answer-the-question)
- [3. Verify capture before trusting it](#3-verify-capture-before-trusting-it)
- [4. Attribute exposed critical-path time](#4-attribute-exposed-critical-path-time)
- [5. Convert the observation into a bounded prediction](#5-convert-the-observation-into-a-bounded-prediction)
- [6. Rank candidates by useful return, not technical appeal](#6-rank-candidates-by-useful-return-not-technical-appeal)
- [7. Write a falsifiable experiment](#7-write-a-falsifiable-experiment)
- [8. Close the experiment and re-map only what changed](#8-close-the-experiment-and-re-map-only-what-changed)

## 1. Start with a cheap time and work map

Use existing metrics first. Split the real boundary into startup, input read and
decode, preparation, host-to-device transfer, forward/backward or generation,
communication, postprocessing and output. Count work as well as time: files,
frames, bytes, actual tokens, padded tokens, launches, synchronizations and
requests. An expensive stage may be doing unnecessary work correctly.

Obtain device model, process placement, memory use and utilization without
changing shared machine settings. `nvidia-smi` is useful for residency, competing
processes and broad utilization clues; its sampled utilization is not a breakdown
of compute versus bandwidth versus stalls. Check CPU utilization, run queue,
I/O wait and disk/network rates when the GPU is starved.

Do not add overlapping stage durations into a total. A host decoder, a copy
engine and GPU compute can be active simultaneously. For a finite workload,
account for startup, pipeline fill, steady-state service rates and final drain.

## 2. Escalate only to the tool that can answer the question

| Missing evidence | First useful tool | What to inspect | What it does not prove |
| --- | --- | --- | --- |
| Which Python path occupies the CPU? | `py-spy`; `cProfile` for a controlled run | Hot stacks, serialization, tokenization, polling | Exact native/GPU execution time from Python stacks alone |
| Is input blocked on disk, network or CPU? | `pidstat`, `iostat`, application queue metrics; bounded `strace` if needed | Utilization, waits, request sizes, open/read patterns | That increasing workers will improve completed throughput |
| Which operators/shapes allocate or consume device time? | PyTorch profiler | CPU/CUDA activities, shapes, memory, operator call counts | Unperturbed final latency with all expensive options enabled |
| Are launches, copies, streams, ranks or synchronization exposed? | Nsight Systems with NVTX | Timeline dependencies and actual worker activity | Kernel microarchitectural cause from duration alone |
| Why is a material kernel slow? | Nsight Compute on one representative kernel | Memory traffic, instructions, occupancy limits, stalls, roofline evidence | Normal application latency under replay |
| What owns memory or grows across requests? | PyTorch allocator statistics/snapshot plus process-level device memory | Live allocations, peaks, retained pools, external allocations gap | Every NCCL/library allocation from the PyTorch snapshot alone |
| Is a collective limited by topology or payload size? | Per-rank timeline, `nccl-tests`, P2P diagnostics | Placement, message-size curve, stragglers | Application scaling without the real computation and overlap |

Enable profiler shapes, stacks and memory only when needed; each can add overhead.
Keep the unprofiled control. Pick a capture window that includes the phenomenon,
and write down its fraction of the workload. A decode-only window cannot supply
a full-job prefill/decode split.

## 3. Verify capture before trusting it

Add stable NVTX ranges around logical stages in the actual worker. A parent
process range does not prove that a spawned model worker was captured. First use
a tiny real-input capture and confirm the expected process, device, stage,
kernel names, invocation count and nonempty activity.

Example templates, after checking the installed tool's `--help`:

```bash
# bench.py and its arguments stand for the actual workload entrypoint.
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output="$PERF_ARTIFACT_DIR/timeline" python bench.py --manifest "$MANIFEST"

# Inspect the generated report before collecting a longer trace.
nsys stats "$PERF_ARTIFACT_DIR/timeline.nsys-rep"
```

Use supported child-process/launch options for the application. If capture fails,
classify it: attachment, permissions, version support, output path, range filtering
or process lifecycle. Spend a bounded repair budget. A successful exit with zero
relevant kernels is inconclusive. Do not run PyTorch profiler inside an Nsight
Compute capture; isolate the instrumentation question instead.

## 4. Attribute exposed critical-path time

Follow dependencies backward from completion: what prevented useful work from
finishing earlier? Record the union of relevant intervals and the blocking edges.
A CPU operation overlapping all GPU work is not immediately recoverable latency.
A tiny `.item()` can matter if it forces the host to wait before issuing the next
batch. A long collective can be hidden until its final tail.

For a pipeline, measure queue depth and per-stage service rates. A permanently
empty GPU-input queue suggests upstream starvation; an ever-growing input queue
with a busy GPU suggests a downstream capacity limit. Confirm with a controlled
probe rather than interpreting one snapshot as causality.

Useful probes include predecoded inputs, a no-op writer, resident weights,
fixed-shape batches or disabled logging. These estimate a ceiling or test a cause.
They are diagnostic substitutions unless the product contract permits them.
Keep the final benchmark's required decode, output and state handling intact.

## 5. Convert the observation into a bounded prediction

For a serial exposed fraction `f` sped up by `s`, the idealized overall speedup is:

```text
S = 1 / ((1 - f) + f / s)
saved time = T * f * (1 - 1/s)
```

Use this only for time that is actually exposed and for unchanged surrounding
work. Include wrapper overhead, layout conversions, new synchronization and
startup. For overlapped pipelines use the measured critical path or stage-rate
model; the serial formula is an upper-bound aid, not a substitute for that model.

Example: a 100-second job exposes 20 seconds in an operator family. A 2x operator
speedup saves at most 10 seconds; a required 3-second conversion reduces that to
7 seconds. A paper's 10x kernel result cannot remove the remaining 80 seconds.
If the observed 20% came only from 32 decode steps, first establish how those
steps relate to the full job before making this calculation.

For compilation or preprocessing caches, calculate amortization:

```text
break-even calls = extra one-time cost / per-call saving
```

An extra 60 seconds of compilation for 2 ms saved per call needs 30,000 calls
just to break even, ignoring eviction and new shape compilations. Compare with
the real process lifetime and cold-start requirements.

## 6. Rank candidates by useful return, not technical appeal

For each plausible lever, estimate low/central/high saved end-to-end seconds,
confidence in the mechanism, implementation time, validation cost and reuse
across expected runs. Prefer an equivalent implementation before numerical or
algorithmic changes. Current user constraints take precedence over a source's
default precision or a previous task's restrictions.

A practical ranking aid is expected saved time times confidence divided by
setup, implementation and validation effort. Keep units and assumptions visible;
do not compare arbitrary scores as if they were measured probabilities. A cheap
probe that sharply reduces uncertainty may rank before the eventual implementation.

Do not add savings estimates for overlapping levers. Bucketing and dynamic-shape
compilation may attack the same padding/recompile cost. Fusion and a backend
swap may eliminate the same launches. Promote one, then recompute the budget.

## 7. Write a falsifiable experiment

```text
Observation: synchronization stalls the host before each new batch.
Change: buffer the existing metric and transfer it at the required report interval.
Prediction: remove most of the measured exposed logging wait; no output or update change.
Falsifier: real-input screen still shows the same waits, or complete-step time is flat.
Controls: same batch order, weights, precision, worker count and cache policy.
Gate: identical required metrics at reporting boundaries; full output/update checks.
Budget: bounded implementation + screen, with final A/B time reserved.
Stop: reject if the logging wait is hidden, required semantics change, or savings are below threshold.
```

Change one causal factor first. A factorial follow-up can study interactions
later if the budget warrants it. Verify the selected backend/configuration in
the worker through logs, import paths or profiler evidence; a flag being accepted
is not proof that the candidate ran.

## 8. Close the experiment and re-map only what changed

If the mechanism is absent, record rejection without a long performance run.
If correctness fails, preserve the failing case and repair within budget or reject.
If the screen succeeds, run the reserved promotion gate. If a trace is empty,
the reference is missing or measurement is too noisy, say inconclusive.

After promotion, refresh cheap timings. Escalate profiling again only when the
remaining bottleneck is unknown or the previous model no longer fits. Stop when
the target is reached, the best affordable bound is immaterial, or the agreed
budget is exhausted. Sources: [profiling tools](sources.md#profiling-and-hardware)
and [experiment evidence](07-validation-and-delivery.md).
