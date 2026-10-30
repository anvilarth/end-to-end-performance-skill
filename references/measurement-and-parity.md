# Measurement and parity

Use this reference when defining a baseline, benchmark protocol, or correctness
gate. Adapt the metrics to the workload, but do not weaken the separation between
correctness and performance.

## Choose the end-to-end surface

Measure the surface the user actually pays for:

- training steady state: completed optimizer steps per unit time;
- training startup: process entry to first completed step;
- checkpointing: exposed training stall and verified resumability;
- offline inference: completed samples per unit time at a fixed batch and shape;
- online inference: request latency, throughput, and tail behavior at a fixed
  arrival pattern and concurrency;
- memory-bound operation: the largest safe real workload, not only allocated
  bytes in a synthetic call.

Do not merge these surfaces into one number. An optimization may improve warm
throughput while regressing cold start or tail latency.

## Freeze the baseline manifest

Record enough state to reproduce both arms:

- source revision and activation flags;
- model and checkpoint identity;
- input identities, preprocessing, order, shapes, and seeds;
- dtype and numerical settings;
- hardware model, GPU count, node count, interconnect, and distributed topology;
- framework, CUDA, compiler, and kernel-library versions;
- warmup, compilation, cache, and synchronization policy;
- benchmark repetitions and aggregation method.

A changed baseline invalidates attribution. Re-run the control instead of
transcribing an old number into a new comparison.

## Build a full stage map

Account for time outside the obvious model call:

- queueing and request preparation;
- data loading, decoding, augmentation, collation, and packing;
- model and checkpoint loading;
- compilation and cache population;
- host-to-device and device-to-host transfers;
- forward, backward, optimizer, and collectives;
- recomputation and activation checkpointing;
- synchronization and allocator behavior;
- postprocessing, serialization, storage, and upload.

Synchronize stage timers only when required for attribution. Avoid adding global
synchronization to the production path merely to make every local timer exact.
Use a timeline profiler for asynchronous boundaries when wall-clock wrappers are
misleading.

A wall-clock timer around an asynchronous launch measures the launch, not the
work, and a timer around the first host read of a device value measures all the
work queued before it. Time GPU stages with `torch.cuda.Event` pairs, and count
the host-device synchronizations per unit of work
(`torch.cuda.set_sync_debug_mode("warn")` in a diagnostic run) as part of the
baseline: an unintended `.item()`, `.cpu()`, or pageable copy inside the hot
loop is itself a stage budget (playbook common.md section 9).

## Estimate headroom

For a stage occupying fraction `p` of end-to-end time and a proposed local
speedup `s`, estimate the ideal total speedup with Amdahl's law:

`total_speedup = 1 / ((1 - p) + p / s)`

Also calculate the impossible best case where the stage becomes free:

`maximum_total_speedup = 1 / (1 - p)`

Stop before implementation when the maximum benefit is below the promotion
threshold or cannot be resolved by benchmark noise.

## Parity hierarchy

Apply the strongest relevant gate.

### Exact behavior-preserving changes

Check as applicable:

- identical sample IDs, order, packing, and augmentation metadata;
- identical RNG state transitions;
- bitwise tensor, state, checkpoint, or output equality;
- identical mode dispatch and operation count.

### Numerically different but intended-lossless changes

Measure deviations rather than asserting equivalence:

- tensor maximum and mean absolute error;
- relative L2 or normalized RMSE;
- loss difference against a repeated-control noise envelope;
- gradient finiteness, norms, direction, and selected update deltas;
- per-step and accumulated learning-curve divergence;
- task-specific output or quality metrics.

Define tolerances before seeing candidate results. A short smoke can reject an
arm but should not promote a numerically different training path whose error may
accumulate.

### Approximate changes

Require explicit authorization for the quality trade-off. Report:

- absolute candidate and baseline quality metrics;
- paired quality deltas and the fraction of examples improved;
- performance, memory, and cost deltas;
- tails and unacceptable outliers;
- the promotion threshold and Pareto position.

Never hide a missed absolute quality gate behind a positive paired delta.

## Performance comparison

- Use the same worker or allocation for small effects.
- Interleave control and candidate when state permits: A/B/A or ABBA.
- Separate cold and warm runs.
- Exclude warmup only by a predeclared rule.
- Use enough repetitions to resolve the target effect.
- Report absolute control and candidate values plus paired deltas.
- Report median and relevant tails; include variability or a confidence interval.
- Track failures, retries, restarts, and out-of-memory events.
- Verify that instrumentation and the candidate path were active.

Do not promote from one fast replica, cross-worker averages, unsynchronized stage
timers, or a microbenchmark alone.

## Scale and topology transfer

Treat every material execution setup as a distinct validation domain. Revalidate
when changing:

- one node to multiple nodes;
- GPU count or mesh dimensions;
- DDP, FSDP, HSDP, tensor, context, or pipeline parallelism;
- microbatch size or gradient accumulation;
- collective algorithms, interconnect, or communication libraries;
- activation checkpointing or sharding boundaries.

Small-scale experiments are useful for mechanism screening. They do not establish
the final speedup because compute-to-communication ratio, exposed collectives,
memory pressure, rank skew, and overlap change with scale. Report a small-scale win
as `promising at the screened setup`, not as a production result.

## Training-specific gates

At minimum, check:

- identical input and optimizer-step semantics;
- finite loss and gradients on every measured step;
- loss deltas against repeated controls;
- step time after warmup and at representative shapes;
- communication and straggler distributions;
- allocated and reserved memory peaks;
- checkpoint save and resume correctness when state handling changes.

For distributed work, measure the complete target topology. A smaller run cannot
prove communication overlap, sharding behavior, or multi-rank memory safety.

## Inference-specific gates

At minimum, check:

- matched inputs, seeds, generation settings, and model state;
- output equality or declared numerical and quality tolerances;
- end-to-end latency and throughput;
- p50 plus relevant p95 or p99 tails;
- batching and concurrency behavior;
- warmup, compilation, and cache state;
- peak memory and failure rate;
- important modes rather than only the simplest request class.

## Failure classification

Use precise verdict boundaries:

- `reject`: a valid candidate fails correctness, quality, performance, memory,
  or another predeclared gate;
- `inconclusive`: the harness, environment, activation, worker health, or sample
  size prevents a valid comparison;
- `promote`: the candidate passes the complete contract under representative
  conditions.
