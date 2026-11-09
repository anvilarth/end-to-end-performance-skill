# 3. Input, shapes, memory and compilation

[Home](../../README.md) / [Library](../README.md) / [Handbook](README.md) / 03

**Open when:** the timeline exposes input stalls, padding, copies, allocations,
Python/launch overhead or repeated compilation. **Exit when:** one intervention
improves the real path under unchanged input/output semantics. First use
[attribution](02-attribution-and-experiments.md) if the symptom is only low GPU
utilization. Sources: [data and compilation](../sources/README.md#data-memory-and-compilation).

## In this chapter

- [1. Diagnose input starvation before increasing workers](#1-diagnose-input-starvation-before-increasing-workers)
- [2. Preserve media semantics while avoiding redundant work](#2-preserve-media-semantics-while-avoiding-redundant-work)
- [3. Make transfer overlap real](#3-make-transfer-overlap-real)
- [4. Reduce padded and repeated work](#4-reduce-padded-and-repeated-work)
- [5. Attribute memory by owner before changing allocators](#5-attribute-memory-by-owner-before-changing-allocators)
- [6. Compile a stable, meaningful region](#6-compile-a-stable-meaningful-region)
- [7. Use CUDA graphs when launches are exposed](#7-use-cuda-graphs-when-launches-are-exposed)
- [Stop and hand off](#stop-and-hand-off)

## 1. Diagnose input starvation before increasing workers

Measure read, decode, transform, collate and transfer time separately, together
with queue depth and consumer wait. Inspect useful bytes/frames per second,
CPU usage, storage latency and native thread counts. An empty consumer queue
does not distinguish slow storage from CPU oversubscription or a serial collator.

Use a predecoded real-input screen to estimate the maximum gain from input work.
If removing input work barely changes end-to-end time, deprioritize it. If it
does, test the relevant stage: bounded prefetch for latency, fewer redundant
reads, batched requests, local reusable cache, in-process decode, or parallel
transforms. Measure each under the real storage path and dataset distribution.

Increase workers coarsely until delivered throughput plateaus, latency/memory
violates the contract, or shared resource limits are reached; then refine around
the knee. Count threads inside each decoder/library as well as Python workers.
Many workers each spawning a full native thread pool can reduce throughput.
Record the limit that stopped scaling, not just the selected worker count.

Persistent workers can amortize startup, but check epoch boundaries, RNG seeding,
file handles and stale state. Sharded iterable datasets need disjoint worker/rank
ownership; apparent speed from duplicated examples is invalid. Preserve sample
order where required and exact coverage where ordering is flexible.

## 2. Preserve media semantics while avoiding redundant work

For video, group reads from the same source when allowed, avoid reopening and
seeking once per frame, and reuse decoded segments when their identity is exact.
For audio, avoid repeated decode/resample cycles. Cache the output of expensive
deterministic transforms only when input, transform configuration and version
are part of the key and invalidation is defined.

Verify timestamp-to-frame selection, variable frame rate, colorspace/range,
orientation, resize/interpolation, channel order, audio sample rate, channel
mixing and clipping behavior. Compare the actual tensors reaching the model,
then the task output. Two decoders producing the same shape is insufficient.
Do not silently lower resolution, frame count or audio quality as an input fix.

Include cache construction and storage cost when the real job pays them. Measure
cold and reusable-cache modes separately. A local cache may solve network I/O
but consume unavailable disk space or serialize population across workers.

## 3. Make transfer overlap real

First reduce transferred data and unnecessary round trips. Find `.cpu()`,
`.numpy()`, `.item()`, scalar-dependent Python branches and implicit copies in
the hot path. Defer logging or aggregate metrics on device when semantics allow.
Avoid making an entire batch contiguous several times in different wrappers.

Pinned host memory and `non_blocking=True` are candidate mechanisms, not proof
of overlap. Check whether data is actually pinned, whether the host call blocks,
whether copies use a suitable stream, and whether the consumer immediately waits.
Pinning synchronously in the consumer loop can cost more than it saves.

For double buffering, prepare batch `n+1` while computing `n`; give the copy and
compute streams correct event dependencies. Do not mutate or reuse the host
source until its transfer is safe. Preserve device allocation lifetime across
streams using the framework's supported stream/lifetime mechanisms. CPU reads
of asynchronously copied device outputs also require completion before access.

Confirm overlap in a timeline and in end-to-end timing. If CPU preprocessing
cannot produce batch `n+1` until a GPU result from `n` returns, adding a stream
does not remove the dependency. If copies already hide behind compute, faster
copies may have no current latency benefit.

## 4. Reduce padded and repeated work

Compute useful tokens/pixels divided by executed tokens/pixels for each shape
family. For attention, account for the relevant sequence-length dependence
rather than assuming cost is linear in padded tokens. Identify outliers that
force the entire batch to a large shape.

Try a small set of length/resolution buckets with bounded waiting. Preserve
input IDs and restore output order where required. Compare total work, queue
delay, batch occupancy and compiler variants: very fine buckets reduce padding
but may fragment batches and create too many compiled graphs.

Packing several sequences requires correct attention boundaries, position IDs,
label shifts, loss masks and normalization. Cross-example attention or a changed
denominator changes the task. For training, altered sample order and RNG calls
can affect dropout and trajectories; apply the agreed equivalence gate rather
than assuming bitwise identity. For serving, waiting for a full bucket may violate
latency even when device throughput improves.

Example: replacing batches padded to 4096 with 1024/2048/4096 buckets may reduce
work, but the candidate succeeds only if the real length distribution fills
those buckets and useful completed throughput improves. A benchmark with one
perfectly full bucket omits the scheduling cost.

## 5. Attribute memory by owner before changing allocators

Measure peaks at load, first forward, backward, optimizer step and steady state.
Separate parameters, gradients, optimizer state, activations, KV/recurrent state,
temporary workspaces, graph pools and caches. Use allocated versus reserved
memory and allocation history to distinguish live tensors from retained blocks.
Compare with process-level device memory; external libraries may allocate memory
outside the framework's snapshot.

If memory grows across iterations, inspect retained outputs, autograd graphs,
callbacks and caches before blaming fragmentation. If the peak is temporary,
look for duplicate casts/copies or overlapping lifetimes. `empty_cache()` does
not release live tensors and is not a default hot-loop optimization.

Prefer removing duplicate materialization or shortening lifetimes first. Then
consider a memory-efficient equivalent operator, selective recomputation,
sharding or offload according to the dominant owner. Include extra compute,
transfers and synchronization in timing. Lower memory is useful when it improves
capacity, stability or throughput under the contract; it is not automatically
a speedup. See [training](05-training-and-distributed.md) for activation/state trade-offs.

## 6. Compile a stable, meaningful region

Start from a valid eager baseline. Confirm whether the proposed path is already
compiled, fused or graph-replayed. Use compiler logs on a real-input screen:

```bash
TORCH_LOGS="graph_breaks,recompiles" \
  python bench.py --manifest "$MANIFEST"
```

Here `bench.py` represents the existing harness. Match log names and compiler
options to the installed PyTorch documentation. Inspect causes before toggling
modes: data-dependent Python, unsupported operations, changing shapes/strides,
Python scalar guards or mutable state. Move nonessential logging out of compiled
regions when safe. Do not conceal required computations to make the graph whole.

Choose among stable-shape specialization, a small bucket set, dynamic shapes
and regional compilation based on the actual shape distribution and process
lifetime. Dynamic support may avoid recompiles but can give up specialization;
many specializations can spend the whole job compiling. Regional compilation
can reduce cold cost for repeated blocks. Measure compile time, first execution,
warm execution and fallback frequency separately.

Use full-graph requirements as a diagnostic when appropriate, not as a universal
product requirement. A small graph break outside the hot path may be harmless.
After any compiler change verify output, gradients where applicable, memory and
the actual selected execution path. Record the compiler/runtime version.

## 7. Use CUDA graphs when launches are exposed

Graph replay is promising for repeated short device work with stable execution
and supported capture behavior. Before capture, settle initialization and
workspaces. Define static buffers, how new inputs enter them, required stream
dependencies, output lifetime and how variable shapes dispatch or fall back.

Account for input copies into graph-owned storage, graph memory pools, capture
cost and output copies needed before the next replay overwrites buffers. Verify
that outputs saved for comparison remain independent across calls. Check RNG,
training state and capturable-optimizer restrictions in the installed framework.

Do not introduce graph replay when CPU data-dependent control makes the required
path incompatible, or when the current engine already replays the same region.
Validate first-use, repeated-use and shape-transition behavior. If the timeline
still has the same gaps, locate the remaining host dependency instead of assuming
a larger graph will fix it.

## Stop and hand off

Close the branch when the screen removes the predicted cost and the
[final gate](07-validation-and-delivery.md) confirms useful savings, or reject
when the cost is hidden, semantics differ, or overhead consumes the benefit.
If the remaining dominant work is device math, go to
[kernel engineering](06-kernel-engineering.md). If waiting is caused by queueing
or inter-rank coordination, use [serving](04-serving-and-concurrency.md) or
[distributed execution](05-training-and-distributed.md).

---

[← Attribution & experiments](02-attribution-and-experiments.md) · [Handbook](README.md) · [Serving →](04-serving-and-concurrency.md)

For source-derived probes and version/architecture boundaries, use the
[extended recipe](../recipes/compilation-and-diffusion.md).
