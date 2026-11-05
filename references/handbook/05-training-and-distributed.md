# 5. Training and distributed execution

[Home](../../README.md) / [Library](../README.md) / [Handbook](README.md) / 05

**Open when:** optimizing complete training updates, fitting a larger useful
batch, or scaling across authorized GPUs/nodes. **Exit when:** the real training
step improves under the agreed numerical and learning-quality contract. Sources:
[training and communication](sources.md#training-and-communication).

## In this chapter

- [1. Define a complete update](#1-define-a-complete-update)
- [2. Establish numerical and state comparators](#2-establish-numerical-and-state-comparators)
- [3. Tune microbatch and accumulation without changing the objective](#3-tune-microbatch-and-accumulation-without-changing-the-objective)
- [4. Remove host stalls and redundant tensor traffic](#4-remove-host-stalls-and-redundant-tensor-traffic)
- [5. Optimize the actual memory owner](#5-optimize-the-actual-memory-owner)
- [6. Test optimizer implementations as full state transitions](#6-test-optimizer-implementations-as-full-state-transitions)
- [7. Separate DDP synchronization from rank imbalance](#7-separate-ddp-synchronization-from-rank-imbalance)
- [8. Treat FSDP and other parallel modes as different execution plans](#8-treat-fsdp-and-other-parallel-modes-as-different-execution-plans)
- [9. Measure scaling at the right boundary](#9-measure-scaling-at-the-right-boundary)

## 1. Define a complete update

Trace data wait, transfer, forward, loss, backward, gradient accumulation,
synchronization, clipping, optimizer, scaler/scheduler and required logging.
Include the first optimizer step when measuring memory: lazy state allocation
may make forward/backward alone appear to fit. Record checkpointing overhead
at its real frequency when it contributes to the objective.

Report useful examples or unmasked training tokens per second and update time.
Specify microbatch, accumulation, data-parallel size and effective batch. For
variable-length training, record the actual loss denominator; multiplying batch
counts is insufficient to describe token-weighted updates.

Distinguish throughput improvement at a fixed learning recipe from a change to
the recipe itself. A larger global batch, changed sample ordering, lower precision
or altered optimizer requires the appropriate acceptance gate. If the objective
is time to a target validation quality, faster updates alone are incomplete evidence.

## 2. Establish numerical and state comparators

Use the same initial model, optimizer, scaler and scheduler state, inputs and
relevant RNG state for paired step checks. Compare loss components, gradients,
gradient norm/clipping behavior, parameter updates and optimizer-state evolution
as appropriate. Checking only forward outputs can miss a broken backward.

Inspect missing versus zero gradients: they can cause different optimizer
behavior. Check tied parameters, frozen parameters and parameter groups. For
mixed precision preserve scale/unscale, overflow handling, clipping order and
step-skipping semantics. Include checkpoint save/reload and one resumed update
when the change touches state layout, sharding or optimizer representation.

Use short trajectories or the agreed validation gate after local checks pass.
Do not promote from one matching loss if a changed reduction order, accumulation
scheme or numeric format can drift over many updates. Conversely, do not impose
bitwise equality when the established contract allows bounded numerical changes.

## 3. Tune microbatch and accumulation without changing the objective

If kernels are underfilled, increase microbatch within memory and task limits.
Adjust accumulation to preserve the effective batch when that is required.
Measure complete-update throughput, not microstep latency alone. Test real
longest examples and difficult shape combinations; an average batch is not an
OOM stress test. Preserve memory headroom for required workspaces and checkpointing.

Recheck loss scaling. Averaging each microbatch equally is not equivalent to a
global token mean when valid token counts differ. Define the intended numerator
and denominator, then account for both local accumulation and distributed
gradient reduction. Include the last partial accumulation group and uneven
final batch in tests. Do not copy a division-by-accumulation recipe without
checking the framework's existing reduction.

Larger microbatches may reduce launch/communication overhead but increase padding
or memory pressure. Smaller microbatches with more accumulation may be necessary
for capacity. Select from measured useful throughput and equivalent updates.

## 4. Remove host stalls and redundant tensor traffic

Count host-device synchronization per microstep. Repeated `loss.item()`, progress
bar updates, CPU gradient inspection and per-parameter Python loops can expose
large gaps. Aggregate detached metrics and transfer them at the required reporting
boundary. Keep essential failure detection and reporting semantics.

Inspect duplicate casts, copies, materialized attention matrices and retained
activations. Prefer supported memory-efficient attention and equivalent fused
operators before custom kernels. Compiling a stable training region may reduce
launch overhead; verify backward and graph-break behavior separately. Use
[input/shapes/compilation](03-input-shapes-and-compilation.md) for that branch.

## 5. Optimize the actual memory owner

If activations dominate, test selective recomputation of expensive-to-store,
cheap-to-recompute regions. Estimate bytes saved and added compute on the full
update. Check RNG preservation and side effects: recomputing a stateful operation
must not update persistent buffers twice or change stochastic semantics outside
the contract. Measure the peak after the final policy, not the sum of tensor sizes.

If optimizer/gradient/parameter state dominates, compare supported sharding or
offload choices. Include communication, CPU cost and transfer overlap. Activation
checkpointing does not solve an optimizer-state peak by itself. Offload that fits
the model may still reduce throughput; describe capacity and speed separately.

If temporary workspaces or duplicate buffers dominate, shorten lifetimes or use
an equivalent lower-workspace path first. Do not enable allocator knobs before
establishing fragmentation as a material cause. See the owner table in the
[memory procedure](03-input-shapes-and-compilation.md#5-attribute-memory-by-owner-before-changing-allocators).

## 6. Test optimizer implementations as full state transitions

Inspect the existing optimizer first; a starter script may already request a
fused implementation. Compare supported scalar-loop, foreach and fused paths
only where the installed version/device/dtype supports them. Include temporary
memory, initialization and actual selected backend, not just the constructor flag.

Keep parameter groups, epsilon, weight decay semantics, step counters, learning
rate schedules and capturability requirements identical. Verify handling of
None gradients, sparse/unsupported gradients and skipped steps. A faster optimizer
that changes Adam versus AdamW semantics is a different algorithm.

Run matched multi-step state checks and save/resume if relevant, then full-update
A/B. Stop if the optimizer's exposed share is too small or additional workspace
forces a smaller microbatch that cancels the gain.

## 7. Separate DDP synchronization from rank imbalance

Collect per-rank timing for input readiness, backward progress and collectives.
A long collective on an early rank may reflect waiting for a late rank rather
than slow links. Compare shapes, data-loader behavior, CPU affinity and topology
before changing the communication algorithm. Avoid barriers added solely for
timing inside the performance path; they can destroy the overlap being studied.

With DistributedDataParallel gradient accumulation, investigate unnecessary
all-reduces on intermediate microsteps. When using the supported `no_sync()`
context, include the forward and backward of those microsteps, and ensure the
final intended microstep synchronizes. Validate partial groups and every rank's
consistent schedule. The optimization is conditional on mathematically equivalent
accumulation; it is not a general instruction to skip collectives.

Tune bucket sizes or overlap only after identifying exposed communication tails.
Small buckets can start communication earlier but increase overhead; large ones
may delay it. Unused-parameter/static-graph options must match actual model behavior.
Check collective order and participation when conditional branches exist.

## 8. Treat FSDP and other parallel modes as different execution plans

Identify the installed FSDP generation/API and current sharding plan. Do not
transfer DDP accumulation instructions mechanically: parameter materialization,
resharding, gradient storage and communication can differ. Measure full-step
memory and synchronization under the exact chosen API.

Inspect wrapping/sharding granularity, all-gather timing, overlap, reshard policy
and supported prefetch controls. Coarse units may create memory peaks or large
unhidden transfers; very fine units may add excessive launches/collectives. Test
one supported policy at a time and retain checkpoint compatibility evidence.

For tensor parallelism, pipeline parallelism or context/sequence sharding, add
the corresponding costs: activation transfers, bubbles, partition imbalance,
padding, attention masks and reduction semantics. Long-context ring/striped
attention needs rank-balanced work and correct global causal positions. An
isolated collective benchmark does not validate these algorithms.

## 9. Measure scaling at the right boundary

Use `nccl-tests` or P2P tools at representative message sizes to investigate a
suspected fabric/placement limit. Record topology and rank mapping. A healthy
large-message all-reduce does not rule out latency problems for many tiny
collectives, and its bandwidth is not the application's exposed communication.

Compare authorized GPU layouts with fixed useful workload and learning semantics.
State strong versus weak scaling assumptions. If a one-GPU comparison does not
fit, use a valid smallest-fitting baseline rather than inventing a speedup from
an unavailable reference. Include input delivery and checkpoint costs that grow
with rank count. Aggregate rate follows job completion, not the best rank.

Example: doubling GPUs cuts compute from 80 to 40 ms but adds 30 ms of exposed
communication to a previous 10 ms tail. The update falls from 90 to 80 ms, not
to 45 ms. Decide whether that gain justifies the resource cost under the contract.

Finish at [validation and delivery](07-validation-and-delivery.md). Preserve
the exact launch, rank mapping, complete-step evidence and restart state needed
to reproduce the selected configuration.

---

[← Serving](04-serving-and-concurrency.md) · [Handbook](README.md) · [Kernel engineering →](06-kernel-engineering.md)
