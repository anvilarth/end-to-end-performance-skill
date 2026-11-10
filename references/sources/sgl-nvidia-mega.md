# SGLang NVIDIA megakernels: whole-step optimization

Reviewed 2026-11-10 at `750004efd8bc0db852cdd4efbebb807c73bedd4b`. External implementation evidence;
no local GPU performance qualification. [Repository README](https://github.com/sgl-project/sgl-nvidia-mega/blob/750004efd8bc0db852cdd4efbebb807c73bedd4b/README.md).

## When this changes the experiment

Use when small-batch decode exposes launch, intermediate-write or collective
boundaries. The implementation routes parts of a model step through mailboxes
and programmatic dependent launch, rather than optimizing isolated operators.
This is an experimental model/hardware-specific implementation, not a generic
replacement for SGLang or a supported GLM backend.

1. Profile the producer, communication and consumer as one dependency chain.
   Propose removing a boundary only when it lies on the measured critical path.
   Account for mailbox storage, synchronization, occupancy and resource lifetime.
2. Build a dispatch matrix by token rows, not request count alone. Speculative
   verification changes rows per request. Retain stock execution outside measured
   regions; check padding boundaries and captured graph widths explicitly.
3. Preserve negative regions. [The slot plan](https://github.com/sgl-project/sgl-nvidia-mega/blob/750004efd8bc0db852cdd4efbebb807c73bedd4b/sgl_nvidia_mega/deepseek_v41/plan.py)
   disables fast routing at widths where it loses and restricts prefetch when
   competing memory traffic outweighs latency hiding. Neither fusion nor overlap
   is a universal improvement.
4. Validate cross-rank agreement and producer/consumer ownership before capture.
   Exercise graph replay, request-slot reuse, retirement, mixed lengths and
   fallback. Inspect synchronization deadlines: a stuck peer must fail visibly,
   not silently hang or return stale data. [Synchronization primitives](https://github.com/sgl-project/sgl-nvidia-mega/blob/750004efd8bc0db852cdd4efbebb807c73bedd4b/sgl_nvidia_mega/common/sync.py).
5. Separate capability admission, component correctness, composed serving
   correctness, task quality and delivered performance. Passing each option alone
   does not qualify their combination. Keep unresolved numerical failures visible.

## Controlled performance versus real acceptance

[Benchmark protocol](https://github.com/sgl-project/sgl-nvidia-mega/blob/750004efd8bc0db852cdd4efbebb807c73bedd4b/benchmark/README.md): fresh servers per arm,
on/off/off/on launch order, forward/reverse batch sweeps, matched launch environments,
and activation evidence on every rank. Estimate the local noise band; do not copy
the repository's thresholds.

The static experiment fixes speculative acceptance using simulation variables to
isolate execution cost. Label this synthetic control; disable simulation for
accuracy and real-serving measurements. Final promotion needs real acceptance,
request completion, quality and latency measurements under the workload contract.
Do not equate per-user static decode tokens/s with fleet throughput or makespan.

## Integration boundary

Pin plugin and compatible SGLang revisions together. Verify model, GPU/SM count,
TP topology, dtype, query width and optional providers against the registered spec.
Startup activation alone does not prove that a particular runtime step used an
optimized slot. Optional interfaces and admitted widths are not performance proof;
retain the stock path and validate the actual selected composition.
