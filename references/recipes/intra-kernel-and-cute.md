# Intra-kernel diagnosis and CuTe correctness

Use after the full-path profile identifies a material kernel budget. For sources
and neighboring frameworks use [the documentation atlas](../sources/gpu-docs-atlas.md).
These are diagnostic hypotheses, not mandatory rewrites or measured local gains.

## Choose the smallest observable boundary

| Missing evidence | Tool / artifact | What it does not establish |
| --- | --- | --- |
| Launch gaps, host sync, transfers, overlap | Nsight Systems timeline | Internal producer/consumer wait attribution |
| Pipe utilization, memory transactions, spills, stalls | Nsight Compute counters plus source/SASS | Complete internal chronology from aggregate counters alone |
| Triton phase/warp timing | Proton scopes and intra-kernel trace | Uninstrumented throughput or general cross-SM clock equivalence |
| CuTe phase/warp timing | CuTe IKET and Perfetto | Async work completion merely because an issue instruction was timed |
| CUDA phase timing without a supported profiler | Small sampled timestamp buffer | Portable timer resolution, zero overhead, or an unchanged resource footprint |

Sources: [Proton](https://github.com/triton-lang/triton/blob/main/third_party/proton/docs/intra-kernel.md),
[IKET](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html),
[Aryan's worked RMSNorm trace](https://a-r-r-o-w.github.io/blog/3_blossom/00005_looking_inside_your_kernels/index.html).

Write the question first: e.g. whether the consumer waits for TMA or the producer
waits for a reusable buffer. Instrument coarse phases and a few relevant warps;
verify event pairing, buffer capacity, dropped/wrapped events and the actual kernel
instance. Place async completion boundaries at the corresponding wait/fence, not
just after submission. A barrier changes scheduling; adding one for measurement
can change the phenomenon being measured.

For IKET, check the installed release's requirements and `run-iket` help. Compile
the instrumented kernel inside the profiled process; a previously cached executor
can omit expected events. Verify Perfetto and JSON contain the intended ranges.
Estimate incremental event overhead by comparing minimally and fully instrumented
profiled variants using trace kernel durations. Keep unprofiled end-to-end timing
as a separate final gate. Do not infer event overhead from host wall-clock alone.

For manual instrumentation, distinguish per-SM cycle counters from a global timer;
check clock frequency, resolution and architecture. Buffer writes change shared
memory/register use and may change residency. Compare resource usage and baseline
latency with tracing disabled. Preserve the application's cache behavior: a
cold-L2 diagnosis and a warm-cache workload answer different questions.

## CuTe failures that look like numerical errors

The [Maharshi case report](https://x.com/maharshii/status/2082921705941651595)
and [identity tensor explanation](https://maharshi.bearblog.dev/identity-tensors-in-cutedsl/)
provide useful probes, not universal API rules:

- **Broadcast/recast aliasing:** inspect strides, offsets and reachable storage
  before linearly indexing packed scale fragments. A stride-zero mode can alias
  stores after recasting. Remove redundant zero-stride modes only when that
  preserves the intended mapping; assert resulting size and test distinct writes.
- **Slices:** equal layouts do not imply equal tensors. Inspect iterators/base
  offsets or use a tiny write/read check. Partition an identity tensor alongside
  the data to recover global coordinates for predication and ragged tiles.
- **Pipeline state:** establish whether a JIT argument is a value or memory-backed
  object. Return updated value state explicitly when needed; test wraparound and
  more iterations than the stage count, not just one tile.
- **Static vs dynamic loops:** a fully unrolled runtime loop is not necessarily
  equivalent to Python/static tracing. Check whether indices must be compile-time
  values and whether a change introduces local-memory traffic or longer live ranges.
- **Barrier arrivals:** derive expected arrivals from the actual pipeline class,
  signaling lanes, multicast mask and cluster topology. Do not adopt “always count
  warps” from one example. Named barriers and mbarriers have different contracts;
  the [pipeline API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html)
  documents distinctions. Check initialization, phase transitions and tail drain.
- **Register allocation:** record registers, local memory, spill instructions and
  achieved occupancy. Local-memory allocation alone is not proof of spills.
  Dynamic register redistribution requires supported target instructions and
  uniform, aligned participation; verify generated SASS and legal operands from
  the installed toolchain rather than copying a blog's register budget.
- **Framework boundary:** check stream, device, dtype, stride, alignment, lifetime,
  aliasing, backward and compile registration. DLPack conversion and Python binding
  overhead can dominate a tiny kernel. Do not invent pointer alignment to suppress
  a compiler warning. Use [framework integration](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/framework_integration.html)
  and compare a supported FFI path only when host overhead is measured.

## Architecture-specific tuning probes

Start from a correct matching example. Change one of tile shape, stages, warp
roles, raster order, cluster shape or epilogue at a time. Measure tails and small
shapes as well as the largest square GEMM.

[Veitner's scheduler explanation](https://veitner.bearblog.dev/tile-scheduling-cute-dsl/)
distinguishes active clusters from CTAs. For 2-CTA kernels verify tile coverage,
CTA rank offsets, peer lifetimes and cluster launch compatibility. His
[2-CTA experiment](https://veitner.bearblog.dev/2-cta-gemm-on-b200/) does not establish
that multicast always helps. His [warp-specialization example](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/)
reports no positive effect from a particular peek/try-wait change. Keep these as
negative controls when choosing experiments.

For persistent L2 experiments, identify a genuinely reused region competing with
streaming traffic; sweep working-set size and hit ratio, inspect actual hit rate,
and restore access-policy state. [Veitner's L2 case](https://veitner.bearblog.dev/gpu-l2-cache-persistence/)
reports little benefit without this competition. Cache policy is not a blanket
optimization for every weight tensor.

## Algorithm before assembly

[Radix Top-K](https://veitner.bearblog.dev/radix-top-k/) motivates selection instead
of sorting discarded values. First freeze exact k, sortedness, indices, ties,
NaNs, infinities, signed zeros and dtype. A threshold-only filter may return the
wrong count; a sampling kernel may not satisfy `torch.topk` semantics. Inspect PTX
and final SASS only after establishing the contract and a measured issue/unroll
bottleneck. Preserve negative experiments from [Alpin's Top-K worklog](https://blog.alpindale.net/posts/top_k_cuda/).

[GDN prefill algebra](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/)
reuses an inverse through a diagonal similarity transform. Check nonzero scales,
underflow/overflow, numerical conditioning and recurrent state over long sequences;
prefer solving a system when appropriate to the implementation. An algebraic
identity in real arithmetic is not bitwise equivalence in low precision.

## Further experiment design

For hierarchical throughput budgets, persistent scheduler regressions, SASS diffs,
layout conversion and AMD wave/register trade-offs, consult the
[blog-derived transfer tests](../sources/gpu-blog-deep-dive.md#decisions-to-carry-into-an-optimization-experiment).
