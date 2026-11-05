# Specialized optimization recipes

[Home](../../README.md) / [Library](../README.md) / [Recipes](README.md)

Use only the branch matching measured exposed cost. These are candidate-selection
rules, not demonstrated gains on the user's workload. Source snapshots were reviewed
2026-11-04; check the installed backend's capabilities before using any API or flag.

## 1. Repeated prefill: prefix reuse and cache-aware scheduling

**Evidence:** repeated token prefixes, prefill cost and achievable hit rate under
the real arrival order. Measure retained KV bytes, eviction and queue latency too.
**Candidate:** supported prefix caching; group independent offline requests by
shared prefix only if output order can be restored and RNG semantics preserved.
**Cheap test:** identical requests with cold and naturally warmed caches, then
changed media, adapter and prefix. Cache identity must include all state that changes
the computation: model/adapter, tokens, positions, masks, media and preprocessing.
Matching image placeholders alone does not identify matching multimodal input.
**Stop:** reuse is prohibited by the benchmark contract, rare, or displaced by
memory pressure; do not manufacture repeated inputs to make the cache look useful.
Report cold/warm/hit distributions and compare both arms under the same cache policy.

Source: [SGLang/RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/),
linked from [lecture 35](https://github.com/gpu-mode/lectures/tree/main/lecture_035).
The cache-identity and evaluation gates above are workflow adaptations.

## 2. Gaps between launches: scheduler and host synchronization

**Evidence:** a timeline shows the GPU waiting for CPU scheduling, metadata, sampling
or scalar reads. Locate `.item()`, `.cpu()`, logging and explicit synchronization.
**Candidate:** batch/defer diagnostics, keep needed metadata on CPU, or use an
existing overlap scheduler. CUDA graphs reduce launch work when capture is supported.
Overlap needs independent next-step work; another stream alone cannot create it.
**Cheap test:** one dependency-preserving change with logging/scheduler time and
GPU gaps measured. For training, reduce detached loss values at reporting boundaries
without retaining autograd graphs. Keep termination/error checks semantically intact.
**Stop:** dependencies serialize the work, gaps remain, or CPU cost is immaterial.
Do not copy lecture-era flags as current defaults or rewrite the server first.

Sources: [lecture 35](https://github.com/gpu-mode/lectures/tree/main/lecture_035),
[training baseline](https://github.com/gpu-mode/paris-hackathon-2026-training/blob/main/train.py),
[PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## 3. Optimizer dominates the step

**Evidence:** time the complete optimizer stage, parameter count, state traffic and
peak memory separately from forward/backward. First inspect the actual implementation.
**Candidate:** the same optimizer's supported fused or foreach implementation;
consider compilation only if remaining cost and run length justify it.
**Cheap test:** clone parameters, gradients and optimizer state; compare multiple
updates, including clipping, accumulation, skipped/absent gradients and scheduler
steps, then measure the complete training step. Foreach can increase peak memory.
**Stop:** already using an effective fused path, small optimizer fraction, state
incompatibility, numerical failure or lost batch capacity. Changing optimizer,
state precision or learning-rate policy is a separate algorithm experiment.

Sources: [training baseline](https://github.com/gpu-mode/paris-hackathon-2026-training/blob/main/train.py),
[PyTorch optimizers](https://docs.pytorch.org/docs/stable/optim.html),
[AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

## 4. Exposed DDP synchronization

**Evidence:** representative per-rank traces show all-reduce starting late, tiny
collectives, a tail after backward, or repeated synchronization during accumulation.
**Candidate:** supported bucket sizing and DDP `no_sync()` on non-final microsteps.
Keep both forward and backward in the intended accumulation context; synchronize
the final microstep, including a partial final accumulation group.
**Cheap test:** compare gradients and updates for a full accumulation cycle, including
unequal valid-token counts where applicable. Preserve global loss normalization,
clipping and step cadence; inspect every relevant rank's completion and collective order.
**Stop:** communication is already hidden, smaller buckets add launch overhead,
or larger buckets delay useful overlap. Do not generalize DDP settings to FSDP.

Sources: [NCCL lecture 17](https://github.com/gpu-mode/lectures/tree/main/lecture_017),
[DDP no_sync](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

## 5. Communication plus GEMM: topology before fusion

**Evidence:** exposed all-gather→GEMM or GEMM→reduce-scatter cost; actual message
sizes, placement, slow ranks and link type. A P2P bandwidth test is a diagnostic,
not an application speedup or a reason to force one NCCL algorithm globally.
**Candidate:** first correct placement/use an existing overlapping backend; then
consider tiled producer/consumer communication if its implementation cost is justified.
At decode, a low-latency fused all-reduce (sentinel-polled buffers, fused with
the residual add, normalization and any quantization) has been reported about
twice as fast as the library collective; it is a candidate only when the exposed
collective is a measured share of the step.
**Cheap test:** original and overlapped subgraphs with all ranks, same buffers,
result ownership and reduction semantics. Include launch, staging and synchronization;
check buffer lifetime, memory ordering and progress when resources are occupied.
**Stop:** communication is hidden already, compute slows from resource competition,
or no matched implementation exists. An AMD Iris example is not a drop-in CUDA kernel.

Sources: [p2p-perf](https://github.com/gpu-mode/p2p-perf),
[distributed reference tasks](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_distributed),
[Iris lecture 78](https://github.com/gpu-mode/lectures/tree/main/lecture_078).

## 6. Causal context parallelism: rank work imbalance

**Evidence:** long-context causal attention is capacity-limited or context-parallel
ranks finish at different times because the causal triangle distributes work unevenly.
**Candidate:** an existing balanced/striped context layout or ring-attention backend,
within authorized GPU resources. Capacity gain and faster execution are separate claims.
**Cheap test:** per-rank work/time and full attention outputs/gradients. Preserve
global positions, causal masks, sequence boundaries and applicable dropout semantics.
**Stop:** short contexts, communication overhead or unsupported masks eliminate the
benefit. Reordering tokens without consistently transforming metadata is incorrect.

Sources: [ring-attention](https://github.com/gpu-mode/ring-attention),
[Striped Attention](https://arxiv.org/abs/2311.09431), linked by that repository.

## 7. Hybrid models: recurrent state, convolution and MoE

**Evidence:** attribute the actual model's time to attention, causal convolution,
DeltaNet/state updates, routing, expert GEMMs and packing. Separate prefill and decode.
**Candidate:** a supported fused/chunked implementation of the measured block;
grouped GEMM only when expert execution, rather than dispatch, is the limiting stage.
**Cheap test:** compare outputs and state at chunk boundaries and after resumed
decode, including uneven lengths, reset, batch reorder and zero/low-token experts.
**Stop:** the candidate assumes a different state/precision/layout contract.
The reference DeltaNet task uses chunk size 64, divisible sequence lengths and a
zero initial state; passing it does not validate arbitrary streaming state or tails.

Sources: [Helion reference tasks](https://github.com/gpu-mode/reference-kernels/tree/main/problems/helion),
[Qwen3.5 inference harness](https://github.com/gpu-mode/paris-hackathon-2026-inference).

## 8. Tensor cores wait: on-chip pipelines and non-matmul work

**Evidence:** a material kernel stalls on transfers, barriers, reductions, special
functions or layout conversions. Elementwise work is not necessarily DRAM-bound.
**Candidate:** existing asynchronous copy/MMA pipelines, producer/consumer warp
specialization, or overlap of softmax with matrix work while preserving precision.
**Cheap test:** one matched shape plus tails; inspect shared-memory/register use,
spills, synchronization and uninstrumented latency. More pipeline stages consume
resources and can lose performance. FMA/reassociation can change numerical results.
**Stop:** no exposed latency remains or added staging costs more than it hides.
Match instructions to architecture: Hopper WGMMA examples and Blackwell-specific
tensor-memory/cluster examples do not share universal tuning constants.

Sources: [FlashAttention-3 lecture 36](https://github.com/gpu-mode/lectures/tree/main/lecture_036),
[SASS lecture 37](https://github.com/gpu-mode/lectures/tree/main/lecture_037),
[ThunderKittens lecture 75](https://github.com/gpu-mode/lectures/tree/main/lecture_075).

## 9. Uneven tiles: persistent scheduling, Stream-K and multicast

**Evidence:** partial final waves, skewed expert/context lengths, static persistent
assignment imbalance, or repeated loads of the same tiles across cooperative CTAs.
**Candidate:** shape-matched work distribution, split/Stream-K, or supported dynamic
cluster scheduling. TMA multicast is relevant only when cooperative blocks reuse data.
**Cheap test:** the real shape distribution, including small jobs; count reduction,
queue, workspace and cluster costs. Confirm active blocks can make progress.
**Stop:** queue/reduction overhead, register pressure or cluster residency loses more
than improved balance/reuse saves. Do not optimize for maximum occupancy alone.

Sources: [lecture 35](https://github.com/gpu-mode/lectures/tree/main/lecture_035),
[Gluon lecture 104](https://github.com/gpu-mode/lectures/tree/main/lecture_104).

## 10. Layout or synchronization bugs: prove the mapping cheaply

**Evidence:** a new custom kernel needs nontrivial index/layout mapping, vectorized
accesses, shared-memory swizzling or asynchronous buffer reuse.
**Candidate:** small CPU/interpreter examples and explicit layout visualization before
expensive GPU sweeps. Escalate from existing Triton to Gluon/CuTe/ThunderKittens only
when required schedule/layout control is missing, not because the DSL is newer.
**Cheap test:** non-power-of-two tails, strides, alignment, boundary masks and large
address arithmetic; use supported race/synchronization instrumentation for async paths.
Triton-Viz/interpreter is not a concurrency proof. Gluon instrumentation and narrow
Proton regions are optional and version-dependent; instrumented timing is diagnostic.
**Stop:** mapping or sanitizer failures. Do not tune an incorrect implementation.

Sources: [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles),
[CuTe copy lecture 57](https://github.com/gpu-mode/lectures/tree/main/lecture_057),
[layout algebra 103](https://github.com/gpu-mode/lectures/tree/main/lecture_103),
[Gluon 104](https://github.com/gpu-mode/lectures/tree/main/lecture_104),
[PyCuTe 114](https://github.com/gpu-mode/lectures/tree/main/lecture_114).

## 11. Repeated hot shapes: bounded autotuning or superoptimization

**Evidence:** stable hot shapes and enough future invocations to amortize search,
compilation, validation and dispatch complexity.
**Candidate:** a small justified search over tiles/warps/stages/layouts or a supported
subgraph superoptimizer such as Mirage; search existing implementations first.
**Cheap test:** fixed search budget, independent reference, disjoint screening/final
inputs and representative weighted shape mix. Cache validated choices by device,
compiler/backend, shape, dtype and layout; retain a general fallback for other cases.
**Stop:** expected lifetime savings cannot repay search, or the final E2E gain is
below the threshold. Algebraic/finite-field equivalence does not establish floating
point, gradient or model-quality equivalence. Validate generated code independently.

Source: [Mirage lecture 79](https://github.com/gpu-mode/lectures/tree/main/lecture_079).
Search-budget and promotion rules are our workflow adaptation.

## 12. Precision, sparsity and alternative attention: separate contracts

**Evidence:** remaining compute/memory cost and explicit authorization for the
specific numerical or algorithmic trade-off; do not infer it from available hardware.
**Candidate:** only the authorized low-bit path, structured sparsity, or alternative
attention. Identify weight/activation/KV/gradient/optimizer/accumulator precision
separately. Include scaling, calibration, packing and conversion costs.
**Cheap test:** dynamic range, cancellation, saturation/non-finite values, sensitive
shapes, model outputs and training trajectory as relevant, then full-path performance.
**Stop:** quality fails, dense fallback dominates, or low-bit preparation erases gain.
Sparse tensor cores require supported structure and layout, not just many zeros.
Streaming/window/sink attention, PaTH and model architecture changes are not presumed
equivalent replacements for the user's full-attention model.

Sources: [sparsity 11](https://github.com/gpu-mode/lectures/tree/main/lecture_011),
[quantized training 30](https://github.com/gpu-mode/lectures/tree/main/lecture_030),
[BitBLAS 33](https://github.com/gpu-mode/lectures/tree/main/lecture_033),
[StreamingLLM 72](https://github.com/gpu-mode/lectures/tree/main/lecture_072),
[PaTH 74](https://github.com/gpu-mode/lectures/tree/main/lecture_074),
[numerics 84](https://github.com/gpu-mode/lectures/tree/main/lecture_084).
