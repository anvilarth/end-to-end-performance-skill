# From a symptom to a decisive experiment

Use only the matching section after measuring the full-path budget. These recipes
select hypotheses; they do not establish an acceleration or replace the contract in
[measurement-and-parity.md](measurement-and-parity.md). Match API examples to the
installed versions. Preserve authorized precision, workload and resource limits.

## 1. Variable shapes and wasted padding

**Measure:** distribution of audio durations, frame counts/resolutions, prompt and
output lengths, batch composition, padding, and time by shape. For a dense token
batch, `1 - sum(lengths)/(batch_size * max_length)` estimates token padding, not
necessarily the fraction of wall time wasted (attention may scale differently).

**First candidate:** length buckets or a supported packed/variable-length operator
when wasted padding is material. For compiled paths, compare bucket count against
compilation count, graph-cache memory and padded work; no universal bucket size.

**Cheap test:** representative short, common and long inputs, including partial
batches. Verify masks, positions, sample boundaries, labels and restored output order.
For training, keep optimizer-step semantics and sample/RNG order as required; changing
batch composition can change numerics and is not automatically exact.

**Stop:** padding is not exposed cost, unsupported shapes require an expensive copy,
or semantic/parity checks fail. Final timing includes bucketing and packing costs.

Source: [PyTorch transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html).

## 2. Compilation stalls, graph breaks and cold startup

**Measure:** compiled regions, recompiles/failed guards, compile time, warm time and
expected lifetime of the process. Use `TORCH_LOGS="graph_breaks,recompiles"` for a
small reproduction; `TORCH_TRACE`/`tlparse` for a large model. Write trace output under
the authorized workspace. Record whether the intended region actually compiled.

**First candidate:** fix the costly break in the hot region; isolate unproductive
preprocessing/logging with `torch.compiler.disable` if appropriate. For varying
shapes compare dynamic compilation with a small bucketed policy. For large repeated
blocks consider regional compilation and valid cache reuse before compiling everything.

**Cheap test:** replay representative changing shapes, not only one warm shape.
If extra setup cost is `C` seconds and warm saving is `d > 0` seconds per call,
`C/d` is the approximate break-even call count; include expected recompiles in `C`.
This estimate assumes comparable calls; use workload-weighted totals otherwise.

**Stop:** expected lifetime does not amortize setup, graph memory exceeds budget, or
fixing a cold-path break has negligible effect. Do not treat zero graph breaks as
the optimization objective, or silently suppress compiler errors as a speed fix.

Sources: [compiler troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html),
[regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html).

## 3. Prefill, decode and serving pressure

**Measure:** preprocessing/encoder, queue, prefill, decode and final serialization
separately; prompt/output length distributions, actual generated tokens, finish
reasons, cache hits, time to first token (TTFT), subsequent token latency, and E2E.
A fixed concurrency closed-loop test and a fixed arrival-rate test answer different
questions: record which is used and include client-side queueing where relevant.

**First candidate:** batching for underfilled decode; chunked prefill if large
prefills interrupt decodes; cache policy only when reuse matches real traffic.
Choose serving throughput subject to the requested latency/error constraints.
For offline work optimize complete fixed-manifest wall time and accepted outputs,
without inventing an online latency requirement.

**Cheap test:** realistic mixtures of prompt/output lengths, not just equal synthetic
requests. Keep generation settings and quality gates fixed. If output length changes,
report it: faster completion may be less generated work. A fixed-token diagnostic
can isolate engine cost but is not a substitute for the actual generation gate.

**Stop:** the apparent gain comes from truncation, rejected outputs, artificial cache
hits, uncounted queue/drain time, or a violated latency requirement.

Sources: [Sarathi-Serve](https://arxiv.org/abs/2403.02310),
[vLLM benchmark metrics](https://docs.vllm.ai/en/latest/cli/bench/serve/),
[GPT-fast case study](https://pytorch.org/blog/accelerating-generative-ai-2/).

## 4. Host/device transfers and claimed overlap

**Measure:** exposed input wait, copy time, CPU preparation and stream dependencies.
A replay from already-prepared inputs can isolate feeding overhead; label it a
counterfactual diagnostic and preserve the real input path for promotion.

**First candidate:** prepare ahead and use reusable pinned buffers, an appropriate
copy stream and independent compute only if feeding is limiting. `non_blocking=True`
removes a host synchronization; it does not prove device-side overlap.

**Cheap test:** timeline showing next-batch copy overlapping current-batch compute.
Check event/wait ordering and tensor lifetime (`record_stream` or explicit lifetime
management). Do not mutate pinned source buffers before transfer completion or read
an asynchronous CPU destination too early. Include pinning/allocation cost; pinning
on the critical path may erase a copy win.

**Stop:** dependencies force serial execution, resources contend, or no exposed wait
is removed. A second stream by itself does not create scheduler lookahead.

Sources: [pinning and non-blocking transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams).

## 5. Large intermediate tensors and fusion

**Measure:** materialized intermediate sizes, lifetimes, reads/writes and launch gaps
at producer/consumer boundaries. Check generated compiler code before implementing
fusion that may already exist.

**First candidate:** remove the materialization, or use an existing fused operator.
For training where logits are only consumed by the loss, test fused/chunked linear
cross entropy instead of storing full `[tokens, vocabulary]` logits. This does not
apply unchanged when callers require full logits. Residual-plus-normalization and
pointwise epilogues are other candidates when their traffic/launch budget is material.

**Cheap test:** same input, loss reduction, ignore-index, token weighting, target
shift and precision; check forward, input/weight gradients and required saved outputs.
Include layout conversions, chunking and backward in timing. In-place and out-of-place
implementations require compatible aliasing/lifetime contracts.

**Stop:** conversion/backward cost consumes the gain, fused semantics differ, or
increased resource pressure slows the surrounding workload.

Sources: [Liger fused linear cross entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py),
[GPU MODE fusion notebook](https://github.com/gpu-mode/lectures/blob/main/lecture_018/FusedKernels.ipynb).

## 6. Memory limits and recomputation

**Measure:** a snapshot at the actual peak plus allocated/reserved and total device
memory. Attribute the peak to activations, logits, optimizer state, temporaries or
retained tensors. PyTorch snapshots cannot see all external allocations, e.g. NCCL.

**First candidate:** eliminate the dominant avoidable allocation; for activation
peaks, compare selective checkpointing with blanket recomputation. Preserve expensive
operations and recompute cheap ones when the measured memory/speed trade-off supports it.

**Cheap test:** a representative optimizer step including the longest supported case,
loss/gradient checks and peak memory. If reduced memory enables a bigger microbatch,
measure the full step at equivalent global-batch/accumulation semantics; GB saved alone
is not a speedup. Different training recipes need a separately authorized quality gate.

**Stop:** extra recomputation costs more than the resulting throughput improvement or
the saved memory does not relieve the real constraint. Do not apply checkpointing to
an unrelated optimizer-state or external-allocation peak.

Sources: [CUDA memory snapshots](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html),
[selective activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/).

## 7. Low utilization and distributed scaling

**Measure:** separate host starvation, too little parallel work, bandwidth limits,
compute limits, communication and rank skew. Arithmetic intensity and a roofline
estimate are screening aids; they assume sufficient parallelism and realistic traffic.
A small kernel can saturate neither compute nor bandwidth.

**First candidate:** select the mechanism matching the evidence, not the utilization
percentage. When PyTorch traces are available, HTA can provide idle/temporal breakdown,
launch statistics, queue lengths, communication overlap and trace comparison. For
multi-GPU work inspect ranks and exposed critical-path collectives before changing
parallelism; an early rank waiting in NCCL may be waiting for a late producer.

**Cheap test:** representative shape/concurrency or collective message-size sweep on
the allowed allocation. Keep isolated collective throughput diagnostic; promotion is
still the full step/request at the intended topology.

**Stop:** the proposed mechanism attacks a different bound, or the communication
benefit is offset by local work, memory, imbalance or smaller per-GPU batches.

Sources: [NVIDIA performance background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html),
[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis).

## 8. Batch changes and numerical behavior

**Measure:** dtype plus actual accumulation/reduction settings, TF32 policy, backend,
batch/packing/order, seeds and generation settings. Batched and individual operations
need not be bit-identical; deterministic decoding does not fix changed model logits.

**Cheap test:** rerun parity when any of these changes, even if weights and dtype did
not. Compare the expected IDs and raw numerical values where available; for generation
also compare final outputs and the agreed quality metrics. For training include
gradients/updates and longer convergence checks when numerical changes may accumulate.

**Stop:** exact contracts fail on any prohibited difference. A tolerance contract uses
its existing bounds; do not widen it after observing a candidate. Numerical tolerance
alone does not establish semantic quality or allow a precision-policy violation.

Source: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
