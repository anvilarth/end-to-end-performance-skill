# 4. Serving and concurrency

[Home](../../README.md) / [Library](../README.md) / [Handbook](README.md) / 04

**Open when:** optimizing online requests, batched generation or an offline job
using an inference server. **Exit when:** the chosen configuration improves the
declared useful-work metric at representative load while meeting quality,
latency, memory and completion constraints. Sources:
[serving and attention](../sources/README.md#serving-and-attention).

## In this chapter

- [1. Define the load before tuning the engine](#1-define-the-load-before-tuning-the-engine)
- [2. Audit feature activation and metric code](#2-audit-feature-activation-and-metric-code)
- [3. Identify the exposed stage](#3-identify-the-exposed-stage)
- [4. Sweep concurrency to the useful knee](#4-sweep-concurrency-to-the-useful-knee)
- [5. Treat caches as semantic state](#5-treat-caches-as-semantic-state)
- [6. Choose replication, tensor parallelism and sharing from evidence](#6-choose-replication-tensor-parallelism-and-sharing-from-evidence)
- [7. Escalate attention and hybrid-model work selectively](#7-escalate-attention-and-hybrid-model-work-selectively)
- [8. Validate the operating region, not one lucky point](#8-validate-the-operating-region-not-one-lucky-point)

## 1. Define the load before tuning the engine

Recover the actual request path: client preparation, serialization, transport,
queue/admission, multimodal encoders, prefill, decode, detokenization and writing.
Declare input/output length distribution, modalities, sampling parameters,
prefix reuse, request arrival process, cancellation and timeout behavior.

Distinguish load models:

- A fixed in-flight client submits new work after completion. It measures a
  closed-loop operating point and can hide overload by reducing arrivals.
- An external arrival schedule tests whether the service can keep up with demand.
  Record intended and achieved submission times; client-side limits can still
  throttle arrivals and change the test.
- A finite offline queue targets makespan or useful throughput and includes fill
  and drain. It does not establish an interactive latency service level.

Measure client CPU, connection pool behavior and event-loop delay. If increasing
client capacity improves throughput while the server was underfed, the engine
was not the bottleneck. Preserve request identity and include all errors and
timeouts in the report. Count useful successes over the full measurement time;
do not delete failed or slow intervals from the denominator.

## 2. Audit feature activation and metric code

Record the engine/runtime version, model implementation, attention backend,
dtype, KV-cache dtype, scheduler limits, graph/compile policy and parallelism.
Check the running worker, not only launch flags. An unsupported feature can
silently fall back; a dtype flag can affect weights but leave KV or reductions
unchanged. Verify model-specific support before trying a generic recipe.

Read the benchmark's formula. Generated-token throughput, total-token throughput
and request throughput answer different questions. Capture actual generated IDs
or trustworthy engine counts when the contract needs exact counts. Keep stop
reasons and truncations. For structured outputs, report both schema completeness
and the separate semantic gate.

Do not import another benchmark's constraints or relax the current user's ones.
No-draft means no speculative draft path. Fixed precision excludes a lower-
precision cache unless separately authorized. Faster empty or truncated responses
are failures, even if the benchmark reports a high token rate.

## 3. Identify the exposed stage

| Observation | First probe | Candidate if the cause is confirmed |
| --- | --- | --- |
| GPU waits while CPU is busy | Client/worker stacks and input queue timing | Batch tokenization/serialization, reuse decoded media, reduce repeated preparation |
| Long queue and high device occupancy | Admission rate versus completed rate; queue growth | Tune scheduler capacity or add authorized replicas; preserve latency limits |
| TTFT grows with prompt length | Separate queue, encoder and prefill time | Equivalent faster attention, prompt reuse, supported chunked prefill |
| Decode dominates at low batch | Per-step kernel timeline, KV traffic and launch gaps | Better batching, graph replay, supported decode kernel |
| Large variation with multimodal content | Encoder and transfer stages by modality/shape | Bounded encoder batching, exact feature reuse, data-path fixes |
| Memory eviction/preemption churn | Live sequence lengths, cache occupancy and retries | Adjust admission, token budget or cache allocation; reassess capacity |
| Long tail after most requests finish | Remaining lengths and batch occupancy during drain | Length-aware scheduling if permitted; report fairness/order effects |

TTFT includes more than prefill in a real service. Inter-token latency can include
scheduler stalls and interference from other requests. Attribute these before
choosing a kernel. Use the [profiling ladder](02-attribution-and-experiments.md)
if engine counters do not establish causality.

## 4. Sweep concurrency to the useful knee

Fix all other factors. Increase in-flight work or offered rate coarsely until
useful throughput plateaus, required latency fails, errors rise, or the allowed
resource limit is reached. Refine around that boundary. Record completed rate,
actual offered rate, queue depth/growth, TTFT/end-to-end tails, memory and failures
at each point. A configuration with more outstanding work but unchanged rate
usually buys queueing delay rather than capacity.

Report sample counts with tail percentiles; a tiny canary cannot establish p99.
At a fixed observation cutoff, report requests still in flight and their ages.
Otherwise the slowest unfinished requests disappear from completed-only latency
statistics and can make an overloaded service look healthy.

Tune one scheduler control at a time: active sequences, tokens per iteration,
prefill chunking or encoder limits. These may interact, so compare the promising
combination after identifying individual effects. Separate prompt-heavy,
decode-heavy and mixed traffic if all are required; a mean can hide regressions.

Example: moving from 16 to 32 clients raises throughput 25%, while 64 adds 1%
and doubles p99. If p99 is constrained, 64 is not the useful operating point.
For a throughput-only offline contract it may still be unattractive because of
memory and failure risk, but the decision follows the declared metric.

Use the engine's benchmark tool when it represents the actual request schema.
For vLLM, inspect `vllm bench serve --help`: request rate and maximum concurrency
control different aspects of load, and a concurrency cap can reduce achieved
arrival rate. Use a custom existing harness when needed for exact multimodal
inputs or output evidence rather than substituting random prompts.

## 5. Treat caches as semantic state

Record cold, warm and realistic reuse policies separately. A repeated-input
screen can accidentally measure a prefix cache while the real workload is
mostly unique. Disable or reset relevant caches for a no-reuse contract, and
verify reset behavior rather than assuming a new client clears the server.

Cache keys must include everything affecting the reusable computation: model
and adapter identity, exact token prefix/positions, relevant configuration and
multimodal content/transformation identity. Equal placeholder tokens do not imply
equal images or audio. Verify invalidation and request isolation.

Measure hit rate, saved work, memory occupancy, eviction and lookup/copy overhead
under the expected reuse distribution. Count cache construction where paid.
Keep accepted outputs independent of cache hits. Prefix reuse is useful only
when the workload and engine support that exact reusable state.

## 6. Choose replication, tensor parallelism and sharing from evidence

First determine whether a complete replica fits with required cache/workspace
headroom. Compare candidate layouts at the same authorized GPU budget, workload
and latency target. More replicas can improve aggregate throughput; wider tensor
parallelism can solve capacity or single-request latency but adds communication.
Measure routing balance and the slowest replica, not just one local worker.

For small independent GPU jobs, several processes or CUDA MPS may improve device
occupancy. Check installed MPS version, permissions, memory duplication, fairness,
fault behavior and actual co-resident throughput. MPS is not a promise of better
throughput, and memory or bandwidth saturation may make sharing worse. Keep
resource scope explicit; do not commandeer additional GPUs to improve a score.

Consider prefill/decode disaggregation only after measuring incompatible resource
needs and checking engine support. Include KV transfer, network, routing,
queueing, extra replicas and low-load behavior. It is usually a larger system
change than first tuning a supported monolithic configuration.

## 7. Escalate attention and hybrid-model work selectively

For long-context low-batch decode, a split-KV/Flash-Decoding-style implementation
may expose more parallelism; include its reduction and workspace cost. For
prefill, check supported masks, head dimensions, dtype and backend before replacing
attention. Attention microbenchmarks must return to the full request path.

For recurrent or hybrid attention models, include state update, reset and chunk
boundary semantics. Test consecutive requests and chunked versus unchunked
execution where equivalence is required. A zero-initial-state fixed-chunk kernel
does not validate arbitrary streaming. For MoE, measure expert load distribution,
routing, packing, all-to-all and restoration; faster expert GEMMs alone may not
improve the request. See [kernel engineering](06-kernel-engineering.md).

## 8. Validate the operating region, not one lucky point

Run the final comparison through the real entrypoint with the frozen manifest
or arrival trace. Include the expected low/normal/high load range when robustness
across it is part of the contract. Preserve raw errors, missing IDs, stop reasons,
actual output lengths, cache settings and per-request timings. Treat load-dependent
regressions as part of the decision rather than averaging them away.

Finish with [validation and delivery](07-validation-and-delivery.md). If there
is no acceptable point before saturation, state whether the limit is compute,
memory, transport, quality or the authorized resource budget. This gives the next
session a concrete decision instead of another blind concurrency sweep.

---

[← Input & compilation](03-input-shapes-and-compilation.md) · [Handbook](README.md) · [Training →](05-training-and-distributed.md)
