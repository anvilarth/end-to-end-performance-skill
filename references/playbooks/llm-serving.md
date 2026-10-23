# Playbook: LLM and VLM serving

For generative models behind a continuous-batching engine (KV cache, tensor or
expert parallelism, chunked prefill), typically multimodal captioning or
generation with hundreds of requests in flight per node and long outputs per
request. Read [common.md](common.md) first; this file adds the engine-side
levers.

## Shape of the bottleneck

Decode step time is dominated by weight reads and is almost flat in batch
size, so throughput is set by how many requests the engine keeps running at
once. The usual limits are not the GPU: an admission cap hidden in cache
sizing, a single-threaded scheduler or tokenizer process moving multimodal
payloads, and the number of independent admission paths per node.

Primary metric: completed requests (or items) per second per node in the
steady window at a fixed request shape. Keep the request contract fixed:
frames, resolution, prompt, sampling settings, and max tokens are the output
contract, not tuning knobs.

## Triage

| Observation | Lever |
| --- | --- |
| Running requests stay well below the memory allows; batch size barely moves step time | 1 |
| Scheduler or tokenizer process at 100% of one core while GPUs idle | 2 |
| One wide tensor-parallel replica per node; prefill chunks stall decode | 3 |
| Replicas hang at start when launched together | 3 (sequential start) |
| OOM on the first GPU of a replica under bursts although the memory fraction looks safe | 4 |
| Client threads busy writing per-chunk streaming output | 5 |

## 1. Lift hidden admission caps

**Symptom:** the engine reports a small running count, decode step time is
flat in batch, KV memory is mostly free.

**Mechanism:** hybrid-attention and recurrent layers keep a per-request state
cache whose slots are consumed several per running request (extra tracking
slots per request in eager cache strategies). An automatic state-cache size
caps concurrency far below the KV budget. Size the state cache at the slot
cost times the intended running count, choose the lazy cache strategy when
it lowers the slot cost, raise the running-request limit explicitly, and
confirm the KV allocation covers running requests times the expected tokens
per request.

**Rung:** 2, scheduling.

**Observed gain:** +57% steady-state throughput on a hybrid-attention
multimodal model after raising running requests from about 70 to about 200
with a correctly sized state cache; the full 1,000-item benchmark showed +50%
because ramp and tail dilute it.

**Activation check:** the engine log prints the configured cache sizes and the
KV token count; the running-request counter reaches the configured value
under load.

**Parity trap:** none from the cache itself; keep kernels and dtypes fixed so
outputs stay comparable. A bf16 recurrent state is a precision rung (+12%)
and needs authorization.

## 2. Shrink multimodal ingestion payloads

**Symptom:** the tokenizer manager or scheduler spends most of its time moving
pixel tensors across inter-process queues, to tensor-parallel peers, and to
the device; per-request admission costs around a second.

**Mechanism:** pixel tensors sent in fp32 are hundreds of megabytes per
request. If the model casts them to bf16 on the device before the vision
encoder, casting at ingestion is bitwise identical and halves every copy.
Zero-copy device IPC removes the copies entirely but costs tens of gigabytes
on the first GPU outside the engine's memory budget.

**Rung:** 1, remove work.

**Observed gain:** per-request ingestion time roughly halved; part of the
+57% above. Tokenizer worker processes did not help on their own: the limit
was per admission path, not per thread.

**Activation check:** the engine log shows the patched cast; scheduler time
per request halves in a profile.

**Parity trap:** bitwise only when the first consumer casts to the same dtype
with the same rounding. Prove it on the encoder input tensor.

## 3. More replicas instead of wider tensor parallelism

**Symptom:** a model that fits on fewer GPUs is served by one wide replica;
node throughput is flat as tensor parallelism grows.

**Mechanism:** each replica has its own single-threaded admission path and
scheduler. Four two-GPU replicas admit four times as many requests as one
eight-GPU replica and avoid stalls where a prefill chunk blocks the whole
wide group. Expert parallelism across the small group keeps the MoE weights
balanced.

**Rung:** 2, topology.

**Observed gain:** +45% per node from two four-GPU replicas to four two-GPU
replicas on the same model and clips; the wider layout decoded faster in
bursts but lost end to end.

**Caveats:** start replicas on a host sequentially (wait for weight loading
to begin before launching the next); two groups initializing symmetric memory
at once hung on some nodes. Keep a fallback that disables symmetric-memory
collectives per host. Spread weight reads from a shared file system by
starting hosts in waves. Disaggregated prefill and decode did not work in the
engine version measured; treat it as untested, not as a lever.

## 4. Keep the memory fraction below the preprocessing overhead

**Symptom:** OOM on the first GPU of a replica under bursts although the
engine's memory fraction leaves room.

**Mechanism:** the image preprocessing of the tokenizer manager, any IPC pool,
and multimodal worker processes allocate on the first GPU outside the
engine's static budget. Observed ceilings: 0.90 at moderate concurrency,
0.88 at about 200 running per replica, lower for wider tensor parallelism.

**Rung:** 2, memory as performance.

**Activation check:** no OOM over a full run at the target concurrency with
the burst pattern of the real client.

## 5. Buffer the streaming client

**Symptom:** client processes show most threads inside file writes while
driving the engine.

**Mechanism:** a client that appends each streamed chunk to disk pays a
system call per token. Buffer per request and write once. At moderate
client thread counts this was not the limit, so measure before changing the
client contract.

**Rung:** 1, remove work.

## Measurement notes for this class

- Report the steady window (for example the 25 to 75% quantile of the run)
  separately from the full run; a 1,000-item run loses about 40% to ramp and
  tail.
- Health probes to a loaded engine time out; do not count probe failures as
  replica failures.
- The configured running count is a ceiling, not a demand: raise client
  concurrency to use it and check any upstream rate limit.
- Validate on the same clips and the same node when comparing engine flags;
  cross-node comparisons hide replica start skew and node-specific hangs.
