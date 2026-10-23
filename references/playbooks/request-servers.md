# Playbook: request servers for short requests

For model servers that answer many short requests over HTTP or RPC: speech
recognition and transcription, embedding services, rerankers, small
multimodal scorers exposed as a service. Clients drive the concurrency; the
server is persistent with a long cold start. Read [common.md](common.md)
first; this file adds the server-side levers.

## Shape of the bottleneck

One server per GPU with its own batching engine usually leaves the GPU half
idle: the engine's internal concurrency is capped, the request path has host
work (decode, feature extraction, serialization) that serializes behind one
process, and the clients upstream are preprocessing-bound. The gain comes
from three things at once: feeding the servers (host-side levers), several
servers per GPU under MPS, and raising the per-server engine concurrency to
match.

Primary metric: processed media seconds (or items) per second per node in
steady state, with outputs bitwise against the single-server reference.

## Triage

| Observation | Lever |
| --- | --- |
| Client workers preprocessing-bound; servers mostly idle | common 1, 2, 3 |
| One server per GPU, GPU utilization well below 80% | 1 |
| Engine queue full while GPU memory is mostly free | 2 |
| Servers restarted per task or per job; minutes of cold start each time | 3 |

## 1. Several servers per GPU under MPS

**Symptom:** one server process per GPU, utilization 50 to 70%, request
latency dominated by host work inside the server.

**Mechanism:** common.md section 4. Three server processes per GPU, each with
its own engine, under one MPS daemon per GPU. Divide the memory fraction so
the three fit with headroom, and keep the client concurrency per server at
or below the engine's running limit.

**Rung:** 2, scheduling.

**Observed gain:** +15 to 21% from MPS alone on a transcription server; the
node went from 372 to about 5,700 media seconds per second (9,800 on
contiguous inputs) once combined with the engine and host-side levers below,
about 26x per node.

**Activation check:** three server processes per GPU in the daemon's client
list; per-GPU throughput rises with the second and third server.

## 2. Raise engine concurrency to match the server count

**Symptom:** the engine queue fills while GPU memory is mostly free.

**Mechanism:** the default engine concurrency (running sequences, total
tokens) was sized for one server per GPU. Raise the running limit (128 per
server worked) and the total token budget, keep the memory fraction at about
0.9 divided across the servers, and cap the client concurrency per server at
or below the engine's limit so requests queue in the client, not in the
server.

**Rung:** 2, scheduling.

**Activation check:** the engine reports the configured limits at start; the
running count under load reaches it.

## 3. Keep servers persistent and feed them from local disk

**Symptom:** servers cold-start from a shared file system for 7 to 20
minutes; workers wait on imports or on a decoder wrapper that loads libraries
remotely.

**Mechanism:** common.md sections 2 and 3. Decode once per group of adjacent
segments (bit-identical, 2.6x less host CPU), run the decoder wrapper from a
local-disk runtime, and keep servers running across tasks instead of
restarting them per job.

**Rung:** 1 and 2.

**Parity trap:** grouped decode must be checked at every segment boundary.

## Measurement notes for this class

- Report media seconds per second on the real input mix; contiguous inputs
  overstate it by about 1.7x versus a sparse sample.
- Health checks against loaded servers time out; count completed requests,
  not probe successes.
- Start servers sequentially per host and in waves across hosts when weights
  come from a shared file system.
