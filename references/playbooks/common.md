# Playbook: shared levers

[Home](../../README.md) / [Library](../README.md) / [Playbooks](README.md)

Levers that apply to every workload class in this library: host-side work
removal, several processes per GPU under CUDA MPS, knowing the ceiling,
precision rungs, and measurement traps. Read this file together with the
class playbook chosen in [README.md](README.md) (LLM and VLM serving, small
filter models, request servers). The class files add their own levers and
numbers and refer back here by section number.

Each entry names the symptom that justifies it, the mechanism, the ladder rung
it belongs to (SKILL.md step 4), the gain range that was actually observed,
how to prove the change activated, and the parity trap. The ranges are prior
expectations for sizing headroom, not promises. Every lever still goes through
the contract, parity, and paired end-to-end gates.

## Quick triage

Before picking a lever, read the stage map and answer:

| Observation | Likely bound | First levers |
| --- | --- | --- |
| GPU utilization well below 80% while the host is busy | host work, decode, process spawn, network file system | 1, 2, 3 |
| GPU busy but with gaps between kernels; many small launches | launch overhead, single-process serialization | 4; small-filters.md lever 2 |
| Serving engine reports few running requests although the GPU has memory | admission or cache-slot cap, ingestion payloads | llm-serving.md |
| GPU busy, kernel profile shows memory-bound kernels far below HBM bandwidth, cast and copy chains, or tiny grids | kernel efficiency | [Kernel backends](../recipes/kernel-backends.md): swap a library backend, then Triton, TileLang, CuTe DSL |
| GPU at its power cap with the SM clock well below maximum under load | power-limited clocks | 7 |
| Two GPU phases of different shape share each device and phase disaggregation is proposed | phase interference or a starved phase | 8: measure per-phase GPU-seconds first |
| Compute pipe utilization above 80 to 85% in a kernel profile, or the GPU at its power cap | the exact-output ceiling | 4 at two or three processes when requests have host phases; then stop, or precision (lever 5) |
| GPU idle gaps aligned with copies or per-step host reads of device values; trace shows `cudaMemcpy`, `cudaStreamSynchronize` or `cudaDeviceSynchronize` on the host thread | synchronization inside one process | 9, then 4 |

## 1. Replace per-item process spawn with in-process decode

**Symptom:** host CPU saturated by a decoder or media tool launched once per
item; GPU idle between batches.

**Mechanism:** each spawn pays `exec` plus dynamic linking of hundreds of
shared libraries. Observed cost was on the order of 100 ms of CPU per item,
which caps a node of a couple of hundred cores below two thousand items per
second regardless of GPU count. Decoding in process through a library binding
removes that cost entirely.

**Rung:** 1, remove work.

**Observed gain:** about 2x per GPU on an audio tagging stage once combined
with lever 4; host CPU per clip halved on a video encoder.

**Activation check:** process creation rate drops to zero in the stage map;
per-item CPU time falls by the spawn cost.

**Parity trap:** keep the same resampler and sample format. Route container
formats the library does not decode identically back to the original tool so
the fast path stays bitwise.

## 2. Decode once per group of adjacent segments

**Symptom:** many segments cut from the same source file, each decoded from
scratch.

**Mechanism:** one decoder invocation over the covering range and a slice per
segment. Output is bit-identical to per-segment decoding when the segment
boundaries land on the same decoded samples.

**Rung:** 1, remove work.

**Observed gain:** 2 to 3x less host CPU on adjacent segments; segments far
apart cost about 3x more (it decodes the gaps), so group only close segments.

**Parity trap:** verify sample alignment at every segment boundary, not only on
the first segment.

## 3. Keep runtimes, libraries, and shared inputs on local disk

**Symptom:** workers stall for minutes at import or at start of each task;
stalls grow with the number of hosts.

**Mechanism:** a Python environment or a decoder wrapper that loads libraries
from a network file system pays a remote lookup per file; many workers on many
hosts reading the same index file amplify it. Copy the environment, the model
cache, and shared index inputs to local disk once per host under a lock, and
point the runtime at the copy.

**Rung:** 2, locality.

**Observed gain:** import stalls from minutes to seconds; preprocessing per
client process about 5x faster on a transcription stage (node still CPU-bound).

**Activation check:** no remote paths in the process's open file list after
warm-up; start-up time per worker is flat as hosts are added.

## 4. Run several processes per GPU under a real MPS daemon

**Symptom:** one process per GPU, GPU utilization 50 to 80%, kernels separated
by host gaps (decode, Python scheduling, serialization, a single-threaded
request scheduler). Adding threads inside the process does not help because of
the interpreter lock or thread oversubscription.

**Mechanism:** the NVIDIA Multi-Process Service lets kernels from several
processes execute concurrently on one GPU instead of time-slicing. Two to four
independent worker processes per GPU fill the host gaps of each other. The
gain is largest when the per-item host work is comparable to the GPU work.

**Rung:** 2, scheduling and overlap. It does not change arithmetic, so outputs
are bitwise when each process is deterministic on its own.

**Observed gain:** from a few percent to almost 40% at two to four processes
per GPU across video encoders, image detectors, frame rankers, speech servers,
and audio taggers; about 2x per GPU on an audio tagger when paired with lever 1
(the spawn cost had hidden the GPU idle time). Beyond three to four processes
the gain flattens or reverses.

**Observed zero:** on a GPU whose compute pipe was already above 85% busy and
at its power cap, a fourth process gained nothing on typical inputs. Two or
three processes still gained there, because they overlap the host phases
between requests; a steady-state idle figure does not show these phases.

**How to run it:**

- Start one control daemon per GPU, before any client, with the pipe and log
  directories unique per GPU, and `CUDA_VISIBLE_DEVICES` set for the daemon
  only.
- Clients must not set `CUDA_VISIBLE_DEVICES` themselves; a client that masks
  devices bypasses the daemon and runs time-sliced, which looks like "MPS does
  nothing".
- One daemon serves at most 48 clients. Count every CUDA context, including
  helper processes, and keep processes per GPU at or below six.
- Give each client process fewer host threads than before (for example 16
  instead of 64) so the processes do not oversubscribe the cores.
- Verify activation: the daemon log lists each client connection, and
  `nvidia-smi` shows the compute mode and the MPS server process.

**Costs and risks:**

- Host CPU per item rises with the number of clients (observed a quarter to
  a half more from one to several clients); budget cores.
- Memory is divided among the clients; size per-process memory fractions so
  the sum fits with headroom.
- A faulting client can take the daemon down with it; one rare MMU fault was
  observed under MPS. Keep a restart path for the daemon.
- When sharing a GPU between a serving engine and its own preprocessing
  workers, the preprocessing memory lives outside the engine's budget (see
  the llm-serving playbook, lever 4).

## 5. Precision rungs as an authorized Pareto, not a default

**Symptom:** the exact-output ceiling is reached (section 6) and the user has
authorized a quality trade-off.

**Mechanism and observed positions:**

- fp32 matmul emulated with bf16 products (each operand split into three bf16
  parts): about 1.4x on an fp32 pipeline with deviation below the
  fp32-versus-fp64 error itself; one trap: a fused bias can route a tall GEMM
  to a much slower legacy kernel, so unfuse the bias and re-measure.
- TF32 matmul: a further 10 to 15%, deviation around 1e-3.
- bf16 recurrent state in a serving engine: about 10% steady state when the
  saved memory admits more running requests.
- Hardware video decode: halves host CPU per window but changes pixels by a
  few LSB.

**Rung:** 6, reduce precision.

**Parity trap:** measure decision flips with the trained production head on
the production input mix. A surrogate head underestimated flips by an order
of magnitude. Report absolute flips and mean logit deviation, never only a
paired delta.

## 6. Know the ceiling before spending more

Stop criteria that closed campaigns honestly:

- the dominant kernel sits above 80 to 85% on its compute pipe in a kernel
  profile and the outputs must stay exact;
- the GPU is at its power cap with clocks already reduced under load, and a
  locked-clock sweep (lever 7) found no setting faster than the default;
- steady-state GPU idle is below about 5% with the exact model, and two or
  three processes per GPU (lever 4) no longer gain. The idle figure does not
  show the host phases between requests.

At that point the remaining exact-output levers are host-side (levers 1 to 3),
lever 4 if it was not measured, or precision (lever 5). Record the ceiling as
negative knowledge with the evidence.

## 7. Lock the SM clock when the GPU sits at its power cap

**Symptom:** under steady load the board draws close to its power limit and
the SM clock sits well below its maximum (for example about two thirds of
it) while the dominant kernels already run near their compute pipe in a
kernel profile.

**Mechanism:** at the cap, the driver's boost logic keeps moving clock and
voltage to hold the power target. A fixed application clock removes that
oscillation and can run more work per watt. Outputs are bitwise unchanged:
only the clock moves, not the kernels.

**Ladder rung:** 2 (environment and scheduling). It needs root on the host
and changes a shared device setting, so it is an operator decision for the
fleet, not a code change.

**How to measure:** read the supported clocks first
(`nvidia-smi -q -d SUPPORTED_CLOCKS`, `clocks.max.sm`, `power.max_limit`);
memory clocks may offer a single value. Sweep a few locked SM clocks
(`nvidia-smi -i <gpu> -lgc <f>,<f>`) against the default in a Latin square
over GPUs, with `nvidia-smi dmon -s pc` logging power and clock for every
arm. Reset with `-rgc` at the end, including on failure.

**Observed range:** no gain on a power-capped video frame ranker (three
workers per GPU, a Latin square over GPUs): several locked clocks stayed
within about 1% of the default, inside the GPU-to-GPU spread. Under load
every arm drew close to the power limit and ran at the same reduced clock;
the power-capping throttle overrode every lock above that clock. A lock can
only pay where the workload leaves power headroom, for example memory-bound or
bursty phases. Check the effective clock under load before reading throughput.

**Activation check:** the effective clock under load (`dmon` `pclk`) equals
the lock. If it sits below the lock and the power-capping throttle reason
counts up, the lock is not active and the arm only measures noise.

**Parity trap:** none for outputs. Throughput can drop when the lock is set
above what the power limit sustains (the driver throttles anyway) or below
the clock the memory-bound phases need. Per-GPU spread of a few percent is
normal, so compare on the same GPUs.

## 8. Disaggregate phases only when a phase is starved, not when the GPU is saturated

**Symptom:** a request runs two GPU phases with different shapes (for example
a vision encoder and a language-model prefill) on the same device, and the
idea is to give each phase its own GPUs, as in prefill/decode disaggregation.

**Mechanism:** separation pays only when co-location wastes something: one
phase idles the GPU, the phases need conflicting batch sizes or parallelism,
or one phase blocks the other's latency. When several processes per GPU
already keep the device saturated, splitting adds transport and leaves the
total GPU time per item unchanged or worse.

**Ladder rung:** 2 (topology). Outputs can stay bitwise if the moved
boundary tensor keeps its producer's memory layout (see the parity trap).

**How to measure first:** run each phase alone on a GPU fed by surplus
capacity of the other phase, and record items per second for each: `r_a` and
`r_b`. The sum `1/r_a + 1/r_b` is the GPU-seconds per item of any split;
compare it with `1/r_co` of the co-located GPU. The best fractional split
gives `N / (1/r_a + 1/r_b)` for N GPUs, and integer splits are lower. If the
sum is not below `1/r_co`, no split can win: stop before testing layouts.

**Observed range:** a power-capped video frame ranker (vision encoder about a
third of GPU time, three processes per GPU under MPS): the per-phase sum was a
few percent above `1/r_co`; the best fractional split lost a few percent, and a
measured integer split about 30% on the mixed input, more on the longest ones.

**Parity trap:** a GEMM library may pick a different algorithm for a
contiguous input than for a permuted view with the same values. After the
boundary tensor went through a file it was contiguous, and nearly half of the
projected features differed by one ulp until the consumer rebuilt the
producer's strides. Compare the boundary tensor and the first op after it
bitwise in isolation before blaming the network or weights.

## 9. Make host-device transfers non-blocking and remove implicit synchronization

**Symptom:** in the timeline the GPU idles at copy boundaries while the host
thread sits in `cudaMemcpy`, `cudaStreamSynchronize`, or
`cudaDeviceSynchronize`. The sources are a pageable `.to(device)`, and
host reads of device values inside the hot loop: `.item()`, `.cpu()`,
`.numpy()`, `.tolist()`, `float(t)`, `print`, `assert`, or `if` on a tensor,
a loss or metric logged every step, a data-dependent shape (`nonzero`,
masked indexing, `unique`). Each one drains the queued GPU work before the
host can continue, so launches never run ahead of execution.

**Mechanism:** pin the host buffers (`pin_memory=True` in the loader or
`tensor.pin_memory()` once for a reused staging buffer) and copy with
`non_blocking=True`, so the copy is a `cudaMemcpyAsync` on the stream and the
host returns immediately. Move host reads out of the loop: accumulate
losses and counters on device and read them every N steps or at the end;
return device tensors from the forward and read them once per batch after
the postprocessing kernels are queued. Where ordering is really required,
record a `torch.cuda.Event` and wait on it instead of synchronizing the
whole device. Run readback on a side stream only when the readback itself
is the exposed cost. Do this lever before lever 4: each synchronization is a
gap that MPS would merely hide behind another process, and a process that
no longer drains the queue needs fewer neighbours.

**Rung:** 2, scheduling and overlap. The arithmetic does not change, so
outputs are bitwise when the ordering rules in the parity trap hold.

**Observed gain:** preliminary. Expect a gain only when the sum of host
blocking time per item is material in the timeline (tens of percent of the
step for small models with per-step logging; a few percent or nothing for
large forwards). **Observed zero:** an async readback and pinned staging
rewrite on a small detector (small-filters.md lever 3), where the model
time was already overlapped with decode in other workers. Size the headroom
from the measured blocking time, not from the number of calls.

**Activation check:** each of these must change, otherwise the arm did
nothing:

- `torch.cuda.set_sync_debug_mode("warn")` around the hot loop (not in
  production): count the warnings per item before and after; the remaining
  ones must be the intentional reads.
- In the torch profiler or Nsight Systems trace, `cudaMemcpy` on the host
  thread becomes `cudaMemcpyAsync`, and the number of `cudaStreamSynchronize`
  plus `cudaDeviceSynchronize` calls per item drops to the intended count.
- `tensor.is_pinned()` is true for every source of a `non_blocking` copy.
- `CUDA_LAUNCH_BLOCKING` is unset in the measured process.
- The launch queue runs ahead: in the trace the gap between a kernel's
  launch and its execution grows instead of staying near zero.

**Parity trap:** two silent ones. `non_blocking=True` from pageable memory is
executed synchronously by the driver: no error, no gain, so the activation
check above is the only way to see it. A `non_blocking` device-to-host copy
into a pinned buffer that is read before the stream reaches it, or a pinned
host source that is overwritten by the next batch before the copy completes,
yields wrong values with no error; require an event or stream
synchronization before every host read of such a buffer and before reusing
it. Keep a bitwise comparison of the final outputs against the synchronous
control in the gate list.

## Measurement traps seen in practice

- Removing a synchronization moves time between stage timers: the stage
  that held the sync looked slow because it absorbed all queued GPU work, and
  after the change the earlier stages look slower. Only end-to-end wall
  clock and the trace are authoritative for a synchronization lever.
- Sharding a benchmark across processes by interleaved rows changed the
  allocation history and therefore the matmul kernel variant, producing
  one-ulp differences that are not the candidate's fault. Shard by contiguous
  blocks.
- Short runs overstate or understate per-process layouts because workers
  start skewed. Compare against a same-length baseline and report a steady
  window (for example the 25 to 75% quantile of the run) for serving. Trust
  that window only with a ramped client start and a run that replaces the
  in-flight requests several times; otherwise prefer an engine-side rate.
- A full-run average for a serving benchmark can lose over a third of its
  wall time to ramp and tail; report steady state and the full run separately.
- Health probes to a loaded engine time out and under-report healthy
  replicas; do not treat probe failures as replica failures.
- A benchmark that masks devices per client while a daemon is running is
  measuring time-slicing, not MPS.
- One engine process that fails on startup (for example a port collision)
  can kill its whole process group. Start every long-lived server in its own
  session, and never use a bare `wait` in a driver that also starts servers:
  it waits for the servers too.
