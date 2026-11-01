# Playbook: small filter models

For small per-item models run inside worker processes: classifiers, audio
taggers, embedders, AI-content detectors, frame rankers, video and image
encoders. One forward per item or per window, the model is a modest fraction
of the item cost, and decode plus host work decide the throughput. Read
[common.md](common.md) first; this file adds the class-specific levers and
numbers.

## Shape of the bottleneck

The GPU is rarely the first limit. Typical stage maps are dominated by media
decode, per-item process spawn, Python scheduling, resizing, and host-to-device
staging, with the model at 20 to 50% of the item time. When the model does
dominate, it is usually launch-bound (many tiny kernels per layer) rather than
compute-bound, until the fixes below expose the real compute ceiling.

Primary metric: items (clips, windows, rows) per second per GPU and per node,
at a fixed input manifest, with outputs bitwise against the released tool
unless a precision rung is authorized. Secondary: host CPU-seconds per item,
because it sets the node-level limit once GPUs are fed.

## Triage

| Observation | Lever |
| --- | --- |
| A decoder process launched per item; cores saturated, GPUs idle | common 1, common 2 |
| Workers stall at import or at task start | common 3 |
| One worker per GPU, utilization 50 to 80%, gaps between kernels | 1 |
| Model time dominated by elementwise kernels, many launches per layer | 2 |
| Compute pipe above 80 to 85% in the kernel profile with exact outputs | common 6, then 4 |
| Host-side readback or pinned staging per batch larger than the model time | 3 |

## 1. Several worker processes per GPU under MPS

The general recipe is in common.md section 4. Class-specific numbers:

- audio tagger, fp32: from one worker per GPU with 64 threads to in-process
  decode plus four workers per GPU with 16 threads each under MPS, about 2x
  per GPU and node, bitwise. Six or eight workers per GPU lost; the daemon
  caps at 48 clients.
- frame ranker (a multimodal model used for ranking, not generation): two to
  three workers per GPU gained a few percent on the mixed input mix and about
  30% on the longest clips; a fourth worker added nothing on typical inputs
  because the tensor pipe was already saturated and the GPU was power-capped.
- video encoder: about 15% at two or three processes, bitwise.
- image detector: two workers about 30% on the stock code, almost 40% on top
  of the kernel fix below.

**Activation check:** the daemon log shows each client connect; throughput
per GPU rises with the second process. If it does not, the clients are
probably masking devices and running time-sliced.

**Parity trap:** sharding the input across processes by interleaved rows
changed the allocation history and the matmul kernel variant, giving one-ulp
differences. Shard by contiguous blocks when comparing to the single-process
reference.

## 2. Fix launch-bound encoders before anything else on the GPU

**Symptom:** a profile shows elementwise kernels at high SM occupancy and low
DRAM bandwidth, and dozens of launches per transformer layer.

**Mechanism and measured ladder on a vision transformer encoder (per window,
bitwise at every step):**

1. rotary tables rebuilt several times per layer: cache them per shape,
   more than 4x faster;
2. batch windows instead of one window per forward: about 40% less time;
3. fuse the Q, K, V projections into one matmul: a few percent;
4. one Triton kernel for the rotary chain with explicit round-to-nearest-even
   when casting to bf16: almost 20% less time.

Total close to 10x per window on the main input shape, with bitwise outputs
across all production resolutions and the short-clip tail. The same rotary
fix on an image detector backbone: about 1.6x per window, no decision flips
on the trained head, about 20% in the real CLI.

**Rung:** 3, then 4, then 5; each step measured on its own.

**Activation check:** launch count per layer in the profiler drops as
predicted and the Triton kernel appears in the timeline.

**Parity trap:** compiler fusion can contract a multiply and an add into a
fused multiply-add and change the last bit; block contraction explicitly in
the kernel. Flash-style attention was slower than the cuDNN path on the
measured GPU; hardware video decode changed pixels by a few LSB and
flipped about 2% of decisions on a trained head, so keep it opt-in.

**Counter-example:** graph compilation gained about 10% in a static benchmark
and lost in the real CLI (dozens of resolutions, partial last batches, dynamic
image size recompiles per shape). Measure compile in the real entry point.

## 3. Host-side staging around the model

**Symptom:** pinned-memory staging or readback per batch costs more than the
forward itself (observed about twice the model time per window).

**Mechanism:** large per-batch pinned buffers and synchronous readback
serialize the host; resize in the decoder output format instead of a second
library; keep the decoder from reading past the last window.

**Rung:** 1 and 2.

**Observed:** async readback and pinned staging rewrites gave zero on the
detector (run the activation check of common.md section 9 before repeating
them elsewhere); the decoder-side resize was about 5x cheaper but not
bitwise. Record such zeros so they are not retried.

## 4. Precision rungs for this class

Only after the exact ceiling is documented (common.md section 6) and with
authorization:

- fp32 matmul emulated with bf16 products (each operand split into three
  bf16 parts): about 1.4x on the audio tagger, deviation a few times 1e-6,
  below the fp32-versus-fp64 error. Trap: with a fused bias the tall GEMM hit
  a much slower legacy kernel; unfuse the bias.
- TF32 matmul: a further 10 to 15%, deviation around 1e-3; bf16 weights a few
  percent more, deviation a few times 1e-2. Both measured only; whether they
  are acceptable is the owner's call.
- Triton normalization kernels on the frame ranker: a few percent, score
  drift about 1e-2, adopted by the owner as a product decision.

Measure decision flips with the trained production head on the production
input mix; a surrogate head underestimated flips by an order of magnitude.

## Measurement notes for this class

- Short runs (under a minute) misstate multi-process layouts because workers
  start skewed; use runs of several minutes and a same-length baseline.
- CPU-seconds per item is the node ceiling: for example, 200 cores at 0.1
  CPU-seconds per item cap the node at 2,000 items/s regardless of GPU count.
- Keep the environment and the model cache on local disk; shared-file-system
  imports stalled workers for minutes on loaded nodes.
- Fingerprints of fp32 outputs are device-specific; compare on the same GPU
  model.
