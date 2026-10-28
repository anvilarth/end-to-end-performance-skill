# Choosing a kernel path: swap, Triton, TileLang, CuTe DSL

Read this at ladder rung 5, after a kernel profile has named the kernels and
their gap to the ceiling. Choose the path from what the profile shows, not
from which language is most familiar. Every path ends in the same gates:
production shapes and dtypes, numerical parity against the control op, then
the paired end-to-end comparison.

## Size the budget first

For each candidate kernel, from the Nsight Compute report:

- achieved memory throughput against peak HBM bandwidth, and achieved
  tensor or FMA pipe utilization against its peak;
- grid size, block size, and occupancy; one-warp blocks and grids smaller than
  the SM count are a common sign that a kernel was written for a different
  shape;
- launch count per step and the share of the step spent in that kernel class.

`recoverable e2e budget = share of e2e time x (1 - achieved / ceiling)`, for
the ceiling that binds. Sum it per kernel class (casts, norms, quantization,
attention, recurrence, GEMM). A class with a budget below the promotion
threshold is not worth a new kernel; a kernel already above 80 to 85% of its
binding ceiling is done.

## Decision table

| Profile evidence | First path | Escalate to |
| --- | --- | --- |
| The framework, engine, or a kernel library already has a faster backend for this op (often behind a flag, dispatcher, or stale guard) | swap it in | none, unless the swapped kernel is still far from its ceiling |
| Chains of elementwise casts, copies, activations, rotary, norms at a fraction of HBM bandwidth; many small launches | compiler fusion, then Triton | none in most cases |
| A quantize, norm, or activation between two GEMMs, round-tripping through memory | fuse it into the GEMM epilogue or prologue of the kernel that already exists | CuTe DSL if the GEMM is a block-scaled or architecture-specific kernel |
| Attention, linear-attention, or recurrent-state kernel with an unusual head dim, state layout, or masking, and no fast library path; memory-bound at low bandwidth or tensor pipe low | TileLang | CuTe DSL if TileLang cannot reach the ceiling on the target architecture |
| GEMM or attention that needs the newest tensor-core path: architecture-specific MMA instructions, tensor memory, cluster or 2-CTA MMA, block-scaled low-precision formats, warp-specialized producer and consumer pipelines with explicit barriers | CuTe DSL | CUDA C++ with CUTLASS |
| Many tiny launches of an already-fast kernel | CUDA graphs, batching, or fusing neighbours | not a rewrite |

## Paths

### Swap an existing implementation

The cheapest kernel win. Engines and libraries ship several backends per op
and default to the most conservative one; dispatchers sometimes reject a newer
backend through a guard written before it existed. List the backends for each
hot op, read the guard, enable the alternative, and run the parity gate. Ask
the upstream library before writing anything.

### Triton

Use for memory-bound fusion: elementwise chains, dtype round trips, rotary,
RMS and layer norm, softmax variants, reductions, gather and scatter, small
custom epilogues, and simple GEMMs where the library GEMM is not the
bottleneck. Iteration is fast, and the autotuner covers block sizes.

Traps:

- autotuned configs chosen on one shape can leave a decode-sized problem with
  tiny grids or one-warp blocks; tune on production shapes, including decode
  batch sizes;
- rounding differs from the eager op unless you match it explicitly, for
  example round-to-nearest-even on the final cast, when bitwise parity is the
  contract;
- Triton GEMM and attention on the newest architectures can trail
  vendor-tuned kernels; when a Triton GEMM or attention kernel loses to the
  library or the unfused path, move down a row instead of tuning further.

### TileLang

A tile-level DSL: you write the tiling, the shared-memory staging, and the
pipeline in Python, and the compiler handles layout, swizzle, and asynchronous
copies. Use it when Triton's code generation is the limit but you do not need
register-level control: GEMM with custom epilogues, dequantizing or
mixed-precision GEMM, attention variants (unusual head dims, sparse or
blockwise masks, latent attention), linear-attention and chunked recurrences.

Traps:

- support for each GPU architecture's newest features lands over time; check
  on the target hardware that the generated kernel uses the expected tensor
  core path before comparing numbers;
- a TileLang kernel that is fast in isolation can still lose end to end when it
  is launched hundreds of times per step; count launches in the timeline.

### CuTe DSL

The Python front end to the CUTLASS CuTe layout and copy abstractions, with
direct access to the architecture's MMA, tensor-memory, async-copy, and
barrier primitives. Use it when the ceiling needs those features: peak tensor
core throughput on the newest architecture, block-scaled low-precision GEMM,
flash-style attention, warp-specialized persistent kernels, and fused
epilogues on such GEMMs. Several current attention and recurrence kernels in
serving engines are written in it, which also makes it the usual source of a
backend swap.

Traps:

- the cost is high: plan for a parity harness and an architecture-specific
  tuning pass, and keep the previous kernel as the switchable control;
- architecture-specific kernels do not run on older GPUs; keep a fallback and
  state the transfer boundary.

### CUDA C++ and CUTLASS C++

Use when the kernel must live inside an existing C++ extension, or when the
DSLs cannot express it (unusual warp-level or inter-kernel synchronization).
Highest cost; the same gates apply.

## Observed cases

- **Swap through a stale guard.** A linear-attention decode recurrence ran in a
  Triton kernel at 2.2 to 2.7 TB/s with one-warp blocks, about 16% of decode
  kernel time. The engine already shipped a CuTe DSL version but its dispatcher
  accepted only the Triton kernel. Relaxing the guard gave +4 to +7% end to
  end on two nodes with judged-equivalent quality.
- **Memory-bound tail in prefill.** A kernel profile of a low-precision VLM
  prefill showed about a quarter of kernel time in elementwise casts,
  activation quantization, a short convolution, and activations, running at 1
  to 2 TB/s of an 8 TB/s ceiling. Estimated recoverable budget: 12 to 15% end
  to end. The levers are cast removal and Triton fusion, and fusing the
  activation quantization into the neighbouring GEMM epilogue. Pursue such
  a tail before declaring the lossless ceiling reached.
- **Unusual head dim.** For attention with head dim 512 the library paths were
  slower or had no kernel. A Triton kernel passed parity but lost to the
  unfused path, 17 + 71 ms against 30 + 39 ms. That is the signal to move to
  TileLang or CuTe DSL rather than tune the Triton kernel further.
- **Bitwise rotary in Triton.** One Triton kernel for the rotary chain with
  explicit round-to-nearest-even replaced an eager chain bitwise; see
  [playbooks/small-filters.md](playbooks/small-filters.md).
- **Launch count, not code.** A pair of small TileLang kernels was launched
  about 360 times per step and took about 7% of the step. That is a fusion or
  graph-capture problem; a faster kernel body would recover little.
