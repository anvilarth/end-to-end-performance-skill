# GPU MODE: use resources at the point of need

Reviewed 2026-11-04. GPU MODE is an organization, not one repository. Follow current
primary documentation for APIs; these resources are optional, not dependencies.
The [organization map](gpu-mode-map.md) covers all public repositories in the reviewed
snapshot. [Advanced recipes](advanced-recipes.md) turn the deeper material into
conditional experiments; reading every resource is not a prerequisite for optimization.

| Need | Source | Useful transfer into this workflow |
| --- | --- | --- |
| Existing implementation | [triton-index](https://github.com/gpu-mode/triton-index) | Search by measured operator; inspect the upstream implementation and its supported contract before custom coding. |
| Reference and shape specification | [reference-kernels](https://github.com/gpu-mode/reference-kernels) | Separate reference implementation, input/output schema and test/benchmark configurations. Adapt the structure, not leaderboard shapes or tolerances. |
| Reliable narrow capture | [NCU guide](https://github.com/gpu-mode/reference-kernels/blob/main/docs/ncu-profiling.md) | Warm up outside capture; mark the actual candidate; keep reference checks outside; verify kernel counters and selected shape. |
| Low-latency kernel harness | [pygpubench](https://github.com/gpu-mode/pygpubench) | Independent input/reference generation and explicit tolerances; process isolation when candidate code may contaminate a benchmark. |
| Targeted background | [lectures](https://github.com/gpu-mode/lectures) | Use the index for profiling, CUDA checklist, NCCL, fusion, Triton and numerics only when that topic answers the current bottleneck question. |

The NCU guide's exact `custom_kernel` NVTX name and hosted runner settings are its
own protocol; local projects can use their own consistent ranges. Its hosted path
is single-GPU NVIDIA evidence, not distributed E2E validation. A nonempty report
with the intended kernel and actual counters matters more than process exit status.
Do not run PyTorch profiler inside an NCU capture path. NCU replay timing is not
normal execution latency.

PyGPUBench targets adversarial short-kernel benchmarking. Its isolation is useful
for generated/untrusted implementations, but is not a reason to install an elaborate
harness for every trusted application edit. Kernel cache-clearing policies are not
necessarily representative of application caches. Keep local benchmark and actual
application timing separate. No leaderboard score replaces full-path parity or
quality acceptance.

These are workflow adaptations, not GPU MODE endorsements or measured improvements
to our workloads. Before copying code, check its current license; documentation links
do not grant permission to transplant every implementation.

The inference-hackathon harness also illustrates why metric code needs inspection:
its reviewed `run_throughput.py` sums prompt and completion counts for
`throughput_tok_per_sec`, while its opening description says output tokens/sec.
Neither metric is inherently wrong; use the one declared by the workload contract
and label it accurately. Source revision is recorded in the organization map.

## Additional candidates after kernel attribution

For general model/pipeline symptoms first use
[bottleneck recipes](bottleneck-recipes.md). The following candidates require a
material kernel budget. Try an available backend implementation before custom code.

| Measured bottleneck | Candidate and source | Decisive check / limit |
| --- | --- | --- |
| Long-context decode attention has too little parallel work at small batch | [Split-KV / Flash-Decoding](https://github.com/gpu-mode/triton-index/blob/main/kernels/0002_Flash-Decoding_Split-KV_Attention.md), [primary explanation](https://pytorch.org/blog/flash-decoding/) | Splitting over KV length adds parallel work and a reduction. Test real context/batch/head shapes, masks and numerics, plus reduction/workspace cost. Not a blanket prefill optimization; do not enable quantized KV merely because the implementation supports it. |
| Many small expert GEMMs, launch overhead or poor scheduling in MoE | [Persistent grouped GEMM tutorial](https://github.com/gpu-mode/triton-tutorials/tree/main/lesson_5/triton_contiguous_group_gemm) | Measure tokens per expert and imbalance; include routing, sorting/packing, padding and output restoration. Tutorial expects aligned contiguous expert groups, common N/K and BF16 output; it is not arbitrary ragged MoE or an FP32-preserving drop-in. |
| Separate residual addition and normalization repeatedly move the same activations | [Residual + RMSNorm/LayerNorm](https://github.com/gpu-mode/triton-index/blob/main/kernels/0001_Triton_RMSNorm%2BResidual.md) | Verify epsilon, normalization axis, residual precision, aliases, returned residual and backward. Reject when the current backend already fuses it or conversions remove the benefit. |
| GEMM/kernel traffic is inefficient after accounting for available parallelism | [Tiling and memory access examples](https://github.com/gpu-mode/lectures/tree/main/lecture_008), [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | Measure strides, actual traffic/cache reuse and tails. Consider tile/program ordering for L2 reuse and a bounded configuration search. Include layout conversion costs. A larger tile, more warps, more stages or higher occupancy is not an objective by itself. |

For the last branch record GPU, input shape/stride/dtype, compiler version and selected
configuration. Tune only supported production-relevant shapes; include compile/tune
startup in the cold budget. Test odd/tail shapes and required views before dispatching
to a specialized path. Use [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
for occupancy/resource trade-offs, not a fixed block-size prescription.

## Transfer tutorial ideas, not unverified benchmark claims

Read the operation and wrapper before using a tutorial as an A/B baseline. Concrete
examples observed in the reviewed GPU MODE code:

- [coalesce.cu](https://github.com/gpu-mode/lectures/blob/77a8df418834e5789c12da23e7d2719e0efabef1/lecture_008/coalesce.cu)
  compares `in[(index * 2) % n]` with `in[index]`: different output mappings. This
  illustrates memory access, not an equivalent faster copy. Preserve the actual
  mapping when building a correctness/performance comparison.
- [occupancy.cu](https://github.com/gpu-mode/lectures/blob/77a8df418834e5789c12da23e7d2719e0efabef1/lecture_008/occupancy.cu)
  fixes block size to 1024 and comments out the occupancy API call. Its printed
  recommendation is not a measured optimum on the current GPU.
- [tiling.cu](https://github.com/gpu-mode/lectures/blob/77a8df418834e5789c12da23e7d2719e0efabef1/lecture_008/tiling.cu)
  uses `width / TILE_WIDTH` and unmasked tile loads. Do not generalize its happy-path
  shape to non-divisible dimensions; add correct tail handling or a guarded fallback.
- [fusion benchmark](https://github.com/gpu-mode/lectures/blob/77a8df418834e5789c12da23e7d2719e0efabef1/lecture_018/kernels/triton_fused_add_mul_activation.py)
  mutates an input in the Triton arm while the PyTorch arm returns a new tensor.
  Repeated calls therefore have different state evolution and allocation semantics.
  Define allowed mutation, reset matched inputs and account for required copies
  before treating its timings as a comparison for another application.

These are static code observations, not claims of measured regressions. Teaching
examples remain useful even when their simplifying assumptions exclude production.
Do not copy their fast-math settings or replace an activation with an approximation
under an exact contract. Restore mutated inputs after warmup/autotuning as well as
between paired arms; timing setup outside the kernel does not remove it from E2E cost.

Repository heads observed during this review (2026-11-04):

- `gpu-mode/lectures`: `77a8df418834e5789c12da23e7d2719e0efabef1`
- `gpu-mode/triton-tutorials`: `dc1532fc128377fc108146e78b4b49439d4bef45`
- `gpu-mode/triton-index`: `6b301ab902d835766a2e9c2914799b514a41fef5`

These versions establish review provenance, not a requirement to install old code.
