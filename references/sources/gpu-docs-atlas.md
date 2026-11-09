# GPU documentation atlas

Reviewed navigation and selected passages on 2026-11-09. This is a route to
primary sources, not a claim that every manual was read or every API tested.
[Machine-readable link graph](gpu-doc-links.jsonl) preserves page URLs, retrieval
status, text hashes and outbound links. Search it by symbol or subsystem; do not
load the whole graph into model context. Discovered outbound links are leads,
not independently verified targets. `latest` and `main` must be matched to the
installed version and target SM before implementation.

## Start from the question

| Question | Primary source | Evidence to take back to the experiment |
| --- | --- | --- |
| Where is the end-to-end gap? | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | Actual process/phase, CPU launch gaps, copies, sync and overlap |
| What limits this kernel? | [Nsight Compute](https://docs.nvidia.com/nsight-compute/index.html) | Relevant pipes, bandwidth hierarchy, stalls, replay effects |
| Is the kernel memory/synchronization-safe? | [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | memcheck/racecheck/initcheck/synccheck coverage and remaining gaps |
| Need activity/callback/profiling integration? | [CUPTI](https://docs.nvidia.com/cupti/index.html) | Supported API, correlation, buffer losses and overhead |
| What did the compiler really emit? | [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | Actual cubin/SASS, resources and instruction sequence, not PTX alone |
| What are CUDA semantics and constraints? | [Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html), [Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | Memory, execution and correctness contract for the device |
| Need kernel fusion or a portable tiled implementation? | [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), [language API](https://triton-lang.org/main/python-api/triton.language.html), [semantics](https://triton-lang.org/main/python-api/triton-semantics.html) | Masking, broadcasting, reductions, casts, layout and launch contract |
| Need explicit lower-level Triton layout control? | [Gluon](https://triton-lang.org/main/gluon/index.html), [tutorials](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html) | A matching operation and hardware example; compare complexity and codegen |
| Triton compiler/numerics diagnosis? | [MLIR dialects](https://triton-lang.org/main/dialects/dialects.html), [debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html), [FpSan](https://triton-lang.org/main/programming-guide/chapter-3/fpsan.html) | Which lowering or numerical transformation explains the failure |
| Internal Triton phase timing? | [Proton intra-kernel](https://github.com/triton-lang/triton/blob/main/third_party/proton/docs/intra-kernel.md) | Trace scopes, buffer/sampling semantics, overhead and exact revision |
| Tensor-core layout and pipeline control? | [CUTLASS Python / CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) | Versioned layout, copy, MMA, pipeline and hardware support |
| Tile-level GEMM/attention/recurrence? | [TileLang](https://tilelang.com/) | Appropriate example, target support, lowering and autotune limits |
| CUDA C++ tiles and warp-specialized kernels? | [ThunderKittens upstream](https://github.com/HazyResearch/ThunderKittens) | Actual supported kernels, bindings, stream and layout assumptions |
| JAX custom kernels? | [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), [OpenXLA](https://openxla.org/xla) | Backend-specific semantics and compiler boundary; not a PyTorch drop-in |
| Compiler IR and scheduling research? | [TVM](https://tvm.apache.org/docs/), [MLIR](https://mlir.llvm.org/docs/) | Dialect/lowering/schedule corresponding to the current stack |
| Simulation or geometry GPU code? | [NVIDIA Warp](https://github.com/NVIDIA/warp) | Differentiation and data model appropriate to the workload |
| AMD implementation/port? | [ROCm](https://rocm.docs.amd.com/en/latest/) | Supported device, HIP/compiler/library/profiler documentation; no assumed PTX equivalence |
| Reusable attention/decode kernels? | [FlashInfer](https://docs.flashinfer.ai/), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Shapes, masks, KV layout, precision, architecture and stream contract |
| Training fusions? | [Liger Kernel](https://github.com/linkedin/Liger-Kernel) | Forward/backward, state and loss parity, supported model integration |
| GEMM/convolution before custom code? | [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html), [cuDNN frontend](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | Layout, accumulation, scales, epilogue and workspace options |
| Engine compilation/export? | [Torch-TensorRT](https://docs.pytorch.org/TensorRT/), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/), [Model Navigator](https://github.com/triton-inference-server/model_navigator) | Partitions/fallback, build cost, shape profiles, accuracy and reload |
| General model serving? | [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) | Batching, instances, backend, queue and client-side latency |
| LLM serving? | [vLLM](https://docs.vllm.ai/en/latest/), [SGLang](https://docs.sglang.ai/), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) | Prefill/decode, KV memory, scheduling, concurrency and backend activation |
| CPU/heterogeneous runtime? | [ONNX Runtime performance](https://onnxruntime.ai/docs/performance/), [OpenVINO](https://docs.openvino.ai/) | Execution-provider partitions, CPU threading/NUMA, conversion overhead |
| Collectives and topology? | [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) | Payload, topology, rank skew, overlap and failure behavior |

Additional primary implementations: [QuACK](https://github.com/Dao-AILab/quack)
for CuTe-based kernels; [CCCL](https://nvidia.github.io/cccl/unstable/) for
CUB/Thrust/libcu++ primitives (the linked branch is explicitly unstable);
[Mojo manual](https://docs.modular.com/mojo/manual/) for the language/runtime.
The Modular [Blackwell hardware walkthrough](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul)
and [optimization continuation](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance)
are primary engineering accounts, not ISA specifications.

## CuTe DSL: go directly to the missing layer

- Foundations: [introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html),
  [dynamic layouts](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.html),
  [control flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html),
  [CuTe API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html).
- MMA families: [WMMA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/wmma_programming.html),
  [WGMMA](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/wgmma_programming.html),
  [tcgen05](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html).
  Distinguish SM generations and consumer/data-center Blackwell targets.
- Movement and ordering: [cp.async API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html),
  [pipeline API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html),
  [SM100 utilities](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html).
- Diagnosis: [debugging](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html),
  [IKET intra-kernel profiling](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html),
  [autotuning](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/autotuning_gemm.html).
- Delivery: [framework integration](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/framework_integration.html),
  [JIT caching](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html),
  [AOT](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/ahead_of_time_compilation.html),
  [TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/tvm_ffi_compilation.html).
- Advanced composition: [task scheduling](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html),
  [validation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_validation.html),
  [limitations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html).

## PTX: read semantics before emitting instructions

Use the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
with its instruction target/version notes. PTX is a virtual ISA; final SASS can
change after ptxas/driver JIT. Record both compiler and driver.

- [Memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model): scopes, ordering and async proxies.
- [cp.async.bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk): transfer direction, alignment and completion mechanism.
- [mbarrier completion](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier): arrivals versus transaction counts and phase.
- [Warpgroup MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions): collective participation, commit/wait and operand lifetime.
- [tcgen05 shapes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-data-movement-shape): supported formats and target-specific operations.
- [setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg): legal operands and participation, not a universal occupancy trick.
- [clock](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock) and
  [globaltimer](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer): counter semantics before cross-warp/SM timeline comparisons.

## Expand the collection deliberately

Start with the exact symbol or missing mechanism. Follow a source's docs and code
links, then official examples, release notes, tests and regression issues. Resolve
one level at a time; deduplicate URLs while retaining section anchors. Verify
redirects, case-sensitive paths and substantive bodies, not HTTP status alone.
A docs index is coverage of navigation, not coverage of all underlying behavior.

For fresh APIs, locate matching source/test implementations and pinned examples.
For blog claims, keep the hardware, shape, dtype, comparator and measurement
boundary; preserve unsuccessful alternatives. Write a transfer test and a stopping
condition, then return to the measured workload. Use
[deep-reading routes](gpu-authors-and-depth.md) for author discovery and
[intra-kernel probes](../recipes/intra-kernel-and-cute.md) for applying evidence.
