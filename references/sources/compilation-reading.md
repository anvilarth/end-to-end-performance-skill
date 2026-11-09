# Compilation collection audit

Reviewed on 2026-11-09. All **21/21** original Dia Compilation tabs were retrieved, using verified case-corrected replacements where necessary. Tabs were left open. This audit distinguishes retrieval, reviewed mechanisms and local validation; no GPU benchmark was run.

The [manifest](compilation-manifest.json) retains the original URLs and replacements. The [link graph](gpu-doc-links.jsonl) includes official documentation, blog navigation and implementation leads without copying source bodies.

## Original tabs

| ID | Source | Extracted route / qualification |
| --- | --- | --- |
| C01 | [GitHub - mit-han-lab/KernelWiki · GitHub ](https://github.com/mit-han-lab/kernelwiki/tree/master) | Implementation discovery; inspect code, license, revision and measured scope before reusing kernels. |
| C02 | [maharshi on X: "Beware of these CuTeDSL gotchas" / X ](https://x.com/maharshii/status/2082921705941651595) | CuTe failure probes: recast aliasing, iterator offsets, value state, unrolling, arrivals and register allocation. Cross-check anecdotes against the pipeline API. |
| C03 | [Blog – Maharshi's blog ](https://maharshi.bearblog.dev/blog/) | Followed the complete blog index; CUDA worklogs and identity tensors routed below, non-performance essays kept outside operational rules. |
| C04 | [Inside ThunderKittens' Python Bindings ](https://www.enbao.me/posts/tk) | Binding boundary: tensor wrappers, argument unpacking, device and CUDA stream propagation; kernel-only timing omits Python/FFI costs. |
| C05 | [Understanding the CUDA Compiler & PTX with a Top-K Kernel / Alpin's Blog ](https://blog.alpindale.net/posts/top_k_cuda/) | Selection instead of full sorting; inspect unroll/codegen, but validate exact Top-K semantics and data distribution first. |
| C06 | [Implementing a fast Tensor Core matmul on the Ada Architecture / spatters.ca ](https://www.spatters.ca/mma-matmul) | Ada GEMM progression: tiling, shared-memory permutation, registers and async pipeline. One shape and accumulation mode do not generalize. |
| C07 | [Ultimate guide to optimizing Stable Diffusion XL - Félix Sanz ](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl) | SDXL deployment options; old package/backend guidance requires current verification. Separate offload, compilation and approximate changes. |
| C08 | [tutorials/Conceptual_Guide/Part_4-inference_acceleration/README.md at main · triton-inference-server/tutorials · GitHub ](https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_4-inference_acceleration/README.md) | Backend and hardware selection with partition/fallback costs; a server accelerator setting does not prove full-request speed. |
| C09 | [Triton Inference Server · GitHub ](https://github.com/triton-inference-server) | Organization discovery entry; linked tools and server documentation belong to serving, not Triton language. |
| C10 | [Compiling FLUX.1-dev model using the Torch-TensorRT dynamo backend — Torch-TensorRT ](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_export_flux_dev.html#sphx-glr-tutorials-rendered-examples-dynamo-torch-export-flux-dev-py) | FLUX backbone export: shape constraints, singleton specialization, precision and engine build memory; validate complete generation and reload. |
| C11 | [Compiling Stable Diffusion model using the torch.compile backend — Torch-TensorRT ](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_stable_diffusion.html) | Stable Diffusion compilation entrypoint; preserve the exact compiled component and first-call versus warm-run boundary. |
| C12 | [Torch-TensorRT Tutorials — Torch-TensorRT v2.12.0.dev0+c0dfd5d documentation ](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/index.html) | Tutorial navigation: use the closest model/export mode and check version; an index is not a benchmark. |
| C13 | [GitHub - pytorch/TensorRT: PyTorch/TorchScript/FX compiler for NVIDIA GPUs using TensorRT · GitHub ](https://github.com/pytorch/tensorrt) | Upstream implementation and supported integration; corrected case-sensitive documentation paths. |
| C14 | [GitHub - triton-inference-server/model_navigator: Triton Model Navigator is an inference toolkit designed for optimizing and deploying Deep Learning models with a focus on NVIDIA GPUs. · GitHub ](https://github.com/triton-inference-server/model_navigator) | Export/runtime configuration search and correctness comparison; benchmark representative requests rather than only optimizer samples. |
| C15 | [Torch-TensorRT — Torch-TensorRT ](https://docs.pytorch.org/TensorRT/) | Official Torch-TensorRT entrypoint, separate from TensorRT-LLM and Triton Inference Server. |
| C16 | [Computing sharding with einsum : ezyang's blog ](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/) | Einsum placements: contracted sharded dimensions yield pending reductions; derive backward and communication as well as forward. |
| C17 | [Looking inside your kernels - Aryan V S ](https://a-r-r-o-w.github.io/blog/3_blossom/00005_looking_inside_your_kernels/index.html) | Intra-kernel phase/warp trace; extracted every documentation and Proton implementation link. Treat cache and timer assumptions as architecture-specific. |
| C18 | [Quick start - Triton Model Navigator ](https://triton-inference-server.github.io/model_navigator/0.13.1/quick_start/#optimize-model) | Versioned Model Navigator 0.13.1 quick start retained as such; do not silently substitute its APIs into a current package. |
| C19 | [Installation — Torch-TensorRT ](https://docs.pytorch.org/TensorRT/getting_started/installation.html) | Installation compatibility route; match PyTorch, CUDA, TensorRT and target platform instead of blindly installing latest. |
| C20 | [Optimizing diffusion for production-ready speeds - IV - Aryan V S ](https://a-r-r-o-w.github.io/blog/3_blossom/00004_productionizing_diffusion-4/index.html) | Followed all four diffusion posts: exact fusion/precomputation, streams, precision/caching trade-offs, LoRA and delivery. |
| C21 | [State of torch.compile for training (August 2025) : ezyang's blog ](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/) | Dated compiler training report: guards, graph breaks, regional compilation, rounding and distributed integration; verify present support. |

## Every docs / Proton implementation link from the kernel-tracing article

These 11 distinct destinations include documentation and implementation references. Repeated links in the article are deduplicated; the source page’s remaining code/paper links remain in the link graph.

- [ScopeIdAllocation.cpp](https://github.com/triton-lang/triton/blob/main/third_party/proton/Dialect/lib/Analysis/ScopeIdAllocation.cpp)
- [ProtonToProtonGPUPass.cpp](https://github.com/triton-lang/triton/blob/main/third_party/proton/Dialect/lib/ProtonToProtonGPU/ProtonToProtonGPUPass.cpp)
- [NvidiaPatternProtonGPUOpToLLVM.cpp](https://github.com/triton-lang/triton/blob/main/third_party/proton/Dialect/lib/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.cpp)
- [intra-kernel.md](https://github.com/triton-lang/triton/blob/main/third_party/proton/docs/intra-kernel.md)
- [ProtonGPUOps.td](https://github.com/triton-lang/triton/blob/main/third_party/proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUOps.td)
- [Utility.cpp](https://github.com/triton-lang/triton/blob/main/third_party/proton/Dialect/lib/ProtonGPUToLLVM/Utility.cpp)
- [CircularLayoutParser.cpp](https://github.com/triton-lang/triton/blob/main/third_party/proton/common/lib/TraceDataIO/CircularLayoutParser.cpp)
- [Dialect](https://github.com/triton-lang/triton/tree/main/third_party/proton/Dialect)
- [index.html](https://docs.nvidia.com/cupti/index.html)
- [index.html](https://docs.nvidia.com/nsight-compute/index.html)
- [index.html#special-registers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers)

## Veitner: complete index, grouped for use

Fetched all **66/66** post bodies listed by the blog index. Reviewed introductions, conclusions and relevant mechanisms; this is not a line-by-line correctness audit of every kernel. Category-theory and background posts support understanding; unrelated essays do not become performance rules. The recipes and study route contain the operational synthesis.

| Article | Route |
| --- | --- |
| [Radix Top-K](https://veitner.bearblog.dev/radix-top-k/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Tile Scheduling in CuTeDSL](https://veitner.bearblog.dev/tile-scheduling-cute-dsl/) | Memory, scheduling, pipelining and profiling |
| [SBO and LBO explained visually](https://veitner.bearblog.dev/sbo-and-lbo-explained-visually/) | Layouts, addressing and correctness |
| [Simple math to speed up GDN prefill](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Chunkwise Gated Delta Rule](https://veitner.bearblog.dev/chunkwise-gated-delta-rule/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Gated Delta Net Decoding](https://veitner.bearblog.dev/gated-delta-net-decoding/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Grouped Blockscaled Gemm - Kernel](https://veitner.bearblog.dev/grouped-blockscaled-gemm-kernel/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Grouped Blockscaled Gemm - Host code](https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Grouped Block scaled Gemm - Intro](https://veitner.bearblog.dev/grouped-block-scaled-gemm-intro/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Warp Specialisation in CuTeDSL](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) | Memory, scheduling, pipelining and profiling |
| [2 CTA GEMM on B200](https://veitner.bearblog.dev/2-cta-gemm-on-b200/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) | Memory, scheduling, pipelining and profiling |
| [B200 Blockscaled GEMM - The setup](https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Scale Tensor construction in CuTeDSL](https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Demystifying numeric conversions in CuTeDSL](https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/) | GEMM / MMA / low precision; match architecture and accumulation |
| [NVFP4 GEMV improved](https://veitner.bearblog.dev/nvfp4-gemv-improved/) | GEMM / MMA / low precision; match architecture and accumulation |
| [NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Bit counting and geometric series](https://veitner.bearblog.dev/bit-counting-and-geometric-series/) | Background algorithms; no automatic GPU optimization rule |
| [Simple reduction in CuTeDSL](https://veitner.bearblog.dev/simple-reduction-in-cutedsl/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [MMA Atoms in CuTe](https://veitner.bearblog.dev/mma-atoms-in-cute/) | GEMM / MMA / low precision; match architecture and accumulation |
| [LRU and LFU in C++](https://veitner.bearblog.dev/lru-and-lfu-in-c/) | Background algorithms; no automatic GPU optimization rule |
| [Mutual Refinement and Composition](https://veitner.bearblog.dev/mutual-refinement-and-composition/) | Layouts, addressing and correctness |
| [Applied introduction to Categorical treatment of CuTe](https://veitner.bearblog.dev/applied-introduction-to-categorical-treatment-of-cute/) | Layouts, addressing and correctness |
| [Layout Gymnastics](https://veitner.bearblog.dev/layout-gymnastics/) | Layouts, addressing and correctness |
| [Swizzles and their usage in CuTeDSL Kernels](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/) | Layouts, addressing and correctness |
| [CuTe partitions](https://veitner.bearblog.dev/cute-partitions/) | Layouts, addressing and correctness |
| [Tensors Slicing in CuTe](https://veitner.bearblog.dev/tensors-slicing-in-cute/) | Layouts, addressing and correctness |
| [Understanding CuTe Swizzling - The Math Behind 32B, 64B, and 128B Patterns](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) | Layouts, addressing and correctness |
| [GPU L2 Cache Persistence](https://veitner.bearblog.dev/gpu-l2-cache-persistence/) | Memory, scheduling, pipelining and profiling |
| [Cuda streams](https://veitner.bearblog.dev/cuda-streams/) | Memory, scheduling, pipelining and profiling |
| [PingPong 🏓 in the CuTeDSL with QuACK 🦆](https://veitner.bearblog.dev/pingpong-in-the-cutedsl-with-quack/) | Memory, scheduling, pipelining and profiling |
| [Bit Hacking in C](https://veitner.bearblog.dev/bit-hacking-in-c/) | Background algorithms; no automatic GPU optimization rule |
| [Intuition behind Hierarchical Layouts](https://veitner.bearblog.dev/intuition-behind-hierarchical-layouts/) | Layouts, addressing and correctness |
| [Persistent Float8 Dense Gemm on Hopper](https://veitner.bearblog.dev/persistent-float8-dense-gemm-on-hopper/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Epilogue in CuTeDSL H100 kernels](https://veitner.bearblog.dev/epilogue-h100-cutedsl/) | Memory, scheduling, pipelining and profiling |
| [Let the compiler do the work in CuTeDSL](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) | Memory, scheduling, pipelining and profiling |
| [Persistent GEMM in CuTeDSL on Hopper](https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Consumer-Producer pattern on H100 in CuTeDSL](https://veitner.bearblog.dev/consumer-producer-pattern-on-h100-in-cutedsl/) | Memory, scheduling, pipelining and profiling |
| [Backprob through Layernorm](https://veitner.bearblog.dev/backprob-through-layernorm/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Backprop through RMSNorm](https://veitner.bearblog.dev/backprop-through-rmsnorm/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Outperform compiled PyTorch code using QuACK 🦆](https://veitner.bearblog.dev/outperform-compiled-pytorch-code-using-quack/) | Memory, scheduling, pipelining and profiling |
| [CuTeDSL on Hopper - Pipelining](https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/) | Memory, scheduling, pipelining and profiling |
| [CuTeDSL on Hopper - WGMMA and TMA intro](https://veitner.bearblog.dev/cutedsl-on-hopper-wgmma-and-tma-intro/) | GEMM / MMA / low precision; match architecture and accumulation |
| [Thread Value Layouts in CuTe](https://veitner.bearblog.dev/thread-value-layouts-in-cute/) | Layouts, addressing and correctness |
| [SGEMM in CuTeDSL](https://veitner.bearblog.dev/sgemm-in-cutedsl/) | GEMM / MMA / low precision; match architecture and accumulation |
| [An applied introduction to CuTeDSL](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/) | Memory, scheduling, pipelining and profiling |
| [Calculating the fibonacci numbers on GPU](https://veitner.bearblog.dev/calculating-the-fibonacci-numbers-on-gpu/) | Background algorithms; no automatic GPU optimization rule |
| [An introduction to Thrust](https://veitner.bearblog.dev/an-introduction-to-thrust/) | Alternative language or library; verify current target/backend |
| [Infinite binary strings](https://veitner.bearblog.dev/infinite-binary-string/) | Background algorithms; no automatic GPU optimization rule |
| [Highly efficient matrix transpose in Mojo 🔥](https://veitner.bearblog.dev/highly-efficient-matrix-transpose-in-mojo/) | Alternative language or library; verify current target/backend |
| [The Bijection Between Natural Numbers and Binary Strings](https://veitner.bearblog.dev/the-bijection-between-natural-numbers-and-binary-strings/) | Background algorithms; no automatic GPU optimization rule |
| [Use TMA without CUDA](https://veitner.bearblog.dev/use-tma-without-cuda/) | Alternative language or library; verify current target/backend |
| [Use PTX instructions in Mojo](https://veitner.bearblog.dev/use-ptx-instructions-in-mojo/) | Alternative language or library; verify current target/backend |
| [Very fast vector sum without CUDA.](https://veitner.bearblog.dev/very-fast-vector-sum-without-cuda/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Short introduction to the Mojo programming language](https://veitner.bearblog.dev/short-introduction-to-the-mojo-programming-language/) | Alternative language or library; verify current target/backend |
| [Bridging Math and Code: CuTe Layout Algebra in CuTeDSL](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/) | Layouts, addressing and correctness |
| [Load and store matrices efficently with PTX instructions](https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/) | Memory, scheduling, pipelining and profiling |
| [How to use reasoning models with SGLang](https://veitner.bearblog.dev/how-to-use-reasoning-models-with-sglang/) | Serving; versioned example, not engine comparison |
| [Making matrix transpose really fast on Hopper GPUs](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) | Memory, scheduling, pipelining and profiling |
| [TMA introduction](https://veitner.bearblog.dev/tma-introduction/) | Memory, scheduling, pipelining and profiling |
| [Analyze CUDA programs by looking at GPU assembly.](https://veitner.bearblog.dev/analyze-cuda-programs-by-looking-at-gpu-assembly/) | Memory, scheduling, pipelining and profiling |
| [Making RMSNorm really fast](https://veitner.bearblog.dev/making-rmsnorm-really-fast/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Making prefix sum really fast](https://veitner.bearblog.dev/making-prefix-sum-really-fast/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Making vector sum really fast](https://veitner.bearblog.dev/making-vector-sum-really-fast/) | Reduction / selection / recurrence; preserve numerics and full operator contract |
| [Predication in Cutlass](https://veitner.bearblog.dev/predication-in-cutlass/) | Layouts, addressing and correctness |
| [Indexing in CUDA](https://veitner.bearblog.dev/indexing-in-cuda/) | Layouts, addressing and correctness |

## Additional blog coverage

- Maharshi: followed all eight posts in the blog index; performance mechanisms from softmax, SGEMV, tensors/broadcasting and identity tensors. Personal and prompting essays were not converted into GPU rules.
- Aryan: followed the whole blog index, including all four diffusion installments; retained performance-related kernels, model and cache boundaries. Editor setup and personal reference pages are background, not optimization evidence.
- AlpinDale: followed the eight-post index, including SIMD tokenizers, Torch registration, weight-quantized GEMM, decode and Vulkan work. Quantization and antislop behavior are not lossless backend replacements.
- Enbao: read the ThunderKittens binding article and inspected the posts index; other reading diaries and personal essays remain navigation leads.
- Patterson: read both GEMM and two-stage accumulation articles; preserved their different precision contracts.
- ezyang: read both requested articles; did not claim to crawl the entire historical blog.
- Newly discovered authors and selected works: see [authors and deep-reading route](gpu-authors-and-depth.md).

## What changed in the skill

- [Intra-kernel and CuTe recipe](../recipes/intra-kernel-and-cute.md): select a profiler, validate traces, diagnose indexing/state/barrier bugs and test architecture-specific ideas.
- [Compilation and diffusion recipe](../recipes/compilation-and-diffusion.md): capture/fallback/build costs, invariants, LoRA switching, quality gates and distributed/CPU boundaries.
- [Official documentation atlas](gpu-docs-atlas.md): route from a missing fact to matching APIs, implementation and validation.

External results are hypotheses for the next workload. Retrieval and documentation checks do not establish measured GPU speedups or behavioral improvement of an agent.
