# Source library: read at the point of need

[Home](../../README.md) / [Library](../README.md) / Sources

This is a decision-oriented index, not a requirement to read every source. Open
the procedure in the [handbook](../handbook/README.md) first, then the source that resolves
its missing implementation detail. Official documentation describes APIs;
research motivates hypotheses; implementation repositories expose actual
contracts; teaching material helps build a mental model. None establishes a
speedup on the current workload without measurement.

Documentation URLs may track a newer release than the installed package. Check
the installed version and use its matching documentation before applying an API.
For copied implementations, record commit, license, supported shapes/dtypes and
local adaptations. The [GPU MODE map](gpu-mode-map.md) separately preserves
the reviewed organization snapshot and exact source revisions.

## Attributed engineering decisions

The [2026 decision corpus](../decisions/README.md) records observations, actions,
reported outcomes, scope and proposed transfer tests. Its [catalog](../decisions/CATALOG.md)
distinguishes reviewed passages from pending extraction and inaccessible bodies.

## Measurement and numerics

Use with [contract and baseline](../handbook/01-contract-and-baseline.md) and
[validation](../handbook/07-validation-and-delivery.md).

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| M1. [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — official API | Choosing a timer, repetitions or operator comparison | A microbenchmark utility does not define the application's input/output boundary. |
| M2. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — official notes | Understanding backend, reduction-order or precision differences | Numerical explanations do not authorize relaxed task tolerances. |
| M3. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — official notes | Controlling RNG and diagnosing nondeterminism | Reproducibility settings can affect performance and do not guarantee cross-platform bitwise identity. |
| M4. [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — official tutorial | Mapping operations, shapes and memory to time | Capture overhead and asynchronous execution require a separate unprofiled baseline. |
| M5. [MLPerf Inference LoadGen](https://github.com/mlcommons/inference/tree/master/loadgen) — reference load generator | Designing meaningful arrivals, latency collection and scenario boundaries | Its compliance rules are specific to MLPerf; a custom application need not adopt the entire suite. |

## Profiling and hardware

Use with [attribution](../handbook/02-attribution-and-experiments.md) and
[kernel engineering](../handbook/06-kernel-engineering.md).

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| P1. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) — official manual | Explaining launch gaps, copies, streams, CPU waits and process activity | Verify the relevant child worker and workload phase were captured. |
| P2. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — official manual | Interpreting replay, caches, rooflines and counters | Profiled execution is not normal application timing; match architecture and metric definitions. |
| P3. [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) — official manual | Selecting kernels, NVTX ranges and metric sets | Check local option/range syntax; collect only the evidence needed. |
| P4. [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) — official manual | Investigating invalid accesses, shared-memory hazards, initialization or synchronization | Different tools have different coverage; a clean run is not a mathematical proof. |
| P5. [py-spy](https://github.com/benfred/py-spy) — upstream tool | Sampling a Python process without first instrumenting its code | Sampling/attach permissions and native-stack support matter; stacks alone do not attribute GPU time. |
| P6. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) — official guide | Understanding coalescing, bandwidth, occupancy, transfer and resource trade-offs | Hardware heuristics need measurement for the actual kernel and shape. |
| P7. [CUDA MPS documentation](https://docs.nvidia.com/deploy/mps/) — official manual | Evaluating several independent processes sharing an authorized GPU | Version, resource controls, memory and fault behavior matter; sharing can reduce throughput. |
| P8. [Nsight Compute performance triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/) — official guide | Turning a slow-kernel observation into a profiling question | Use as a diagnostic route, not a fixed utilization target or universal tuning sequence. |

## Data, memory and compilation

Use with [input, shapes and compilation](../handbook/03-input-shapes-and-compilation.md).

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| D1. [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — official recipe | Looking for low-cost framework, CPU and data-path candidates | Check whether each recommendation addresses an observed cost; do not toggle everything together. |
| D2. [Pinned memory and non-blocking transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) — official tutorial | Reasoning about host blocking, transfer overlap and source/destination lifetime | Timing and correctness depend on transfer direction, pinning and synchronization. |
| D3. [Understanding CUDA memory usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — official notes | Inspecting allocation history, peaks and retained blocks | Framework snapshots may omit memory allocated directly by external libraries. |
| D4. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) — official notes | Checking streams, allocator lifetime, precision controls and CUDA graphs | Match the installed version and preserve cross-stream dependencies. |
| D5. [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html) — official guide | Finding graph breaks, recompilations and unsupported behavior | A graph break outside the critical path is not automatically worth repairing. |
| D6. [Dynamic shapes](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html) — official guide | Choosing specialization versus dynamic-shape execution | Measure the actual shape distribution, recompilation frequency and code quality. |
| D7. [Regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) — official recipe | Reducing compilation startup for repeated model structure | Include cold-start amortization and verify the chosen region remains a hot path. |
| D8. [Activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/) — PyTorch technical article | Selecting memory versus recomputation trade-offs | Check current API availability, RNG/state behavior and complete-update time. |
| D9. [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html) — official API | Worker lifecycle, pinning, sampling and iterable-dataset sharding | Preserve sample coverage, ordering/seed requirements and real storage behavior. |
| D10. [FFmpeg documentation](https://ffmpeg.org/documentation.html) — upstream manuals | Checking decoder, seeking, timestamps, filters and pixel/audio conversion semantics | Decoder throughput does not prove equivalent frames or model inputs. |

## Serving and attention

Use with [serving and concurrency](../handbook/04-serving-and-concurrency.md).

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| S1. [vLLM serving benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/) — official API | Configuring arrival rate, in-flight limits and per-request results | Verify actual arrivals, request schema and token-count definitions. |
| S2. [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) — official guide | Investigating batching, scheduler and memory trade-offs in that engine | Model support and defaults vary by release; validate activation in the worker. |
| S3. [SGLang and RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — authors' technical article | Understanding prefix reuse and serving runtime design | Use current engine docs for APIs; reported workloads do not establish local cache hit rates. |
| S4. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — research paper | Understanding chunked prefill and throughput/latency scheduling trade-offs | Results motivate an experiment; engine support and local traffic determine value. |
| S5. [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) — authors' technical article | Understanding split-KV parallelism for long-context, small-batch decode | Include reduction/workspace costs and do not transfer the result to prefill automatically. |
| S6. [FlashAttention](https://github.com/Dao-AILab/flash-attention) — upstream implementations | Looking for supported efficient attention kernels | Check device, dtype, mask, head dimension, backward and version constraints. |
| S7. [FlashInfer documentation](https://docs.flashinfer.ai/) — upstream library | Investigating serving-oriented attention and related operators | Wrapper, cache layout, planning and integration costs belong in the benchmark. |
| S8. [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) — official guide | Checking supported prefix-reuse behavior and configuration | Include model/adapter/content identity, invalidation, hit rate and memory pressure. |
| S9. [Together Compiler: Continual Compilation for AI Inference](https://www.alphaxiv.org/abs/2611.together-compiler-continual-ai-inference) — industry paper | Building a persistent, tiered measurement harness; measurement fidelity for data-dependent operators; parallelism cost model with measured ceilings; ablations of curated agent context | Results come from parallel agents, a production engine, day-long budgets and two nodes per model; evidence of technique and of measurement pitfalls, not of local gains. |

## Training and communication

Use with [training and distributed execution](../handbook/05-training-and-distributed.md).

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| T1. [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — official API | Gradient synchronization, accumulation, bucket and graph assumptions | Include forward in `no_sync()` scope where required and preserve final synchronization. |
| T2. [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — official tutorial | Understanding parameter sharding and the current FSDP execution model | FSDP1/FSDP2 and DDP behavior are not interchangeable; use the installed API. |
| T3. [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — official API | Comparing supported optimizer paths and state semantics | Check actual backend, None-gradient behavior, parameter groups, precision and memory. |
| T4. [PyTorch optimizers](https://docs.pytorch.org/docs/stable/optim.html) — official guide | State dictionaries, parameter groups, schedules and implementation choices | Preserve the complete update and save/resume behavior. |
| T5. [NCCL tests](https://github.com/NVIDIA/nccl-tests) — upstream diagnostic suite | Measuring collective performance versus payload size and rank placement | Synthetic bandwidth is diagnostic; application overlap and rank skew need a real trace. |
| T6. [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — official manual | Collective semantics, topology, environment controls and troubleshooting | Avoid copying tuning environment variables without an identified cause and version match. |
| T7. [Striped Attention](https://arxiv.org/abs/2311.09431) — research paper | Reasoning about load balance in distributed causal attention | Preserve global positions/masks and include communication on all ranks. |
| T8. [PyTorch tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) — official tutorial | Understanding tensor/sequence-parallel partitioning and composition | Capacity and throughput gains depend on model, layout, topology and scale. |
| T9. [Activation checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html) — official API | Checking recomputation variants, RNG handling and limitations | Side effects and stateful regions need explicit validation. |

## Kernel implementation

Use with [kernel engineering](../handbook/06-kernel-engineering.md) and the
[DSL selection guide](../recipes/kernel-backends.md).

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| K1. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — upstream examples | Finding a minimal implementation pattern for a specific operator | Tutorial shapes, cache policy and numerical settings are not the application contract. |
| K2. [Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) — upstream tutorial | Tiling, program ordering, masks and configuration search | Include tails, strides and any required layout conversion. |
| K3. [Triton grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) — upstream tutorial | Scheduling many independent matrix multiplications | Use actual group sizes and include routing/packing/restoration. |
| K4. [TileLang documentation](https://tilelang.com/) — upstream guide | Tile-oriented programming, pipelines, layout inspection and autotuning | Check backend support, generated code, numerics and compilation cost. |
| K5. [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) — official guide | Explicit layouts, tensor-core primitives and architecture-aware composition | More control adds implementation and validation work; require measured headroom. |
| K6. [CUTLASS](https://github.com/NVIDIA/cutlass) — upstream library | Finding tuned GEMM building blocks and supported schedules | Match architecture, dtype, alignment, epilogue and workspace requirements. |
| K7. [PyTorch custom Python operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html) — official guide | Registering schemas, mutation, fake implementations and autograd behavior | `opcheck` checks integration; separate tests must check mathematical correctness. |
| K8. [Gluon tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) — upstream examples | Exploring explicit layout control within the Triton ecosystem | APIs evolve; use only when required control justifies moving below the current abstraction. |
| K9. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — official manual | Checking memory model, execution, synchronization and architecture features | Use the relevant toolkit/device version; examples still require application-level validation. |
| K10. [Transformer building blocks in PyTorch](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html) — official tutorial | Investigating supported attention/representation/composition improvements | Confirm mask, layout, numerical and model compatibility before adoption. |

## GPU MODE and implementation discovery

These complement the [26-repository map](gpu-mode-map.md), its lecture route
and the [tutorial transfer warnings](gpu-mode.md). Curated indexes are discovery
sources; follow the linked original implementation or paper for technical claims.

| ID / source | Open when / question answered | Apply with this boundary |
| --- | --- | --- |
| G1. [GPU MODE lectures](https://github.com/gpu-mode/lectures/tree/77a8df418834e5789c12da23e7d2719e0efabef1) — pinned teaching corpus | Learning the mechanism behind a measured memory, layout, serving or communication bottleneck | Folder numbering and external links need inspection; slides are not benchmark evidence. |
| G2. [resource-stream](https://github.com/gpu-mode/resource-stream) — community discovery index | Finding further reading on a specific unresolved topic | Inclusion is not validation; inspect the primary source and its date. |
| G3. [triton-index](https://github.com/gpu-mode/triton-index) — operator discovery index | Searching for an existing implementation before writing one | Verify the upstream contract and license; index descriptions can lag the code. |
| G4. [Reference-kernel NCU guide](https://github.com/gpu-mode/reference-kernels/blob/main/docs/ncu-profiling.md) — harness documentation | Learning a narrow kernel-capture protocol | Its range names and hosted-runner requirements are local conventions, not universal rules. |
| G5. [PyGPUBench](https://github.com/gpu-mode/pygpubench) — benchmark implementation | Isolating short or potentially contaminating kernel candidates | Adversarial isolation and cache policies do not replace application timing. |
| G6. [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles) — exercises and debugging entrypoints | Practicing indexing, masks and inspecting small kernels | Educational correctness is not production shape coverage or speed evidence. |
| G7. [KernelWiki](https://github.com/mit-han-lab/KernelWiki) — curated kernel knowledge | Looking up techniques relevant to a diagnosed operator | Trace claims to the original paper/code and local measurements. |
| G8. [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) — profiling workflow | Structuring collection and interpretation of kernel counters | Optional aid; use only available tools and preserve the application's promotion gate. |
| G9. [Mirage](https://github.com/mirage-project/mirage) — research implementation | Considering automated kernel composition/search after simpler options | Bound search cost and verify supported operators, numerics, compilation and integration. |

## Turn reading into one experiment

After reading a source, write the transferable mechanism in one sentence, then
record the assumptions that must hold locally. Find the cheapest measurement
that checks those assumptions. Estimate recoverable end-to-end time, include
integration and validation cost, and either schedule that experiment or stop.
More references are useful only while they resolve a live decision.

When adding sources, keep these distinctions explicit: linked, content reviewed,
code inspected, locally reproduced, and promoted on a representative workload.
Do not upgrade one evidence level to another by wording alone. HTTP link checks
verify reachability, not implementation correctness or full review of external
videos, papers and every transitive link.

---

[← Validation & delivery](../handbook/07-validation-and-delivery.md) · [Handbook](../handbook/README.md) · [Library](../README.md)

## Compilation, GPU documentation and author routes

- [Compilation collection audit](compilation-reading.md): all original tabs,
  source corrections, complete Veitner index and extracted mechanisms.
- [Official GPU documentation atlas](gpu-docs-atlas.md): Triton/Gluon/Proton,
  CuTe/CUTLASS/IKET, PTX, alternative DSLs, libraries and serving engines.
- [Authors and deep study routes](gpu-authors-and-depth.md): specific primary
  articles, prerequisites, architecture boundaries and concrete learning outputs.
- [Link graph](gpu-doc-links.jsonl): retrieved pages and discovered outbound links;
  search with `python3 scripts/source_links.py QUERY` from the skill root.

## Further blog research

See [additional authors, AMD route and twelve transfer tests](gpu-blog-deep-dive.md).
