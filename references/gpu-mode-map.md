# GPU MODE organization map

[Home](../README.md) / [Library](README.md) / Research

Snapshot reviewed 2026-11-04: 26 public repositories, including one empty repository.
This is a purpose map, not a certification of all code or external links. Use the
[advanced recipes](advanced-recipes.md) for experiments and [measurement rules](measurement-and-parity.md)
for acceptance. Sources are optional; use the installed library documentation for current APIs.

## Repositories and pinned revisions

| Repository | Revision | When it helps |
| --- | --- | --- |
| [resource-stream](https://github.com/gpu-mode/resource-stream) | [5c0efa14c48f](https://github.com/gpu-mode/resource-stream/tree/5c0efa14c48f4f53c6c204380dc4bb5a386dae3e) | Primary-link discovery for a diagnosed topic; linked material is not automatically reviewed. |
| [profiling-cuda-in-torch](https://github.com/gpu-mode/profiling-cuda-in-torch) | [333ba64033f6](https://github.com/gpu-mode/profiling-cuda-in-torch/tree/333ba64033f6fdcbd690bf47f3529f833e45e0c1) | Archived profiling introduction; prefer the consolidated lectures. |
| [lectures](https://github.com/gpu-mode/lectures) | [77a8df418834](https://github.com/gpu-mode/lectures/tree/77a8df418834e5789c12da23e7d2719e0efabef1) | Main teaching corpus: kernels, profiling, collectives, scheduling, layouts and numerics. |
| [lecture2](https://github.com/gpu-mode/lecture2) | [0ae6ca3af8bc](https://github.com/gpu-mode/lecture2/tree/0ae6ca3af8bc0b57cbd2bead8fe1fd075273a45c) | Archived PMPP introduction, duplicated by lectures/lecture_002. |
| [ring-attention](https://github.com/gpu-mode/ring-attention) | [28a95e4c830f](https://github.com/gpu-mode/ring-attention/tree/28a95e4c830f36b8a52ff21fe8f0956c7fae74c4) | Long-context sharding and link to Striped Attention; validate masks and rank balance. |
| [axolotl](https://github.com/gpu-mode/axolotl) | [9b6ee83a73d5](https://github.com/gpu-mode/axolotl/tree/9b6ee83a73d5ffbdc33cfb383a131a08c2b594ff) | Training framework fork; inspect recipe provenance and upstream currency before reuse. |
| [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | [4d794abed022](https://github.com/gpu-mode/Triton-Puzzles/tree/4d794abed02292081500ebf4b1e35cd5fc5fb019) | Small indexing/masking exercises and Triton-Viz; functional learning/debugging, not timing evidence. |
| [p2p-perf](https://github.com/gpu-mode/p2p-perf) | [521e8ac2fe84](https://github.com/gpu-mode/p2p-perf/tree/521e8ac2fe84dfe0bcc80fc0713221f2ed666584) | P2P/NCCL topology diagnostics using submodules; local message sizes and placement matter. |
| [awesomeMLSys](https://github.com/gpu-mode/awesomeMLSys) | [49031c21e586](https://github.com/gpu-mode/awesomeMLSys/tree/49031c21e586ae93829d700fcb46543aecfc0f27) | Broader systems reading list: parallelism, memory, inference and communication. |
| [triton-index](https://github.com/gpu-mode/triton-index) | [6b301ab902d8](https://github.com/gpu-mode/triton-index/tree/6b301ab902d835766a2e9c2914799b514a41fef5) | Operator-indexed implementation discovery; inspect linked upstream contracts and licenses. |
| [cutlass-tutorial](https://github.com/gpu-mode/cutlass-tutorial) | empty | Empty repository in this snapshot. Use lectures 57/86/103/114 and upstream CUTLASS. |
| [amd-cluster](https://github.com/gpu-mode/amd-cluster) | [32aa12263907](https://github.com/gpu-mode/amd-cluster/tree/32aa12263907222442bdfff792f485a2541cf9a8) | AMD examples and CI; architecture-specific candidates, not NVIDIA defaults. |
| [kernelbot](https://github.com/gpu-mode/kernelbot) | [ec4669669d36](https://github.com/gpu-mode/kernelbot/tree/ec4669669d368c395675565831f98b509db2b0b3) | Competition execution backend and profiling protocols; infrastructure, not an inference engine. |
| [popcorn](https://github.com/gpu-mode/popcorn) | [edf2827a1ec5](https://github.com/gpu-mode/popcorn/tree/edf2827a1ec51978078ceee41a214f23babf0b36) | Research-project overview and links, useful for discovery. |
| [reference-kernels](https://github.com/gpu-mode/reference-kernels) | [f3295bb6bd55](https://github.com/gpu-mode/reference-kernels/tree/f3295bb6bd559bae4659d2d8fc45edb1394251b1) | Task schemas, independent references, shape suites, checkers and profiling harnesses. |
| [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) | [036b3ead604f](https://github.com/gpu-mode/popcorn-cli/tree/036b3ead604f1e9f404cbee5cf0e48377ef0766f) | Submission/profiling client. Remote use sends code to a service; not required for local optimization. |
| [KernelSwarm](https://github.com/gpu-mode/KernelSwarm) | [0b30fcd11eb2](https://github.com/gpu-mode/KernelSwarm/tree/0b30fcd11eb2ddb71f8ff19b067711270c39a88c) | VPN/lighthouse coordination infrastructure; not a ready-made kernel superoptimizer. |
| [kernelboard](https://github.com/gpu-mode/kernelboard) | [51dde2b22e5c](https://github.com/gpu-mode/kernelboard/tree/51dde2b22e5c18c218ac682ee5d0e8af5d8baba2) | Competition dashboard and submission browsing, not an optimization library. |
| [triton-tutorials](https://github.com/gpu-mode/triton-tutorials) | [dc1532fc1283](https://github.com/gpu-mode/triton-tutorials/tree/dc1532fc128377fc108146e78b4b49439d4bef45) | Focused implementations including grouped GEMM; check dtype, alignment, tails and mutation. |
| [kernelbot-data](https://github.com/gpu-mode/kernelbot-data) | [5da7a0fa9b6d](https://github.com/gpu-mode/kernelbot-data/tree/5da7a0fa9b6de2153958d7ccebcd9da8c402419a) | Database-to-Hugging-Face dataset exporter; not acceleration code. |
| [leaderboard_cli](https://github.com/gpu-mode/leaderboard_cli) | [a374ef9f6600](https://github.com/gpu-mode/leaderboard_cli/tree/a374ef9f6600922c02d74fac18a2d7ef61016297) | Submission registry/client/server; useful experiment metadata pattern, no automatic speedup. |
| [pygpubench](https://github.com/gpu-mode/pygpubench) | [286841bf9734](https://github.com/gpu-mode/pygpubench/tree/286841bf973450f5d426198f744c3a12defe7b07) | Low-latency adversarial benchmark isolation and verification; keep separate from application timing. |
| [kernelguard](https://github.com/gpu-mode/kernelguard) | [6797d26ddb52](https://github.com/gpu-mode/kernelguard/tree/6797d26ddb52f4a7f8d6020af317eda9291359fc) | Rule-based detection of competition hacks; neither numerical proof nor application-quality validation. |
| [paris-hackathon-2026-inference](https://github.com/gpu-mode/paris-hackathon-2026-inference) | [44959f783d9a](https://github.com/gpu-mode/paris-hackathon-2026-inference/tree/44959f783d9a7c5bd2e81e234f09fdd8eb102dc9) | Qwen3.5 BF16 serving task and client harness; inspect metric formula and content assumptions. |
| [paris-hackathon-2026-training](https://github.com/gpu-mode/paris-hackathon-2026-training) | [8ffcabe37430](https://github.com/gpu-mode/paris-hackathon-2026-training/tree/8ffcabe374308ab94770dcba1a7f32a81d4192ea) | Time-budgeted training starter: optimizer, accumulation and checkpoint contract; inspect hot-loop overhead. |
| [ncu-service](https://github.com/gpu-mode/ncu-service) | [71bd64c22c42](https://github.com/gpu-mode/ncu-service/tree/71bd64c22c421c23eece2d39bd5f68f76c9fe036) | Hosted per-shape NCU capture infrastructure; does not establish multi-GPU end-to-end performance. |

## Operator task atlas

These are lookup starting points inside `reference-kernels/problems`, not contracts
for the user's workload. Read reference.py, task/schema and actual benchmark inputs together.

| Task family | Lookup / transfer limit |
| --- | --- |
| Low-bit matmul, GEMV, grouped/dual GEMM | `nvidia`, `amd`, `amd_202602`: precision, scale layout and hardware are part of the task. |
| MLA decode and MoE | `amd`, `amd_202602`: head dimensions, expert routing and packing are workload-specific. |
| All-gather GEMM, GEMM reduce-scatter, all-to-all | `amd_distributed`: include all ranks and the communication boundary. |
| Causal convolution and gated DeltaNet | `helion`: chunk size, initial/recurrent state, precision and tails need independent checking. |
| Cross entropy | `princeton`: compare loss semantics and backward, not forward speed alone. |
| QR, Cholesky, eigendecomposition | `linalg`: preserve the factorization convention and numerical conditioning contract; compare appropriate residuals. |
| Triangular multiplicative update | `bioml/trimul`: specialized tensor contraction, not general attention. |
| Reduction, scan, histogram, sort, convolution, matmul | `pmpp`, `pmpp_v2`: learn algorithm/parallel decomposition before low-level tuning. |

## Lecture route

- Profiling, basic memory access, reductions, fusion: 1, 8, 9, 16, 18.
- Communication/context sharding: 13, 17, 67, 70, 78.
- Serving/scheduling and attention: 22, 35, 36, 40, 72.
- Optimizers and fused training operators: 6, 28, 30.
- Layouts, compiler and GPU execution: 29, 37, 57, 75, 86, 103, 104, 114.
- Precision/sparsity: 7, 11, 30, 33, 34, 38, 42/43, 69, 84.
- Automatic kernel search: 79 (Mirage). Alternative model architectures: 71, 74.

The lecture README is not a complete or consistently numbered syllabus. Some entries
only link external slides/videos; a folder number can differ from a lecture number
(for example `lecture_042` contains the Turing INT8 slides listed under lecture 43).
Check the actual file title. ROCm/Composable Kernel, Metal and ARM material is useful
on its target hardware; it is not a direct recipe for the user's NVIDIA workload.

## Transfer limits found by inspection

- The inference-hackathon throughput field includes prompt plus completion tokens,
  despite output-token wording in its introduction. Recompute/label the intended metric.
- Its random-vocabulary request stream and mathematical quality gate are task choices,
  not representative caption/audio quality tests. Text retokenization is not raw token-ID capture.
- The training starter reads `loss.item()` every microstep; investigate exposed synchronization
  before importing that logging pattern. The current baseline already requests fused AdamW.
- The DeltaNet state task assumes chunk size 64 and zero initial state; it is not a streaming validator.
- Gluon/FA3 slides demonstrate architecture-specific primitives. Tutorial performance claims
  are not measurements of the user's shapes or end-to-end pipeline.
- The older tutorial issues in [GPU MODE resources](gpu-mode.md) still apply.

No external submissions, competition enrollment, framework migration or precision changes
are required by this map. Follow the authorization and resource scope of the actual task.
