# Further GPU blog research: mechanisms and counterexamples

Reviewed on 2026-11-09. These are additions to the author route, not a claim of
exhaustive coverage. Retrieved 28 substantive pages out of 29 attempted URLs;
three are indexes. Selected introductions, mechanism passages and experiment
outcomes were reviewed; full code reproduction and GPU validation were not run.
Stephen Diehl's attempted FlashAttention URL did not yield a body and is excluded
from recommendations. Older articles remain useful but are not current API specs.

## Ten additional author and engineering routes

| Route | Concrete reading | What to extract and where it stops transferring |
| --- | --- | --- |
| Pranjal Shankhdhar / CUDA for Fun | [H100 worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), [NVFP4 on Blackwell](https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4) | Persistent scheduling, TMA, register redistribution, tile ordering, partition-aware L2 reuse. Keep failed variants: persistence alone regressed in the H100 worklog; earlier accumulator release lost in the NVFP4 experiment. Shape, numerical format and hardware are essential boundaries. |
| Daniel Vegamyhre / ML Perf Notes | [MXFP8 GEMM](https://danielvegamyhre.github.io/2026/03/29/mxfp8-gemm.html) | CUDA/PTX block-scaled GEMM, scale layouts, staged movement and remaining gaps. The B200 comparison varies by shape; the headline maximum does not describe the 4096 case. |
| Axel Feldmann | [Shared-memory microbenchmarks](https://feldmann.nyc/blog/smem-microbenchmarks) | Controlled tests for bank throughput, conflicts and broadcasts. Experiments use A100; reproduce transaction width and emitted instructions before applying results to another architecture. |
| Yifan Yang | [Making a compute-bound problem actually compute-bound](https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html) | Match operand-delivery throughput at each memory level, not only HBM arithmetic intensity. Derive reuse requirements from the chosen dataflow; power and instruction delivery can still limit performance. |
| Amanzhol Salykov | [Advanced NVIDIA SGEMM](https://salykova.github.io/sgemm-gpu) | FP32 CUDA-core GEMM, inline PTX, buffering, coalesced epilogues, clock and cache controls. Compare the same alpha/beta semantics and precision; profiler-controlled measurements and application timing answer different questions. |
| Florian Mattana | [SASS King, Part 1](https://florianmattana.com/posts/sass_king/) | Controlled source variations, uniform/general registers, scoreboard fields, FMA and loop lowering. Treat reverse-engineered explanations as hypotheses scoped to the tested compiler/SM; do not adopt unsupported claims that other research does not exist. |
| Lei Zhang / Lei.Chat() | [Linear layouts: concept](https://www.lei.chat/posts/triton-linear-layout-concept/), [examples](https://www.lei.chat/posts/triton-linear-layout-examples/), [bespoke layouts](https://www.lei.chat/posts/triton-bespoke-layouts/), [Gluon](https://www.lei.chat/posts/gluon-explicit-performance/) | Compiler ownership mappings and conversions, then explicit control through Gluon. Also follow the author's MLIR/vector/linalg articles. Distinct author from Lei Mao. |
| Pavel Belevich | [My Triton from Scratch: verification](https://pbelevich.github.io/2026/06/26/My_Triton_From_Scratch_Part_5_Verification.html) | Trace to typed SSA to CUDA; validate IR after transformations. This is an educational compiler, not an alternative performance result for production Triton. Earlier/later series links are discovery leads, not reviewed bodies. |
| Mahmoud Zalt | [A Cache Hit Is Not Free](https://zalt.me/blog/cache-hit-not-free) | Inspect work remaining on a Triton cache-hit path. Useful code-reading lead, not a measured speedup claim; main-branch source links must be pinned before reproducing. |
| Modular engineering | [Blackwell matrix multiplication, Part 4](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota) | Compare the Mojo implementation route with CUDA/PTX and CUTLASS. Read comparator conditions and supported workload before using a vendor headline as evidence. |

## AMD route, with named implementers

Amanzhol Salykov, Christian Gilli, Andy Luo and collaborators provide a separate
hardware route; do not translate NVIDIA warp, bank, barrier or instruction rules
by replacing names:

1. [CDNA3/CDNA4 Matrix Cores](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html): MFMA operand mappings, formats and scaling.
2. [CDNA4 FP8 GEMM](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html): LDS tiling/swizzling, direct loads and eight-wave scheduling.
3. [Four-wave interleave](https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html): trade wave occupancy for register capacity and explicitly scheduled overlap.
4. [HIP to FlyDSL](https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html): retain layout and MFMA control through an MLIR-based Python DSL.
5. [FlyDSL mono kernels in ATOM](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-mono-atom/README.html): connect kernel scheduling to decode latency; inspect serving workload and integration costs.

## Decisions to carry into an optimization experiment

These are our transfer tests derived from the linked articles, not locally
validated performance results. Record the source, hardware, compiler revision,
shape, precision and negative outcome alongside each attempted optimization.

| Trigger | Experiment | Evidence required / rejection condition |
| --- | --- | --- |
| A roofline predicts compute-bound, achieved compute is low | Build bandwidth/reuse budgets for HBM, L2, shared memory and registers using Yifan's method. | Identify the actual limiting stage; reject a larger tile if resource pressure offsets reuse. |
| Persistent GEMM gets slower | Compare ordinary launch, persistent row-major, grouped and alternative tile orders (Pranjal). | Same work and precision, cache-controlled runs, L2 traffic and end-to-end time. Persistence is not independently a win. |
| A locality-preserving order stops helping | Test ownership of reused operands separately from order within each ownership group (NVFP4 worklog). | Reproduce topology and cache behavior on the actual GPU; do not hardcode observed SM-to-side mappings as an API guarantee. |
| An earlier async release appears beneficial | Benchmark the extra control/unrolling against the shorter lifetime (NVFP4 worklog). | Prove safe completion first, then retain only measured wins; earlier release can cost more instructions. |
| Shared-memory traffic looks unexpectedly slow | Construct conflict-free, conflicting and broadcast microbenchmarks (Feldmann). | Inspect generated loads and transaction widths; separate requests from distinct bank-serviced words. |
| A source change or compiler update changes speed | Save cubins and diff SASS for fusion, spills, loop size and scheduling (Mattana). | One controlled change, equal compiler flags and numerical contracts; assembly differences alone do not prove causality. |
| Triton layout conversion dominates | Trace ownership through TTGIR and identify conversions before choosing explicit Gluon layouts (Lei Zhang). | Verify masks, boundaries and output first; compare conversion cost and register/shared-memory usage. |
| Adding an optimization pass to a compiler | Run a typed SSA verifier after each pass (Belevich). | Reject malformed IR at the first offending pass; verification does not replace numerical tests of generated kernels. |
| A warm cache still has startup latency | Time in-process reuse, fresh-process disk-cache reuse and cold compilation separately (Zalt). | Attribute lookup, deserialization, module loading and launch setup only where traces show them; do not label all residual time compilation. |
| GEMM headline speedups differ between runs | Repeat under explicit cache/clock policies and realistic application conditions (Salykov, Pranjal). | Report warm versus rotated inputs, thermal/power state and shape distribution; no mixed precision or mismatched beta comparisons. |
| More occupancy fails to improve AMD GEMM | Compare four-wave interleave and eight-wave schedules with equal math (ROCm series). | Track VGPR budget, LDS, MFMA issue and overlap on the specified CDNA target; maximum occupancy is not the objective. |
| A low-precision kernel approaches cuBLAS only on some sizes | Keep per-shape outcomes, scale layouts and error checks (Vegamyhre). | Preserve unresolved shapes in the report and dispatcher; never promote the best point to a universal speedup. |

## Reading order

For kernel engineering: Yifan → Axel → Salykov → Pranjal H100 → Daniel MXFP8 →
Pranjal NVFP4. For compiler work: Lei's concept → examples → bespoke layouts →
Gluon → Pavel's verifier. Read Mattana alongside a real SASS diff. For another
hardware stack: Matrix Cores → CDNA4 GEMM → four-wave interleave → FlyDSL.

## Retrieval inventory

`body retrieved` means fetch succeeded, not full review or reproduction. Indexes
and source-code links are not counted as independent optimization experiments.

- body retrieved: https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
- body retrieved: https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4
- body retrieved: https://feldmann.nyc/blog/smem-microbenchmarks
- body retrieved: https://florianmattana.com/posts/sass_king/
- body retrieved: https://www.lei.chat/categories/compiler/
- body retrieved: https://pbelevich.github.io/2026/06/26/My_Triton_From_Scratch_Part_5_Verification.html
- body retrieved: https://zalt.me/blog/cache-hit-not-free
- body retrieved: https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html
- body retrieved: https://salykova.github.io/sgemm-gpu
- body unavailable: https://stephendiehl.com/posts/flash_attention/
- body retrieved: https://rocm.blogs.amd.com/authors/amanzhol-salykov.html
- body retrieved: https://danielvegamyhre.github.io/2026/03/29/mxfp8-gemm.html
- body retrieved: https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota
- body retrieved: https://www.lei.chat/posts/
- body retrieved: https://www.lei.chat/posts/gluon-explicit-performance/
- body retrieved: https://www.lei.chat/posts/triton-bespoke-layouts/
- body retrieved: https://www.lei.chat/posts/triton-linear-layout-examples/
- body retrieved: https://www.lei.chat/posts/triton-linear-layout-concept/
- body retrieved: https://www.lei.chat/posts/triton-compiler-development-tips/
- body retrieved: https://www.lei.chat/posts/mlir-linalg-dialect-and-patterns/
- body retrieved: https://www.lei.chat/posts/mlir-vector-dialect-and-patterns/
- body retrieved: https://www.lei.chat/posts/mlir-codegen-dialects-for-machine-learning-compilers/
- body retrieved: https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/
- body retrieved: https://www.lei.chat/posts/codegen-performant-convolution-kernels-for-mobile-gpus/
- body retrieved: https://rocm.blogs.amd.com/software-tools-optimization/flydsl-mono-atom/README.html
- body retrieved: https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html
- body retrieved: https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
- body retrieved: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- body retrieved: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html

## Further expansion: attention, compiler inspection and runtime behavior

Additional search on 2026-11-09: 24 of 25 attempted pages returned bodies.
This includes indexes, project pages and a repository README. Selected technical
passages were reviewed, not every line of every article. The following routes
extend the previous collection; they are not all personal blogs.

| Author / engineering collection | Read next | Distinct value and boundary |
| --- | --- | --- |
| Hamza Elshafie | [ThunderKittens dissection](https://hamzaelshafie.bearblog.dev/dissecting-thunderkittens-anatomy-of-a-compact-dsl-for-high-performance-ai-kernels/), [H100 worklog](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/), [paged attention](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/), [AWQ](https://hamzaelshafie.bearblog.dev/awq-activation-aware-weight-quantisation/) | Especially close to Veitner: register/shared/tensor tiles, producer-consumer ownership, TMA, WGMMA, tcgen05 and an attention implementation. H100 worklog explicitly remains WIP; FP32 and mixed-precision phases use different comparators. |
| Paul Chan | [B200 GEMM worklog](https://www.chanpaul.com/articles/outperforming-cublas-b200) | 2-CTA MMA, double-buffered TMEM, epilogues, Hilbert ordering and Cluster Launch Control. Useful comparative scheduling experiment; do not copy its hardware-spec table without verification against the exact SKU. |
| Charles Frye, Nathan Wang, Timothy Feng / Modal | [Reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) | Read actual code by warp role: MMA, softmax, correction, epilogue. Explains polynomial exponentiation and SFU pressure. A pinned historical implementation, with some explicitly speculative explanations; not a specification of every current FA4 kernel. |
| FlashInfer authors | [Sorting-free sampling](https://flashinfer.ai/2025/03/10/sampling.html), [cascade attention](https://flashinfer.ai/2024/02/02/cascade-inference.html), [autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html), [MegaMoE](https://flashinfer.ai/2026/09/22/mega-moe.html) | Original engineering articles on sampling semantics/numerics, mergeable attention states, execution-aware tuning and fused communication/computation. Preserve authors and benchmark configurations per article. |
| Nicholas Wilt / CUDA Handbook | [Asynchronous errors](https://www.cudahandbook.com/blog/asynchronous-error-handling-is-hard), [follow-up on cudaGetLastError](https://www.cudahandbook.com/blog/cudagetlasterror-its-raison-detre) | Runtime API design and error attribution. Read both: the follow-up qualifies the earlier recommendation for triple-chevron launch error reporting. Never convert an author's preference into a universal ban. |
| Graphics Programming / Nabla | [FFT bloom optimization](https://graphics-programming.org/blog/fft-bloom-optimized-to-the-bone-in-nabla) | A non-LLM route into FFT convolution, subgroup operations, memory footprint, padding and data movement. Small-filter crossover and image boundary semantics matter. |
| Kevin Huai | [Triton Pass Explorer](https://test.kevinhuai.com/triton/) | Interactive compiler companion, not a benchmark article: recorded Triton/Gluon lowering examples on CUDA and AMD. Landing-page inventory retrieved; individual interactive traces were not inspected in this review. |
| DefTruth | [LeetCUDA notes and article index](https://github.com/xlite-dev/LeetCUDA/blob/main/README.md) | CUDA/MMA/CuTe implementations and links to Chinese-language technical articles. README reviewed selectively; linked article bodies are discovery leads, not bulk-reviewed content. |
| Nathan Nguyen | [Projects and measurements](https://nathan.ng/) | Consumer-GPU INT8 KV-cache alignment and attention integration, with links to patches. Project-page claims are leads; linked PRs and benchmark code have not been independently audited here. |
| vLLM and SGLang contributors | [Kimi K3 decode path](https://vllm-project.github.io/2026/10/16/kimi-k3-megakernel.html), [DeepSeek-V4.1 optimization](https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization) | Connect specialized kernels to serving. Read hardware topology, decode batch range, fallback and accuracy boundaries before transferring a model-specific path. Introductory/scope passages reviewed; headline speedups not reproduced. |

### Additional transfer tests

- **Tune at the execution boundary actually deployed.** FlashInfer's autotuner
  article separates eager calls from CUDA Graph replay. Measure both when both
  are used; key persisted choices by compatible environment and timing policy.
  Evaluate selection on independent samples and report latency regret, not just
  whether the selected tactic matches the nominal winner.
- **Sampling optimization must preserve the specified distribution.** From the
  sorting-free sampling article: distinguish top-k-first and joint filtering;
  test degenerate/tied distributions, rounding around thresholds and validity
  of the sampled index. Matching one seed's token output is not a sufficient
  statistical correctness test, and faster sorting alone may miss the algorithmic opportunity.
- **Separate attention partitioning from approximation.** Cascade merges partial
  output and normalization state; preserve stable log-sum-exp scaling and test
  the merged result within floating-point tolerance. Reuse shared K/V reads,
  but do not assume queries for different requests are identical.
- **SFU bottlenecks invite a numerical experiment.** The FA4 code analysis
  motivates mixing polynomial exponentials with hardware exponentials. Measure
  the relevant input range, tails and error propagation before applying an
  approximation; a bf16-specific observation does not establish fp32 parity.
- **Choose scheduler against the tail, not only steady-state throughput.** Paul
  Chan's static-persistent/CLC comparison motivates testing awkward tile counts,
  load imbalance and work-acquisition overhead. Re-prove cluster synchronization
  and paired-CTA ownership when changing schedules.
- **Debug asynchronous execution without timing the debug synchronization.**
  Wilt's articles distinguish launch-time errors from later execution failures.
  Check immediate launch status and use an explicit completion boundary to
  attribute execution failures; keep extra diagnostic synchronization out of
  production benchmark timings.
- **Algorithm substitutions have crossover points.** The Nabla FFT article
  motivates comparing direct and FFT convolution across filter/image sizes,
  including transforms, padding, temporary allocation and output cropping.
  Preserve linear-versus-circular convolution semantics.

### Unverified lead

[Poppy: Writing PTX for Consumer Blackwell](https://getpoppy.app/blog/writing-ptx-for-consumer-blackwell)
was discovered through search. Both direct retrieval and the web reader failed
(the latter returned HTTP 502). Its search excerpt describes a bank-conflict
removal regression, but this has NOT been promoted to a reviewed optimization
rule. Retrieve the full article and code before using that claim.

### Additional retrieval inventory

- body retrieved: https://www.chanpaul.com/
- body retrieved: https://flashinfer.ai/posts/
- body retrieved: https://test.kevinhuai.com/triton/
- body retrieved: https://vllm-project.github.io/2026/10/16/kimi-k3-megakernel.html
- body retrieved: https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization
- body retrieved: https://hamzaelshafie.bearblog.dev/
- body retrieved: https://www.cudahandbook.com/blog/
- body retrieved: https://modal.com/blog/reverse-engineer-flash-attention-4
- body retrieved: https://deftruth.github.io/
- body retrieved: https://nathan.ng/
- body retrieved: https://www.chanpaul.com/articles/outperforming-cublas-b200
- body retrieved: https://hamzaelshafie.bearblog.dev/blog/
- body retrieved: https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/
- body unavailable: https://getpoppy.app/blog/writing-ptx-for-consumer-blackwell
- body retrieved: https://flashinfer.ai/2026/09/22/autotuner-v2.html
- body retrieved: https://flashinfer.ai/2026/09/22/mega-moe.html
- body retrieved: https://flashinfer.ai/2025/03/10/sampling.html
- body retrieved: https://flashinfer.ai/2024/02/02/cascade-inference.html
- body retrieved: https://graphics-programming.org/blog/fft-bloom-optimized-to-the-bone-in-nabla
- body retrieved: https://www.cudahandbook.com/blog/asynchronous-error-handling-is-hard
- body retrieved: https://www.cudahandbook.com/blog/cudagetlasterror-its-raison-detre
- body retrieved: https://github.com/xlite-dev/LeetCUDA/blob/main/README.md
- body retrieved: https://hamzaelshafie.bearblog.dev/dissecting-thunderkittens-anatomy-of-a-compact-dsl-for-high-performance-ai-kernels/
- body retrieved: https://hamzaelshafie.bearblog.dev/awq-activation-aware-weight-quantisation/
- body retrieved: https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/
