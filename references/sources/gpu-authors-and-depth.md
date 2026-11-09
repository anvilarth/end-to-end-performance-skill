# Authors and a deep GPU engineering route

Primary pages opened on 2026-11-09. This is a curated route, not a ranking by
popularity or a claim of exhaustive coverage of the web. Blog measurements remain
external evidence; no kernels were run for this collection. Work through one
mechanism, its official specification and its implementation together.

## Closest matches to Veitner

| Author / collection | Start here | Why read it / transfer boundary |
| --- | --- | --- |
| Simon Veitner | [Full index](https://veitner.bearblog.dev/blog/), [layout algebra](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/), [Blackwell pipelines](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) | A progression from layout math to complete CuTe kernels. Match the exact CUTLASS release; not all posts are performance experiments. |
| gau-nernst | [tcgen05](https://gau-nernst.github.io/tcgen05/), [CuTe TMA](https://gau-nernst.github.io/cutedsl-tma/), [5090 attention](https://gau-nernst.github.io/fa-5090/) | Plain CUDA/PTX and concrete kernel iterations; explicitly separates SM100 from SM120. A single shape near peak is not broad dispatcher coverage. |
| Jay Shah / Ganesh Bikshandi and Colfax Research | [WGMMA](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/), [TMA](https://research.colfax-intl.com/tutorial-hopper-tma/), [pipelines](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/), [persistent/Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) | Connect instruction semantics to working CUTLASS schedules. Read their [transpose study](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/) for mixed TMA outcomes rather than assuming newer instructions win. |
| Alex Armbruster | [Tensor-core matmul from scratch](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) | Deep derivation of rooflines, hierarchy, indexing, swizzles and double buffering on T4. Newer async hardware changes the implementation. |
| Sonny / lubits.ch | [FlashAttention series](https://lubits.ch/flash/), [A100 profiling](https://lubits.ch/flash/Part-7), [instruction reduction](https://lubits.ch/flash/Part-8), [benchmark appendix](https://lubits.ch/flash/Appendix) | Repeated kernel evolution, profiling and SASS diagnosis. Parts 9/10 are listed as forthcoming in the index; do not pretend they were reviewed. Read comparator modifications in the appendix. |
| Sam Patterson | [Ada tensor-core GEMM](https://spatters.ca/mma-matmul), [two-stage accumulation](https://www.spatters.ca/two-stage-fp16-mma) | `mma`, `ldmatrix`, shared-memory permutation and async stages. The second article is a precision/speed trade-off, not FP32-accumulation parity. |
| Maharshi | [SGEMV](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/), [softmax](https://maharshi.bearblog.dev/optimizing-softmax-cuda/), [identity tensors](https://maharshi.bearblog.dev/identity-tensors-in-cutedsl/), [CuTe gotchas](https://x.com/maharshii/status/2082921705941651595) | Accessible worklogs plus real indexing/JIT failure modes. Early CUDA results use a 1050 Ti; verify each CuTe anecdote against the current API. |
| Simon Boehm | [CUDA SGEMM worklog](https://siboehm.com/articles/22/CUDA-MMM) | A clean causal progression from memory coalescing to tiling and tuning. FP32 CUDA-core comparison is different from tensor-core mixed precision. |
| Lei Mao | [GEMM optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/), [CuTe transpose](https://leimao.github.io/article/CuTe-Matrix-Transpose/), [MMA peak probes](https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/) | Independent derivations and executable details for alignment, bank conflicts and instruction ceilings. Microbenchmarks require anti-optimization safeguards. |
| Kapil Sharma | [CUTLASS the hard way I](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/), [II](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/) | Long worklogs connecting elementary GEMM to low precision and Hopper. Use the matching hardware section instead of merging 4090 and H100 assumptions. |
| Tri Dao and collaborators | [FlashAttention-3](https://tridao.me/blog/2024/flash3/), [SonicMoE on Blackwell](https://tridao.me/blog/2026/sonicmoe-blackwell/), [Gram Newton-Schulz](https://tridao.me/blog/2026/gram-newton-schulz/) | Algorithm/hardware co-design, overlap, grouped work and algebraic reformulation. Separate exact reorganization from changes to precision or optimization behavior. |
| Horace He | [Making Deep Learning Go Brrr](https://horace.io/brrr_intro.html) | Systems-level basis for compute, memory and overhead bottlenecks. Use to form a budget, not as current API documentation. |
| Hazy Research | [Research blog](https://hazyresearch.stanford.edu/blog), [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | Kernel architecture and tiled abstractions from their implementers. Check source revision and actual supported kernels. |
| AlpinDale | [Top-K](https://blog.alpindale.net/posts/top_k_cuda/), [5090 decode](https://blog.alpindale.net/posts/5090_decode_optimization/), [blog](https://blog.alpindale.net/) | Assembly, CPU tokenization and whole decode paths, including unsuccessful tuning. Inspect comparator semantics and do not transplant persistent-kernel barriers without proving residency/progress. |
| Aryan V S | [Intra-kernel trace](https://a-r-r-o-w.github.io/blog/3_blossom/00005_looking_inside_your_kernels/index.html), [diffusion series](https://a-r-r-o-w.github.io/blog/) | Links kernel mechanisms to multi-component diffusion deployment. Preserve transformer-only versus full generation boundaries. |
| Enbao Cao | [ThunderKittens Python bindings](https://enbao.me/posts/tk) | Explains the launch/binding boundary, streams and tensor wrappers. Measure host interop as well as kernel time. |
| Raph Levien | [A good parallel computer](https://raphlinus.github.io/gpu/2025/03/21/good-parallel-computer.html) | Broadens thinking beyond dense ML into dynamic work and execution models. A conceptual complement, not a CUDA recipe. |

## Study route with concrete outputs

These are optional learning exercises. Do not launch paid compute or alter a live
workload merely because a reading route suggests an experiment.

1. **Bounds and evidence.** Horace → Boehm → Armbruster. Derive FLOPs and bytes for
   one real shape; distinguish HBM/L2/shared-memory traffic and instruction issue.
   Deliver a predicted bound, a measured baseline and a statement of what is unknown.
2. **Layouts as executable mappings.** Veitner layout algebra → hierarchical
   layouts → slicing → partitions → swizzles → Maharshi identity tensors. Map a
   tiny non-square tile by hand and verify coordinate coverage, uniqueness and
   masks with a host-side enumerator. Distinguish an address, layout and iterator.
3. **Movement and synchronization.** Patterson → Colfax TMA/WGMMA/pipelines →
   Veitner producer-consumer. Draw ownership of every shared buffer by stage,
   expected arrivals/transactions and release condition. Check tails, phase wrap
   and async proxy ordering against PTX before timing.
4. **Blackwell details.** gau-nernst tcgen05 → Veitner block-scale setup, scale
   layout, conversions, grouped host code and kernel → 2-CTA → tile scheduler.
   Deliver an SM-specific feature matrix and a correct ragged/grouped case.
   SM100, SM120 and Hopper are different targets, not one “modern GPU” target.
5. **See inside execution.** Aryan trace → Proton → IKET → Nsight/SASS. Answer one
   producer/consumer question with a small trace; quantify instrumentation's
   resource/timing effect and validate again without it.
6. **Complete attention.** lubits Parts 1–8 and appendices → gau-nernst 5090 →
   Tri Dao FA3. Preserve masks, scaling, output/LSE and backward requirements.
   Compare scalar issue pressure and register use, not just tensor FLOPs. The
   A100/3090 divergence in lubits Part 7 is a useful architecture-transfer test.
7. **Non-GEMM and algebra.** Veitner reductions/radix Top-K/GDN → Alpin tokenizers
   and decode. Prove the operator contract and recurrent state; keep negative
   tuning results. Avoid optimizing sorting when selection suffices, or repeating
   algebraically redundant work before descending to instruction tuning.
8. **Deliver to the application.** Aryan diffusion → ezyang compile/sharding →
   engine docs → Enbao bindings. Measure export/build/load, stream interop,
   batching and complete output production. End with a composed end-to-end gate.

## Finding the next layer of authors

Use primary-source citation chains, not generic “top CUDA blogs” lists: follow
acknowledgements, benchmark code, compiler issues and implementation authors.
For each new lead retain a concrete article and code, hardware/precision/shape,
negative result or caveat, and a question the source can resolve. Search exact
mechanisms (`mbarrier phase`, `tcgen05 TMEM`, `scale layout`, `SASS register moves`)
with author domains. Prefer an original measurement over reposts of the same work.
A title found in a list remains a lead until its body is opened.

## Further blog research

See [additional authors, AMD route and twelve transfer tests](gpu-blog-deep-dive.md).
