# Broad GPU optimization search — 2026-11-06

This pass added 10 sources and 13 decisions (D070–D082). Total: 45 sources,
82 decisions, 12 negative-tagged records. All are external evidence; no local
performance experiment ran. Five new papers were reviewed at abstract level only:
instruction-wall, M2N, JPEG v3, power-caps v2, and sparse FFT. Their records do not
claim to extract the full papers. Blog records also cover selected passages only.

## Coverage map

| Mechanism | Current evidence or next route |
| --- | --- |
| Hidden host/device sync | D070; indexing and logging are part of the hot path |
| Coalescing, fetch granularity, caches | D071–D072; a driver hint did not produce the expected change |
| Instruction throughput | D073; database workloads expose a different bound than HBM bandwidth |
| Resharding and interconnect topology | D074; distinguish layer transfer from whole RL step |
| CPU image decode | D075; CPU and image distribution change decoder rankings |
| Hardware video decode | D076–D077; throughput and decoder/KV memory competition |
| Graph segmentation, startup, diffusion | D078–D079; persistent buffers and shape coverage matter |
| Storage and GPUDirect | D080; direct path lost to host staging in one setup |
| Power and energy | D081; workload-dependent trade-off, no global cap recommendation |
| Sparse algorithms and capacity | D082; keep lossless and approximate variants distinct |
| GEMM, attention, TMA, warp specialization | Existing corpus; inspect linked kernel revisions for more failed variants |
| Training input overlap and worker contention | Existing Aram cases; expand distributed input imbalance |
| Compilation/autotuning | Existing Helion cases; compile controls lead in MODELS-2026.md |
| Allocator fragmentation and graph pools | Lead below; extract exact allocation/lifetime cases next |
| AMD profiling and portability | Lead below; need architecture-specific case extraction |
| Checkpointing and restart | Search returned mostly 2024/2025 writeups; find verified 2026 experiments |
| Multi-tenancy/MPS/MIG, NUMA locality | Lead below; distinguish on-GPU locality from host NUMA |
| Communication overlap and stragglers | Leads below; rank timelines before assuming network saturation |

## Retained leads, not counted as new extracted decisions

- [CUDA allocator fragmentation, Edward Yang, June 1](https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/): date and introduction inspected; read pool and expandable-segment cases next. Author discloses AI-assisted drafting.
- [GPU stragglers, k10s, May 25](https://blog.k10s.dev/understanding-straggler-ranks-and-nccl-basics/): opening logs inspected. Introductory glossary calls world size GPU cores; do not inherit that terminology (world size counts processes).
- [MIG/locality domains, NVIDIA](https://developer.nvidia.com/blog/accelerating-data-processing-with-nvidia-multi-instance-gpu-and-numa-node-localization/): search returned February 19; verify full page and experiments before extracting.
- [AMD profiling tools](https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html): search snippet shows April 10, 2026 but index metadata differs; verify revision/date and follow hands-on series.
- [CUDA graph batch sizes, TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html): verify date and extract padding/startup/memory trade-off.
- [Graph traversal on tensor cores](https://arxiv.org/abs/2606.05081): verify full method and correctness/performance evidence before adding cases.
- [GPU memory deduplication, gKSM](https://arxiv.org/abs/2610.25219): inspect driver/VMM assumptions and isolation trade-offs.
- [Network failures and AllReduce](https://arxiv.org/abs/2606.01680): search result reports SimAI evaluation; distinguish simulation from hardware measurements.

## Search log and exclusions

Twelve query routes used combinations of `2026 GPU optimization` with:
memory bandwidth/engineering; NCCL/stragglers; data loading/decode; register spills;
PyTorch checkpointing; AMD profiling; power/NUMA/PCIe; video pipelines;
sparse/graph/FFT/stencil; CUDA graphs/fragmentation; GPUDirect; power capping.
Followed primary blog and arXiv URLs and read selected passages. Search snippets
were discovery leads, not sufficient support for unseen implementation details.

Dropped old results despite recent crawl dates: NVIDIA shared-memory register
spilling (2025), PyTorch async checkpoint blogs (2024/2025), and a 2025 energy
paper. Did not extract an introductory 2026 matmul retelling as an independent
experiment: it explicitly reuses an older worklog's measurements. Excluded a
search result claiming TF32 halves parameter storage: that is not a sound basis
for a memory optimization recommendation.

The JPEG search result still described an older title/workload while arXiv v3
contained a different experiment. D075 is explicitly tied to v3. This illustrates
why dates, versions and primary passages must be checked before adding a case.

## Next bounded collection pass

Read the retained leads, follow their implementation PRs and reverts, and record
one decision per verified attempt. Expand gaps in collectives, checkpointing,
allocator behavior, AMD, sparse/scientific kernels and multi-tenancy. Keep absent
measurements unknown and all search/review gaps explicit. No background search
or scheduled maintenance is running; this is not exhaustive internet coverage.
