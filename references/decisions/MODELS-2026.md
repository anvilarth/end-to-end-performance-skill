# Models and search policies — 2026-11-06 research snapshot

Historical research note. Model deployment and comparisons are outside the active
documentation-only scope; follow [WORKFLOW.md](WORKFLOW.md) instead.

These are candidates, not locally validated performance improvements. This note is separate from the 35-source / 69-decision corpus counts.

## Concrete model candidates

[daVinci-kernel paper, June 15](https://arxiv.org/abs/2606.16497) couples selection, generation and skill summarization. Its central transferable idea is to verify reuse by execution before admitting a skill.

The [official repository](https://github.com/GAIR-NLP/daVinci-kernel) publishes Qwen3-based 8B/14B SFT and RL checkpoints. Selection uses BM25 then reranking; the RL releases include a skill library. This makes it unusually relevant to a decision-corpus experiment. Keep its kernel scope explicit: it does not establish end-to-end training optimization ability.

Verified public model pages:
- [14B RL](https://huggingface.co/SII-GAIR-NLP/daVinci-kernel-14B-RL)
- [8B RL](https://huggingface.co/SII-GAIR-NLP/daVinci-kernel-8B-RL)

Both pages expose BF16 safetensors metadata but have no model card and no hosted inference provider listed. We have not downloaded, loaded or benchmarked either checkpoint. Recommendation: use 8B as the cheaper experimental arm and 14B as a capacity comparison, not assume 14B wins locally.

[CUDA Agent, February 27](https://arxiv.org/abs/2602.24286) is a useful training-method reference. Its [official repository](https://github.com/BytedTsinghua-SIA/CUDA-Agent) announces data, skill and agent environment releases; a trained-weight release was not found there in this review. Do not treat it as an immediately available checkpoint.

## Policies worth borrowing

- [AutoProfiler, October 27](https://arxiv.org/abs/2610.36576): learns a kernel-conditioned profiling tree using evaluated REWRITE/SPLIT proposals. The abstract reports that more counters do not monotonically help. Candidate transfer: choose the next measurement by its ability to distinguish competing bottleneck hypotheses. Only abstract reviewed; local value untested.
- [KernelPro, 2026, section 3.4](https://arxiv.org/html/2606.26453v2): structured search memory had statistically equivalent final speedup in a 42-task ablation, with an early-convergence trend. Treat memory utility as an experimental question. Its semantic microprofilers suggest converting evidence into ranked, falsifiable hypotheses rather than dumping raw counters.
- [Helion/CompileIQ, October 19](https://pytorch.org/blog/one-search-two-layers-compiler-controls-inside-pytorch-helion-using-nvidia-compileiq/): searches compiler controls alongside kernel configuration, starting from already tuned kernels. Candidate transfer: when source-level tuning plateaus, check compatible compiler-control catalogs before rewriting the algorithm. Confirm toolchain and GPU support first.

## Proposed decision experiment (not run)

Hold task set, hardware, wall-clock budget and correctness contract fixed. Compare no retrieved cases, relevant cases, and shuffled cases. Include held-out workload families, failed cases and an explicit no-change option. Measure correct full-path speedup, time to first accepted improvement, total search cost and regressions. Separately score retrieval relevance and whether the chosen experiment discriminated between hypotheses. Evaluate model size independently from memory/policy changes.

Do not import repository thresholds as local acceptance criteria. Kernel-level gains remain diagnostic until representative full-path validation passes. All sources here concern 2026; coverage is selective and upstream revisions are not pinned.
