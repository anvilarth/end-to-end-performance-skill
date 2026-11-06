# Engine-specific reading map — 2026-11-06

This pass adds seven sources and ten decisions, D083–D092. Corpus totals:
52 sources, 92 decisions, 13 negative-tagged records. Selected passages have
been reviewed; no article or engine is exhaustively covered. No local engine
benchmark ran. Search remains scoped to 2026 publications or explicit updates.

## Read and extracted

| Engine/layer | Source | Decision focus |
| --- | --- | --- |
| SGLang | [DSpark, July 6](https://www.lmsys.org/blog/2026-07-06-dspark-sglang) | Variable verification budget and actually smaller CUDA graph replay |
| SGLang | [HiCache, November 2](https://www.lmsys.org/blog/2026-11-02-hicache-distributed-serving) | Sharded cache, bounded gather buffers and restore ordering |
| SGLang | [Request isolation, October 26](https://www.lmsys.org/blog/2026-10-26-cf-security) | Negative case: reusing KV slots before remote writers finish |
| vLLM | [Push NIXL, October 14](https://vllm.ai/blog/2026-10-14-nixl-push-connector) | Move allocation and transfer setup off critical path |
| vLLM | [Bytecode startup, September 22](https://vllm.ai/blog/2026-09-22-python-bytecode-startup) | Build-time Python compilation; separate image delivery cost |
| TensorRT-LLM | [Overlap scheduler, June 27 update](https://nvidia.github.io/TensorRT-LLM/1.3.0rc20/torch/features/overlap_scheduler.html) | CPU/GPU overlap versus extra decode work |
| Dynamo | [Agentic inference, April 17](https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/) | Cache-aware routing and backend priority normalization |

Dynamo is an orchestration layer, not a replacement inference kernel engine.
The previously extracted SGLang CUDA Graph and vLLM video-decode cases remain
in the same corpus; they were not counted again.

## Blog entry points and targeted follow-ups

- [LMSYS blog](https://www.lmsys.org/blog/) and [SGLang blog](https://www.sglang.io/blog): compare canonical links before adding mirrored posts. Index results expose Waterfill/LPLB MoE balance, elastic EP, unified radix cache, fast recovery and model-specific optimization diaries. Bodies not extracted in this pass.
- [vLLM blog](https://vllm.ai/blog): follow disaggregated serving, bidirectional KV transfer, adaptive verification, elastic EP and AMD optimization diaries. Model release announcements are useful only when they describe actual engineering decisions.
- [TensorRT-LLM expert parallelism part 3](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.html): body/date still need targeted review. Do not infer a publication year from the docs footer or search crawl date.
- [TensorRT-LLM guided plus speculative decoding](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html): investigate feature-combination costs; date and decisions pending.
- [vLLM WideEP on Blackwell](https://vllm-project.github.io/2026/02/03/dsr1-gb200-part1.html): February 3 index result; inspect overlap, batching and communication trade-offs next.
- [SGLang unified radix cache](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache): resolve canonical body and compare hybrid-state requirements, not just hit rate.

These follow-ups are leads, not additional reviewed sources. Other engines
such as llama.cpp and LMDeploy are not yet covered by this pass.

## How to extend by engine

For each engine, traverse official blog -> design/feature docs -> linked PR ->
benchmark script -> regressions/reverts. Record version, GPU, batch/concurrency,
prompt/output distribution, cache state, precision and measurement boundary.
Search the same mechanisms across engines: scheduling, cache ownership,
disaggregation, speculative execution, graph capture, quantization and startup.
A feature name shared by engines does not guarantee equivalent semantics.

For each source distinguish: stated design, measured decision, unsupported
claim, and open question. Extract failed combinations and lifetime bugs as well
as speedups. Keep engine-specific limits attached to the record rather than
turning a release claim into a general skill rule. The concrete checklist lives
in WORKFLOW.md; the searchable records remain in decisions.jsonl.
