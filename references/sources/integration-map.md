# Integration of the Compilation and GPU-blog collection

Integrated on 2026-11-09. This map covers all collected source families, not every
outbound link as a reviewed technical claim. Downloading a page does not establish
exhaustive review. Recipes contain conditional tests; none of these imported
results is labeled locally GPU-validated.

| Collected material | Operational owner | Evidence / remaining boundary |
| --- | --- | --- |
| 21 Compilation tabs: compilers, export, engines and diffusion | [Compilation recipe](../recipes/compilation-and-diffusion.md), handbook 03 | [Audit and original manifest](compilation-reading.md); versioned examples, not default backend settings |
| Aryan intra-kernel article and its 11 docs/code links | [Profiler selection and instrumentation](../recipes/intra-kernel-and-cute.md#choose-the-smallest-observable-boundary) | Distinguishes issue/completion, instrumented/uninstrumented timing and cache behavior |
| Veitner 66-post index and selected technical passages | [CuTe correctness, architecture probes and algorithm checks](../recipes/intra-kernel-and-cute.md) | Full inventory in audit; not every code line or experiment reproduced |
| Original 17 author/collection routes | [Backend selection](../recipes/kernel-backends.md), [kernel diagnostics](../recipes/intra-kernel-and-cute.md) | [Author route](gpu-authors-and-depth.md) retains source-specific hardware/numerical boundaries |
| Pranjal, Daniel, Yifan, Axel, Salykov, Mattana | [Operand delivery, layout and scheduler experiments](../recipes/intra-kernel-and-cute.md#operand-delivery-layout-and-scheduler-experiments) | D093–D095 preserve selected 2026 outcomes; older sources stay in recipes |
| Lei Zhang, Pavel, Zalt, Kevin Huai | [Compiler/tuning recipe](../recipes/compilation-and-diffusion.md#execution-aware-tuning-and-compiler-changes), [backend routes](../recipes/kernel-backends.md#additional-implementation-routes-when-evidence-calls-for-them) | D099–D100; Pass Explorer inventory is discovery, interactive traces not reviewed |
| AMD authors, Modular, other framework docs | [Additional implementation routes](../recipes/kernel-backends.md#additional-implementation-routes-when-evidence-calls-for-them) | D098 and [docs atlas](gpu-docs-atlas.md); no blanket framework recommendation |
| Hamza, Paul Chan, Modal FA4 | [Kernel scheduling probes](../recipes/intra-kernel-and-cute.md#operand-delivery-layout-and-scheduler-experiments), [attention gates](../recipes/specialized-bottlenecks.md#attention-states-sampling-and-alternative-algorithms) | Full provenance in [deep-dive](gpu-blog-deep-dive.md); WIP, historical code and approximation limits preserved |
| FlashInfer sampling, cascade, autotuner, MegaMoE | [Algorithm probes](../recipes/specialized-bottlenecks.md#attention-states-sampling-and-alternative-algorithms), [execution-aware tuning](../recipes/compilation-and-diffusion.md#execution-aware-tuning-and-compiler-changes) | D096–D097; runtime boundary, sampling distribution and shape-dependent regressions matter |
| Wilt CUDA runtime articles | [Validation handbook](../handbook/07-validation-and-delivery.md#runtime-errors-during-kernel-validation) | Original recommendation and corrective follow-up read together |
| Nabla graphics/FFT worklog | [Algorithm crossover probe](../recipes/specialized-bottlenecks.md#attention-states-sampling-and-alternative-algorithms) | Includes transform costs and convolution boundary semantics |
| DefTruth, Nathan Nguyen, vLLM/SGLang project posts | [Discovery graph](gpu-doc-links.jsonl), [review scope](gpu-blog-deep-dive.md) | Useful project/patch leads; unaudited linked code is not promoted to a proven recipe |
| Poppy and failed Stephen Diehl URL | [Unavailable leads](gpu-blog-deep-dive.md) | No body-based decision extracted; search snippets do not establish experiments |
| LMSYS GLM-5.3-Flash H200 PD case (November 1, added November 10) | [Serving playbook](../playbooks/llm-serving.md#hybrid-models-with-disaggregated-prefill-and-decode) | [Reviewed case and transfer experiments](glm53-h200-pd.md); custom revision, external evidence only |
| SGLang NVIDIA Mega (reviewed November 10) | [Whole-step decode route](../recipes/intra-kernel-and-cute.md#optimize-a-whole-decode-dependency-chain) | [Pinned code and benchmark review](sgl-nvidia-mega.md); conditional dispatch, synthetic acceptance and composition boundaries |
| Remaining outbound docs, code, blogs | [Searchable link graph](gpu-doc-links.jsonl) | Explicit discovery records; consult exact target and revision for a live decision |

## Retrieval from a real task

1. Start with the relevant handbook chapter and its recipe, not the whole source graph.
2. Search an observed mechanism, e.g. `python3 scripts/decision_library.py search "autotune" --limit 3`.
3. Find adjacent official APIs/code with `python3 scripts/source_links.py "Proton" --limit 5`.
4. Inspect the source conditions, predict recoverable full-path time and run one falsifier.
5. Record the local outcome in the workload log; never overwrite the external author's outcome.

Structural and navigation checks establish usable documentation, not improved
kernel speed or behavioral evaluation on a new optimization workload.
