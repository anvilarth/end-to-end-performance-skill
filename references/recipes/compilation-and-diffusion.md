# Compilation, engine selection and diffusion delivery

Use the [Compilation source audit](../sources/compilation-reading.md) for provenance
and the [framework atlas](../sources/gpu-docs-atlas.md) for version-matched APIs.

## Compiler versus serving engine

Triton the GPU language/compiler and NVIDIA Triton Inference Server are distinct.
Torch-TensorRT is a PyTorch/TensorRT integration; Model Navigator searches export
and runtime configurations; neither replaces application-level measurement.

For a compiler/backend comparison, record capture coverage, partition boundaries,
fallback operations, layout/dtype conversion, host launches, build memory, first
request latency, warmed latency, supported shapes and artifact load behavior.
A successful export or one fast engine segment does not establish a fast pipeline.
Test min/opt/max shapes and shape transitions, including singleton dimensions.
The FLUX Torch-TensorRT tutorial explicitly discusses 0/1 specialization and
build-memory constraints; treat its settings as a versioned example, not current
universal requirements.

Compare cold compile, warm disk cache, warm process and serialized artifact load
separately. The incremental setup break-even is
`ceil(extra_setup_seconds / (control_seconds_per_unit - candidate_seconds_per_unit))`
only when the denominator is positive and both per-unit boundaries match. Include
rebuilds, shape changes and restart frequency in the expected lifetime. Validate
reload in the real consumer, not just export in the builder.

For training, inspect graph breaks, guards and recompiles across forward,
backward, optimizer and distributed communication. Regional compilation can trade
cross-block fusion for reusable compilation. Do not assume a compiled region is
bitwise equivalent: fused casts, reductions and backend selection can change
rounding. A larger backward region can delay gradient availability and affect
communication overlap. These are diagnostic routes from
[ezyang's dated training report](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/);
verify current behavior instead of carrying forward its old support matrix.

## Diffusion probes in increasing semantic risk

Sources: [Aryan I](https://a-r-r-o-w.github.io/blog/3_blossom/00001_productionizing_diffusion-1/index.html),
[II](https://a-r-r-o-w.github.io/blog/3_blossom/00002_productionizing_diffusion-2/index.html),
[III](https://a-r-r-o-w.github.io/blog/3_blossom/00003_productionizing_diffusion-3/index.html),
[IV](https://a-r-r-o-w.github.io/blog/3_blossom/00004_productionizing_diffusion-4/index.html).

1. Measure text/tokenization, embeddings, denoiser steps, VAE, staging, load and
   output encoding. Retain the full generation boundary; transformer-only source
   measurements omit real deployment costs.
2. Compare attention backends, supported layouts and compilation on actual shapes.
   Remove proven redundant materialization; consider norm/modulation/gate fusion
   and the model's exact RoPE convention. Reuse buffers only with valid stream and
   lifetime handling.
3. Precompute only values invariant for the stated request/step. Cross-attention
   K/V reuse is exact only while conditioning, weights/adapters and relevant
   transformations remain unchanged. Invalidate caches on all dependencies.
4. Test streams only for independent operations with complementary resource use.
   Concurrent compute-saturating GEMMs can contend. Observe overlap, events and
   memory lifetimes; do not infer speed from asynchronous API calls.
5. For LoRA, measure merge/unmerge, per-request switching and first request after a
   switch. Preserve scales, adapter identity, multi-adapter composition and cache
   invalidation. A benchmark with one fixed merged adapter does not cover hotswap.
6. Quantization, approximate attention and reusing changing hidden states require
   an authorized quality trade-off. Test multiple prompts/seeds and complete
   trajectories; small one-step errors can accumulate. Retain failures as well as
   accepted samples. Do not label these lossless because one image looks similar.

## Distributed and CPU boundaries

[ezyang's einsum method](https://blog.ezyang.com/2026/01/computing-sharding-with-einsum/)
helps derive placements: sharding a contracted dimension creates a partial result
that must be reduced when a replicated result is needed. Derive backward
placements too; verify communication on all ranks and overlap in the real trace.

The ARM64 tokenizer posts in the audit suggest checking CPU preprocessing when it
is on the critical path. Require exact token IDs, normalization, special tokens,
Unicode/byte fallback and multilingual parity. A fast unknown-token path is not
proof of an equivalent tokenizer. Native SIMD changes are architecture-specific.

For execution-aware autotuning, independent selection evaluation and persisted
cache boundaries, see the [additional blog transfer tests](../sources/gpu-blog-deep-dive.md#additional-transfer-tests).

## Execution-aware tuning and compiler changes

Source: [FlashInfer Autotuner v2](https://flashinfer.ai/2026/09/22/autotuner-v2.html).
When the selected tactic wins a kernel benchmark but loses the application:

1. Measure eager calls including recurring host/backend work separately from CUDA
   Graph replay. Keep setup/capture out only if the deployment amortizes them.
2. Tune using the deployment boundary; record environment, shape, dtype, backend
   revision and measurement policy with persisted choices. Test cache invalidation,
   fresh-process loading and rank agreement where the operation requires it.
3. Evaluate selection using independent samples. Report latency regret
   `(selected_latency - best_latency) / best_latency`, alongside absolute time;
   ties and tiny differences should not count like catastrophic misses.
4. Reject the tuning change if its search/setup cost cannot be amortized or the
   held-out end-to-end comparison fails. Kernel winner identity alone is not success.

For a warm compilation cache that still starts slowly, separately time in-process
reuse, fresh-process disk-cache reuse and cold compilation. Trace lookup, artifact
loading and runtime initialization before attributing residual time to compilation
([Zalt's code-reading lead](https://zalt.me/blog/cache-hit-not-free)). Pin the actual
compiler revision; the article's main-branch links are not immutable evidence.

When modifying compiler passes, validate typed SSA and use-def/type contracts after
each transformation; locate the first invalid pass before debugging generated CUDA.
[Pavel's educational verifier](https://pbelevich.github.io/2026/06/26/My_Triton_From_Scratch_Part_5_Verification.html)
is a model for this diagnostic boundary, not a production compiler performance
claim. IR verification still requires numerical execution tests of lowered code.
