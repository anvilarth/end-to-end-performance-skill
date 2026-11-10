# GLM-5.3-Flash: H200 disaggregated serving

[Serving playbook](../playbooks/llm-serving.md)

[Primary source: LMSYS, November 1, 2026](https://www.lmsys.org/blog/2026-11-01-glm53-h200-pd).
Reviewed November 10; not locally reproduced.

## Reported scope

Text-only SGLang, 64K inputs. Integrated
3P1D uses sixteen H200s, not four. Its gains span 2.11–4.28× across prefix
regimes; baseline and optimized client loads differ. Mean latency targets
are not tail-latency guarantees. Vision encoding is excluded.

## Conditional experiments

- Budget KV, recurrent state and speculative buffers separately. Share prefix
  KV without sharing request-local recurrent state; verify boundary correctness.
- Compare speculative cycle time per accepted token, not acceptance alone.
- Tune MoE with real routing distributions and held-out routes, not token
  count alone; validate quantization with nonuniform scales.
- Probe FP8 communication/reuse and fusion while preserving rounding;
  measure handoff and synchronization, not just GEMMs.
- Sweep prefix reuse and P/D allocation. Low reuse can remain prefill-bound
  despite faster decode. Report actual output and logical cached-input rates
  separately, with tails and quality gates.

Reproduction requires the article's custom SGLang revision
`3cb45d3dc4e9a0d6aae90c4efc0e87aa9db75f2e`; linked launch commands are
version-specific evidence, not portable defaults.
