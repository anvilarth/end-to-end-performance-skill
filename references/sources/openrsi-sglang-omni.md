# OpenRSI / SGLang-Omni: independent promotion

[Article, November 9, 2026](https://index.openrsi.foundation/blog/2026/openrsi-sglang-omni/).
Reviewed November 10. External evidence, not local performance validation.

## Transfer to the optimization loop

Separate candidate search from acceptance: reproduce the exact submitted patch
with a protected evaluator in a fresh environment. Only confirmed candidates
advance the accepted baseline. Preserve a frozen reference for cumulative gains.
Check affected neighboring models and streaming/non-streaming paths, not merely
the optimized lane. Quality metrics must cover the actual output contract:
transcription WER alone does not establish perceptual speech quality.

## Concrete profiling probes

- Idle worker spinning can steal CPU capacity from other stages. Compare CPU
  budgets and end-to-end completion, not just isolated encoder latency.
- Inspect tensor/list/tensor conversions at stage boundaries, duplicate frontend
  work, unused reconstruction and repeated reference preprocessing.
- Measure graph/fusion and asynchronous-stage changes together with startup,
  peak memory and first-output/tail latency.

## Evidence limits

Article commit ranges sometimes include unrelated changes, different images,
new defaults or a baseline bug fix; do not attribute their full gain to one patch.
[PR #2524](https://github.com/sgl-project/sglang-omni/pull/2524) isolates CPU spinning
and retains conflicting low-concurrency evidence.
[PR #2881](https://github.com/sgl-project/sglang-omni/pull/2881) retains latency and
quality tradeoffs. Faster throughput and passing CI alone do not establish
acceptance against a workload's gates.
