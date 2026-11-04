# End-to-End Performance Skill

An agent skill for optimizing PyTorch and CUDA training and inference workloads
as complete systems.

The skill first recovers actual usage from launchers and existing artifacts: hardware, scale,
topology, workload shapes, modes, concurrency, cold or warm behavior, and the
correctness contract. It then profiles the full execution path, ranks measured
bottlenecks by recoverable end-to-end impact, applies lossless changes first,
and screens candidates cheaply before representative correctness and performance gates.
Existing authorization is reused; only decision-changing gaps require questions.

[The decision map](references/decision-map.md) brings the workflow into one
flowchart: what to check first, which diagnostic tool to use, where to optimize
next, and when to stop or promote a change.

Its central rule is simple:

> Optimize the delivered workload, not an isolated component.

## How to use it

Install the skill, then invoke it on a PyTorch or CUDA training or inference
workload:

```text
Use $end-to-end-performance-skill to profile and optimize this PyTorch workload.
```

Provide the repository or code path and describe the real usage setup when it is
known. The skill will ask for any missing details needed to define a
representative benchmark and correctness contract.

The workflow is:

```text
Establish the real usage setup
-> Freeze the correctness and performance contract
-> Measure the complete end-to-end path
-> Estimate recoverable headroom
-> Rank lossless-first candidates
-> Implement one causal change
-> Validate parity before performance
-> Measure the representative end-to-end workload
-> Promote, reject, or mark inconclusive
-> Re-profile because the bottleneck may have moved
```

Every contract, baseline, hypothesis, gate result and verdict is recorded using
the workload's existing experiment log, or `.perf-log/log.jsonl` if none exists.
A new session reads the record first, so rejected and inconclusive ideas are not
repeated and open next actions are picked up. The loop checks control/comparator
readiness before expensive runs and reserves time for final delivery validation.

## Kernel-level escalation

The skill does not begin with custom kernels, but it descends to them as soon
as a kernel profile shows recoverable end-to-end budget.
[references/kernel-dsls.md](references/kernel-dsls.md) picks the path from the
profile evidence: swap an existing library backend, Triton for memory-bound
fusion, TileLang for tile-level GEMM, attention, and recurrence variants, CuTe
DSL when the ceiling needs architecture-specific tensor-core features. It also
uses:

- [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) to collect
  and interpret Nsight Compute evidence;
- [KernelWiki](https://github.com/mit-han-lab/KernelWiki) to research relevant
  GPU-kernel techniques and implementations.

The candidate kernel must then return to numerical parity and representative
end-to-end validation. A faster microbenchmark is not itself a model speedup.

## Installation

Clone the repository into a skill directory supported by your coding agent. For
Codex, a user-level installation can be created with:

```bash
git clone <repository-url> ~/.codex/skills/end-to-end-performance-skill
```

The exact installation location may differ between Agent Skills-compatible
tools.

## Repository structure

```text
end-to-end-performance-skill/
├── SKILL.md
├── references/
│   ├── measurement-and-parity.md
│   ├── decision-map.md
│   ├── kernel-dsls.md
│   ├── bottleneck-recipes.md
│   ├── advanced-recipes.md
│   ├── gpu-mode.md
│   ├── gpu-mode-map.md
│   ├── experiment-loop.md
│   ├── playbooks/
│   │   ├── README.md
│   │   ├── common.md
│   │   ├── llm-serving.md
│   │   ├── small-filters.md
│   │   └── request-servers.md
│   ├── experiment-log.md
│   └── abstract-cases.md
├── README.md
└── LICENSE
```

- `SKILL.md` contains the step loop (each step with its output and exit
  condition), the rules that apply at every step, kernel escalation, and the
  reporting format.
- `references/kernel-dsls.md` chooses between a library backend swap, Triton,
  TileLang, and CuTe DSL from kernel-profile evidence.
- `references/measurement-and-parity.md` defines benchmark, parity, topology,
  and verdict guidance.
- `references/decision-map.md` contains the unified flowchart, tool-selection
  matrix and per-branch stopping rules.
- `references/playbooks/` is a library of measured levers by workload class:
  `common.md` (in-process decode, local-disk runtimes, CUDA MPS with several
  processes per GPU, ceilings, precision rungs, measurement traps),
  `llm-serving.md` (admission caps, ingestion payloads, replicas versus tensor
  parallelism), `small-filters.md` (workers per GPU, launch-bound encoder
  fixes, host staging), `request-servers.md` (servers per GPU, engine
  concurrency, persistent servers). `README.md` there picks the class.
- `references/experiment-log.md` defines the local append-only log of
  hypotheses, gate results and verdicts, with reuse of existing workspace records.
- `references/experiment-loop.md` covers screening, validation budgets, reuse and stop decisions.
- `references/bottleneck-recipes.md` covers eight common measured symptoms;
  `references/advanced-recipes.md` adds twelve conditional optimization paths.
- `references/gpu-mode.md` records source and tutorial-transfer guidance;
  `references/gpu-mode-map.md` indexes the reviewed organization and pinned revisions.
- `references/abstract-cases.md` contains generalized examples of promotion,
  rejection, and inconclusive decisions.

## Acknowledgements

Special thanks to [MIT HAN Lab](https://github.com/mit-han-lab),
[Dongyun Zou](https://github.com/DongyunZou), and all contributors to
[KernelWiki](https://github.com/mit-han-lab/KernelWiki) and
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill).

Their work provides an excellent GPU-kernel knowledge base and a rigorous
Nsight Compute profiling workflow. This skill is designed to complement those
projects by identifying when kernel-level investigation is justified and by
bringing every local optimization back to a representative end-to-end
correctness and performance decision.

## License

MIT
