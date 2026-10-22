# End-to-End Performance Skill

An agent skill for optimizing PyTorch and CUDA training and inference workloads
as complete systems.

The skill starts by asking how the code is actually used: hardware, scale,
topology, workload shapes, modes, concurrency, cold or warm behavior, and the
correctness contract. It then profiles the full execution path, ranks measured
bottlenecks by recoverable end-to-end impact, applies lossless changes first,
and validates every candidate with correctness and performance gates.

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

Every contract, baseline, hypothesis, gate result, and verdict is appended to a
local experiment log (`.perf-log/log.jsonl` in the workload repository) as it
happens. A new session reads the log first, so rejected and inconclusive ideas
are not repeated and open next actions are picked up.

## Kernel-level escalation

This is intentionally a higher-level skill. It does not begin by writing custom
CUDA kernels.

When end-to-end profiling proves that a specific kernel has enough recoverable
budget to matter, use:

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
│   ├── experiment-log.md
│   └── abstract-cases.md
├── README.md
└── LICENSE
```

- `SKILL.md` contains the core principles, workflow, escalation rules, and
  required result format.
- `references/measurement-and-parity.md` defines benchmark, parity, topology,
  and verdict guidance.
- `references/experiment-log.md` defines the local append-only log of
  hypotheses, gate results, and verdicts.
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
