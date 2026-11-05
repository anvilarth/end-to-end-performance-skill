<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/readme/cover-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/readme/cover-light.svg">
  <img alt="End-to-End Performance — measure the full path, prove the gain." src="assets/readme/cover-light.svg" width="1200">
</picture>

<p align="center">
  <strong>A field guide and agent skill for optimizing complete PyTorch and CUDA workloads.</strong><br>
  Training · inference · serving · distributed systems · GPU kernels
</p>

<p align="center">
  <a href="#start-here">Start here</a> ·
  <a href="#the-handbook">Handbook</a> ·
  <a href="references/README.md">Library</a> ·
  <a href="#use-the-skill">Use the skill</a> ·
  <a href="references/handbook/sources.md">Sources</a>
</p>

---

## Start here

Find the next useful experiment, estimate what it can save, and validate the
complete delivered workload. Start with the question you need to answer.

<table>
<tr>
<td width="50%" valign="top">
<h3>01 / Choose the next move</h3>
<p>A decision tree from the real workload to a measured bottleneck, a bounded experiment and a verdict.</p>
<a href="references/decision-map.md"><strong>Open the decision map →</strong></a>
</td>
<td width="50%" valign="top">
<h3>02 / Work through the details</h3>
<p>Seven practical chapters: what to check, which tool to use, how to interpret the result and when to stop.</p>
<a href="references/handbook/README.md"><strong>Read the handbook →</strong></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<h3>03 / Match your workload</h3>
<p>Focused recipes for LLM/VLM serving, small model workers and request servers, with activation and parity checks.</p>
<a href="references/playbooks/README.md"><strong>Pick a playbook →</strong></a>
</td>
<td width="50%" valign="top">
<h3>04 / Find the evidence</h3>
<p>59 annotated sources, a reviewed GPU MODE map and implementation references matched to the current question.</p>
<a href="references/handbook/sources.md"><strong>Browse the sources →</strong></a>
</td>
</tr>
</table>

## The handbook

**Contract → measure → attribute → change → validate → decide.**

Read in order for the complete workflow, or open the chapter that resolves your
current bottleneck. Each chapter includes entry evidence, procedures, failure
routes and an exit condition.

| | Chapter | The question it answers |
| :--- | :--- | :--- |
| **01** | [Contract & baseline](references/handbook/01-contract-and-baseline.md) | What must become faster, and can we measure it reliably? |
| **02** | [Attribution & experiments](references/handbook/02-attribution-and-experiments.md) | Where is the recoverable time, and what should we try first? |
| **03** | [Input, shapes & compilation](references/handbook/03-input-shapes-and-compilation.md) | Are decode, padding, copies, memory or launch overhead limiting us? |
| **04** | [Serving & concurrency](references/handbook/04-serving-and-concurrency.md) | How do we increase useful throughput while meeting latency and quality limits? |
| **05** | [Training & distributed execution](references/handbook/05-training-and-distributed.md) | How do we improve the complete update and scale across GPUs? |
| **06** | [Kernel engineering](references/handbook/06-kernel-engineering.md) | When is a custom kernel worth it, and which technique fits the bound? |
| **07** | [Validation & delivery](references/handbook/07-validation-and-delivery.md) | Is the gain real, reproducible and ready to use? |

## Use the skill

Clone into a skill directory supported by your coding agent. For a standard
Codex user-level installation:

```bash
git clone https://github.com/anvilarth/end-to-end-performance-skill.git \
  ~/.codex/skills/end-to-end-performance-skill
```

Then invoke it with a repository or launcher and the outcome you need:

```text
Use $end-to-end-performance-skill to optimize this workload.
Entry point: <repository and launch command>
Target: <latency, useful throughput, memory or cost>
Constraints: <hardware, precision, quality and time budget>
```

The skill recovers actual usage from code and existing artifacts, loads only the
relevant references, and reuses the workload's experiment log. New sessions can
resume from prior baselines, decisions and rejected hypotheses.

[Read the agent instructions](SKILL.md) · [Experiment record](references/experiment-log.md)

## Explore the library

| Collection | What you will find |
| :--- | :--- |
| [Workflow & measurement](references/README.md#workflow--measurement) | Decision map, measurement rules, experiment budgets, logs and worked cases |
| [Handbook](references/handbook/README.md) | The full sequence, with tools, commands, calculations and failure routes |
| [Workload playbooks](references/playbooks/README.md) | Serving engines, small models, request servers and shared techniques |
| [Optimization recipes](references/README.md#optimization-recipes) | 8 common bottleneck recipes, 12 advanced paths and a kernel/DSL guide |
| [Research & implementations](references/README.md#research--implementations) | Annotated sources, GPU MODE repositories, lectures and transfer limits |

**The complete catalog is in [references/README.md](references/README.md).**
Existing document paths remain stable, so bookmarks and installed skill links
continue to work.

## What counts as a win

- **Measured headroom.** Rank changes by recoverable end-to-end time and the cost of proving them.
- **Preserved behavior.** Start with equivalent changes; validate the agreed numerical and quality contract.
- **A complete result.** Promote from representative full-path evidence. Record rejected and inconclusive experiments too.

An isolated kernel speedup is one piece of evidence. Success is a faster delivered
training step, batch, generation or served request.

---

Thanks to [GPU MODE](https://github.com/gpu-mode),
[MIT HAN Lab](https://github.com/mit-han-lab),
[Dongyun Zou](https://github.com/DongyunZou), and the contributors to
[KernelWiki](https://github.com/mit-han-lab/KernelWiki) and
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) for the teaching,
implementation and profiling resources that inform this library.

[MIT License](LICENSE)
