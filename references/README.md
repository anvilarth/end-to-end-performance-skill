# Library

[Home](../README.md) / Library

The complete reading map for the performance skill. Choose a collection by the
decision you need to make; use the [decision map](decision-map.md) when the next
step is unclear.

[Workflow](#workflow--measurement) · [Handbook](#handbook) ·
[Playbooks](#workload-playbooks) · [Recipes](#optimization-recipes) ·
[Research](#research--implementations)

## Workflow & measurement

Establish the workload, make evidence trustworthy, and keep the experiment loop
short enough to reach a decision.

| Read | Use it for |
| :--- | :--- |
| [Decision map](decision-map.md) | The full branching workflow, tool-selection matrix and stopping rules |
| [Measurement & parity](measurement-and-parity.md) | Timing boundaries, correctness, topology and promotion gates |
| [Experiment loop](experiment-loop.md) | Screening, profiler repair, validation budgets and reuse of evidence |
| [Experiment log](experiment-log.md) | Contracts, hypotheses, gate results and resumable decisions |
| [Worked cases](abstract-cases.md) | Examples of promotion, rejection and inconclusive results |

## Handbook

Detailed operating procedures. Every chapter has a contents list and links to
its neighbors. Start at the [handbook overview](handbook/README.md) for the
sequence and the question-to-chapter routing table.

| | Chapter | Main topics |
| :--- | :--- | :--- |
| **01** | [Contract & baseline](handbook/01-contract-and-baseline.md) | Useful work, timing, representative inputs, references, cold/warm state, A/A |
| **02** | [Attribution & experiments](handbook/02-attribution-and-experiments.md) | Profiling ladder, critical path, ceilings, savings, falsifiers |
| **03** | [Input, shapes & compilation](handbook/03-input-shapes-and-compilation.md) | Decode, workers, transfers, packing, memory, compile, CUDA graphs |
| **04** | [Serving & concurrency](handbook/04-serving-and-concurrency.md) | Arrival processes, queues, caching, replicas, attention, tails |
| **05** | [Training & distributed execution](handbook/05-training-and-distributed.md) | Complete updates, accumulation, optimizer, checkpointing, DDP/FSDP, scaling |
| **06** | [Kernel engineering](handbook/06-kernel-engineering.md) | Contracts, backends/DSLs, counters, bounded tuning, sanitizers, integration |
| **07** | [Validation & delivery](handbook/07-validation-and-delivery.md) | Gate ladder, independent references, paired timing, composition, handoff |

## Workload playbooks

Start at the [playbook selector](playbooks/README.md), then read the shared
techniques and the class matching your deliverable. Reported ranges are priors,
not promises for a new workload.

| Read | Use it for |
| :--- | :--- |
| [Shared techniques](playbooks/common.md) | Work removal, decode/locality, GPU sharing, precision and measurement traps |
| [LLM / VLM serving](playbooks/llm-serving.md) | Continuous batching, admission limits, payloads, replicas versus tensor parallelism |
| [Small model workers](playbooks/small-filters.md) | Classifiers, encoders and filters; workers per GPU, launch overhead and staging |
| [Request servers](playbooks/request-servers.md) | Short HTTP requests, persistent servers, concurrency and startup amortization |

## Optimization recipes

Use a recipe after establishing a material budget for its mechanism. Return to
full-path validation after the local screen.

| Read | Use it for |
| :--- | :--- |
| [Common bottlenecks](bottleneck-recipes.md) | 8 paths covering shapes, compilation, generation, transfers, materialization, memory, utilization and numerics |
| [Advanced paths](advanced-recipes.md) | 12 paths covering reuse, scheduling, optimizers, collectives, hybrid models and kernel techniques |
| [Kernels & DSL selection](kernel-dsls.md) | Choosing an existing backend, Triton, TileLang or CuTe from profile evidence |

## Research & implementations

Find the explanation or implementation needed for the current hypothesis.
Check the source's version, supported contract and transfer limits before reuse.

| Read | Use it for |
| :--- | :--- |
| [Annotated source library](handbook/sources.md) | 59 documentation, research, implementation and discovery references grouped by decision |
| [GPU MODE organization map](gpu-mode-map.md) | A reviewed 26-repository snapshot, pinned revisions, operator atlas and lecture routes |
| [GPU MODE transfer guide](gpu-mode.md) | Applying tutorial ideas, narrow captures and implementation candidates without importing invalid benchmark assumptions |

## Where things live

```text
SKILL.md                  Agent entrypoint and essential rules
README.md                 Repository front page
references/
  README.md               This catalog
  handbook/               Detailed procedures and annotated sources
  playbooks/              Workload-specific levers
  *.md                    Workflow, recipes, cases and GPU MODE maps
assets/readme/            Light and dark SVG covers
```

To extend the library, follow the handbook's
[maintenance guidance](handbook/README.md#keeping-the-library-useful): add evidence
that changes a decision, place it with the relevant procedure, and link the
primary source. Keep workload artifacts and private session data in the workload's
own workspace.
