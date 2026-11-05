# Library

[Home](../README.md) / Library

Start with the question you are trying to answer. The folders separate the
optimization protocol, detailed procedures, candidate changes, workload context
and supporting evidence. Each document has one home.

## Choose a collection

| Question | Collection | What belongs here |
| :--- | :--- | :--- |
| What happens next, and what counts as evidence? | [Workflow](workflow/README.md) | Decision map, measurement rules, experiment budgets, records and decision cases |
| How do I carry out this stage? | [Handbook](handbook/README.md) | Seven detailed procedures, from baseline to delivery |
| What change could remove this measured bottleneck? | [Recipes](recipes/README.md) | A mechanism, applicability, cheap test and stop rule |
| What is different about this kind of workload? | [Playbooks](playbooks/README.md) | Workload-specific levers, activation checks, observed ranges and parity traps |
| Where is the underlying explanation or implementation? | [Sources](sources/README.md) | Annotated references, source revisions and transfer limits |

The [agent entrypoint](../SKILL.md) contains the essential loop and routes into
these collections. The [repository front page](../README.md) introduces the
project. Neither needs to duplicate the entire library.

## Follow a short reading path

- **Unknown bottleneck:** [decision map](workflow/decision-map.md) →
  [attribution procedure](handbook/02-attribution-and-experiments.md) → one measured candidate.
- **Serving throughput has plateaued:** [serving procedure](handbook/04-serving-and-concurrency.md) →
  [serving playbook](playbooks/llm-serving.md) → a relevant recipe or source only if needed.
- **A kernel microbenchmark improved:** [kernel procedure](handbook/06-kernel-engineering.md) →
  [measurement rules](workflow/measurement-and-parity.md) →
  [delivery gate](handbook/07-validation-and-delivery.md).
- **Resuming work:** the workload's existing journal →
  [experiment loop](workflow/experiment-loop.md) → the next unresolved decision.

These are routes into the library, not required reading lists. Reuse valid
artifacts and skip sections that do not change the current decision.

## Where new material goes

| New material | Its home | Keep out of that document |
| :--- | :--- | :--- |
| A cross-workload acceptance rule or log field | The relevant `workflow/` reference | Model-specific tuning instructions |
| A missing step, command or failure route in a procedure | The relevant `handbook/` chapter | A second copy of the measurement contract |
| A reusable optimization hypothesis | An existing `recipes/` file | Claims of success without a workload and gate |
| A lever whose value depends on workload behavior | The matching `playbooks/` file | Generic procedure text already in the handbook |
| A paper, implementation or tutorial | `sources/README.md` or its GPU MODE maps | New default requirements inferred from the source |
| An illustrative promote/reject/inconclusive pattern | [Decision cases](workflow/decision-cases.md) | Private session transcripts or invented measurements |

Workflow references own the measurement and record definitions. Handbook
chapters explain how to use them. Recipes and playbooks link back to those gates;
a source supports an idea without becoming a new authority over the task.
When a statement affects several collections, update its home and the links to
it rather than pasting the same rule everywhere.

## Repository layout

```text
README.md                      Project introduction and human entrypoint
SKILL.md                       Agent entrypoint and essential loop
references/
  README.md                    Collection map and placement rules
  workflow/                    Decision protocol and evidence definitions
  handbook/                    Step-by-step procedures
  recipes/                     Candidates by measured bottleneck
  playbooks/                   Candidates in workload context
  sources/                     References, provenance and transfer limits
assets/readme/                 Light and dark cover artwork
```

Keep runnable helpers in `scripts/` only when a repeated task actually needs
one; this documentation library does not require an empty scripts directory or
an application build. Workload logs, datasets, traces and private evidence stay
in their own workspace, outside this repository.
