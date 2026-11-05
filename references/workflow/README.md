# Workflow

[Home](../../README.md) / [Library](../README.md) / Workflow

The compact protocol for making and recording an optimization decision. These
references define the common measurement and evidence rules; the
[handbook](../handbook/README.md) provides detailed procedures for carrying them out.

| Reference | Open when | Leave with |
| :--- | :--- | :--- |
| [Decision map](decision-map.md) | The next diagnostic or optimization branch is unclear | A branch, a suitable tool and an exit condition |
| [Measurement and parity](measurement-and-parity.md) | Defining a benchmark, comparator or acceptance gate | The useful-work boundary and evidence required for a verdict |
| [Experiment loop](experiment-loop.md) | Planning repeated screens and final validation | A bounded experiment with time reserved for the final gate |
| [Experiment log](experiment-log.md) | Starting or resuming a session | A recovered contract, previous decisions and one consistent record |
| [Decision cases](decision-cases.md) | Interpreting a local win, regression or incomplete result | An illustrative decision pattern and its limits |

Start from the actual workload and its existing artifacts. The protocol should
shorten the path to a decisive result; it does not require a new framework, every
profiler, or a separate document for every minor change.

For implementation choices use [recipes](../recipes/README.md) and
[workload playbooks](../playbooks/README.md). For detailed API or algorithm
questions use the [source library](../sources/README.md).
