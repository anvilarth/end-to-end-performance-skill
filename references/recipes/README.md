# Optimization recipes

[Home](../../README.md) / [Library](../README.md) / Recipes

Candidate changes indexed by measured bottleneck. Open a recipe after the
[attribution procedure](../handbook/02-attribution-and-experiments.md) identifies
material exposed cost. A recipe proposes an experiment; it does not promise a
speedup or replace the workload's acceptance gate.

| Collection | Use for | Scope |
| :--- | :--- | :--- |
| [Common bottlenecks](common-bottlenecks.md) | Shape waste, compilation, generation pressure, transfers, materialization, memory, utilization or numerical validation | Eight broadly recurring symptoms |
| [Specialized bottlenecks](specialized-bottlenecks.md) | Prefix reuse, scheduling gaps, optimizers, collectives, hybrid models, on-chip pipelines, advanced GEMM/layouts or automated search | Twelve conditional paths requiring more specific evidence |
| [Kernel backends and DSLs](kernel-backends.md) | Choosing an existing implementation or a kernel language after kernel attribution | Backend swap, Triton, TileLang and CuTe trade-offs |

Choose the smallest relevant experiment and record its expected end-to-end
saving, assumptions, cheapest falsifier and stop rule. If the missing information
is how to perform a step, return to the [handbook](../handbook/README.md). If it is
how that mechanism behaves in a particular service or model-worker layout, use
the [playbooks](../playbooks/README.md).

## Adding a recipe

Extend the relevant collection with a measured symptom, mechanism, applicability,
activation check, correctness trap, cheap test and stop condition. Link the
primary explanation or implementation through [sources](../sources/README.md).
Distinguish a proposed candidate from a result validated on a named workload.
Create a separate recipe file only when an existing collection no longer offers
a clear, bounded lookup; update this index and the callers together.
