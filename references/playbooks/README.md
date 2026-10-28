# Playbook library

Levers that repeatedly delivered material end-to-end gains, grouped by
workload class. Use this index in workflow step 3 to pick the playbook that
matches the deliverable, then read `common.md` plus one class file. Do not read
every file; the classes have different bottleneck shapes and the ranges do not
transfer across them.

Each entry in a playbook has the same skeleton: symptom in the stage map,
mechanism, ladder rung (SKILL.md step 4), observed gain range, activation
check, and parity trap. The ranges are priors for sizing headroom, not
promises; every lever still passes the contract, parity, and paired gates.

## Pick a class

| Deliverable | Typical shape | Playbook |
| --- | --- | --- |
| Generative LLM or VLM behind a serving engine (continuous batching, KV cache, tensor parallel), long outputs per request | few large replicas per node, hundreds of requests in flight, decode-dominated | [llm-serving.md](llm-serving.md) |
| Small per-item models run in worker processes: classifiers, taggers, embedders, detectors, frame rankers, encoders | one forward per item or window, model is a fraction of the item cost, decode and host work matter | [small-filters.md](small-filters.md) |
| Model servers answering many short requests over HTTP: speech recognition, embedding services, rerankers | several server processes per GPU, clients drive concurrency, persistent servers with long cold start | [request-servers.md](request-servers.md) |

Mixed pipelines (a filter cascade that ends in a VLM) use one playbook per
stage; the stage with the largest measured budget picks the first lever.

## Shared levers

[common.md](common.md) applies to all classes:

- host-side work removal: in-process decode, grouped decode, local-disk
  runtimes and inputs;
- several processes per GPU under a CUDA MPS daemon: the full recipe,
  activation checks, pitfalls, and when it yields zero;
- knowing the ceiling: compute-pipe utilization, power cap, idle floor;
- precision rungs as an authorized Pareto and how to measure decision flips;
- measurement traps seen in practice.

The class files reference these by section instead of repeating them and add
only the class-specific levers and numbers.

## Adding a class or a lever

Add a new class file when the bottleneck shape differs from the three above
(for example training steps, diffusion sampling, or multi-node collectives),
not when only the model differs. Add a lever to an existing file when it has
been measured at least once on a real deliverable with a parity result; record
the observed range, the activation check, and the trap. Keep entries
organization-neutral: no hostnames, internal paths, ticket numbers, or
internal model names.
