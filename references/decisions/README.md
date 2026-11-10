# Engineering decisions from 2026 sources

For collection and maintenance, follow [the documentation workflow](WORKFLOW.md).
It requires no model training or deployment.

Use this collection after measuring the current workload. Search by the observed
symptom, then read the source section before proposing an experiment.

`python3 scripts/decision_library.py search "synchronization"`

Each record separates an author's reported observation, action and outcome from
our proposed transfer test. A reported outcome is external evidence, not a local
acceptance result. Unknown alternatives and measurements remain unknown.

Selection procedure:

1. Match the measured symptom, execution stage, hardware and numerical constraints.
2. Retrieve both successful and unsuccessful cases for that mechanism.
3. Explain which source assumptions hold locally and which do not.
4. Predict recoverable full-path time from the local baseline.
5. Run the cheapest test that could disprove the proposed mechanism.
6. Keep the existing parity and full-path promotion gates.

Do not treat frequency in this collection as probability of success. Sources are
selected publications, not an unbiased sample of experiments. Several articles
may describe one implementation; they are not independent confirmations.
Do not copy their batch sizes, tolerance thresholds or speedup factors as defaults.
A source can contain overgeneralizations; preserve its scope and verify the actual
runtime. A cleaner trace, higher occupancy or higher utilization alone is no win.

The corpus is a dated research snapshot. Source coverage and decision extraction
coverage are different. `sources.json` includes discovered but unextracted items;
`decisions.jsonl` contains only decisions supported by inspected source passages.
No article is marked exhaustively extracted. This avoids claiming to know omitted
experiments or private reasoning. Publication dates are distinguished from updates.
The search is bounded to 2026 through November 9, not a claim to cover the internet.

Check: `python3 scripts/decision_library.py check`.
The check validates data and references, not factual truth or agent improvement.

## Snapshot coverage and next pass

This edition contains 58 sources and 100 decision records from 56 sources.
Two discovered sources (Uber and Baseten) could not be read through the web tool.
The main gaps are distributed training case studies, CPU/decode pipelines, and
complete extraction of all decisions within long articles. The current records
cover inspected passages, not every decision in each article.

Search routes used: author blogs, Colfax article index, Hugging Face profiling
series, PyTorch release and engineering blogs, serving case studies, and 2026
agent/kernel papers. Negative search results are not proof that a source is absent.
Next: recover unavailable bodies, inspect linked code and earlier/later series
parts, expand training and input-pipeline coverage, and retain source revisions.
This static snapshot does not itself run a crawler. An optional separately deployed
[private performance radar](../../scripts/performance_watch/README.md) discovers and
reviews candidates without silently changing this corpus.

Latest broad GPU pass: [search coverage and remaining leads](GPU-SEARCH-2026.md).

Engine-specific collection: [SGLang and serving-engine map](ENGINES-2026.md).
