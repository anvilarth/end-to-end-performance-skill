# Collect and reuse engineering experience

This workflow builds experience through documentation and ordinary agent work.
No training, specialized model, embedding service or background crawler is required.
Run it when asked to expand the library, when a current decision lacks evidence,
or when an authorized experiment produces a useful verdict.

## 1. Choose the question and bound the search

Start from a concrete question: symptom, pipeline stage, hardware, workload and
constraints. For this collection, search 2026 publications through the date of
review. Record queries, dates, sources inspected and remaining leads in the task
research notes. Stop at the agreed scope or search budget; report gaps rather
than claiming that all articles or all author decisions have been found.

## 2. Find decisions, including failed attempts

Search author blogs, engineering writeups, papers and their linked code. Follow
article series, benchmark scripts, PR discussions, issues, regressions and
reverts: these can reveal alternatives omitted from the final writeup. Search
both the symptom and why a promising optimization failed. Prefer original
accounts; group posts about the same experiment rather than counting them as
independent evidence. Verify publication year separately from later updates.

## 3. Register the source and read the evidence

Add or update sources.json with the existing schema. Record access limits and
revision information honestly. A search snippet or abstract cannot support
unseen experimental details. Keep unavailable sources as leads, not decisions.
Locate the section, figure, code revision or discussion supporting each record.
If evidence is missing, write unknown; do not reconstruct the author's thoughts.

## 4. Extract one decision per record

Use the existing decisions.jsonl schema and the next unused D identifier:

- author_observation: the measured symptom and relevant baseline;
- author_action: the change actually made;
- author_outcome: reported result, including failure or inconclusive outcome;
- alternatives_reported: only alternatives explicitly described, otherwise null;
- scope: hardware, shapes, software, numerical contract and measurement boundary
  when available; state consequential missing details;
- section: an exact locator within the source;
- transfer_test: our proposed cheapest test of applicability, explicitly separate
  from the author's conclusions;
- tags: mechanism, stage and negative-result tag when appropriate;
- local_validation: not_run for external evidence.

Preserve absolute timings and comparison conditions when reported. Distinguish
kernel timing from full-path timing. Separate several attempts into several
records; preserve corrections and contradictions in source notes. Paraphrase
briefly and link evidence instead of copying whole articles.

## 5. Check and publish the documentation snapshot

Run from the skill directory:

```sh
python3 scripts/decision_library.py check
python3 scripts/decision_library.py catalog > references/decisions/CATALOG.md
git diff --check
```

Inspect new records against their cited passages. Update README counts and
coverage gaps, then report what was added and what remains unread. Structural
validation does not establish factual correctness or improved agent performance.
Follow the repository's normal review and delivery rules.

## 6. Use experience in the next task

After measuring the current workload, search the library by symptom. Read a few
relevant successful and negative cases, then return to their primary sources.
In the task's lever table record: case IDs, matching conditions, mismatches,
expected recoverable full-path time and the cheapest falsifier. Choose a test
from present evidence; a previously successful recipe is not a default setting.
Do not load the whole corpus into the prompt.

## 7. Close the loop after an experiment

Write the actual result to the workload's established experiment log: contract,
change, artifact paths, baseline/candidate measurements, correctness verdict,
transfer limits and next action. Include rejected and inconclusive attempts.
When library maintenance is authorized, distill a reusable local case with links
to those artifacts in a separate local-case document. Do not overwrite external
author outcomes or label them locally validated. Respect source confidentiality
and repository boundaries. Amend the general skill instructions only when the
lesson is supported beyond one narrow example.

The durable loop is: question -> sources -> documented decisions -> relevant
retrieval -> experiment -> documented verdict. Model comparisons are outside
this collection workflow.
