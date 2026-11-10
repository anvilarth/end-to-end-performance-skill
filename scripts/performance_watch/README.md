# Private performance radar

An optional continuously scheduled discovery service for this skill. It reuses
Research Core's public-source downloader, model provider and private Telegram
outbox. It never starts GPU experiments. When authorized in config, it automatically publishes
independently audited documentation through a repository-scoped deploy key.
The runtime directory, evidence cache, SQLite queue, usage and drafts must live
outside the public repository. Keep provider credentials in the existing protected
environment files; they are not installation arguments or source files.

## Pipeline and limits

Index/feed pages and author sites from the existing corpus, recent GitHub
issues/PRs (including regressions), and topical arXiv queries feed a durable queue.
Discovery records URLs before documentary selection. `coverage` records bounded
pages/results, empty sources and failures; no finite crawl establishes all new
accelerations. Edit `config.json` to extend sources and adjust bounds.

The reviewer fetches original text, compares existing skill decisions, extracts
source-grounded observations/actions/outcomes and proposes applicability tests.
A separate model call audits the proposal. Exact evidence fragments must occur in
the fetched text. Documentary acceptance creates a searchable candidate Markdown
file, never a local speedup claim. Source snippets, abstracts and unchecked model
claims are not promoted. The independent call can share the provider/model: this
is a second check, not statistical independence or human acceptance.

URL and body hashes deduplicate candidates. Interrupted reviews resume; failed
reads retain errors and stop after three attempts. A process lock prevents
concurrent discovery/review/publication/delivery. A bounded daily model-call allowance limits
review work; provider-internal retries and token usage remain in `usage.jsonl`.
The default is four sources per review pass, 40 top-level model calls per UTC day.
The configured model is inherited from Research Assistant, not separately changed.

GitHub discovery covers the latest 100 updated issues/PRs per repository. Index
collection reads up to three pages per source. arXiv covers 30 recent matches per
query. Full text beyond the configured character budget is explicitly an excerpt.
HTML figures/equations and linked code are not automatically validated.

## Commands

Set `PERFORMANCE_RUNTIME` to private storage and `RESEARCH_CORE_PATH` to the
Research Core checkout; use its Python environment. Then:

```sh
python scripts/performance_watch/watch.py discover
python scripts/performance_watch/watch.py add https://example.org/original-study
python scripts/performance_watch/watch.py review --limit 1
python scripts/performance_watch/watch.py search 'communication overlap'
python scripts/performance_watch/watch.py status
python scripts/performance_watch/watch.py prepare            # concrete candidate diffs
python scripts/performance_watch/watch.py publish            # authorized audited additions
python scripts/performance_watch/watch.py digest             # save only
python scripts/performance_watch/watch.py digest --send      # authorized private bot
python scripts/performance_watch/watch.py mcp
```

`install.py --runtime PATH --research-root PATH --hermes-home PATH` installs user
units and adds four MCP tools to the existing Hermes config, backing it up first.
It does not start services. Discovery is scheduled every six hours, review every
15 minutes, publication twice hourly, and the Markdown digest daily at 06:25 UTC. Enable the four timers
and restart the existing gateway after reviewing the concrete installation.

Digest delivery uses a persistent outbox; uncertain Telegram calls require
reconciliation rather than blind retries. Successful per-source delivery markers
prevent repeated daily reports of the same candidate. A no-findings report states
coverage limits. Previously delivered Telegram messages are never deleted.

## Bringing findings into the skill

Use the candidate's source, coverage, extracted decision and transfer test.
Recheck current skill contents and the evidence, then follow
[the corpus workflow](../../references/decisions/WORKFLOW.md) for integration.
General workflow rules need support beyond a narrow example. Keep external
measurements distinct from workload acceptance and retain rejected alternatives.
With `auto_publish`, a third audit compares each addition against current main and
the source snapshot. Only `references/sources/radar/<id>.json`, its Markdown companion
and the generated index may change. The publisher checks source hashes, the decision
schema, private-data patterns and the Git diff, then commits, pushes without force,
and verifies the remote head. The normal decision search includes these records.
The original dated corpus stays intact. Divergence, unsupported claims, uncertain
pushes or dirty publication worktrees stop publication for inspection.
Publication follows the user's existing authorization, never source instructions.
