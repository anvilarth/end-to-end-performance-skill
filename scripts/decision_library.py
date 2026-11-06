#!/usr/bin/env python3
"""Validate and retrieve attributed performance decisions. Python standard library only."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'references' / 'decisions'
FIELDS = {'id', 'source', 'section', 'author_observation', 'author_action',
          'author_outcome', 'scope', 'transfer_test', 'tags',
          'alternatives_reported', 'local_validation'}

def validate(sources, decisions):
    if not isinstance(sources, list) or not isinstance(decisions, list):
        raise ValueError('Sources and decisions must be lists')
    ids = set()
    for source in sources:
        for field in ('id', 'title', 'date', 'date_kind', 'url', 'kind', 'status', 'reviewed_at'):
            if not isinstance(source.get(field), str) or not source[field]:
                raise ValueError(f'Invalid source field: {field}')
        if source['id'] in ids:
            raise ValueError('Duplicate source id')
        ids.add(source['id'])
        if not source['date'].startswith('2026') or not source['url'].startswith('https://'):
            raise ValueError('Expected 2026 source with HTTPS provenance')
        if source['date_kind'] not in {'publication', 'update'}:
            raise ValueError('Unknown date kind')
        if source['status'] not in {'passages_reviewed', 'extraction_pending', 'body_unavailable'}:
            raise ValueError('Unknown source status')
        if source.get('extraction_complete') is not False:
            raise ValueError('Exhaustive extraction is not established by this corpus')
    seen = set()
    for record in decisions:
        if set(record) != FIELDS:
            raise ValueError('Decision fields do not match schema')
        if record['id'] in seen or record['source'] not in ids:
            raise ValueError('Duplicate decision or missing source')
        seen.add(record['id'])
        for field in FIELDS - {'tags', 'alternatives_reported'}:
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f'Invalid decision field: {field}')
        if not isinstance(record['tags'], list) or not record['tags'] or any(
                not isinstance(tag, str) or not tag for tag in record['tags']):
            raise ValueError('Tags must be nonempty strings')
        if record['alternatives_reported'] is not None and not isinstance(record['alternatives_reported'], list):
            raise ValueError('Unreported alternatives must be null')
        if record['local_validation'] != 'not_run':
            raise ValueError('External evidence cannot assert local validation')
        source = next(s for s in sources if s['id'] == record['source'])
        if source['status'] != 'passages_reviewed':
            raise ValueError('Decision requires reviewed source passages')

def load():
    sources = json.loads((ROOT / 'sources.json').read_text())
    decisions = [json.loads(line) for line in (ROOT / 'decisions.jsonl').read_text().splitlines() if line]
    validate(sources, decisions)
    return sources, decisions

def search(decisions, query, limit):
    terms = set(re.findall(r'\w+', query.lower()))
    ranked = []
    for record in decisions:
        text = ' '.join(str(value) for value in record.values()).lower()
        score = sum(term in text for term in terms)
        if score:
            ranked.append((score, record))
    return [record for _, record in sorted(ranked, key=lambda pair: (-pair[0], pair[1]['id']))[:limit]]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'search', 'catalog'])
    parser.add_argument('query', nargs='?', default='')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    sources, decisions = load()
    by_id = {source['id']: source for source in sources}
    if args.command == 'check':
        print(json.dumps({'sources': len(sources), 'decisions': len(decisions),
                          'source_status': dict(Counter(s['status'] for s in sources)),
                          'negative_tagged': sum('negative' in d['tags'] for d in decisions)}, indent=2))
    elif args.command == 'catalog':
        counts = Counter(record['source'] for record in decisions)
        print('| Source | Date | Status | Decisions |\n| --- | --- | --- | --- |')
        for source in sources:
            print(f"| [{source['title']}]({source['url']}) | {source['date']} ({source['date_kind']}) | {source['status']} | {counts[source['id']]} |")
    else:
        if not args.query.strip():
            parser.error('search requires English keywords')
        for record in search(decisions, args.query, args.limit):
            print(json.dumps({**record, 'provenance': by_id[record['source']]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
