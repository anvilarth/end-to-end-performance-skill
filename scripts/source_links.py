#!/usr/bin/env python3
"""Search the collected GPU documentation link graph without loading whole manuals."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', nargs='?', help='Case-insensitive substring in URL or title')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    records = [json.loads(line) for line in
               (ROOT / 'references/sources/gpu-doc-links.jsonl').read_text().splitlines()]
    if args.check:
        urls = [record['url'] for record in records]
        if len(urls) != len(set(urls)):
            raise SystemExit('Duplicate source records')
        for record in records:
            if not record['url'].startswith('https://'):
                raise SystemExit(f"Invalid source URL: {record['url']}")
            if len(record['links']) != len(set(record['links'])):
                raise SystemExit(f"Duplicate edges: {record['url']}")
        manifest = json.loads((ROOT / 'references/sources/compilation-manifest.json').read_text())
        if len(manifest['tabs']) != 21 or any(r['status'] != 200 for r in manifest['tabs']):
            raise SystemExit('Original-tab coverage is incomplete')
        print(json.dumps({'source_pages': len(records), 'original_tabs': 21,
                          'unique_link_targets': len({u for r in records for u in r['links']}),
                          'article_docs_and_code': len(manifest['article_doc_and_implementation_links'])}))
        return
    if not args.query or args.limit < 1:
        parser.error('Provide a query and a positive --limit, or use --check')
    needle = args.query.casefold()
    by_url = {r['url']: r for r in records}
    found = {}
    for record in records:
        if needle in (record['url'] + ' ' + record['title']).casefold():
            found.setdefault(record['url'], record['url'])
    for record in records:
        for url in record['links']:
            if needle in url.casefold():
                found.setdefault(url, record['url'])
    print(f'{len(found)} matches; showing at most {args.limit}. Discovered links are not reviewed claims.')
    for url, source in list(found.items())[:args.limit]:
        record = by_url.get(url)
        status = (f"HTTP {record['http_status']}; {record['body_status']}"
                  if record else 'discovered link; target not independently retrieved')
        print(f'{url}\n  From: {source}\n  Status: {status}')


if __name__ == '__main__':
    main()
