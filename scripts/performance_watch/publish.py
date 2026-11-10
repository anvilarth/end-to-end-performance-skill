"""Publish audited documentary records, never executable or runtime material."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def public_record(item, review):
    source = {'id':item['id'],'url':item['url'],'title':item['title'],
              'reviewed_at':item['updated'],'content_hash':item['content_hash'],
              'status':'passages_reviewed','date':item.get('published'),
              'date_kind':'source_metadata' if item.get('published') else 'unknown'}
    decisions=[]
    for n,d in enumerate(review['decisions'],1):
        decisions.append({'id':'R'+item['id']+'-'+str(n),'source':item['id'],
            'section':d['section'],'author_observation':d['observation'],
            'author_action':d['action'],'author_outcome':d['outcome'],
            'scope':d['scope'],'transfer_test':d['transfer_test'],
            'tags':review.get('tags') or ['performance','external-evidence'],
            'alternatives_reported':None,'local_validation':'not_run'})
    return {'source':source,'verification':'source_supported','decisions':decisions}


def validate_public(text):
    # Public payload cannot include runtime locations, credentials or private hosts.
    if re.search(r'/mnt/train-nfs|/home/ubuntu|/Users/|BEGIN .*PRIVATE KEY|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|\bb300-\d+\b|filatov-cpu|https?://[^/\s]+:[^/\s]+@',text):
        raise ValueError('Public record contains private/runtime material')


def publish(w):
    if not w.CONFIG.get('auto_publish'):
        return {'state':'disabled'}
    checkout=w.RUNTIME/'publication-checkout'
    repo=w.CONFIG['publish_repository']
    def git(*args):
        return subprocess.check_output(['git','-C',str(checkout),*args],text=True,stderr=subprocess.PIPE,timeout=180).strip()
    if not checkout.exists():
        subprocess.run(['git','clone','--single-branch','--branch','main',repo,str(checkout)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        git('config','user.name','Performance Radar')
        git('config','user.email','anvilarth@users.noreply.github.com')
    if git('status','--porcelain'):
        raise RuntimeError('Publication checkout is dirty; retained for inspection')
    git('fetch','origin','main')
    ahead,behind=map(int,git('rev-list','--left-right','--count','HEAD...origin/main').split())
    if ahead:
        # Retry only a previously audited publication commit, never force push.
        pending=w.RUNTIME/'pending-publication.json'
        if not pending.exists() or json.loads(pending.read_text())['commit']!=git('rev-parse','HEAD'):
            raise RuntimeError('Unrecognized local publication commit; manual reconciliation needed')
        if behind:
            # Only our recognized, unpublished generated commit can be rebased.
            # Archive its patch, return to remote main and re-audit below.
            archive=w.RUNTIME/'publication-retries';archive.mkdir(exist_ok=True)
            (archive/(git('rev-parse','HEAD')+'.patch')).write_text(git('format-patch','-1','--stdout'))
            git('reset','--hard','origin/main')
        else:
            git('push','origin','HEAD:main')
    else:
        git('merge','--ff-only','origin/main')
    rows=[]
    with w.connect() as db:
        for row in db.execute("SELECT * FROM items WHERE state='verified' ORDER BY updated"):
            if not db.execute('SELECT 1 FROM publications WHERE id=?',(row['id'],)).fetchone():
                rows.append(dict(row))
    changed=[]; published=[]
    for item in rows[:1]:
        relative='references/sources/radar/'+item['id']+'.json'
        target=checkout/relative
        if target.exists():
            existing=json.loads(target.read_text())
            if existing['source']['content_hash']!=item['content_hash']:
                raise RuntimeError('Published source revision differs; keep both for review')
            published.append(item['id']);continue
        review=json.loads(item['review'])
        if review.get('verification',{}).get('accepted') is not True:
            raise ValueError('Missing independent documentary audit')
        evidence=(w.RUNTIME/'evidence'/f'{item["content_hash"]}.txt').read_text()
        if hashlib.sha256(evidence.encode()).hexdigest()!=item['content_hash']:
            raise ValueError('Source evidence hash mismatch')
        w.validate_review(review,evidence)
        record=public_record(item,review)
        public=json.dumps(record,ensure_ascii=False,indent=2)+'\n';validate_public(public)
        # Audit against the current accepted library, not the search-time snapshot.
        prior=[json.loads(p.read_text()) for p in (checkout/'references/sources/radar').glob('*.json')]
        existing_decisions=(checkout/'references/decisions/decisions.jsonl').read_text()
        verdict=w.model_call('publication-audit',
            'Independently decide whether to publish this source-grounded performance decision in a reusable skill. Require a concrete useful addition, missing condition or contradictory evidence beyond the existing library. Check attribution and limitations against original text. Reject unsupported claims, merely promotional descriptions, redundant entries and private information. No local speedup is established. Return {accepted:bool,reasons:[strings]}.',
            {'proposed':record,'source':evidence[:w.CONFIG['max_document_chars']],
             'existing_radar':prior[-30:],'existing_library':existing_decisions})
        (w.RUNTIME/'candidates'/f'{item["id"]}-publication-audit.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2))
        if verdict.get('accepted') is not True:
            with w.connect() as db:db.execute("UPDATE items SET state='integration_review',error=?,updated=? WHERE id=?",(json.dumps(verdict,ensure_ascii=False),w.now(),item['id']))
            continue
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(public)
        text=Path(item['report']).read_text();validate_public(text)
        target.with_suffix('.md').write_text(text)
        changed.extend([relative,str(Path(relative).with_suffix('.md'))]);published.append(item['id'])
    if changed:
        index=checkout/'references/sources/radar/INDEX.md'
        body='# Automatically reviewed performance evidence\n\n'
        body+='Source-supported external decisions; local validation has not run. Read comparison conditions and proposed transfer tests before reuse. Search with `python3 scripts/decision_library.py search QUERY`.\n\n'
        for p in sorted(index.parent.glob('*.json')):
            v=json.loads(p.read_text())['source'];body+='- ['+v['title'].replace('[','(').replace(']',')')+']('+p.with_suffix('.md').name+') — '+v['url']+'\n'
        validate_public(body);index.write_text(body);changed.append('references/sources/radar/INDEX.md')
        try:
            subprocess.run([sys.executable,str(checkout/'scripts/decision_library.py'),'check'],check=True,capture_output=True,text=True,timeout=60)
            git('diff','--check')
            for path in changed:
                if not re.fullmatch(r'references/sources/radar/(?:[0-9a-f]{24}\.(?:json|md)|INDEX\.md)',path):
                    raise ValueError('Publication path outside allowlist')
            git('add','--',*changed)
            staged=git('diff','--cached','--name-only').splitlines()
            if set(staged)!=set(changed):raise ValueError('Unexpected staged paths')
            git('commit','-m','Add audited performance discoveries')
            commit=git('rev-parse','HEAD')
            (w.RUNTIME/'pending-publication.json').write_text(json.dumps({'commit':commit,'items':published}))
            git('push','origin','HEAD:main')
        except Exception:
            # Preserve failures, but permit inspection/recovery instead of hiding them.
            raise
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=git('rev-parse','HEAD'):
        raise RuntimeError('Remote publication head differs from verified commit')
    with w.connect() as db:
        for ident in published:db.execute('INSERT OR IGNORE INTO publications VALUES(?,?,?)',(ident,remote,w.now()))
    # Refresh the service's retrieval corpus without changing its running source.
    radar=checkout/'references/sources/radar'
    if radar.exists():
        import shutil
        shutil.copytree(radar,w.ROOT/'references/sources/radar',dirs_exist_ok=True)
    return {'published':len(published),'commit':remote,'url':'https://github.com/anvilarth/end-to-end-performance-skill/commit/'+remote}
