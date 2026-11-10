import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1]/'scripts/performance_watch/watch.py'
spec = importlib.util.spec_from_file_location('watch', P)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(w, 'RUNTIME', Path(self.tmp.name))
        self.patch.start()
    def tearDown(self):
        self.patch.stop(); self.tmp.cleanup()
    def test_download_has_wall_clock_deadline(self):
        import types, time, sys
        transport = types.ModuleType('research_core.documents')
        transport.download = lambda url: time.sleep(1)
        with patch.dict(sys.modules, {'research_core.documents': transport}):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                w.get('https://example.org/', deadline=0.02)
            self.assertLess(time.monotonic()-started, 0.5)
    def test_canonical_identity_and_priority(self):
        a=w.add('https://example.org/post/?utm_source=a','Paper')
        b=w.add('https://example.org/post#section','Other title',priority=10)
        self.assertEqual(a,b)
        with w.connect() as db:
            self.assertEqual(db.execute('select count(*) from items').fetchone()[0],1)
            self.assertEqual(db.execute('select priority from items').fetchone()[0],10)
        self.assertNotEqual(w.canonical('https://example.org/?page=1'),w.canonical('https://example.org/?page=2'))
    def test_rejects_unattributed_or_local_claim(self):
        d={k:'valid field' for k in ('observation','action','outcome','scope','transfer_test','section')}
        d.update(evidence_quote='The operation measured twice as fast.',local_validation='not_run')
        r={'relevant':True,'decisions':[d]}
        w.validate_review(r,d['evidence_quote'])
        with self.assertRaises(ValueError):w.validate_review(r,'unrelated source')
        d['local_validation']='passed'
        with self.assertRaises(ValueError):w.validate_review(r,d['evidence_quote'])
    def test_failures_retained_with_bounded_retries(self):
        ident=w.add('https://example.org/paper')
        with patch.object(w,'document',side_effect=RuntimeError('source unavailable')):
            for _ in range(4):w.review_batch(1)
        with w.connect() as db:
            row=db.execute('select * from items where id=?',(ident,)).fetchone()
        self.assertEqual(row['state'],'failed');self.assertEqual(row['attempts'],3)
        self.assertIn('unavailable',row['error'])
    def test_independent_rejection_not_published(self):
        ident=w.add('https://example.org/paper','Kernel performance')
        text='This observed optimization improves throughput in the measured setup.'
        d={k:'field' for k in ('observation','action','outcome','scope','transfer_test','section')}
        d.update(evidence_quote=text,local_validation='not_run')
        with patch.object(w,'document',return_value=(text,'body')),patch.object(w,'model_call',side_effect=[{'relevant':True,'decisions':[d]}, {'accepted':False,'reasons':['unsupported']}, {'relevant':True,'decisions':[d]}, {'accepted':False,'reasons':['still unsupported']} ]):
            w.review_batch(1)
        with w.connect() as db:row=db.execute('select * from items where id=?',(ident,)).fetchone()
        self.assertEqual(row['state'],'needs_review');self.assertIsNone(row['report'])
    def test_recovery_and_verified_search(self):
        ident=w.add('https://example.org/paper','CUDA graph reuse')
        with w.connect() as db:db.execute("update items set state='reviewing'")
        with patch.object(w,'document',return_value=('source body','body')),patch.object(w,'model_call',return_value={'relevant':False}):w.review_batch(1)
        self.assertEqual(w.search('CUDA')[0]['state'],'irrelevant')
    def test_unsafe_url(self):
        for u in ['file:///etc/passwd','https://user:password@example.org']:
            with self.assertRaises(ValueError):w.add(u)

class PublicationTests(unittest.TestCase):
    def test_publishes_only_documentary_records_and_deduplicates(self):
        import subprocess, json, hashlib, shutil
        pubspec=importlib.util.spec_from_file_location('publisher',P.with_name('publish.py'))
        pub=importlib.util.module_from_spec(pubspec);pubspec.loader.exec_module(pub)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bare=root/'origin.git'; seed=root/'seed'; runtime=root/'runtime'
            def git(*args):
                return subprocess.run(['git',*map(str,args)],check=True,capture_output=True,text=True)
            git('init','--bare',bare);git('init','-b','main',seed)
            git('-C',seed,'config','user.email','test@example.org');git('-C',seed,'config','user.name','Test')
            (seed/'scripts').mkdir();shutil.copy2(P.parents[1]/'decision_library.py',seed/'scripts/decision_library.py')
            d=seed/'references/decisions';d.mkdir(parents=True)
            (d/'sources.json').write_text('[]');(d/'decisions.jsonl').write_text('')
            git('-C',seed,'add','.');git('-C',seed,'commit','-m','base');git('-C',seed,'remote','add','origin',bare);git('-C',seed,'push','origin','main')
            with patch.object(w,'RUNTIME',runtime),patch.object(w,'ROOT',seed),patch.dict(w.CONFIG,{'auto_publish':True,'publish_repository':str(bare)}):
                ident=w.add('https://example.org/kernel','Kernel experiment')
                text='The measured kernel executes faster in this specific setup.'
                digest=hashlib.sha256(text.encode()).hexdigest();(runtime/'evidence').mkdir();(runtime/'evidence'/f'{digest}.txt').write_text(text)
                fields={k:'condition' for k in ('observation','action','outcome','scope','transfer_test','section')}
                fields.update(evidence_quote=text,local_validation='not_run')
                review={'relevant':True,'summary':'External evidence','decisions':[fields],'verification':{'accepted':True}}
                report=runtime/'candidates'/f'{ident}.md';report.parent.mkdir();report.write_text('# External evidence\n\nhttps://example.org/kernel\n')
                with w.connect() as db:db.execute("update items set state='verified',content_hash=?,review=?,report=? where id=?",(digest,json.dumps(review),str(report),ident))
                with patch.object(w,'model_call',return_value={'accepted':True,'reasons':[]}):
                    first=pub.publish(w);second=pub.publish(w)
                self.assertEqual(first['published'],1);self.assertEqual(second['published'],0)
                paths=git('--git-dir',bare,'diff-tree','--no-commit-id','--name-only','-r','main').stdout.splitlines()
                self.assertEqual(len(paths),3)
                self.assertTrue(all(p.startswith('references/sources/radar/') for p in paths))
    def test_private_data_is_not_public(self):
        pubspec=importlib.util.spec_from_file_location('publisher',P.with_name('publish.py'))
        pub=importlib.util.module_from_spec(pubspec);pubspec.loader.exec_module(pub)
        with self.assertRaises(ValueError):pub.validate_public('Stored in /mnt/train-nfs/andrei/private')

if __name__=='__main__':unittest.main()
