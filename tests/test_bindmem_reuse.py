"""Ensure priority runs reuse only unchanged, compatible, complete model replies."""
import json
import tempfile
import unittest
from pathlib import Path
from bindmem.common import write_json,write_jsonl
from bindmem_extra.inference import reuse_completed,REUSABLE_RUNNER_SHA256


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.old=dict(dataset_sha256='data',checkpoint={'weights':'same'},processor_files={'config':'same'},backend='llava',
            max_new_tokens=80,max_pixels=802816,seed=20261022,do_sample=False,attention='sdpa',dtype='float16',capture=False,
            image_policy='same',prompt_policy='same',packages={'torch':'same'},
            source={'inference.py':REUSABLE_RUNNER_SHA256,'common.py':'same'})
        self.new=dict(self.old,source={'inference.py':'new-selection-only-version','common.py':'same'})
        write_json(self.root/'manifest.json',self.old)
        self.rows=[dict(trial_id='correct',raw_response='YES',error=None),
                   dict(trial_id='invalid',raw_response='long explanation',error=None),
                   dict(trial_id='failed',raw_response=None,error='CUDA failed')]
        write_jsonl(self.root/'predictions.jsonl',self.rows)
    def tearDown(self):self.temp.cleanup()
    def test_preserves_raw_and_invalid_but_not_failed(self):
        before=(self.root/'predictions.jsonl').read_bytes()
        rows=reuse_completed(self.root,self.new,{'correct','invalid','failed'})
        self.assertEqual([r['trial_id'] for r in rows],['correct','invalid'])
        self.assertEqual(rows[1]['raw_response'],'long explanation')
        self.assertEqual((self.root/'predictions.jsonl').read_bytes(),before)
    def test_refuses_changed_protocol_or_model(self):
        for key,value in [('max_new_tokens',4),('checkpoint',{'weights':'other'}),('image_policy','crop'),('prompt_policy','new')]:
            with self.subTest(key=key),self.assertRaises(ValueError):reuse_completed(self.root,dict(self.new,**{key:value}),{'correct'})
    def test_refuses_changed_dependency(self):
        m=dict(self.new,source=dict(self.new['source'],**{'common.py':'changed'}))
        with self.assertRaises(ValueError):reuse_completed(self.root,m,{'correct'})
    def test_torn_tail_skipped_without_modifying_source(self):
        p=self.root/'predictions.jsonl'
        with p.open('ab') as f:f.write(b'{"trial_id":')
        before=p.read_bytes()
        rows=reuse_completed(self.root,self.new,{'correct'})
        self.assertEqual(len(rows),1);self.assertEqual(p.read_bytes(),before)
    def test_unknown_runner_rejected(self):
        old=dict(self.old,source=dict(self.old['source'],**{'inference.py':'unknown'}));write_json(self.root/'manifest.json',old)
        with self.assertRaises(ValueError):reuse_completed(self.root,self.new,{'correct'})


if __name__=='__main__':unittest.main()
