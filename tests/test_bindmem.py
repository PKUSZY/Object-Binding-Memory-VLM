"""Protocol tests use tiny procedural fixtures and synthetic outputs, never model scores."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from PIL import Image, ImageChops

from bindmem.common import canonical, parse_answer, read_jsonl, sha_file, write_json, write_jsonl
from bindmem.generate import Builder, parity_sets, lure_type
from bindmem.run import recover_rows, make_messages
from bindmem.score import score
from bindmem.stats import cluster_ci, paired_delta
from bindmem.validate import marginals, validate


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)/'data'
        Builder(cls.root, 4, 17, ['L0','L1','L2','L3','L4','L5'], [f'Q{i}' for i in range(1,9)]).run()

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_all_invariants(self):
        result = validate(self.root)
        self.assertGreater(result['counterfactual_pairs'], 100)
        inputs = read_jsonl(self.root/'inputs/test.jsonl')
        self.assertEqual({r['question'] for r in inputs}, {'CAL',*[f'Q{i}' for i in range(1,9)]})
        self.assertEqual({r['level'] for r in inputs}, {f'L{i}' for i in range(6)})
        for row in inputs:
            messages = make_messages(row,self.root,802816)
            self.assertNotIn('trial_id',canonical(messages))
        labels = read_jsonl(self.root/'labels/test.jsonl')
        for level in ('L1','L2','L3','L4','L5'):
            ys = Counter(r['answer'] for r in labels if r['level'] == level and r['question'] in ('Q1','Q2','Q3','Q8'))
            self.assertEqual(ys['YES'],ys['NO'])

    def test_reject_incorrect_semantic_label(self):
        path = self.root/'labels/test.jsonl'
        original = path.read_text()
        manifest_path = self.root/'manifest.json'
        original_manifest = manifest_path.read_text()
        labels = read_jsonl(path)
        row = next(r for r in labels if r['level'] == 'L2' and r['question'] == 'Q2' and r['answer'] == 'NO')
        row['answer'] = 'YES'
        try:
            write_jsonl(path,labels)
            manifest = json.loads(original_manifest)
            manifest['files']['labels/test.jsonl'] = sha_file(path)
            write_json(manifest_path,manifest)
            with self.assertRaises(AssertionError): validate(self.root,check_images=False)
        finally:
            path.write_text(original)
            manifest_path.write_text(original_manifest)

    def test_packed_candidates_preserve_object_pixels(self):
        inputs = {r['trial_id']:r for r in read_jsonl(self.root/'inputs/test.jsonl')}
        labels = read_jsonl(self.root/'labels/test.jsonl')
        for mode in ('2d','3d_fixed','3d_cross','3d_occluded'):
            a = next(r for r in labels if r['level'] == 'L5' and r['question'] == 'Q4' and r['mode'] == mode)
            b = next(r for r in labels if r['semantic_id'] == a['semantic_id'] and r['question'] == 'Q5')
            packed_path = next(x['image'] for x in inputs[a['trial_id']]['query'] if x['type'] == 'image')
            separate = [x['image'] for x in inputs[b['trial_id']]['query'] if x['type'] == 'image']
            with Image.open(self.root/packed_path) as packed:
                for i,path in enumerate(separate):
                    with Image.open(self.root/path) as single:
                        # Exclude only the top letter-label area from the packed image.
                        self.assertIsNone(ImageChops.difference(packed.crop((i*224,32,(i+1)*224,224)),
                                                               single.crop((0,32,224,224))).getbbox())

    def test_parity_requires_triples(self):
        a,b = parity_sets([['red','blue'],['dots','solid'],['circle','star']])
        self.assertEqual(marginals(a),marginals(b))
        self.assertEqual(lure_type(b,a[0]), 'pairwise_complete')
        self.assertNotEqual(marginals(a),marginals(b[:-1]))

    def test_strict_parsing(self):
        self.assertEqual(parse_answer(' yes ','binary'),'YES')
        self.assertIsNone(parse_answer('YES because ...','binary'))
        self.assertIsNone(parse_answer('A.','choice2'))
        self.assertIsNone(parse_answer('{"color":"red"}','attributes'))
        self.assertIsNone(parse_answer('{"color":[],"texture":"grid","shape":"star"}','attributes'))

    def test_recover_truncated_append(self):
        p = Path(self.temp.name)/'tail.jsonl'
        p.write_bytes(b'{"trial_id":"a"}\n{"trial_')
        self.assertEqual(recover_rows(p),[{'trial_id':'a'}])
        self.assertEqual(p.read_bytes(),b'{"trial_id":"a"}\n')
        p.write_bytes(b'bad\n{"trial_id":"a"}\n')
        with self.assertRaises(ValueError): recover_rows(p)

    def test_score_oracle_and_partial(self):
        run = Path(self.temp.name)/'oracle'; run.mkdir(exist_ok=True)
        write_json(run/'manifest.json',dict(dataset_sha256=sha_file(self.root/'manifest.json'),split='both',levels=None,questions=None))
        labels = read_jsonl(self.root/'labels/test.jsonl')
        predictions = [dict(trial_id=r['trial_id'],raw_response=json.dumps(r['answer']) if isinstance(r['answer'],dict) else r['answer'],error=None) for r in labels]
        write_jsonl(run/'predictions.jsonl',predictions)
        report = score(self.root,run,run/'scored',repetitions=20)
        self.assertTrue(report['complete'])
        self.assertTrue(all(v['accuracy']['value'] == 1 for v in report['by_condition'].values()))
        self.assertEqual(report['paired_formats']['Q1_vs_Q2']['delta_b_minus_a'],0)
        # An invalid actual output counts as wrong; missing inference is a coverage issue.
        predictions[0]['raw_response'] = 'not a valid answer'
        write_jsonl(run/'predictions.jsonl',predictions[:-1])
        with self.assertRaises(ValueError): score(self.root,run,run/'partial',repetitions=20)
        partial = score(self.root,run,run/'partial',repetitions=20,allow_partial=True)
        self.assertFalse(partial['complete'])
        self.assertEqual(sum(v['invalid'] for v in partial['by_condition'].values()),1)

    def test_cluster_bootstrap(self):
        self.assertIsNone(cluster_ci([0,1],['a','a'],20))
        self.assertEqual(cluster_ci([1,1,1],['a','b','c'],20),[1.0,1.0])


if __name__ == '__main__': unittest.main()
