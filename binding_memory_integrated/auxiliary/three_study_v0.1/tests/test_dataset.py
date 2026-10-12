import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate import build, image_record
from validate import validate
from difficulty import annotate_existing


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix="binding-dataset-check-")
        cls.root=Path(cls.tmp.name)/"first"
        with contextlib.redirect_stdout(io.StringIO()):
            build(cls.root,seed=17,dev=6,test=6,calibration_variants=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_valid_dataset(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report=validate(self.root,write_report=False)
        self.assertTrue(report["passed"])
        self.assertEqual(report["trials_checked"],60)

    def test_reproducible_files(self):
        other=Path(self.tmp.name)/"second"
        with contextlib.redirect_stdout(io.StringIO()):
            build(other,seed=17,dev=6,test=6,calibration_variants=0)
        first={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        second={p.relative_to(other):p.read_bytes() for p in other.rglob('*') if p.is_file()}
        self.assertEqual(first.keys(),second.keys())
        self.assertTrue(all(first[p]==second[p] for p in first),"Same seed/config must reproduce every artifact byte")

    def check_corruption(self,relative,mutate,error):
        path=self.root/relative;before=path.read_text()
        rows=[json.loads(line) for line in before.splitlines()]
        mutate(rows[0]);path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        try:
            with self.assertRaisesRegex(ValueError,error):
                validate(self.root,write_report=False)
        finally:
            path.write_text(before)

    def test_reject_wrong_answer(self):
        self.check_corruption('episodes/dev.jsonl',lambda row:row['tests'][0].update(answer='NO'),'Membership label wrong')

    def test_reject_label_in_model_input(self):
        self.check_corruption('inputs/dev.jsonl',lambda row:row.update(answer='YES'),'extra fields/labels')

    def test_reject_wrong_feature_origin(self):
        self.check_corruption('episodes/dev.jsonl',lambda row:row['tests'][0]['feature_sources'].update(color=2),'Feature provenance wrong')

    def test_reject_target_ranked_as_hard(self):
        self.check_corruption('episodes/dev.jsonl',lambda row:row['tests'][0].update(structural_difficulty='hard'),'Structural difficulty wrong')

    def test_reject_lure_in_shared_target_index(self):
        path=self.root/'difficulty/dev.json';before=path.read_text()
        data=json.loads(before)
        data['shared_target_ids'][0]=data['tiers']['hard']['lure_ids'][0]
        path.write_text(json.dumps(data))
        try:
            with self.assertRaisesRegex(ValueError,'Difficulty group index wrong'):
                validate(self.root,write_report=False)
        finally:
            path.write_text(before)

    def test_annotation_preserves_stimuli_and_safe_inputs(self):
        other=Path(self.tmp.name)/'annotation'
        with contextlib.redirect_stdout(io.StringIO()):
            build(other,seed=31,dev=6,test=0,calibration_variants=0)
        paths=[p for folder in ('images','inputs','labels') for p in (other/folder).glob('*') if p.is_file()]
        before={p:p.read_bytes() for p in paths}
        # Simulate the old JSONL schema, with no difficulty fields.
        for folder in ('episodes','manifests'):
            path=other/folder/'dev.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            tests=[t for ep in rows for t in ep['tests']] if folder=='episodes' else rows
            for test in tests:
                test.pop('structural_difficulty');test.pop('novel_feature_count')
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report=annotate_existing(other)
        self.assertTrue(report['images_inputs_labels_unchanged'])
        self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
        self.assertEqual(report,annotate_existing(other))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(validate(other,write_report=False)['passed'])

    def test_pixel_duplicate_is_resampled_deterministically(self):
        attrs={'color':'red','shape':'circle','texture':'solid'}
        for folder in ['collision1','collision2']:
            out=Path(self.tmp.name)/folder;(out/'images').mkdir(parents=True)
            hashes=set();a=image_record(out,attrs,123,hashes);b=image_record(out,attrs,123,hashes)
            self.assertNotEqual(a['pixel_sha256'],b['pixel_sha256'])
            self.assertGreater(b['render_resample_count'],0)
            if folder=='collision1': expected=(a,b)
            else: self.assertEqual(expected,(a,b))


if __name__=='__main__':
    unittest.main()
