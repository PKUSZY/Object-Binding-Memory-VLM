"""CPU checks of experimental matching, selection and honest scoring; no model scores simulated as results."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from bindmem.common import digest, read_jsonl, sha_file, write_json, write_jsonl
from bindmem_extra.gradient import catalogue, generate, validate, pair_counts
from bindmem_extra.inference import llava_prompt, pad_square
from bindmem_extra.score import load_rows, write_report, compare
from bindmem_extra.cases import export_cases
from PIL import Image


class ExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.data=cls.root/'n6';generate(cls.data,groups=4,seed=19,n=6)
        cls.n4=cls.root/'n4';generate(cls.n4,groups=4,seed=19,n=4)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def fixture_run(self,name,answer='NO'):
        out=self.root/name;out.mkdir()
        items=read_jsonl(self.data/'inputs/test.jsonl')
        write_json(out/'manifest.json',dict(dataset_sha256=sha_file(self.data/'manifest.json'),split='test',
            levels=None,questions=None,seed=20261022,do_sample=False,max_new_tokens=80))
        write_jsonl(out/'predictions.jsonl',[dict(trial_id=r['trial_id'],raw_response=answer,error=None) for r in items])
        return out

    def test_gradient_invariants(self):
        for n,path,ks in [(4,self.n4,[1,2,3]),(6,self.data,[0,1,2,3])]:
            result=validate(path)
            self.assertEqual(result['gradient_levels'],ks)
            self.assertEqual(result['trials'],4*len(ks)*2*3)
        self.assertNotIn(0,catalogue(4)[0])
        # Enumerated n=6 design uses every k with the same entire column multiset.
        for k,matrices in catalogue(6)[0].items():
            for matrix in matrices:
                self.assertEqual([sum(o[d]==0 for o in matrix) for d in range(3)],[2,2,2])
                self.assertEqual(sum(x>0 for x in pair_counts(matrix,(0,0,0))),k)

    def test_validator_catches_relabelled_k(self):
        p=self.data/'labels/test.jsonl';old=p.read_text();mp=self.data/'manifest.json';oldm=mp.read_text()
        rows=read_jsonl(p);r=next(r for r in rows if r['answer']=='NO');r['gradient_k']=9
        try:
            write_jsonl(p,rows);m=json.loads(oldm);m['files']['labels/test.jsonl']=sha_file(p);write_json(mp,m)
            with self.assertRaises(AssertionError):validate(self.data,False)
        finally:p.write_text(old);mp.write_text(oldm)

    def test_llava_order_and_no_crop(self):
        item=read_jsonl(self.data/'inputs/test.jsonl')[0]
        prompt,paths=llava_prompt(item,self.data)
        self.assertEqual(prompt.count('<image>'),len(paths))
        self.assertLess(prompt.index('Study image 1:'),prompt.index('<image>'))
        self.assertLess(prompt.index('Study image 3:'),prompt.index('End of study.'))
        self.assertNotIn(item['trial_id'],prompt)
        im=Image.new('RGB',(448,224),'red');im.putpixel((447,223),(0,255,0))
        padded=pad_square(im,(0,0,0))
        self.assertEqual(padded.size,(448,448));self.assertEqual(padded.getpixel((447,335)),(0,255,0))

    def test_selection_failures_and_scoring(self):
        out=self.fixture_run('selection',answer='INVALID')
        rows=read_jsonl(out/'predictions.jsonl');plan=dict(selected=[{'trial_id':r['trial_id']} for r in rows[:-1]],
            excluded=[dict(trial_id=rows[-1]['trial_id'],split='test',reason='native_context_limit')])
        write_json(out/'selection.json',plan);m=json.loads((out/'manifest.json').read_text());m['selection_sha256']=digest(plan);write_json(out/'manifest.json',m)
        selected,excluded=load_rows(self.data,out)
        self.assertEqual(len(selected),len(rows)-1);self.assertEqual(len(excluded),1)
        report=write_report(selected,excluded,out/'score',bootstrap=20)
        self.assertTrue(report['complete']);self.assertTrue(all(c['accuracy']['value']==0 for c in report['by_condition'].values()))
        self.assertEqual(sum(c['invalid'] for c in report['by_condition'].values()),len(selected))
        rows[0]['error']='out of memory';write_jsonl(out/'predictions.jsonl',rows)
        with self.assertRaises(ValueError):load_rows(self.data,out)
        selected,_=load_rows(self.data,out,allow_partial=True);self.assertFalse(selected[0]['observed'])
        plan['selected']=[];write_json(out/'selection.json',plan)
        with self.assertRaises(ValueError):load_rows(self.data,out,allow_partial=True)

    def test_actual_error_export_and_pairing(self):
        a=self.fixture_run('model_a','YES');b=self.fixture_run('model_b','NO')
        doc=export_cases(self.data,[str(a)],self.root/'cases',limit=1,levels=['G'])
        self.assertEqual(len(doc['cases']),2)
        self.assertTrue(all(c['label']['gradient_k']==3 and c['prediction']['raw_response']=='YES' for c in doc['cases']))
        self.assertEqual([x['false_acceptances'] for x in doc['denominators']],[3,3])
        empty=export_cases(self.data,[str(b)],self.root/'no_cases',levels=['G'])
        self.assertFalse(empty['cases'])
        compare(self.data,[str(a)],[str(b)],self.root/'comparison',bootstrap=20)
        result=json.loads((self.root/'comparison/comparison.json').read_text())
        self.assertEqual(result['shared'],72)
        self.assertTrue(all(c['delta_b_minus_a']['delta_b_minus_a']==0 for c in result['by_condition'].values()))


if __name__=='__main__':unittest.main()
