"""CPU end-to-end probe plumbing with synthetic vectors (not experimental results)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bindmem.common import ATTRS, VALUES, write_json, write_jsonl, sha_file


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data, run = Path(tmp)/'data', Path(tmp)/'run'
        (run/'activations').mkdir(parents=True)
        inputs, labels, raw = [], [], []
        for g in range(512):
            for task in ('cal','yes','no'):
                tid = f'{g}-{task}'
                cal = task == 'cal'
                truth = {a:v[g % len(v)] for a,v in zip(ATTRS,VALUES)} if cal else ('YES' if task == 'yes' else 'NO')
                label = dict(trial_id=tid,split='dev' if g < 128 else 'test',cluster_id=f'g{g}',
                             level='L0' if cal else 'L2',question='CAL' if cal else 'Q2',answer=truth,
                             mode='2d',total_objects=1 if cal else 4,objects_per_image=1,
                             probe_type='calibration' if cal else ('target' if task == 'yes' else 'pairwise_complete'))
                labels.append(label)
                inputs.append(dict(trial_id=tid,split=label['split'],response_kind='attributes' if cal else 'binary'))
                x = np.zeros(18,dtype='float32')
                if cal:
                    for offset,v in zip((0,6,10),VALUES): x[offset+g % len(v)] = 1
                else: x[16] = 1 if task == 'yes' else -1
                np.savez_compressed(run/'activations'/f'{tid}.npz', **{'decoder.0.answer':x})
                raw.append(dict(trial_id=tid,raw_response=json.dumps(truth) if cal else truth,
                                activation_file=f'activations/{tid}.npz',error=None))
        for split in ('dev','test'):
            write_jsonl(data/'inputs'/f'{split}.jsonl',[r for r in inputs if r['split'] == split])
            write_jsonl(data/'labels'/f'{split}.jsonl',[r for r in labels if r['split'] == split])
        write_json(data/'manifest.json',{'synthetic_test':True,
            'files':{str(p.relative_to(data)):sha_file(p) for p in data.glob('*/*.jsonl')}})
        write_json(run/'manifest.json',dict(dataset_sha256=sha_file(data/'manifest.json'),split='both'))
        write_jsonl(run/'predictions.jsonl',raw)
        subprocess.run([sys.executable,'-m','bindmem.probes','--data',str(data),'--run',str(run),
                        '--epochs','150','--penalties','.01','--bootstrap','20','--threads','2'],check=True)
        results = json.loads((run/'probes/summary.json').read_text())
        assert len(results) == 8
        split = json.loads((run/'probes/split.json').read_text())
        assert not set(split['train']) & set(split['test'])
        assert all(r['test']['n'] > 0 for r in results)
        assert all(r['test']['accuracy'] > .95 for r in results), results
        assert (run/'probes/layers.svg').exists()
        assert json.loads((run/'probes/manifest.json').read_text())['complete']
        import torch
        state = torch.load(next((run/'probes/weights').glob('*.pt')), weights_only=True)
        assert state['mean'].shape == (18,)
        print('PASS: synthetic feature/binding, linear/MLP, held-out splits, permutation control and plots')


if __name__ == '__main__': main()
