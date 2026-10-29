"""Compare two model runs on exactly matched trials and within each task type."""
import argparse
import json
from pathlib import Path

from .common import write_json
from .score import collect
from .stats import paired_delta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2')
    p.add_argument('--run-a', required=True); p.add_argument('--run-b', required=True)
    p.add_argument('--output', default='runs/bindmem_v2/comparison.json')
    p.add_argument('--split', choices=['dev','test','both'], default='test')
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    manifests = [json.loads((Path(path)/'manifest.json').read_text()) for path in (args.run_a,args.run_b)]
    for field in ('max_new_tokens','max_pixels','do_sample','seed','attention'):
        if manifests[0].get(field) != manifests[1].get(field):
            raise SystemExit(f'Model comparison requires matching inference settings: {field}')
    a,b = [collect(args.data, path, args.split) for path in [args.run_a,args.run_b]]
    if {r['trial_id'] for r in a} != {r['trial_id'] for r in b}:
        raise SystemExit('Model selections differ; run matching conditions before comparing')
    if any(not r['observed'] for r in a+b): raise SystemExit('Both model runs must be complete')
    result = dict(run_a=args.run_a, run_b=args.run_b, split=args.split,
                  by_level_question={f'{l}/{q}': paired_delta([r for r in a if (r['level'],r['question']) == (l,q)],
                        [r for r in b if (r['level'],r['question']) == (l,q)], args.bootstrap)
                        for l,q in sorted({(r['level'],r['question']) for r in a})})
    write_json(Path(args.output), result)
    print(args.output)


if __name__ == '__main__': main()
