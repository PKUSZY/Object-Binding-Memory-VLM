"""Score outputs, report coverage, clustered CIs and semantic paired comparisons."""
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

from .common import parse_answer, read_jsonl, sha_file, write_json, write_jsonl
from .stats import summarize, paired_delta, logistic_load


def collect(data, run, split='test'):
    data, run = Path(data), Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['dataset_sha256'] != sha_file(data/'manifest.json'):
        raise ValueError('Run and dataset fingerprints differ')
    splits = ['dev','test'] if split == 'both' else [split]
    if manifest['split'] != 'both': splits = [s for s in splits if s == manifest['split']]
    if not splits: raise ValueError('Requested split is not in this run')
    data_manifest = json.loads((data/'manifest.json').read_text())
    for split_name in splits:
        for folder in ('inputs','labels'):
            name = f'{folder}/{split_name}.jsonl'
            if sha_file(data/name) != data_manifest['files'][name]:
                raise ValueError(f'Dataset file fingerprint mismatch: {name}')
    raw = {}
    for r in read_jsonl(run/'predictions.jsonl'):
        if r['trial_id'] in raw and not raw[r['trial_id']].get('error'):
            raise ValueError(f'Duplicate completed trial: {r["trial_id"]}')
        raw[r['trial_id']] = r
    kinds = {r['trial_id']: r['response_kind'] for s in splits for r in read_jsonl(data/'inputs'/f'{s}.jsonl')}
    rows = []
    for s in splits:
        for label in read_jsonl(data/'labels'/f'{s}.jsonl'):
            if manifest.get('levels') and label['level'] not in manifest['levels']: continue
            if manifest.get('questions') and label['question'] not in manifest['questions']: continue
            rec = raw.get(label['trial_id'], {})
            observed = bool(rec) and not rec.get('error')
            kind = kinds[label['trial_id']]
            pred = parse_answer(rec.get('raw_response'), kind) if observed else None
            rows.append(dict(label, kind=kind, observed=observed, prediction=pred,
                             valid=pred is not None, correct=observed and pred == label['answer'],
                             accepted=pred == 'YES', error=rec.get('error')))
    return rows


def svg_chart(path, cells):
    selected = [(k, v['accuracy']) for k,v in cells.items() if v['accuracy']['value'] is not None]
    width, height = 1000, 55+len(selected)*25
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="15" y="22" font-size="16">Accuracy with cluster bootstrap 95% intervals</text>']
    for i,(name,metric) in enumerate(selected):
        y = 46+i*25; x = 480; scale = 440
        parts += [f'<text x="12" y="{y+4}" font-size="12">{html.escape(name)}</text>',
                  f'<rect x="{x}" y="{y-8}" width="{scale*metric["value"]}" height="15" fill="#3979b5"/>']
        if metric['ci95']:
            lo,hi = metric['ci95']
            parts += [f'<path d="M {x+scale*lo} {y} H {x+scale*hi}" stroke="black" stroke-width="2"/>']
        parts += [f'<text x="935" y="{y+4}" font-size="12">{metric["value"]:.1%}</text>']
    parts += ['</svg>']
    path.write_text('\n'.join(parts))


def score(data, run, out, split='test', repetitions=1000, allow_partial=False):
    rows = collect(data, run, split)
    if not rows: raise ValueError('No selected rows')
    missing = sum(not r['observed'] for r in rows)
    if missing and not allow_partial:
        raise ValueError(f'{missing}/{len(rows)} trials missing or failed; finish inference or explicitly use --allow-partial')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for r in rows:
        key = f'{r["level"]}/{r["question"]}/{r["mode"]}/n{r["total_objects"]}/per{r["objects_per_image"]}'
        if r['level'] == 'L4': key += '/'+r['relation']
        grouped[key].append(r)
    conditions = {k: summarize(v, repetitions) for k,v in sorted(grouped.items())}
    # No pooled benchmark accuracy: task types and chance levels differ.
    report = dict(split=split, expected=len(rows), observed=len(rows)-missing,
                  complete=not missing, partial_policy='Rates use observed rows only; invalid outputs count as errors.',
                  bootstrap_unit='cluster_id (all formats, worlds and conditions of a replicate)',
                  by_condition=conditions,
                  by_level_question={f'{l}/{q}': summarize([r for r in rows if r['level'] == l and r['question'] == q], repetitions)
                                     for l,q in sorted({(r['level'],r['question']) for r in rows})},
                  paired_formats={}, exploratory_load_regression=logistic_load(rows))
    for a,b in [('Q1','Q2'),('Q1','Q3'),('Q4','Q5')]:
        report['paired_formats'][f'{a}_vs_{b}'] = paired_delta(
            [r for r in rows if r['question'] == a], [r for r in rows if r['question'] == b], repetitions,
            key=lambda r: r['semantic_id'])
    report['paired_load'], report['paired_total_load'], report['paired_3d'] = {}, {}, {}
    for q in sorted({r['question'] for r in rows}):
        for n in (4,8):
            for per in (2,4,8):
                if per > n: continue
                selected = [r for r in rows if r['level'] == 'L3' and r['total_objects'] == n and r['question'] == q]
                if not selected: continue
                report['paired_load'][f'{q}/n{n}_per1_vs_per{per}'] = paired_delta(
                    [r for r in selected if r['objects_per_image'] == 1],
                    [r for r in selected if r['objects_per_image'] == per], repetitions,
                    key=lambda r: (r['cluster_id'],r['world']))
        for per in (1,2,4):
            selected = [r for r in rows if r['level'] == 'L3' and r['objects_per_image'] == per and r['question'] == q]
            if not selected: continue
            report['paired_total_load'][f'{q}/per{per}_n4_vs_n8'] = paired_delta(
                [r for r in selected if r['total_objects'] == 4], [r for r in selected if r['total_objects'] == 8], repetitions,
                key=lambda r: (r['cluster_id'],r['world']))
        for mode in ('3d_fixed','3d_cross','3d_occluded'):
            selected = [r for r in rows if r['level'] == 'L5' and r['question'] == q]
            if not selected: continue
            report['paired_3d'][f'{q}/2d_vs_{mode}'] = paired_delta(
                [r for r in selected if r['mode'] == '2d'], [r for r in selected if r['mode'] == mode], repetitions,
                key=lambda r: (r['cluster_id'],r['world']))
    write_json(out/'summary.json', report)
    write_jsonl(out/'scored.jsonl', rows)
    with (out/'conditions.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['condition','expected','observed','accuracy','ci_low','ci_high','invalid'])
        for key, v in conditions.items():
            writer.writerow([key,v['expected'],v['observed'],v['accuracy']['value'],*(v['accuracy']['ci95'] or [None,None]),v['invalid']])
    svg_chart(out/'accuracy.svg', conditions)
    print(f'Scored {len(rows)-missing}/{len(rows)} rows -> {out}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2'); p.add_argument('--run', required=True)
    p.add_argument('--output', default=None); p.add_argument('--split', choices=['dev','test','both'], default='test')
    p.add_argument('--bootstrap', type=int, default=1000); p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    if args.bootstrap < 1: p.error('--bootstrap must be positive')
    score(args.data, args.run, args.output or str(Path(args.run)/'scored'/args.split), args.split,args.bootstrap,args.allow_partial)


if __name__ == '__main__': main()
