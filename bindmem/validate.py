"""Validate labels, counterfactual marginals, cross-format matching and assets."""
import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .common import ATTRS, VALUES, read_jsonl, sha_file, write_json
from .generate import lure_type


def marginals(study):
    return [Counter(tuple(o[i] for i in dims) for o in study)
            for k in (1, 2) for dims in itertools.combinations(range(3), k)]


def validate(root, check_images=True):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for p, sha in manifest['files'].items():
        assert sha_file(root/p) == sha, f'Manifest mismatch: {p}'
    inputs = sum([read_jsonl(root/'inputs'/f'{s}.jsonl') for s in ('dev', 'test')], [])
    labels = sum([read_jsonl(root/'labels'/f'{s}.jsonl') for s in ('dev', 'test')], [])
    assets = {a['path']: a for a in read_jsonl(root/'assets.jsonl')}
    assert len(inputs) == len(labels) == manifest['trials']
    assert len({r['trial_id'] for r in inputs}) == len(inputs), 'Duplicate trial ID'
    indexed = {r['trial_id']: r for r in labels}
    assert set(indexed) == {r['trial_id'] for r in inputs}
    pairs, semantic, clusters = defaultdict(list), defaultdict(list), defaultdict(set)
    for item in inputs:
        assert set(item) == {'trial_id','split','level','question','response_kind','instruction','study_images','query'}
        label = indexed[item['trial_id']]
        assert all(item[k] == label[k] for k in ('split', 'level', 'question'))
        clusters[label['cluster_id']].add(item['split'])
        study = [tuple(o) for scene in label['study_scenes'] for o in scene]
        assert len(study) == label['total_objects']
        assert len(study) == len(set(study)), 'Duplicate study identity'
        for obj in study:
            assert all(x in v for x, v in zip(obj, VALUES))
        assert len(item['study_images']) == len(label['study_scenes'])
        for path, scene in zip(item['study_images'], label['study_scenes']):
            assert assets[path]['objects'] == scene, 'Rendered study disagrees with labels'
            assert assets[path]['relation'] == label.get('layout')
            assert assets[path]['mode'] == label.get('study_mode','2d')
        for block in item['query']:
            assert set(block) == ({'type','text'} if block['type'] == 'text' else {'type','image'})
        used = item['study_images'] + [b['image'] for b in item['query'] if b['type'] == 'image']
        assert all(p in assets for p in used)
        if item['response_kind'] == 'binary' and item['level'] != 'L4':
            obj = tuple(label['probe'])
            assert label['answer'] == ('YES' if obj in study else 'NO')
            assert label['probe_type'] == lure_type(study, obj)
        if item['response_kind'].startswith('choice'):
            candidates = [tuple(o) for o in label['candidates']]
            hits = [i for i, o in enumerate(candidates) if o in study]
            assert len(candidates) == len(set(candidates))
            assert len(hits) == 1 and label['answer'] == chr(65+hits[0])
            rendered = [tuple(o) for b in item['query'] if b['type'] == 'image' for o in assets[b['image']]['objects']]
            assert candidates == rendered
        if item['response_kind'] == 'recall':
            hits = [o for o in study if [o[0], o[2]] == label['cue']]
            assert len(hits) == 1 and hits[0][1].upper() == label['answer']
        if item['response_kind'] == 'attributes':
            assert dict(zip(ATTRS, study[0])) == label['answer']
        if item['level'] == 'L4':
            a, b = map(tuple, label['probe'])
            scenes = [[tuple(o) for o in sc] for sc in label['study_scenes']]
            truth = any(a in sc and b in sc for sc in scenes) if label['relation'] == 'cooccur' else scenes[0] == [a, b]
            assert label['answer'] == ('YES' if truth else 'NO')
        if label.get('pair_id'):
            pairs[(label['pair_id'], item['question'])].append((item, label))
        if item['question'] in ('Q1','Q2','Q3'):
            semantic[label['semantic_id']].append(item)
    for values in clusters.values():
        assert len(values) == 1, 'Cluster crosses dev/test'
    for rows in semantic.values():
        assert all(r['study_images'] == rows[0]['study_images'] for r in rows)
    for (pair_id, question), rows in pairs.items():
        assert len(rows) == 2, f'Incomplete pair: {pair_id} {question}'
        (ia, a), (ib, b) = rows
        assert ia['query'] == ib['query'], 'Counterfactual query must be identical'
        assert a['answer'] != b['answer'], 'Pair answers must differ'
        sa = [tuple(o) for sc in a['study_scenes'] for o in sc]
        sb = [tuple(o) for sc in b['study_scenes'] for o in sc]
        if a['level'] != 'L4':
            assert marginals(sa) == marginals(sb), 'Unequal first/second order counts'
            if question in ('Q1','Q2','Q3'):
                assert {a['probe_type'], b['probe_type']} == {'target','pairwise_complete'}
        else:
            assert set(sa) == set(sb), 'Relation pair must preserve object identities'
    if check_images:
        for path, asset in assets.items():
            assert sha_file(root/path) == asset['sha256'], f'Changed image: {path}'
            with Image.open(root/path) as im:
                assert im.format == 'PNG' and im.mode == 'RGB'
                im.verify()
    report = dict(ok=True, trials=len(inputs), images=len(assets),
                  counterfactual_pairs=len(pairs), clusters=len(clusters), image_hashes_checked=check_images)
    write_json(root/'validation.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2')
    p.add_argument('--skip-images', action='store_true')
    args = p.parse_args()
    print(validate(args.data, not args.skip_images))


if __name__ == '__main__': main()
