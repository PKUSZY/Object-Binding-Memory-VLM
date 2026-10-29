"""Strictly marginal-matched recombination gradient; no model dependencies."""
import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from bindmem.common import VALUES, VERSION, digest, read_jsonl, sha_file, write_json, write_jsonl
from bindmem.generate import Builder
from bindmem.validate import validate as validate_base

PAIRS = ((0, 1), (0, 2), (1, 2))


def pair_counts(study, probe):
    return [sum(all(o[d] == probe[d] for d in pair) for o in study) for pair in PAIRS]


def single_counts(study):
    return [Counter(o[d] for o in study) for d in range(3)]


@lru_cache(None)
def catalogue(n):
    # Every column has exactly the same multiset, including the candidate value 0.
    # Holding the first column sorted removes redundant permutations of rows.
    column = (0, 0, 1, 2) if n == 4 else (0, 0, 1, 1, 2, 2)
    permutations = sorted(set(itertools.permutations(column)))
    neg, pos = defaultdict(list), []
    for b in permutations:
        for c in permutations:
            rows = tuple(zip(column, b, c))
            if len(set(rows)) != n:
                continue
            if (0, 0, 0) in rows:
                pos.append(rows)
            else:
                counts = pair_counts(rows, (0, 0, 0))
                # In the full six-object gradient, each present pair occurs exactly once.
                # For n=4,k=1 its one present pair must occur twice; keep this distinction explicit.
                if n == 6 and max(counts) > 1:
                    continue
                neg[sum(v > 0 for v in counts)].append(rows)
    expected = {1, 2, 3} if n == 4 else {0, 1, 2, 3}
    assert set(neg) == expected and pos
    return dict(neg), pos


def generate(root, groups=320, seed=20261028, n=6):
    root = Path(root)
    if groups < 4 or groups % 4 or n not in (4, 6):
        raise ValueError('groups must be a positive multiple of 4; n must be 4 or 6')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Output must be empty; existing experiments are never overwritten')
    b = Builder(root, groups, seed, ['G'], ['Q1', 'Q2', 'Q3'])
    negative, positive = catalogue(n)
    per = 1 if n == 4 else 2
    for g in range(groups):
        rng = random.Random(int(digest([seed, g, 'gradient', n])[:16], 16))
        banks = [rng.sample(list(v), 3) for v in VALUES]
        probe = tuple(bank[0] for bank in banks)
        # Permute attribute roles so k=1 and k=2 do not privilege a specific pair.
        axes = list(range(3)); rng.shuffle(axes)
        def decode(matrix):
            objects = [tuple(banks[d][r[axes[d]]] for d in range(3)) for r in matrix]
            rng.shuffle(objects)
            return objects
        pos = decode(rng.choice(positive))
        for k in sorted(negative):
            neg = decode(rng.choice(negative[k]))
            assert single_counts(neg) == single_counts(pos)
            for world, study in [('positive', pos), ('negative', neg)]:
                base = b.base(g, 'G', n=n, per=per, gradient_k=k, world=world,
                              matched_block_id=f'gradient-n{n}-seed{seed}-g{g:05d}',
                              candidate_attribute_counts=[2, 2, 2],
                              candidate_pair_counts=pair_counts(study, probe))
                b.object_tasks(base, study, probe, f'gradient{k}_{world}', rng)
    for split in ('dev', 'test'):
        write_jsonl(root/'inputs'/f'{split}.jsonl', [r for r in b.inputs if r['split'] == split])
        write_jsonl(root/'labels'/f'{split}.jsonl', [r for r in b.labels if r['split'] == split])
    assets = [{'path': p, 'sha256': sha_file(root/p), **s} for p, s in sorted(b.renderer.specs.items())]
    write_jsonl(root/'assets.jsonl', assets)
    files = [root/'assets.jsonl', *sorted((root/'inputs').glob('*.jsonl')), *sorted((root/'labels').glob('*.jsonl'))]
    write_json(root/'manifest.json', dict(version=VERSION, extension='matched-gradient-1', seed=seed,
        groups=groups, total_objects=n, objects_per_image=per, levels=['G'], questions=b.questions,
        gradient_levels=sorted(negative), candidate_attribute_counts=[2, 2, 2],
        positive_policy='Identical positive input repeated across k within cluster; not independent evidence.',
        control='All first-order attribute value counts, query identity/pixels, n, layout, rendering and 1:1 labels fixed within cluster.',
        pair_multiplicity_policy='Each present candidate pair occurs once.' if n == 6 else 'k=1 necessarily repeats its one pair twice; k=2/3 have one occurrence per present pair.',
        limitation='Pair co-occurrence arrangements necessarily change; this is not a pixel-identical manipulation.',
        generator_source={str(p): sha_file(p) for p in [Path(__file__), Path('bindmem/generate.py'), Path('bindmem/render.py'), Path('bindmem/common.py')]},
        trials=len(b.inputs), images=len(assets), by_level={'G': len(b.inputs)},
        files={str(p.relative_to(root)): sha_file(p) for p in files}))
    return validate(root)


def validate(root, check_images=True):
    root = Path(root)
    result = validate_base(root, check_images)
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['extension'] == 'matched-gradient-1'
    inputs = {r['trial_id']: r for split in ('dev', 'test') for r in read_jsonl(root/'inputs'/f'{split}.jsonl')}
    blocks = defaultdict(list)
    for split in ('dev', 'test'):
        for r in read_jsonl(root/'labels'/f'{split}.jsonl'):
            blocks[(r['cluster_id'], r['question'])].append(r)
    for rr in blocks.values():
        assert len(rr) == 2 * len(manifest['gradient_levels'])
        first = rr[0]; study0 = [o for scene in first['study_scenes'] for o in scene]
        positives = []
        for r in rr:
            item = inputs[r['trial_id']]
            study = [o for scene in r['study_scenes'] for o in scene]
            assert r['total_objects'] == manifest['total_objects'] and len(study) == r['total_objects']
            assert all(len(scene) == manifest['objects_per_image'] for scene in r['study_scenes'])
            assert single_counts(study) == single_counts(study0)
            assert r['probe'] == first['probe'] and item['query'] == inputs[first['trial_id']]['query']
            assert [sum(o[d] == r['probe'][d] for o in study) for d in range(3)] == [2, 2, 2]
            counts = pair_counts(study, r['probe'])
            assert counts == r['candidate_pair_counts']
            if r['answer'] == 'NO':
                assert sum(c > 0 for c in counts) == r['gradient_k']
                if manifest['total_objects'] == 6:
                    assert max(counts) <= 1
            else:
                positives.append(item['study_images'])
        assert all(p == positives[0] for p in positives)
        for k in manifest['gradient_levels']:
            assert Counter(r['answer'] for r in rr if r['gradient_k'] == k) == {'YES': 1, 'NO': 1}
    result.update(gradient_checked=True, gradient_levels=manifest['gradient_levels'])
    write_json(root/'validation.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/bindmem_gradient_n6')
    p.add_argument('--groups', type=int, default=320)
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--objects', type=int, choices=[4, 6], default=6)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    print(validate(args.output) if args.validate_only else generate(args.output, args.groups, args.seed, args.objects))


if __name__ == '__main__': main()
