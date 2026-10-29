"""Generate the independent v2 benchmark: python -m bindmem.generate --help."""
import argparse
import itertools
import random
import platform
from collections import Counter
from pathlib import Path

import PIL
import numpy as np

from .common import ATTRS, VALUES, VERSION, describe, digest, write_json, write_jsonl, sha_file
from .render import Renderer

UNIVERSE = list(itertools.product(*VALUES))
RULE = ('Remember the study objects. Identity requires the SAME color, texture AND shape on one object. '
        'Position, size and viewpoint do not change identity. Shapes of 3D objects refer to their front silhouettes. ')


def text(s):
    return {'type': 'text', 'text': s}


def image(s):
    return {'type': 'image', 'image': s}


def lure_type(study, obj):
    if obj in study:
        return 'target'
    seen = sum(obj[i] in {x[i] for x in study} for i in range(3))
    pairs = sum(any((x[i], x[j]) == (obj[i], obj[j]) for x in study)
                for i, j in [(0, 1), (0, 2), (1, 2)])
    return ('unrelated' if seen == 0 else f'near{seen}' if seen < 3 else
            'pairwise_complete' if pairs == 3 else 'critical')


def parity_sets(banks):
    sets = [[], []]
    for bits in itertools.product(range(2), repeat=3):
        sets[sum(bits) % 2].append(tuple(banks[i][b] for i, b in enumerate(bits)))
    return sets


class Builder:
    def __init__(self, root, groups, seed, levels, questions):
        self.root, self.groups, self.seed = Path(root), groups, seed
        self.levels, self.questions = levels, questions
        self.renderer = Renderer(root)
        self.inputs, self.labels = [], []

    def add(self, meta, scenes, query, kind, answer, **private):
        tid = digest([VERSION, self.seed, meta])[:28]
        study_images = [self.renderer.image(s, mode=meta.get('study_mode', '2d'), variant=self.seed+meta['replicate'],
                                            relation=meta.get('layout')) for s in scenes]
        self.inputs.append(dict(trial_id=tid, split=meta['split'], level=meta['level'],
                                question=meta['question'], response_kind=kind, instruction=RULE,
                                study_images=study_images, query=query))
        self.labels.append(dict(trial_id=tid, **meta, answer=answer, study_scenes=scenes, **private))

    def base(self, g, level, n=1, per=1, mode='2d', **kw):
        return dict(replicate=g, cluster_id=f'seed{self.seed}-g{g:05d}',
                    split='dev' if g < self.groups//4 else 'test', level=level, total_objects=n,
                    objects_per_image=per, mode=mode, **kw)

    def object_tasks(self, base, study, probe, task, rng, pair_id=None, choices=None, include_recognition=True):
        per = base['objects_per_image']
        scenes = [study[i:i+per] for i in range(0, len(study), per)]
        query_mode = {'3d_cross': '3d_cross_query', '3d_occluded': '3d_occluded_query'}.get(base['mode'], base['mode'])
        study_mode = '2d' if base['mode'] == '2d' else '3d_fixed'
        query_im = self.renderer.image([probe], mode=query_mode, variant=self.seed+base['replicate']+100000)
        yes = probe in study
        private = dict(probe=probe, probe_type=lure_type(study, probe), pair_id=pair_id,
                       semantic_id=digest([base, task])[:24])
        for q in ('Q1', 'Q2', 'Q3'):
            if q not in self.questions or not include_recognition:
                continue
            content = []
            if q in ('Q1', 'Q3'):
                content += [image(query_im)]
            phrase = f'The candidate is a {describe(probe)}. ' if q in ('Q2', 'Q3') else ''
            content += [text(phrase+'Did this complete object occur in the study set? Output only YES or NO.')]
            self.add(dict(base, question=q, task=task, study_mode=study_mode), scenes, content, 'binary',
                     'YES' if yes else 'NO', **private)
        if choices is not None:
            for q in ('Q4', 'Q5', 'Q6'):
                if q not in self.questions:
                    continue
                candidates = choices[:2] if q in ('Q4', 'Q5') else choices
                # Rotate, independent of the answer. The same order is used on both counterfactual sides.
                offset = base['replicate'] % len(candidates)
                candidates = candidates[offset:]+candidates[:offset]
                hits = [i for i, obj in enumerate(candidates) if obj in study]
                if len(hits) != 1:
                    raise ValueError('Choice tasks require exactly one study object')
                content = [text('Which ONE candidate occurred as a complete study object? Output only its letter.')]
                if q == 'Q5':
                    for i, obj in enumerate(candidates):
                        content += [text('Candidate '+chr(65+i)), image(self.renderer.image([obj], mode=query_mode, variant=5))]
                else:
                    content += [image(self.renderer.image(candidates, mode=query_mode, variant=5, labels=True))]
                self.add(dict(base, question=q, task=task, study_mode=study_mode), scenes, content,
                         'choice2' if len(candidates) == 2 else 'choice4', chr(65+hits[0]),
                         **dict(private, candidates=candidates))
        if 'Q7' in self.questions and task == 'recall_anchor':
            cue = (probe[0], probe[2])
            matches = [o for o in study if (o[0], o[2]) == cue]
            if len(matches) != 1:
                raise ValueError('Recall cue must identify exactly one object')
            content = [text(f'What texture did the {cue[0]} {cue[1]} have in the study set? '
                            'Output only SOLID, STRIPES, DOTS or GRID.')]
            self.add(dict(base, question='Q7', task=task, study_mode=study_mode), scenes, content, 'recall',
                     matches[0][1].upper(), **dict(private, cue=cue))

    def calibration(self, g, rng):
        obj = UNIVERSE[(g*53 + self.seed) % len(UNIVERSE)]
        for mode in ('2d', '3d_fixed', '3d_cross', '3d_occluded'):
            rm = {'3d_cross': '3d_cross_query', '3d_occluded': '3d_occluded_query'}.get(mode, mode)
            query = [text('Identify the single study object. Output one JSON object with exactly color, texture, shape. '
                          'Allowed values: '+ '; '.join(f'{a}: {", ".join(v)}' for a, v in zip(ATTRS, VALUES)) + '.')]
            meta = self.base(g, 'L0', mode=mode, question='CAL', task='attributes', study_mode=rm)
            self.add(meta, [[obj]], query, 'attributes', dict(zip(ATTRS, obj)), probe=obj, probe_type='calibration',
                     semantic_id=digest(['cal', g, self.seed])[:24], pair_id=None)

    def basic(self, g, rng):
        banks = [rng.sample(list(v), 3) for v in VALUES]
        for n in (2, 3, 4):
            study = [tuple(b[i] for b in banks) for i in range(min(n, 3))]
            if n == 4:
                study += [(banks[0][0], banks[1][1], banks[2][2])]
            base = self.base(g, 'L1', n=n)
            negatives = {}
            for typ in ('unrelated', 'near1', 'near2', 'critical'):
                negatives[typ] = rng.choice([x for x in UNIVERSE if lure_type(study, x) == typ])
            for k, (typ, neg) in enumerate(negatives.items()):
                target = study[k % n]
                others = [x for x in negatives.values() if x != neg][:2]
                self.object_tasks(base, study, target, 'positive_'+typ, rng, choices=[target, neg]+others)
                self.object_tasks(base, study, neg, 'negative_'+typ, rng)
            self.object_tasks(base, study, study[0], 'recall_anchor', rng, include_recognition=False)

    def counterfactual(self, g, rng, level):
        banks = [rng.sample(list(v), 2) for v in VALUES]
        even, odd = parity_sets(banks)
        # Query is identical in both worlds. Permute sides and image order independently of labels.
        probe = rng.choice(even)
        foil = rng.choice(odd)
        novel = [x for x in UNIVERSE if x not in even+odd]
        extras = rng.sample(novel, 2)
        outside = list(itertools.product(*[[x for x in v if x not in b] for v, b in zip(VALUES, banks)]))
        fillers = rng.sample(outside, 4)
        conditions = [('2d', 4, 1)] if level == 'L2' else (
            [('2d', n, per) for n in (4, 8) for per in (1, 2, 4, 8) if n % per == 0] if level == 'L3' else
            [(mode, 4, 1) for mode in ('2d', '3d_fixed', '3d_cross', '3d_occluded')])
        # Do not use filler objects as alternative choices.
        extras = [o for o in novel if o not in fillers][:2]
        for mode, n, per in conditions:
            pair = digest([self.seed, g, level, mode, n, per])[:24]
            for side, core in enumerate((even, odd)):
                study = core + (fillers if n == 8 else [])
                random.Random(self.seed + g*1009 + side*31).shuffle(study)
                base = self.base(g, level, n, per, mode, world=side, banks=banks)
                self.object_tasks(base, study, probe, 'recall_anchor', rng, pair_id=pair,
                                  choices=[probe, foil]+extras)

    def relations(self, g, rng):
        # Distinct identities, same set in both worlds; only positions/co-occurrence differ.
        objs = rng.sample(UNIVERSE, 4)
        a, b, c, d = objs
        for relation in ('left', 'above', 'front', 'cooccur'):
            pair = digest(['relation', self.seed, g, relation])[:24]
            for world in (0, 1):
                scenes = ([[a, b], [c, d]] if world == 0 else [[a, c], [b, d]]) if relation == 'cooccur' else (
                    [[a, b]] if world == 0 else [[b, a]])
                question = (f'Did the {describe(a)} and the {describe(b)} occur together in the SAME study image?' if relation == 'cooccur'
                            else f'In the study image, was the {describe(a)} {dict(left="to the left of", above="above", front="in front of (closer to the camera than)")[relation]} the {describe(b)}?')
                meta = self.base(g, 'L4', sum(map(len, scenes)), 2, '3d_fixed' if relation == 'front' else '2d',
                                 question='Q8', task=relation, relation=relation, world=world,
                                 layout=relation if relation in ('above', 'front') else None,
                                 study_mode='3d_fixed' if relation == 'front' else '2d')
                self.add(meta, scenes, [text(question+' Output only YES or NO.')], 'binary',
                         'YES' if world == 0 else 'NO', pair_id=pair, probe=[a, b], probe_type='relation',
                         semantic_id=digest([self.seed, g, relation, world])[:24])

    def run(self):
        for g in range(self.groups):
            for level in self.levels:
                rng = random.Random(int(digest([self.seed, g, level])[:16], 16))
                if level == 'L0': self.calibration(g, rng)
                elif level == 'L1': self.basic(g, rng)
                elif level in ('L2', 'L3', 'L5'): self.counterfactual(g, rng, level)
                elif level == 'L4' and 'Q8' in self.questions: self.relations(g, rng)
        for split in ('dev', 'test'):
            write_jsonl(self.root/'inputs'/f'{split}.jsonl', [r for r in self.inputs if r['split'] == split])
            write_jsonl(self.root/'labels'/f'{split}.jsonl', [r for r in self.labels if r['split'] == split])
        assets = [{'path': p, 'sha256': sha_file(self.root/p), **s} for p, s in sorted(self.renderer.specs.items())]
        write_jsonl(self.root/'assets.jsonl', assets)
        manifest = dict(version=VERSION, seed=self.seed, groups=self.groups, levels=self.levels, questions=self.questions,
                        pillow=PIL.__version__, numpy=np.__version__, python=platform.python_version(),
                        generator_source={name: sha_file(Path(__file__).with_name(name)) for name in
                                          ('generate.py','render.py','common.py')},
                        trials=len(self.inputs), images=len(assets),
                        by_level=dict(Counter(r['level'] for r in self.inputs)),
                        files={str(p.relative_to(self.root)): sha_file(p) for p in
                               [self.root/'assets.jsonl'] + sorted((self.root/'inputs').glob('*.jsonl')) +
                               sorted((self.root/'labels').glob('*.jsonl'))})
        write_json(self.root/'manifest.json', manifest)
        print(manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/bindmem_v2')
    p.add_argument('--groups', type=int, default=320, help='Independent clusters; 25%% dev, 75%% test')
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--levels', nargs='+', choices=['L0','L1','L2','L3','L4','L5'], default=['L0','L1','L2','L3','L4','L5'])
    p.add_argument('--questions', nargs='+', choices=[f'Q{i}' for i in range(1,9)], default=[f'Q{i}' for i in range(1,9)])
    args = p.parse_args()
    if args.groups < 4 or args.groups % 4:
        p.error('--groups must be a positive multiple of 4 (at least 4)')
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        p.error('Output must be empty; use a new directory to protect existing experiments')
    Builder(root, args.groups, args.seed, args.levels, args.questions).run()


if __name__ == '__main__': main()
