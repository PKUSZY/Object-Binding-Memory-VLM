"""Frozen-representation linear/MLP probes with group-disjoint validation/test.

Features: L0 labels only (not attributes explicitly named by a text query).
Binding: Q2 strict counterfactual membership only, balanced within every pair.
All normalization is fitted on training groups. Test labels never select models.
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ATTRS, VALUES, digest, read_jsonl, sha_file, write_json, write_jsonl
from .score import collect
from .stats import cluster_ci


def fit_probe(X, y, train, valid, classes, architecture, penalty, epochs, seed):
    import torch
    torch.manual_seed(seed)
    mean = X[train].mean(0); std = np.maximum(X[train].std(0), 1e-5)
    Z = torch.from_numpy(((X-mean)/std).astype('float32'))
    target = torch.tensor(y, dtype=torch.long)
    if architecture == 'linear': model = torch.nn.Linear(X.shape[1], classes)
    else: model = torch.nn.Sequential(torch.nn.Linear(X.shape[1],64), torch.nn.ReLU(), torch.nn.Linear(64,classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=penalty)
    loss_fn = torch.nn.CrossEntropyLoss()
    best, best_loss, best_epoch, stale = None, float('inf'), 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(np.flatnonzero(train))
        for begin in range(0, len(order), 128):
            ix = order[begin:begin+128]
            optimizer.zero_grad(); loss = loss_fn(model(Z[ix]), target[ix]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_loss = float(loss_fn(model(Z[valid]), target[valid]))
        if val_loss < best_loss-1e-6:
            best_loss, best, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch+1, 0
        else: stale += 1
        if stale >= 20: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): probabilities = model(Z).softmax(-1).numpy()
    return probabilities, best_loss, dict(state_dict=best, mean=mean, std=std, epochs=best_epoch,
                                        input_dim=X.shape[1], classes=classes, architecture=architecture, penalty=penalty)


def evaluate(indices, pred, y, rows, repetitions):
    ix = np.flatnonzero(indices)
    vals = (pred[ix] == y[ix]).astype(int)
    return dict(n=len(ix), accuracy=float(vals.mean()) if len(ix) else None,
                ci95=cluster_ci(vals.tolist(), [rows[i]['cluster_id'] for i in ix], repetitions))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2'); p.add_argument('--run', required=True)
    p.add_argument('--output', default=None); p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--penalties', type=float, nargs='+', default=[.01,.1])
    p.add_argument('--architectures', nargs='+', choices=['linear','mlp'], default=['linear','mlp'])
    p.add_argument('--keys', nargs='+', default=None, help='Optional activation keys, e.g. decoder.0.answer')
    p.add_argument('--bootstrap', type=int, default=1000); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.epochs < 1 or args.bootstrap < 1 or any(x < 0 for x in args.penalties): p.error('Invalid training parameters')
    import torch
    torch.set_num_threads(args.threads)
    run = Path(args.run); out = Path(args.output) if args.output else run/'probes'
    out.mkdir(parents=True, exist_ok=True)
    probe_manifest = dict(settings=vars(args), source_sha256=sha_file(Path(__file__)),
                          run_manifest_sha256=sha_file(run/'manifest.json'), complete=False)
    write_json(out/'manifest.json', probe_manifest)
    raw = {r['trial_id']:r for r in read_jsonl(run/'predictions.jsonl')}
    scored = collect(args.data, run, 'both')
    rows = [r for r in scored if (r['level'] == 'L0' or (r['question'] == 'Q2' and r['level'] in ('L2','L3','L5')))]
    if not rows or any(not raw.get(r['trial_id'], {}).get('activation_file') or not r['observed'] for r in rows):
        raise SystemExit('All selected probe trials need successful --capture inference before fitting')
    groups = sorted({r['cluster_id'] for r in rows if r['split'] == 'dev'}, key=lambda g: digest([args.seed,g]))
    test_groups = {r['cluster_id'] for r in rows if r['split'] == 'test'}
    if len(groups) < 8 or len(test_groups) < 4:
        raise SystemExit('Probes need >=8 dev clusters and >=4 test clusters; generate at least --groups 32')
    cut = max(1, int(.75*len(groups)))
    train_groups, valid_groups = set(groups[:cut]), set(groups[cut:])
    assert not (train_groups & valid_groups or train_groups & test_groups or valid_groups & test_groups)
    write_json(out/'split.json', dict(train=sorted(train_groups),validation=sorted(valid_groups),test=sorted(test_groups),
                                    source_run_sha256=sha_file(run/'manifest.json'), seed=args.seed,
                                    note='Train/validation partition uses dev clusters only. Test clusters are untouched.'))
    with np.load(run/raw[rows[0]['trial_id']]['activation_file']) as f: keys = sorted(f.files)
    keys = args.keys or keys
    tasks = [('feature_'+a, a) for a in ATTRS]+[('binding', None)]
    summaries, prediction_rows = [], []
    for key in keys:
        for task, attr in tasks:
            selected = [r for r in rows if (r['level'] == 'L0') == (attr is not None)]
            if not selected: continue
            vectors = []
            for r in selected:
                with np.load(run/raw[r['trial_id']]['activation_file']) as f:
                    if key not in f: raise ValueError(f'Missing activation key: {key}')
                    vectors.append(f[key])
            X = np.stack(vectors).astype('float32')
            classes = list(VALUES[ATTRS.index(attr)]) if attr else ['NO','YES']
            y = np.array([classes.index(r['answer'][attr] if attr else r['answer']) for r in selected])
            train = np.array([r['cluster_id'] in train_groups for r in selected])
            valid = np.array([r['cluster_id'] in valid_groups for r in selected])
            test = np.array([r['cluster_id'] in test_groups for r in selected])
            if not train.any() or not valid.any() or not test.any():
                raise ValueError(f'{task} lacks train/validation/test rows')
            if len(np.unique(y[train])) < len(classes):
                raise ValueError(f'{task} training split lacks classes; use more groups')
            for architecture in args.architectures:
                fits = [fit_probe(X,y,train,valid,len(classes),architecture,penalty,args.epochs,args.seed)
                        for penalty in args.penalties]
                probability, val_loss, state = min(fits, key=lambda result: result[1])
                prediction = probability.argmax(1)
                # Group-preserving label permutation control: feature labels are reassigned
                # between clusters; binding labels flip together within a whole cluster.
                shuffled = y.copy()
                for group_set in (train_groups, valid_groups):
                    ordered = sorted(group_set); rng = np.random.default_rng(args.seed)
                    if attr:
                        mapping = dict(zip(ordered, rng.permutation(ordered)))
                        targets = {r['cluster_id']: y[i] for i,r in enumerate(selected) if r['cluster_id'] in group_set}
                        for i,r in enumerate(selected):
                            if r['cluster_id'] in targets: shuffled[i] = targets[mapping[r['cluster_id']]]
                    else:
                        flips = dict(zip(ordered, rng.integers(0,2,len(ordered))))
                        for i,r in enumerate(selected):
                            if r['cluster_id'] in flips: shuffled[i] = y[i] ^ int(flips[r['cluster_id']])
                baseline, _, _ = fit_probe(X,shuffled,train,valid,len(classes),architecture,state['penalty'],state['epochs'],args.seed)
                result = dict(task=task, key=key, architecture=architecture, chosen_penalty=state['penalty'],
                              chosen_epochs=state['epochs'], validation_loss=val_loss,
                              train=evaluate(train,prediction,y,selected,args.bootstrap),
                              validation=evaluate(valid,prediction,y,selected,args.bootstrap),
                              test=evaluate(test,prediction,y,selected,args.bootstrap), chance=1/len(classes),
                              permuted_label_test=evaluate(test,baseline.argmax(1),y,selected,args.bootstrap),
                              majority_test_accuracy=float(np.mean(y[test] == np.bincount(y[train],minlength=len(classes)).argmax())),
                              by_condition={}, correct_vs_error={})
                conds = {(r['level'],r['mode'],r['total_objects'],r['objects_per_image']) for r in selected}
                for condition in sorted(conds):
                    mask = test & np.array([(r['level'],r['mode'],r['total_objects'],r['objects_per_image']) == condition for r in selected])
                    result['by_condition']['/'.join(map(str,condition))] = evaluate(mask,prediction,y,selected,args.bootstrap)
                for success in (True,False):
                    mask = test & np.array([r['correct'] == success for r in selected])
                    metric = evaluate(mask,prediction,y,selected,args.bootstrap)
                    ix = np.flatnonzero(mask)
                    metric['mean_probability_of_truth'] = float(probability[ix,y[ix]].mean()) if len(ix) else None
                    result['correct_vs_error']['model_correct' if success else 'model_error'] = metric
                summaries.append(result)
                for i in np.flatnonzero(test):
                    prediction_rows.append(dict(trial_id=selected[i]['trial_id'],task=task,key=key,architecture=architecture,
                        truth=classes[y[i]],prediction=classes[prediction[i]],probability_of_truth=float(probability[i,y[i]]),
                        model_correct=selected[i]['correct'],cluster_id=selected[i]['cluster_id']))
                model_file = out/'weights'/f'{task}-{key}-{architecture}.pt'
                model_file.parent.mkdir(exist_ok=True)
                torch.save(dict(state, classes=classes, mean=torch.from_numpy(state['mean']),
                                std=torch.from_numpy(state['std'])),model_file)
                write_json(out/'summary.json', summaries)
                print(f'{task} {key} {architecture}: held-out accuracy {result["test"]["accuracy"]:.3f}', flush=True)
    write_jsonl(out/'predictions.jsonl',prediction_rows)
    probe_manifest['complete'] = True
    write_json(out/'manifest.json', probe_manifest)
    # Standalone layer-wise plot; no matplotlib installation needed.
    from .score import svg_chart
    svg_chart(out/'layers.svg', {f'{r["task"]}/{r["key"]}/{r["architecture"]}':
              {'accuracy':dict(value=r['test']['accuracy'],ci95=r['test']['ci95'])} for r in summaries})


if __name__ == '__main__': main()
