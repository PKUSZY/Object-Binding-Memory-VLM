"""Cluster-aware summaries, paired comparisons and exploratory logistic regression."""
from collections import defaultdict
import math

import numpy as np


def cluster_ci(values, clusters, repetitions=1000, seed=0):
    if not values: return None
    grouped = defaultdict(list)
    for y, g in zip(values, clusters): grouped[g].append(float(y))
    groups = list(grouped.values())
    if len(groups) < 2: return None
    sums = np.array([sum(g) for g in groups]); counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(groups), (repetitions, len(groups)))
    stats = sums[samples].sum(1)/counts[samples].sum(1)
    return np.quantile(stats, [.025, .975]).tolist()


def rate(rows, field, repetitions=1000):
    return dict(n=len(rows), value=sum(r[field] for r in rows)/len(rows) if rows else None,
                ci95=cluster_ci([r[field] for r in rows], [r['cluster_id'] for r in rows], repetitions))


def summarize(rows, repetitions=1000):
    observed = [r for r in rows if r['observed']]
    result = dict(expected=len(rows), observed=len(observed), coverage=len(observed)/len(rows) if rows else None,
                  accuracy=rate(observed, 'correct', repetitions),
                  invalid=sum(not r['valid'] for r in observed))
    binary = [r for r in observed if r['kind'] == 'binary' and r['level'] != 'L4']
    targets = [r for r in binary if r['answer'] == 'YES']
    result['target_hit_rate'] = rate(targets, 'correct', repetitions)
    result['lure_false_acceptance'] = {typ: rate([r for r in binary if r['probe_type'] == typ], 'accepted', repetitions)
                                      for typ in sorted({r['probe_type'] for r in binary if r['answer'] == 'NO'})}
    critical = [r for r in binary if r['probe_type'] == 'critical']
    unrelated = [r for r in binary if r['probe_type'] == 'unrelated']
    if critical and unrelated:
        # Match seed/list and format before computing the false-acceptance difference.
        key = lambda r: (r['cluster_id'], r['total_objects'], r['objects_per_image'], r['mode'], r['question'])
        left = {key(r): r for r in critical}; right = {key(r): r for r in unrelated}
        matched = sorted(left.keys() & right.keys())
        diffs = [int(left[k]['accepted'])-int(right[k]['accepted']) for k in matched]
        result['critical_lure_gap'] = dict(n_pairs=len(matched), value=float(np.mean(diffs)) if diffs else None,
                                          ci95=cluster_ci(diffs, [k[0] for k in matched], repetitions))
    cf = [r for r in binary if r['level'] in ('L2','L3','L5')]
    yes, no = ([r for r in cf if r['answer'] == a] for a in ('YES','NO'))
    result['triple_binding_balanced_accuracy'] = (
        (np.mean([r['correct'] for r in yes])+np.mean([r['correct'] for r in no]))/2 if yes and no else None)
    paired = defaultdict(list)
    for r in observed:
        if r.get('pair_id'): paired[(r['pair_id'], r['question'])].append(r)
    pairs = [dict(correct=all(x['correct'] for x in rr), cluster_id=rr[0]['cluster_id'])
             for rr in paired.values() if len(rr) == 2]
    result['pair_accuracy'] = rate(pairs, 'correct', repetitions)
    result['relation_accuracy'] = rate([r for r in observed if r['level'] == 'L4'], 'correct', repetitions)
    cal = [r for r in observed if r['kind'] == 'attributes']
    result['feature_accuracy'] = {a: rate([dict(r, feature_correct=isinstance(r['prediction'], dict) and
                                                        r['prediction'].get(a) == r['answer'][a]) for r in cal],
                                        'feature_correct', repetitions) for a in ('color','texture','shape')}
    return result


def paired_delta(rows_a, rows_b, repetitions=1000, key=lambda r: r['trial_id']):
    a = {key(r): r for r in rows_a if r['observed']}; b = {key(r): r for r in rows_b if r['observed']}
    keys = sorted(a.keys() & b.keys())
    delta = [int(b[k]['correct'])-int(a[k]['correct']) for k in keys]
    return dict(n_pairs=len(keys), delta_b_minus_a=float(np.mean(delta)) if delta else None,
                ci95=cluster_ci(delta, [a[k]['cluster_id'] for k in keys], repetitions),
                only_a_correct=sum(a[k]['correct'] and not b[k]['correct'] for k in keys),
                only_b_correct=sum(b[k]['correct'] and not a[k]['correct'] for k in keys))


def logistic_load(rows):
    """Ridge logistic fit; cluster sandwich SE. Exploratory, not a causal test."""
    rows = [r for r in rows if r['observed'] and r['level'] == 'L3' and r['question'] in ('Q1','Q2','Q3')]
    if len({r['cluster_id'] for r in rows}) < 10:
        return {'status': 'insufficient_clusters', 'minimum': 10}
    names = ['intercept','n8','per2','per4','per8','Q2','Q3','n8:Q2','n8:Q3']
    def design(r):
        n = int(r['total_objects'] == 8); q2 = int(r['question'] == 'Q2'); q3 = int(r['question'] == 'Q3')
        return [1,n,*[int(r['objects_per_image'] == k) for k in (2,4,8)],q2,q3,n*q2,n*q3]
    X = np.array([design(r) for r in rows], float)
    y = np.array([r['correct'] for r in rows], float)
    if len(np.unique(y)) < 2: return {'status': 'constant_outcome'}
    if np.linalg.matrix_rank(X) < X.shape[1]: return {'status': 'rank_deficient_selection'}
    penalty = np.eye(X.shape[1]); penalty[0,0] = 0
    beta = np.zeros(X.shape[1])
    converged = False
    for _ in range(100):
        pr = 1/(1+np.exp(-np.clip(X@beta, -30, 30)))
        hess = (X.T*(pr*(1-pr)))@X + penalty
        step = np.linalg.solve(hess, X.T@(y-pr)-penalty@beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True; break
    pr = 1/(1+np.exp(-np.clip(X@beta, -30, 30)))
    hess = (X.T*(pr*(1-pr)))@X + penalty
    scores = defaultdict(lambda: np.zeros(X.shape[1]))
    for r, row, resid in zip(rows, X, y-pr): scores[r['cluster_id']] += row*resid
    meat = sum(np.outer(v,v) for v in scores.values())
    inv = np.linalg.inv(hess)
    se = np.sqrt(np.maximum(np.diag(inv@meat@inv)*len(scores)/(len(scores)-1), 0))
    return dict(status='ok' if converged else 'not_converged', n=len(rows), clusters=len(scores), ridge=1.0,
                reference='L3, total=4, per_image=1, Q1; outcome=correct',
                caveat='Exploratory penalized estimates; cluster sandwich intervals are approximate. No causal interpretation.',
                coefficients={name: dict(log_odds=float(b), odds_ratio=float(math.exp(b)),
                    cluster_se=float(s), log_odds_ci95=[float(b-1.96*s),float(b+1.96*s)])
                    for name,b,s in zip(names,beta,se)})
