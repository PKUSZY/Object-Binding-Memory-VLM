"""Score extended runs and compare models on explicit identical trial intersections."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from bindmem.common import digest, read_jsonl, write_json, write_jsonl
from bindmem.score import collect
from bindmem.stats import paired_delta, rate, summarize


def load_rows(data, run, split='test', allow_partial=False):
    run=Path(run); rows=collect(data,run,split)
    manifest=json.loads((run/'manifest.json').read_text())
    exclusions=[]
    if manifest.get('selection_sha256'):
        plan=json.loads((run/'selection.json').read_text())
        if digest(plan)!=manifest['selection_sha256']: raise ValueError('Selection fingerprint mismatch')
        selected={r['trial_id'] for r in plan['selected']}
        if len(selected)!=len(plan['selected']):raise ValueError('Duplicate selected trial')
        rows=[r for r in rows if r['trial_id'] in selected]
        exclusions=[r for r in plan['excluded'] if split=='both' or r['split']==split]
    if not rows:raise ValueError('No selected rows in this split')
    if not allow_partial and any(not r['observed'] for r in rows):
        raise ValueError(f'{sum(not r["observed"] for r in rows)}/{len(rows)} selected trials unfinished')
    return rows,exclusions


def condition(r):
    if r['level']=='G':return f'G/{r["question"]}/n{r["total_objects"]}/k{r["gradient_k"]}'
    key=f'{r["level"]}/{r["question"]}/{r["mode"]}/n{r["total_objects"]}/per{r["objects_per_image"]}'
    if r['level']=='L4':key+='/'+r['relation']
    return key


def metrics(rr,bootstrap):
    observed=[r for r in rr if r['observed']]
    out=summarize(rr,bootstrap)
    negatives=[r for r in observed if r['kind']=='binary' and r['answer']=='NO']
    out['negative_correct_rejection']=rate(negatives,'correct',bootstrap)
    out['negative_false_acceptance']=rate(negatives,'accepted',bootstrap)
    out['correct']=sum(r['correct'] for r in observed)
    return out


def write_report(rows,exclusions,out,bootstrap=1000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for r in rows:groups[condition(r)].append(r)
    cells={k:metrics(v,bootstrap) for k,v in sorted(groups.items())}
    contrasts={}
    for q in sorted({r['question'] for r in rows if r['level']=='G'}):
        neg=[r for r in rows if r['level']=='G' and r['question']==q and r['answer']=='NO']
        ks=sorted({r['gradient_k'] for r in neg})
        for a,b in zip(ks,ks[1:]):
            contrasts[f'{q}/k{a}_vs_k{b}/correct_rejection']=paired_delta(
                [r for r in neg if r['gradient_k']==a],[r for r in neg if r['gradient_k']==b],bootstrap,key=lambda r:r['cluster_id'])
    summary=dict(expected=len(rows),observed=sum(r['observed'] for r in rows),excluded=len(exclusions),
        complete=all(r['observed'] for r in rows),bootstrap_unit='cluster_id',bootstrap_repetitions=bootstrap,
        interpretation='Invalid responses count as wrong. Rates use observed selected rows. Excluded context-overflow trials are not model errors and are reported separately.',
        by_condition=cells,paired_gradient=contrasts)
    write_json(out/'summary.json',summary);write_jsonl(out/'scored.jsonl',rows);write_jsonl(out/'excluded.jsonl',exclusions)
    table=[]
    for key,m in cells.items():
        for metric in ['accuracy','target_hit_rate','negative_correct_rejection','negative_false_acceptance','pair_accuracy']:
            v=m[metric]
            if not v['n']:continue
            table.append(dict(condition=key,metric=metric,n=v['n'],value=v['value'],
                ci_low=v['ci95'][0] if v['ci95'] else None,ci_high=v['ci95'][1] if v['ci95'] else None,
                invalid_in_condition=m['invalid']))
    with (out/'rates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','metric','n','value','ci_low','ci_high','invalid_in_condition']);w.writeheader();w.writerows(table)
    def show(v):
        if not v['n']:return '—'
        ci='' if v['ci95'] is None else f' [{v["ci95"][0]:.2%}, {v["ci95"][1]:.2%}]'
        return f'{v["value"]:.2%}'+ci
    md=['# 分条件行为评分',f'完成 {summary["observed"]}/{len(rows)} 道选中题；另有 {len(exclusions)} 道因上下文长度不支持而排除。',
        '方括号为按随机组重采样的 95% 区间。无效输出按错计分；没有把排除题记作正确或模型错误。',
        '| 条件 | 完成题数 | 准确率 [95% CI] | 目标命中 | 负例正确拒绝 | 负例误认 | 无效回答 |',
        '|---|---:|---|---|---|---|---:|']
    for key,m in cells.items():
        md.append(f'| {key} | {m["observed"]}/{m["expected"]} | '+ ' | '.join(show(m[k]) for k in ['accuracy','target_hit_rate','negative_correct_rejection','negative_false_acceptance'])+f' | {m["invalid"]} |')
    md+=['','梯度实验的正例在同组各 k 中重复，不能视为独立样本。相邻 k 的负例正确拒绝率配对差及区间见 summary.json。',
          '全对或全错时经验 bootstrap 区间可能退化，不表示总体没有不确定性；条件较多，区间未作多重比较校正。']
    (out/'report.md').write_text('\n\n'.join(md[:3])+'\n\n'+'\n'.join(md[3:])+'\n')
    print(f'Scored {summary["observed"]}/{len(rows)}; excluded {len(exclusions)} -> {out}')
    return summary


def compare(data,runs_a,runs_b,out,split='test',bootstrap=1000,allow_subset=False):
    def merge(paths):
        rows=[]
        for path in paths:
            r,_=load_rows(data,path,split);rows+=r
        if len(rows)!=len({r['trial_id'] for r in rows}):raise ValueError('Overlapping runs within one model')
        return {r['trial_id']:r for r in rows}
    a,b=merge(runs_a),merge(runs_b)
    if not allow_subset and set(a)!=set(b):raise ValueError('Selections differ; inspect exclusions and use --allow-subset explicitly for shared trials')
    shared=sorted(set(a)&set(b))
    if not shared:raise ValueError('No shared trials')
    groups=defaultdict(list)
    for tid in shared:
        if a[tid]['answer']!=b[tid]['answer']:raise ValueError('Inconsistent labels')
        groups[condition(a[tid])].append(tid)
    cells={}
    for k,ids in sorted(groups.items()):
        ra,rb=[a[i] for i in ids],[b[i] for i in ids]
        cells[k]=dict(a=metrics(ra,bootstrap),b=metrics(rb,bootstrap),delta_b_minus_a=paired_delta(ra,rb,bootstrap))
    manifests={str(path):json.loads((Path(path)/'manifest.json').read_text()) for path in [*runs_a,*runs_b]}
    for key in ('seed','do_sample','max_new_tokens'):
        if len({json.dumps(m.get(key),sort_keys=True) for m in manifests.values()})>1:raise ValueError(f'Incompatible inference setting: {key}')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'comparison.json',dict(run_a=runs_a,run_b=runs_b,shared=len(shared),
        only_a=len(set(a)-set(b)),only_b=len(set(b)-set(a)),split=split,
        caveat='Same source trials and strict parser; model-native chat wrappers, vision resolution, image padding and training differ. This compares complete model pipelines.',
        manifests=manifests,by_condition=cells))
    write_jsonl(out/'shared_trials.jsonl',[{'trial_id':i} for i in shared])
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','n','accuracy_a','accuracy_b','delta_b_minus_a','ci_low','ci_high'])
        for k,c in cells.items():
            d=c['delta_b_minus_a'];w.writerow([k,d['n_pairs'],c['a']['accuracy']['value'],c['b']['accuracy']['value'],d['delta_b_minus_a'],*(d['ci95'] or [None,None])])
    print(f'Compared {len(shared)} identical trials -> {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/bindmem_v2');p.add_argument('--run');p.add_argument('--output')
    p.add_argument('--split',choices=['dev','test','both'],default='test');p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--run-a',nargs='+');p.add_argument('--run-b',nargs='+')
    p.add_argument('--allow-subset',action='store_true');args=p.parse_args()
    if args.bootstrap<1:p.error('bootstrap must be positive')
    if args.run_a or args.run_b:
        if not(args.run_a and args.run_b and args.output):p.error('Comparison requires --run-a, --run-b and --output')
        compare(args.data,args.run_a,args.run_b,args.output,args.split,args.bootstrap,args.allow_subset)
    else:
        if not args.run:p.error('--run required')
        rows,excluded=load_rows(args.data,args.run,args.split,args.allow_partial)
        write_report(rows,excluded,args.output or str(Path(args.run)/'scored'/args.split),args.bootstrap)


if __name__=='__main__':main()
