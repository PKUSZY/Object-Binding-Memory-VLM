#!/usr/bin/env python3
"""Audit disjoint runs against all dataset IDs and summarize behavior only."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bindmem.common import read_jsonl, sha_file, write_json
from bindmem.score import collect, svg_chart
from bindmem.stats import paired_delta, summarize, logistic_load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2')
    p.add_argument('--runs', nargs='+', default=[f'runs/bindmem_v2/qwen8b_{s}' for s in
                   ('core', 'image', 'choices', 'formats', 'basic', 'extensions')])
    p.add_argument('--output', default='runs/bindmem_v2/qwen8b_report')
    args = p.parse_args()
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifests = [json.loads((Path(r)/'manifest.json').read_text()) for r in args.runs]
    # Capture differs in the historical core run; retain it in provenance.
    ignore = {'levels', 'questions', 'capture', 'layers'}
    base = {k:v for k,v in manifests[0].items() if k not in ignore}
    for r,m in zip(args.runs, manifests):
        if {k:v for k,v in m.items() if k not in ignore} != base:
            raise ValueError(f'Incompatible manifest: {r}')
    report = {'data': args.data, 'runs': [dict(path=r, manifest=m,
        predictions_sha256=sha_file(Path(r)/'predictions.jsonl')) for r,m in zip(args.runs,manifests)],
        'bootstrap_unit': 'cluster_id', 'bootstrap_repetitions': 1000,
        'analysis_script_sha256': sha_file(Path(__file__)), 'splits': {}}
    invalid_rows = []
    for split in ('dev','test'):
        rows = [row for r in args.runs for row in collect(args.data,r,split)]
        ids = [r['trial_id'] for r in rows]
        expected = {r['trial_id'] for r in read_jsonl(Path(args.data)/'labels'/f'{split}.jsonl')}
        if len(set(ids)) != len(ids) or set(ids) != expected or not all(r['observed'] for r in rows):
            raise ValueError(f'{split}: duplicate, missing, unexpected or failed trials')
        cells = {f'{l}/{q}': summarize([r for r in rows if (r['level'],r['question'])==(l,q)])
                 for l,q in sorted({(r['level'],r['question']) for r in rows})}
        grouped = defaultdict(list)
        for r in rows:
            key = f'{r["level"]}/{r["question"]}/{r["mode"]}/n{r["total_objects"]}/per{r["objects_per_image"]}'
            if r['level'] == 'L4': key += '/' + r['relation']
            grouped[key].append(r)
        conditions = {k:summarize(v) for k,v in sorted(grouped.items())}
        paired = {}
        for level in sorted({r['level'] for r in rows}):
            for a,b in [('Q1','Q2'),('Q1','Q3'),('Q4','Q5')]:
                aa = [r for r in rows if r['level']==level and r['question']==a]
                bb = [r for r in rows if r['level']==level and r['question']==b]
                if not aa or not bb: continue
                for rr in (aa,bb):
                    if len({r['semantic_id'] for r in rr}) != len(rr):
                        raise ValueError('Duplicate semantic pairing keys')
                if {r['semantic_id'] for r in aa} != {r['semantic_id'] for r in bb}:
                    raise ValueError('Incomplete semantic pairing')
                paired[f'{level}/{a}_vs_{b}'] = paired_delta(aa,bb,key=lambda r:r['semantic_id'])
        design_pairs = {}
        for q in ('Q1','Q2','Q3','Q4','Q5','Q6','Q7'):
            def compare_conditions(name, selected, field, a, b):
                left = [r for r in selected if r[field] == a]
                right = [r for r in selected if r[field] == b]
                key = lambda r: (r['cluster_id'],r['world'])
                for rr in (left,right):
                    if len({key(r) for r in rr}) != len(rr): raise ValueError(name)
                if not left or {key(r) for r in left} != {key(r) for r in right}: raise ValueError(name)
                design_pairs[name] = paired_delta(left,right,key=key)
            for n in (4,8):
                selected = [r for r in rows if r['level']=='L3' and r['question']==q and r['total_objects']==n]
                for per in (2,4,8):
                    if per <= n:
                        compare_conditions(f'L3/{q}/n{n}/per1_vs_per{per}',selected,'objects_per_image',1,per)
            for per in (1,2,4):
                selected = [r for r in rows if r['level']=='L3' and r['question']==q and r['objects_per_image']==per]
                compare_conditions(f'L3/{q}/per{per}/n4_vs_n8',selected,'total_objects',4,8)
            selected = [r for r in rows if r['level']=='L5' and r['question']==q]
            for mode in ('3d_fixed','3d_cross','3d_occluded'):
                compare_conditions(f'L5/{q}/2d_vs_{mode}',selected,'mode','2d',mode)
            for mode in ('3d_cross','3d_occluded'):
                compare_conditions(f'L5/{q}/3d_fixed_vs_{mode}',selected,'mode','3d_fixed',mode)
        report['splits'][split] = dict(complete=True, expected=len(expected), observed=len(rows),
            clusters=len({r['cluster_id'] for r in rows}),
            invalid=sum(not r['valid'] for r in rows), by_level_question=cells,
            by_condition=conditions, paired_formats=paired, paired_design=design_pairs,
            exploratory_load_regression=logistic_load(rows))
        invalid_ids = {r['trial_id']:r for r in rows if not r['valid']}
        for run in args.runs:
            for raw in read_jsonl(Path(run)/'predictions.jsonl'):
                tid = raw['trial_id']
                if tid in invalid_ids and not raw.get('error'):
                    r = invalid_ids[tid]
                    invalid_rows.append([split,run,tid,r['level'],r['question'],r['mode'],
                                         r['answer'],raw.get('raw_response','')])
    write_json(out/'summary.json',report)
    with (out/'accuracy.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','task','n','accuracy','ci_low','ci_high','invalid'])
        for split,s in report['splits'].items():
            for k,v in s['by_level_question'].items():
                w.writerow([split,k,v['observed'],v['accuracy']['value'],*v['accuracy']['ci95'],v['invalid']])
    with (out/'conditions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','condition','n','accuracy','ci_low','ci_high','invalid'])
        for split,s in report['splits'].items():
            for k,v in s['by_condition'].items():
                w.writerow([split,k,v['observed'],v['accuracy']['value'],*v['accuracy']['ci95'],v['invalid']])
    with (out/'paired_comparisons.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','comparison','n_pairs','delta_b_minus_a','ci_low','ci_high'])
        for split,s in report['splits'].items():
            for family in ('paired_formats','paired_design'):
                for k,v in s[family].items():
                    w.writerow([split,k,v['n_pairs'],v['delta_b_minus_a'],*v['ci95']])
    with (out/'invalid_responses.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['split','run','trial_id','level','question','mode','answer','raw_response'])
        w.writerows(invalid_rows)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    test=report['splits']['test']
    svg_chart(figures/'task_accuracy.svg',test['by_level_question'])
    for name,prefix in [('load','L3/'),('spatial','L4/'),('3d','L5/')]:
        svg_chart(figures/f'{name}_accuracy.svg',{k:v for k,v in test['by_condition'].items() if k.startswith(prefix)})
    lines=['# Qwen3-VL-8B 行为结果汇总','',
        '完整性已按原始数据集 trial_id 核对：各运行无重叠，所有题均已获得回答。',
        'dev 与 test 分开报告；无效回答按错误计分，不同任务不计算混合总分。',
        '核心历史运行启用了表示采集，其余运行未启用；本报告仅分析行为回答。','',
        '| 测试任务 | 题数 | 准确率 | 95% 置信区间 | 无效回答 |',
        '|---|---:|---:|---|---:|']
    for k,v in report['splits']['test']['by_level_question'].items():
        lo,hi=v['accuracy']['ci95']
        lines.append(f'| {k} | {v["observed"]} | {v["accuracy"]["value"]:.2%} | {lo:.2%}–{hi:.2%} | {v["invalid"]} |')
    lines += ['', '配对结果见 summary.json，差值方向为后一个问法减前一个问法。',
        '置信区间使用 1,000 次按独立 cluster_id 重采样；多条件结果应结合多重比较谨慎解释。',
        'L5 的 Q2/Q7 没有查询图片，跨视角与遮挡的重复文本控制不构成独立视觉效应证据。',
        '各模式、数量与关系的细分结果保留在原运行目录 scored/test/summary.json。']
    def percent(v):
        return '—' if v is None else f'{v:.2%}'
    def metric(v):
        if v['value'] is None: return '—'
        lo,hi=v['ci95']
        return f'{v["value"]:.2%} [{lo:.2%}, {hi:.2%}]'
    lines += ['', '## 数据与分析范围', '',
        f'正式 test 共 {test["observed"]:,} 题、{test["clusters"]} 个独立组；dev 共 '
        f'{report["splits"]["dev"]["observed"]:,} 题、{report["splits"]["dev"]["clusters"]} 个组。',
        f'test 无效回答 {test["invalid"]} 条（{test["invalid"]/test["observed"]:.2%}），'
        f'dev 无效回答 {report["splits"]["dev"]["invalid"]} 条；均按错误计分。',
        '六批结果使用同一数据集、模型权重指纹、代码版本、分辨率、解码设置和随机种子。'
        '失败尝试保留在原始日志中，成功重试覆盖失败状态；当前无缺失或最终运行失败题。',
        'Q1 图片判断；Q2 文字判断；Q3 图片加文字判断；Q4 同图二选一；'
        'Q5 分图二选一；Q6 四选一；Q7 属性回忆；Q8 关系判断。',
        '下文均为 test 结果，区间按组 bootstrap，默认未做多重比较校正；属于单模型行为分析。',
        '', '## 主要发现', '',
        '1. **基础属性识别接近满分，严格绑定判断仍有明显错误。** '
        'L0 联合属性识别为 99.90%；L1/Q1 为 99.25%，L2/Q1 为 71.46%。'
        '这支持在该受控基准上区分属性识别与严格绑定表现；跨层级任务不同，不能把分数差视为单一因素的因果效应。',
        '2. **文字 Yes/No 查询表现出强烈拒绝倾向。** '
        'L1/Q2 target 命中率仅 6.63%，L2/Q2 为 0.83%，L3/Q2 为 14.40%。'
        '因此其低诱饵错误接受率不能直接解释为绑定能力强，需同时报告命中率。',
        '3. **加入文字说明的收益依任务而变。** '
        'Q3 相对 Q1 在 L1 下降，在 L2 提升；L3 的差值区间包含 0，未显示明确改善。'
        '不能声称图文查询普遍优于纯图片查询。',
        '4. **分图二选一在 L1–L3 优于同图二选一。** '
        'L5 汇总差值区间包含 0。布局同时改变图片数量与视觉 token，不能把差异完全归因为记忆负载。',
        '5. **数量效应不呈简单单调下降。** '
        '例如 L3/Q1 在总物体数为 8 时，每图 1/2/4/8 个物体的准确率分别为 '
        '76.25%/81.88%/79.17%/73.12%。应固定总数量或每图数量分别做配对比较。',
        '6. **关系类型与查询图遮挡带来明显差异。** '
        '左右、上下判断均为 100%；深度前后和共现约为 79%。'
        'L5 遮挡条件下 Q1/Q3 均为 50%，但这并不单独证明绑定记忆丢失，也可能涉及查询图识别或回答倾向。',
        '', '## 命中率、诱饵错误接受率与严格配对正确率', '',
        '| 任务 | Target 命中率 | 关键诱饵错误接受率 | 正负两题均正确 |',
        '|---|---|---|---|']
    for level in ('L1','L2','L3','L5'):
        for q in ('Q1','Q2','Q3'):
            k=f'{level}/{q}';v=test['by_level_question'][k]
            lure='critical' if level=='L1' else 'pairwise_complete'
            lines.append(f'| {k} | {metric(v["target_hit_rate"])} | '
                         f'{metric(v["lure_false_acceptance"][lure])} | {metric(v["pair_accuracy"])} |')
    lines += ['', 'L1 的关键诱饵是属性重组；L2/L3/L5 是一阶与二阶统计量匹配的严格反事实诱饵。'
        '表中两题均正确指标衡量同一反事实对的正负两侧，不能与单题准确率混用。',
        '', '### L1 各类诱饵', '',
        '| 问法 | Unrelated | Near1 | Near2 | Critical | Critical − Unrelated（百分点，95% CI） |',
        '|---|---:|---:|---:|---:|---|']
    for q in ('Q1','Q2','Q3'):
        v=test['by_level_question']['L1/'+q];g=v['critical_lure_gap'];lo,hi=g['ci95']
        vals=[percent(v['lure_false_acceptance'][t]['value']) for t in ('unrelated','near1','near2','critical')]
        lines.append(f'| {q} | '+ ' | '.join(vals)+f' | {g["value"]*100:.2f} [{lo*100:.2f}, {hi*100:.2f}] |')
    lines += ['', '## 查询方式的配对比较', '',
        '| 比较（后者 − 前者） | 配对题数 | 准确率差（百分点） | 95% CI（百分点） |',
        '|---|---:|---:|---|']
    for k,v in test['paired_formats'].items():
        lo,hi=v['ci95']
        lines.append(f'| {k} | {v["n_pairs"]} | {v["delta_b_minus_a"]*100:.2f} | [{lo*100:.2f}, {hi*100:.2f}] |')
    lines += ['', '## 数量、空间和 3D 的分条件表现', '',
        '| 条件 | 准确率 [95% CI] | 无效回答 |', '|---|---|---:|']
    for k,v in test['by_condition'].items():
        if k.startswith(('L4/','L5/')) or any(k.startswith('L3/'+q+'/') for q in ('Q1','Q2','Q3')):
            lines.append(f'| {k} | {metric(v["accuracy"])} | {v["invalid"]} |')
    lines += ['', '3D 的视角/遮挡解释应优先比较 3d_fixed 与 3d_cross/3d_occluded；'
        '相关配对区间已保存在 paired_comparisons.csv。Q2/Q7 的重复控制没有独立查询视觉变化。',
        '', '## 无效回答与解释边界', '',
        '所有无效回答的题目 ID、标准答案与原始模型回复见 invalid_responses.csv；'
        '本报告没有放宽解析规则或删除无效回答。',
        'Q6 的两个额外候选在两种世界中均未出现，四个选项并非同等熟悉；'
        'Q7 是线索条件下的属性回忆，不能将其高分直接视为与 Yes/No 相同任务上的改进。',
        '本数据为程序化物体及受控挤出形状，不能直接推广到自然场景或所有 VLM。'
        '目前仅有一个模型的完整行为证据，不进行内部机制或探针解释。',
        'L3 探索性回归结果保存在 summary.json，按组计算 sandwich 区间；'
        '布局与图片/token 数量存在关联，因此该回归不提供独立的因果归因。',
        '', '## 交付文件', '',
        '- accuracy.csv：dev/test 分任务准确率及区间。',
        '- conditions.csv：每个任务、模式、数量和关系条件的分数。',
        '- paired_comparisons.csv：跨问法、数量、布局与 3D 模式的配对差异。',
        '- invalid_responses.csv：无效回答原文，供人工复查。',
        '- summary.json：完整指标、运行来源、文件指纹及探索性回归。',
        '- figures/*.svg：可独立打开或插入报告的矢量图，含按组 bootstrap 区间。',
        '', '![各任务准确率](figures/task_accuracy.svg)',
        '', '![空间关系准确率](figures/spatial_accuracy.svg)',
        '', '![3D 条件准确率](figures/3d_accuracy.svg)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(out)
    for split,s in report['splits'].items():print(split,s['observed'],'complete',s['complete'],'invalid',s['invalid'])


if __name__ == '__main__':
    main()
