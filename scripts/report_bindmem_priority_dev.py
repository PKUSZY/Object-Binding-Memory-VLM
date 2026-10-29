#!/usr/bin/env python3
"""Immediate exploratory DEV comparison from existing outputs; no new inference."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bindmem_extra.score import load_rows,metrics,condition
from bindmem.stats import paired_delta
from bindmem.common import write_json,write_jsonl,sha_file


def main():
    data='data/bindmem_v2';root=Path('runs/bindmem_priority/dev_preview');root.mkdir(parents=True,exist_ok=True)
    bpath='runs/bindmem_v2/llava7b_behavior'
    apaths=['runs/bindmem_v2/qwen8b_core','runs/bindmem_v2/qwen8b_image','runs/bindmem_v2/qwen8b_formats']
    selected=lambda r:(r['level']=='L0' and r['mode']=='2d') or (r['level']=='L2' and r['question'] in ('Q1','Q2','Q3'))
    b={r['trial_id']:r for r in load_rows(data,bpath,'dev',allow_partial=True)[0] if selected(r)}
    a={r['trial_id']:r for path in apaths for r in load_rows(data,path,'dev')[0] if selected(r)}
    assert set(a)==set(b) and len(a)==560
    assert all(r['observed'] for r in [*a.values(),*b.values()])
    assert len({r['cluster_id'] for r in a.values()})==80
    cells={}
    for cond in sorted({condition(r) for r in a.values()}):
        ids=sorted(i for i,r in a.items() if condition(r)==cond)
        aa,bb=[a[i] for i in ids],[b[i] for i in ids]
        cells[cond]=dict(qwen=metrics(aa,1000),llava=metrics(bb,1000),delta=paired_delta(aa,bb,1000))
    write_json(root/'comparison.json',dict(split='dev',clusters=80,trials=560,exploratory=True,
        scope='Task-type selection only; no filtering by model correctness. Not formal test results.',
        source={path:dict(manifest_sha256=sha_file(Path(path)/'manifest.json'),predictions_sha256=sha_file(Path(path)/'predictions.jsonl')) for path in [*apaths,bpath]},by_condition=cells))
    write_jsonl(root/'qwen_scored.jsonl',list(a.values()));write_jsonl(root/'llava_scored.jsonl',list(b.values()))
    parts=['# 已完成开发集的初步同题对比',
        '**这是 dev 的探索性结果，不是正式 test 结论。** 全部使用已有原始回复，无新推理；80 个开发组、560 道相同题目。正式 test 的 240 组正在精简作业中补齐。',
        '判断题的目标命中和负例误认必须一起阅读；无效回答计错。表内百分比为准确率，方括号为按组 bootstrap 95% 区间。',
        '| 任务 | 题数 | Qwen3-VL-8B | LLaVA-1.5-7B | LLaVA 目标命中 | LLaVA 强重组误认 | LLaVA 无效回答 |',
        '|---|---:|---|---|---|---|---:|']
    def fmt(m):
        if not m['n']:return '—'
        lo,hi=m['ci95'];return f'{m["value"]:.2%} [{lo:.2%}, {hi:.2%}]'
    names={'CAL':'二维属性识别','Q1':'图片判断','Q2':'文字判断','Q3':'图片＋文字判断'}
    for cond,c in cells.items():
        a,b=c['qwen'],c['llava'];q=cond.split('/')[1]
        parts.append(f'| {names[q]} | {a["observed"]} | {fmt(a["accuracy"])} | {fmt(b["accuracy"])} | {fmt(b["target_hit_rate"])} | {fmt(b["negative_false_acceptance"])} | {b["invalid"]} |')
    parts+=['','这里只保留预先指定的核心任务，不能据此代表所有能力；原选择题长回复与格式失败的记录仍保留。范围收缩是在观察到这些现象后作出的，正式报告会说明这一点。']
    (root/'report.md').write_text('\n\n'.join(parts[:3])+'\n\n'+'\n'.join(parts[3:])+'\n')
    print((root/'report.md').read_text())


if __name__=='__main__':main()
