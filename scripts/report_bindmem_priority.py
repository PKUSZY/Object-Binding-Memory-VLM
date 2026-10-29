#!/usr/bin/env python3
"""Readable same-trial comparison; generated after each completed priority phase."""
import json
from pathlib import Path


def main():
    root=Path('runs/bindmem_priority');parts=['# 精简实验：Qwen3-VL-8B 与 LLaVA-1.5-7B 同题比较',
    '保留原 240 个正式测试组；只比较完全相同的题目。模型输入包装与原生视觉分辨率不同，结论针对各自完整推理流程。无效回答计错。方括号为按组 bootstrap 95% 区间。']
    def fmt(m):
        if not m['n']:return '—'
        ci=m['ci95'];interval=f' [{ci[0]:.2%}, {ci[1]:.2%}]' if ci else ''
        return f'{m["value"]:.2%}'+interval
    for name,title in [('core','二维识别与严格绑定'),('gradient','匹配的重组难度梯度')]:
        path=root/f'{name}_comparison/comparison.json'
        parts.append('## '+title)
        if not path.exists():parts.append('尚未完成这一阶段，暂不报告完整结果。');continue
        report=json.loads(path.read_text());parts.append(f'双方共有 {report["shared"]} 道正式测试题。差值为 LLaVA 减 Qwen，单位为百分点。')
        lines=['| 条件 | 题数 | Qwen 准确率 | LLaVA 准确率 | 差值 [95% CI] |', '|---|---:|---|---|---|']
        binary=[]
        for cond,c in report['by_condition'].items():
            a,b=c['a'],c['b'];d=c['delta_b_minus_a'];lo,hi=d['ci95']
            lines.append(f'| {cond} | {d["n_pairs"]} | {fmt(a["accuracy"])} | {fmt(b["accuracy"])} | {d["delta_b_minus_a"]*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] |')
            if a['target_hit_rate']['n']:
                binary.append(f'| {cond} | {fmt(a["target_hit_rate"])} | {fmt(b["target_hit_rate"])} | {fmt(a["negative_false_acceptance"])} | {fmt(b["negative_false_acceptance"])} |')
        parts.append('\n'.join(lines))
        if binary:parts.append('\n'.join(['| 条件 | Qwen 目标命中 | LLaVA 目标命中 | Qwen 负例误认 | LLaVA 负例误认 |','|---|---|---|---|---|']+binary))
    parts.append('范围调整在发现选择题长回复和格式失败后作出，未按单题正确率筛选；原选择题记录保留。本报告不代表原完整基准。梯度正例在同组各 k 下重复，统计按组处理。')
    (root/'report.md').write_text('\n\n'.join(parts)+'\n')
    print(root/'report.md')


if __name__=='__main__':main()
