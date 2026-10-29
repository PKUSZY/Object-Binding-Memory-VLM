#!/usr/bin/env python3
"""Freeze a result-independent primary-task selection on all 240 existing test clusters."""
import json
from pathlib import Path
from collections import Counter


def main():
    dest=Path('runs/bindmem_priority');dest.mkdir(parents=True,exist_ok=True)
    source=Path('data/bindmem_v2/labels/test.jsonl')
    labels=[json.loads(line) for line in source.open()]
    chosen=[r for r in labels if (r['level']=='L0' and r['mode']=='2d') or
            (r['level']=='L2' and r['question'] in ('Q1','Q2','Q3'))]
    gradient=[json.loads(line) for line in Path('data/bindmem_gradient_n6/labels/test.jsonl').open()]
    gradient=[r for r in gradient if r['question'] in ('Q1','Q3')]
    for name,rows,n in [('core',chosen,1680),('gradient',gradient,3840)]:
        assert len(rows)==n and len({r['cluster_id'] for r in rows})==240
        ids=[r['trial_id'] for r in rows]
        path=dest/f'{name}_test_ids.json'
        if path.exists() and json.loads(path.read_text())!=ids:raise ValueError(f'Selection changed: {path}')
        path.write_text(json.dumps(ids,indent=2)+'\n')
        print(name,n,'trials; 240 test clusters;',dict(Counter(r['level']+'/'+r['question'] for r in rows)))
    (dest/'selection_rationale.md').write_text('''# 精简实验选择

只按预先明确的任务类型选择，不按任何单题对错筛选。保留原有全部 240 个正式测试组。

- 核心：二维属性识别 240 题；L2 严格绑定 Q1/Q2/Q3 各 480 题，共 1,680 题。
- 梯度：六物体 0–3 属性对，Q1/Q3 各 1,920 题，共 3,840 题。
- 暂停选择题、回忆、多物体布局、空间关系、三维扩展及梯度 Q2。
- 保持原始提示、图片预处理、80 token 上限、严格解析和随机种子；没有因 LLaVA 产生长回复而降低 token 上限。
- 原完整作业中已完成的同题回复，只有在模型、数据、预处理、生成参数、依赖和兼容源代码均核对通过后才复用。原目录不变。
- 精简范围是在观察到选择题格式问题后决定的，属于后续分析范围调整；不声称覆盖原完整基准，也不隐藏已观察到的选择题问题。
- 两个模型仅比较相同 test trial_id，并报告正确数、命中率、误认率及按随机组 bootstrap 的 95% 区间。
''')


if __name__=='__main__':main()
