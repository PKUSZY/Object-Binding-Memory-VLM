"""Detailed task success rates, with strong recombination reported separately."""
import csv
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bindmem.stats import rate

QN={'Q1':'图片判断','Q2':'文字判断','Q3':'图片＋文字判断','Q4':'同图二选一',
    'Q5':'分图二选一','Q6':'四选一','Q7':'属性回忆'}
LN={'L0':'属性识别','L1':'基础绑定','L2':'严格绑定','L3':'多物体绑定','L4':'空间关系','L5':'三维扩展'}
MN={'2d':'二维','3d_fixed':'固定视角三维','3d_cross':'跨视角三维','3d_occluded':'查询图遮挡'}


def build_task_section(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());s=summary['splits']['test']
    rows=[]
    for run in summary['runs']:
        rows.extend(json.loads(line) for line in (Path(run['path'])/'scored/test/scored.jsonl').open())
    assert len(rows)==69840 and len({r['trial_id'] for r in rows})==len(rows)
    assert all(r['observed'] for r in rows)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['level'],r['question'])].append(r)
    records=[]
    def metric(rr,name):
        assert rr
        n=len(rr);correct=sum(r['correct'] for r in rr)
        m=rate(rr,'correct'); pct=(Decimal(correct)*100/Decimal(n)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
        records.append({'task':name,'n':n,'correct':correct,'success_rate':correct/n,
                        'ci_low':m['ci95'][0],'ci_high':m['ci95'][1],
                        'invalid':sum(not r['valid'] for r in rr)})
        return f'**{pct}%**（{correct}/{n}）',m
    def cell(rr,name):return metric(rr,name)[0]
    def table(headers,body):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(r)+' |' for r in body])
    def select(level,q,**kwargs):return [r for r in grouped[(level,q)] if all(r.get(k)==v for k,v in kwargs.items())]
    parts=[];add=lambda text:parts.append(text.strip())
    add('## 逐任务成功率：把强重组诱饵单独列出')
    add('这里的“成功率”统一指模型答对的比例，不是程序运行完成率。以下全部来自正式 test；无效回答计错。每格同时列出正确题数/总题数，百分比按整数计数四舍五入。不要把“目标命中率”“诱饵正确拒绝率”和“正负合计准确率”混为一项。')
    add('**两类重组必须区分：** 基础绑定中的普通重组，只要求三种属性分别出现过，不要求所有属性对都出现；严格绑定、多物体绑定和三维扩展中的强重组，则要求三种两两属性组合全部在学习物体上出现过，但完整三元组从未出现。强重组题的正确答案是 NO。')
    add('### 先看最关心的强重组任务')
    add('每个条件将真实目标与强重组诱饵配平。下表里的“拒绝成功”是回答正确的 NO；拒绝率很高但目标命中率很低，可能只是偏向一律回答 NO。')
    body=[];strong={}
    for level in ['L2','L3','L5']:
        for q in ['Q1','Q2','Q3']:
            allrows=grouped[(level,q)];yes=[r for r in allrows if r['answer']=='YES'];no=[r for r in allrows if r['probe_type']=='pairwise_complete' and r['answer']=='NO']
            assert len(yes)==len(no) and len(yes)+len(no)==len(allrows)
            hit,hm=metric(yes,f'{level}/{q}/target_hit');reject,rm=metric(no,f'{level}/{q}/strong_lure_correct_rejection')
            strong[(level,q)]=(hm,rm)
            body.append([LN[level],QN[q],hit,reject,cell(allrows,f'{level}/{q}/accuracy')])
    add(table(['任务','问法','真实目标命中率','强重组正确拒绝率','正负合计准确率'],body))
    add('例如严格绑定的图片查询：240 道目标答对 238 道，240 道强重组负例只正确拒绝 105 道，因此总准确率是 (238+105)/480=71.46%。56.25% 是误认率，43.75% 才是这类负例的成功率。文字查询虽然拒绝了全部 240 道强重组负例，但只认出 2 道目标，因此不能认为它解决了绑定问题。')
    add('三维汇总含四种模式。文字查询的三个三维模式输入重复，因此这个汇总不是三份独立的视觉变化证据；后面的分模式表更适合解释具体条件。')

    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(14,4.9),layout='constrained')
    for ax,level in zip(axes,['L2','L3','L5']):
        for i,q in enumerate(['Q1','Q2','Q3']):
            for j,(m,color) in enumerate(zip(strong[(level,q)],['#24699C','#D67A28'])):
                x=i+(j-.5)*.34;v=m['value']*100;lo,hi=np.array(m['ci95'])*100
                ax.bar(x,v,.3,color=color,zorder=3)
                ax.errorbar(x,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='none',ecolor='#263645',capsize=3,zorder=4)
                ax.annotate(f'{v:.1f}%',(x,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
        ax.set_ylim(0,115);ax.set_xticks(range(3),['图片','文字','图文']);ax.set_ylabel('答对比例（%）');ax.set_title(LN[level]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].plot([],[],color='#24699C',lw=7,label='真实目标命中');axes[0].plot([],[],color='#D67A28',lw=7,label='强重组正确拒绝')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('强重组单独评分：两类成功率都应高，才表示可靠区分',fontsize=15)
    figdir=root/'task_figures';figdir.mkdir(exist_ok=True)
    for ext in ['png','svg','pdf']:fig.savefig(figdir/f'strong_recombination_success.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    relative='task_figures/strong_recombination_success.png'
    add(f'![真实目标命中与强重组正确拒绝]({relative})\n\n蓝色：真实目标命中率；橙色：强重组正确拒绝率。两者越高越好。误差线为按组计算的 95% 区间。')

    add('### 所有物体任务的问法总览')
    body=[]
    for q in QN:
        body.append([q+' '+QN[q]]+[cell(grouped[(level,q)],f'{level}/{q}/accuracy') for level in ['L1','L2','L3','L5']])
    add(table(['输入与输出形式','基础绑定','严格绑定','多物体绑定','三维扩展'],body))
    add('Q1：给候选图片，回答是否见过；Q2：只用文字描述候选，学习阶段仍是图片；Q3：同时给图片与属性文字；Q4/Q5：同图或分图的二选一；Q6：四选一；Q7：给颜色和形状、回忆纹理。不同问法的线索和输出空间不同，不计算跨题型总分。')

    add('### 属性识别：看清一个物体的三个属性')
    body=[]
    for mode in MN:body.append([MN[mode],cell(select('L0','CAL',mode=mode),f'L0/{mode}/joint_attributes')])
    body.append(['合计',cell(grouped[('L0','CAL')],'L0/joint_attributes')])
    add(table(['呈现条件','三个属性同时答对'],body))
    add('这里要求颜色、纹理、形状全部正确。遮挡校准中的一个无效输出按错计分。100% 只表示当前样本全部答对，不保证未见样本零错误。')

    add('### 基础绑定：目标和四类诱饵分别表现如何')
    body=[]
    for typ,meaning in [('target','真实出现过的目标：应答 YES'),('unrelated','三种属性全新：应答 NO'),('near1','仅一类属性见过：应答 NO'),('near2','仅两类属性见过：应答 NO'),('critical','普通属性重组：应答 NO')]:
        body.append([meaning]+[cell(select('L1',q,probe_type=typ),f'L1/{q}/{typ}/correct') for q in ['Q1','Q2','Q3']])
    add(table(['候选类型与正确行为','图片查询','文字查询','图片＋文字'],body))
    add('目标行是命中率，四类负例行是正确拒绝率。本表普通重组不等于“两两组合全出现”的强重组，不能用它替代前面的强重组成绩。')
    body=[]
    for n in (2,3,4):
        body.append([str(n)]+[cell(select('L1',q,total_objects=n),f'L1/{q}/n{n}/accuracy') for q in QN])
    add('基础任务再按学习物体总数拆分，每张学习图始终只有一个物体：')
    add(table(['学习物体数']+[QN[q] for q in QN],body))

    add('### 严格绑定：两道反事实题能否都答对')
    body=[]
    for q in QN:
        m=s['by_level_question'][f'L2/{q}']['pair_accuracy'];n=m['n'];c=round(m['value']*n)
        pct=(Decimal(c)*100/Decimal(n)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
        pairs=defaultdict(list)
        for r in grouped[('L2',q)]:pairs[r['pair_id']].append(r)
        assert len(pairs)==n and all(len(rr)==2 for rr in pairs.values())
        assert sum(all(r['correct'] for r in rr) for rr in pairs.values())==c
        records.append({'task':f'L2/{q}/pair_accuracy','n':n,'correct':c,
                        'success_rate':c/n,'ci_low':m['ci95'][0],'ci_high':m['ci95'][1],
                        'invalid':sum(any(not r['valid'] for r in rr) for rr in pairs.values())})
        body.append([QN[q],f'**{pct}%**（{c}/{n} 对）'])
    add(table(['问法','同一正负世界的两题都正确'],body))
    add('每题仍是独立模型输入；这里事后将对应的两题放在一起评分。总是输出同一个答案可能得到约 50% 的单题分数，却无法同时答对正负两题。选择题两世界正确选项不同，回忆题两世界正确纹理也不同。')

    add('### 多物体任务：每种数量和布局的成功率')
    body=[]
    for n,per in [(4,1),(4,2),(4,4),(8,1),(8,2),(8,4),(8,8)]:
        body.append([str(n),str(per)]+[cell(select('L3',q,total_objects=n,objects_per_image=per),f'L3/{q}/n{n}/per{per}/accuracy') for q in QN])
    add(table(['总物体数','每图物体数']+[QN[q] for q in QN],body))
    add('每格 480 题；二元判断内含 240 个目标和 240 个强重组诱饵。各条件成绩不能简单概括为“数量越大越低”。')
    body=[]
    for n,per in [(4,1),(4,2),(4,4),(8,1),(8,2),(8,4),(8,8)]:
        body.append([f'总数 {n} / 每图 {per}']+[cell(select('L3',q,total_objects=n,objects_per_image=per,probe_type='pairwise_complete'),f'L3/{q}/n{n}/per{per}/strong_lure_correct_rejection') for q in ['Q1','Q2','Q3']])
    add('同一批数量实验中的**强重组负例单独正确拒绝率**：')
    add(table(['数量条件','图片查询','文字查询','图片＋文字'],body))

    add('### 空间关系：四类关系分别评分')
    body=[]
    for rel,name in [('left','左右：2 个物体'),('above','上下：2 个物体'),('front','深度前后：2 个物体'),('cooccur','同图共现：4 个物体、每图 2 个')]:
        body.append([name,cell(select('L4','Q8',relation=rel),f'L4/{rel}/accuracy')])
    add(table(['关系任务','答对率'],body))
    add('共现任务问两个指定物体是否在同一张学习图出现。不同关系的材料和物体数量不同，跨关系分数差不能完全解释为关系本身的难度差。')

    add('### 三维任务：分模式报告，避免遮挡被平均掉')
    body=[]
    for mode in MN:
        body.append([MN[mode]]+[cell(select('L5',q,mode=mode),f'L5/{q}/{mode}/accuracy') for q in QN])
    add(table(['模式']+[QN[q] for q in QN],body))
    add('每格 480 题。跨视角/遮挡施加在查询图上。Q2/Q7 无查询图片，三个三维模式的输入重复；重复分数不是独立的视觉效应证据。')
    body=[]
    for mode in MN:
        body.append([MN[mode]]+[cell(select('L5',q,mode=mode,probe_type='pairwise_complete'),f'L5/{q}/{mode}/strong_lure_correct_rejection') for q in ['Q1','Q2','Q3']])
    add('各模式中的**强重组负例单独正确拒绝率**：')
    add(table(['模式','图片查询','文字查询','图片＋文字'],body))
    add('遮挡条件的图片与图文判断均将 480 题全部回答 NO：240 道负例全对、240 道目标全错，因此负例拒绝率为 100%，目标命中率为 0%，总准确率只有 50%。这不是可靠识别强重组的证据。')
    add('注意：负例拒绝率不能单独代表整体能力。例如全部答 NO 会拒绝所有负例，却漏掉所有目标。因此应结合前面的目标命中率和正负合计分数阅读。遮挡也可能降低候选本身可识别性，不能只归因于记忆失败。')
    add('### 无效输出是否被排除了？')
    add('没有。正式测试共 436 条无效输出，全部按错误计分；例如二选一回答 C/D 会直接记错。本报告显示的是含无效输出的成功率。所有分母为实际完成题数，当前无最终运行失败或缺失题。')
    add('完整分数与按组置信区间见 `task_success_rates.csv`；统计来源为已评分的正式逐题结果，读取后重新核对了全量覆盖。解释输入与原始回复的例子见 `input_output_report.html`。')
    # Duplicated overview cells are written once in the machine-readable export.
    unique={r['task']:r for r in records}
    for key,r in unique.items():assert 0<=r['correct']<=r['n']
    with (root/'task_success_rates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task','n','correct','success_rate','ci_low','ci_high','invalid']);w.writeheader();w.writerows(unique.values())
    return '\n\n'.join(parts)+'\n',{relative:figdir/'strong_recombination_success.png'}
