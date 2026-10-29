#!/usr/bin/env python3
"""Build the complete, audited multi-model project report without running inference.
Use the existing base Python; PDF extras may be supplied in /tmp/bindmem-report-deps.
"""
import base64
import csv
import html
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import build_bindmem_visual_report as plot_style
import matplotlib.pyplot as plt
import numpy as np
import markdown
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from bindmem.common import read_jsonl,sha_file,parse_answer,write_json
from bindmem_extra.score import load_rows
from bindmem.stats import rate,paired_delta,cluster_ci
from bindmem_extra.gradient import pair_counts
from bindmem_report_io import describe_zh
from report_pdf import render_pdf

OUT=ROOT/'reports/bindmem_complete_20261029'
QN={'Q1':'图片判断','Q2':'文字判断','Q3':'图文判断','Q4':'同图二选一','Q5':'分图二选一','Q6':'四选一','Q7':'纹理回忆'}
BLUE='#24699C';ORANGE='#D67A28';GREEN='#249382'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def pct(x):return str((Decimal(str(x))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
def measure(m,counts=False):
    if not m['n']:return '—'
    s=pct(m['value'])
    if counts:s+=f'（{round(m["value"]*m["n"])}/{m["n"]}）'
    if m.get('ci95'):s+=f' [{100*m["ci95"][0]:.2f}, {100*m["ci95"][1]:.2f}]'
    return s

def load_json(path):return json.loads(Path(path).read_text())


def main():
    for p in [OUT,OUT/'figures',OUT/'tables',OUT/'evidence']:p.mkdir(parents=True,exist_ok=True)
    baseline=load_json('runs/bindmem_v2/qwen8b_report/summary.json');s=baseline['splits']['test']
    assert s['complete'] and s['observed']==69840
    sources={};audit=[]
    def source(path):
        p=Path(path);sources[str(p)]=dict(sha256=sha_file(p),bytes=p.stat().st_size)
    qrows=[]
    for run in baseline['runs']:
        rr,_=load_rows('data/bindmem_v2',run['path'],'test');qrows+=rr
        for f in ['manifest.json','predictions.jsonl']:source(Path(run['path'])/f)
    assert len(qrows)==69840 and len({r['trial_id'] for r in qrows})==69840
    assert sum(not r['valid'] for r in qrows)==436
    for key,m in s['by_level_question'].items():
        rr=[r for r in qrows if r['level']+'/'+r['question']==key]
        assert len(rr)==m['observed'] and abs(sum(r['correct'] for r in rr)/len(rr)-m['accuracy']['value'])<1e-12
    core=load_json('runs/bindmem_priority/core_comparison/comparison.json')
    grad=load_json('runs/bindmem_priority/gradient_comparison/comparison.json')
    lc,_=load_rows('data/bindmem_v2','runs/bindmem_priority/llava7b_core','test')
    lg,_=load_rows('data/bindmem_gradient_n6','runs/bindmem_priority/llava7b_gradient','test')
    qg,_=load_rows('data/bindmem_gradient_n6','runs/bindmem_gradient_n6/qwen8b','test')
    assert len(lc)==1680 and len(lg)==3840 and len(qg)==5760
    assert len({r['cluster_id'] for r in lc})==len({r['cluster_id'] for r in lg})==240
    qi={r['trial_id']:r for r in qrows};qgi={r['trial_id']:r for r in qg}
    for comp,brows,ai in [(core,lc,qi),(grad,lg,qgi)]:
        assert comp['shared']==len(brows)
        for cond,c in comp['by_condition'].items():
            if cond.startswith('G/'):
                q=cond.split('/')[1];k=int(cond.rsplit('k',1)[1]);br=[r for r in brows if r['question']==q and r['gradient_k']==k]
            else:
                level,q=cond.split('/')[:2];br=[r for r in brows if r['level']==level and r['question']==q]
            ar=[ai[r['trial_id']] for r in br]
            for side,rr in [('a',ar),('b',br)]:
                assert len(rr)==c[side]['accuracy']['n']
                assert abs(sum(r['correct'] for r in rr)/len(rr)-c[side]['accuracy']['value'])<1e-12
                assert all(r['observed'] for r in rr)
    for path in ['runs/bindmem_priority/llava7b_core','runs/bindmem_priority/llava7b_gradient','runs/bindmem_gradient_n6/qwen8b']:
        for f in ['manifest.json','predictions.jsonl','scored/test/summary.json']:source(Path(path)/f)
    for path in ['data/bindmem_v2/manifest.json','data/bindmem_gradient_n6/manifest.json',
                 'runs/bindmem_v2/qwen8b_report/summary.json','runs/bindmem_priority/core_comparison/comparison.json',
                 'runs/bindmem_priority/gradient_comparison/comparison.json']:source(path)
    # Copy existing Qwen charts and their underlying chart audit, not old conclusions about model coverage.
    for p in Path('runs/bindmem_v2/qwen8b_report/visual_figures').glob('*'):
        if p.suffix in ('.png','.svg'):shutil.copy2(p,OUT/'figures'/p.name)
    shutil.copy2('runs/bindmem_v2/qwen8b_report/visual_figures/figure_data.csv',OUT/'tables/qwen_figure_data.csv')
    for f in ['accuracy.csv','conditions.csv','paired_comparisons.csv','task_success_rates.csv']:
        shutil.copy2(Path('runs/bindmem_v2/qwen8b_report')/f,OUT/'tables'/('qwen_'+f))
    figures=[]
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        figures.append(name);plt.close(fig)
    def remember(key,m):
        assert m['n'] and m['value'] is not None
        audit.append(dict(metric=key,n=m['n'],value=m['value'],ci_low=m['ci95'][0],ci_high=m['ci95'][1]))
    def bar(ax,x,m,color,key,width=.35):
        remember(key,m);v=100*m['value'];lo,hi=np.array(m['ci95'])*100
        ax.bar(x,v,width,color=color,zorder=3)
        ax.errorbar(x,v,yerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='none',ecolor='#253746',capsize=3,zorder=4)
        ax.annotate(f'{v:.1f}%',(x,max(v,hi)),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
    def axes_percent(ax,label='答对率（%）'):
        ax.set_ylim(0,114);ax.set_yticks(range(0,101,20));ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    # Model comparison: calibration and the three strictly matched binary tasks.
    ck=['L0/CAL/2d/n1/per1']+[f'L2/{q}/2d/n4/per1' for q in ['Q1','Q2','Q3']]
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.2),layout='constrained',gridspec_kw={'width_ratios':[1,3]})
    for j,side in enumerate(['a','b']):bar(axs[0],j,core['by_condition'][ck[0]][side]['accuracy'],[BLUE,ORANGE][j],f'core/cal/{side}/accuracy',.55)
    axs[0].set_xticks([0,1],['Qwen','LLaVA']);axs[0].set_title('单物体二维：三属性同时答对')
    for i,key in enumerate(ck[1:]):
        for j,side in enumerate(['a','b']):bar(axs[1],i+(j-.5)*.38,core['by_condition'][key][side]['accuracy'],[BLUE,ORANGE][j],f'{key}/{side}/accuracy')
    axs[1].set_xticks(range(3),['图片 Q1','文字 Q2','图文 Q3']);axs[1].set_title('四物体严格绑定：同一批 480 题')
    axs[1].axhline(50,color='#777',ls='--',lw=1)
    for ax in axs:axes_percent(ax)
    fig.legend([plt.Rectangle((0,0),1,1,color=BLUE),plt.Rectangle((0,0),1,1,color=ORANGE)],['Qwen3-VL-8B','LLaVA-1.5-7B'],loc='outside lower center',ncol=2,frameon=False)
    save(fig,'09_model_comparison')
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.1),layout='constrained')
    for ax,metric,title in zip(axs,['target_hit_rate','negative_false_acceptance'],['真实目标命中：越高越好','强重组误认：越低越好']):
        for i,key in enumerate(ck[1:]):
            for j,side in enumerate(['a','b']):bar(ax,i+(j-.5)*.38,core['by_condition'][key][side][metric],[BLUE,ORANGE][j],f'{key}/{side}/{metric}')
        ax.set_xticks(range(3),['图片 Q1','文字 Q2','图文 Q3']);ax.set_title(title);axes_percent(ax,'比例（%）')
    fig.legend([plt.Rectangle((0,0),1,1,color=BLUE),plt.Rectangle((0,0),1,1,color=ORANGE)],['Qwen3-VL-8B','LLaVA-1.5-7B'],loc='outside lower center',ncol=2,frameon=False)
    save(fig,'10_response_bias')
    # Show actual gradient stimuli from the first test cluster, not invented examples.
    glabels=read_jsonl('data/bindmem_gradient_n6/labels/test.jsonl');ginputs={r['trial_id']:r for r in read_jsonl('data/bindmem_gradient_n6/inputs/test.jsonl')}
    gfirst=glabels[0]['cluster_id'];gcases=[next(r for r in glabels if r['cluster_id']==gfirst and r['question']=='Q1' and r['answer']=='NO' and r['gradient_k']==k) for k in range(4)]
    fig,axs=plt.subplots(4,4,figsize=(12.8,10.8),layout='constrained',gridspec_kw={'width_ratios':[1.45,1.45,1.45,1]})
    traces=[]
    for row,r in enumerate(gcases):
        item=ginputs[r['trial_id']];paths=item['study_images']+[b['image'] for b in item['query'] if b['type']=='image']
        for col,path in enumerate(paths):
            axs[row,col].imshow(Image.open(Path('data/bindmem_gradient_n6')/path));axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            for spine in axs[row,col].spines.values():spine.set_visible(False)
            axs[row,col].set_title(f'学习图 {col+1}（2 个物体）' if col<3 else '同一候选：应答 NO',fontsize=10)
        axs[row,0].set_ylabel(f'k = {row}\n已见属性对 {row} 种',fontsize=12)
        traces.append(dict(trial_id=r['trial_id'],label=r,input=item,images={p:sha_file(Path('data/bindmem_gradient_n6')/p) for p in paths}))
    fig.suptitle('实际匹配梯度：每行 6 个学习物体；候选及单属性频数相同',fontsize=15)
    save(fig,'11_gradient_design');write_json(OUT/'evidence/gradient_design.json',traces)
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for row,q in enumerate(['Q1','Q3']):
        for col,metric in enumerate(['negative_false_acceptance','accuracy']):
            ax=axs[row,col]
            for side,color,label in [('a',BLUE,'Qwen3-VL-8B'),('b',ORANGE,'LLaVA-1.5-7B')]:
                mm=[grad['by_condition'][f'G/{q}/n6/k{k}'][side][metric] for k in range(4)]
                for k,m in enumerate(mm):remember(f'G/{q}/k{k}/{side}/{metric}',m)
                y=np.array([m['value']*100 for m in mm]);lo=np.array([m['ci95'][0]*100 for m in mm]);hi=np.array([m['ci95'][1]*100 for m in mm])
                ax.errorbar(range(4),y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],marker='o',capsize=4,lw=2,color=color,label=label)
                for k,v in enumerate(y):ax.annotate(f'{v:.1f}',(k,v),xytext=(0,10 if side=='b' or v<5 else -17),textcoords='offset points',ha='center',fontsize=10,color=color)
            axes_percent(ax,'误认率（%）' if col==0 else '准确率（%）');ax.set_ylim(-7,114)
            ax.set_xticks(range(4));ax.set_xlabel('负例中已见候选属性对的种类数 k')
            ax.set_title(QN[q]+('：负例误认，越低越好' if col==0 else '：正负合计准确率'))
            if col==1:ax.axhline(50,color='#999',ls='--',lw=1)
    axs[0,0].legend(loc='center left',frameon=False);save(fig,'12_gradient_results')
    # Same-case cross-model examples; deterministic selection with visible selection rule.
    inputs={r['trial_id']:r for r in read_jsonl('data/bindmem_v2/inputs/test.jsonl')};li={r['trial_id']:r for r in lc}
    qraw={}
    for run in baseline['runs']:
        for r in read_jsonl(Path(run['path'])/'predictions.jsonl'):
            if r['trial_id'] in qi and not r.get('error'):qraw[r['trial_id']]=r
    lraw={r['trial_id']:r for r in read_jsonl('runs/bindmem_priority/llava7b_core/predictions.jsonl')}
    eligible=sorted([r for r in lc if r['level']=='L2' and r['question']=='Q1' and r['answer']=='NO'],key=lambda r:r['trial_id'])
    example_both=next(r for r in eligible if qi[r['trial_id']]['prediction']=='YES' and r['prediction']=='YES')
    example_diff=next(r for r in eligible if qi[r['trial_id']]['correct'] and r['prediction']=='YES')
    cases=[]
    for r in [example_both,example_diff]:
        tid=r['trial_id'];item=inputs[tid];study=[o for sc in r['study_scenes'] for o in sc]
        assert r['probe'] not in study and all(pair_counts(study,r['probe']))
        cases.append(dict(trial_id=tid,input=item,label=r,qwen_raw=qraw[tid]['raw_response'],llava_raw=lraw[tid]['raw_response'],
            images={p:sha_file(Path('data/bindmem_v2')/p) for p in item['study_images']+[b['image'] for b in item['query'] if b['type']=='image']}))
    fig,axs=plt.subplots(2,5,figsize=(14,6.5),layout='constrained')
    for i,c in enumerate(cases):
        paths=c['input']['study_images']+[b['image'] for b in c['input']['query'] if b['type']=='image']
        for j,p in enumerate(paths):
            axs[i,j].imshow(Image.open(Path('data/bindmem_v2')/p));axs[i,j].axis('off');axs[i,j].set_title(f'学习物体 {j+1}' if j<4 else '候选：标准 NO',fontsize=11)
        axs[i,0].text(0,1.3,f'案例 {i+1}：Qwen = {c["qwen_raw"]}；LLaVA = {c["llava_raw"]}',transform=axs[i,0].transAxes,fontsize=13)
    fig.suptitle('真实强重组负例：属性对均见过，完整物体未出现',fontsize=16)
    save(fig,'13_real_errors');write_json(OUT/'evidence/shared_error_cases.json',cases)
    # A compact workflow diagram, data-native rather than generated illustrations.
    fig,ax=plt.subplots(figsize=(13.5,3.4));ax.set_xlim(0,13.5);ax.set_ylim(0,3.4);ax.axis('off')
    texts=['生成学习集合\n与匹配候选','同一条用户消息\n规则＋学习图＋查询','独立确定性推理\n保存原始回复','严格解析＋标签核对\n无效输出计错','按任务与随机组统计\n同题比较＋置信区间']
    for i,t in enumerate(texts):
        x=.15+i*2.7;ax.text(x+1.15,1.9,t,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.7',fc='#EAF1F6',ec=BLUE))
        if i<4:ax.annotate('',xy=(x+2.65,1.9),xytext=(x+2.28,1.9),arrowprops=dict(arrowstyle='->',lw=1.8,color=BLUE))
    ax.text(6.7,.45,'标签不进入模型输入；每题独立；学习图仍在当前上下文中；不更新模型参数',ha='center',fontsize=12,color='#465E70')
    save(fig,'14_protocol')
    # Gradient endpoint/adjacent contrasts use signed false-acceptance differences, clustered by replicate.
    contrasts={}
    for model,rr in [('Qwen',qg),('LLaVA',lg)]:
        for q in sorted({r['question'] for r in rr}):
            neg=[r for r in rr if r['question']==q and r['answer']=='NO']
            for a,b in [(0,1),(1,2),(2,3),(0,3)]:
                left={r['cluster_id']:r for r in neg if r['gradient_k']==a};right={r['cluster_id']:r for r in neg if r['gradient_k']==b}
                ids=sorted(left.keys() & right.keys());diff=[int(right[i]['accepted'])-int(left[i]['accepted']) for i in ids]
                contrasts[f'{model}/{q}/k{a}_to_k{b}']=dict(n=len(ids),delta=float(np.mean(diff)),ci95=cluster_ci(diff,ids))
    write_json(OUT/'tables/gradient_paired_effects.json',contrasts)
    with (OUT/'tables/new_figure_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['metric','n','value','ci_low','ci_high']);w.writeheader();w.writerows(audit)
    # Full precision tables for both completed model comparisons.
    for name,comp in [('core',core),('gradient',grad)]:
        shutil.copy2(f'runs/bindmem_priority/{name}_comparison/comparison.csv',OUT/'tables'/f'{name}_comparison.csv')
        records=[]
        for cond,c in comp['by_condition'].items():
            for side,model in [('a','Qwen3-VL-8B'),('b','LLaVA-1.5-7B')]:
                for metric in ['accuracy','target_hit_rate','negative_false_acceptance','negative_correct_rejection','pair_accuracy']:
                    m=c[side][metric]
                    if m['n']:records.append(dict(condition=cond,model=model,metric=metric,n=m['n'],value=m['value'],ci_low=m['ci95'][0],ci_high=m['ci95'][1],invalid_in_condition=c[side]['invalid']))
        with (OUT/'tables'/f'{name}_all_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # The narrative below is regenerated from audited result objects.
    parts=[];add=lambda x:parts.append(x.strip())
    def fig(name,caption):add(f'![{caption}](figures/{name}.png)\n\n*{caption}*')
    add('# BindMem：视觉语言模型物体属性绑定评测\n\n**完整项目报告 · Qwen3-VL-8B 与 LLaVA-1.5-7B · 2026-10-29**\n\n数据版本：BindMem v2 与六物体匹配重组梯度。本文汇总已完成的行为实验，不包含探针或内部机制实验。所有主要结论使用正式 test；开发集初步结果不混入正式成绩。')
    add('## 摘要')
    add('本项目检验视觉语言模型能否区分“见过一些属性”与“见过由这些属性共同构成的完整物体”。程序生成颜色、纹理、形状可控的物体，要求模型在同一上下文中的学习图片之后回答识别、选择、属性回忆和关系问题；进一步构造单属性与两两属性统计相同的严格正负世界，以及固定单属性频数、只按候选已见属性对数量分级的重组梯度。')
    add('Qwen3-VL-8B 完成原基准全部 93,120 题，其中正式 test 69,840 题；另完成六物体梯度 7,680 题，其中 test 5,760 题。LLaVA-1.5-7B 完成精简的 5,520 道正式 test，与 Qwen 逐题配对比较。Qwen 在单物体属性识别、基础图片判断和简单左右／上下关系上接近满分，但严格图片判断中能认出 99.17% 的目标，却误认 56.25% 的强重组负例。新增梯度中，Qwen 图片查询的负例误认率随已见属性对数量 0→1→2→3，依次为 0.00%、0.83%、10.42%、39.17%。')
    add('LLaVA 在严格图片判断和图片梯度中全部回答 YES，因而目标命中率和负例误认率同时为 100%，平衡题集准确率为 50%。这反映当前多图协议下的接受偏向，不能称为随机猜测，也不足以单独证明其绑定机制弱于 Qwen：LLaVA 的严格二维三属性识别仅为 67.08%，存在基础识别及格式问题；模型的原生分辨率、提示包装和多图训练经历也不同。总体而言，单一准确率会掩盖不同的失败模式，需同时报告目标命中、负例拒绝、误认和原始回复。')
    add('## 1. 研究问题与项目范围')
    add('物体身份定义为三元组 `(颜色, 纹理, 形状)`。例如见过“红色、纯色、星形”和“红色、条纹、正方形”，并不等于见过“红色、纯色、正方形”。如果模型把不同物体的属性组合成一个从未出现的物体，就会产生重组误认。研究的目的不是单纯提高任务分数，而是辨别这种误认与视觉识别、回答偏向、查询形式及场景难度之间的关系。')
    add(table(['研究问题','实验对应','需要观察的证据'],[
        ['看清单个特征后，是否仍然绑定错误？','L0 校准＋L1/L2','识别准确率、目标命中、不同诱饵的误认'],
        ['单个属性和属性对都熟悉时，还能认出完整三元组吗？','L2 严格反事实','同一查询在正负世界的单题及成对正确率'],
        ['问法、数量、布局和空间／三维因素如何影响表现？','Q1–Q8、L3–L5','分条件成绩与有效的配对差'],
        ['重组难度是否有梯度，另一模型是否复现？','G 梯度＋双模型同题比较','各 k 的误认曲线、命中曲线和原始案例']]))
    add('本项目是上下文内的视觉行为评测：每题生成时学习图仍在输入中，模型权重不更新。它不测试跨会话长期记忆，也没有把中间隐藏状态、探针结果或神经机制作为证据。当前范围不包含独立的上下文距离或额外视觉干扰图研究。旧版 `PROJECT_REPORT.md` 的历史数据不与 v2 混合。')
    add('## 2. 数据集与已完成实验')
    add(table(['属性','取值','数量'],[
        ['颜色','红、黄、蓝、绿、紫、橙','6'],['纹理','纯色、条纹、圆点、网格','4'],['形状','圆形、三角形、正方形、星形、五边形、六边形','6']]))
    add('共有 144 种完整属性组合。二维物体由 Pillow 绘制，三维物体使用挤出轮廓、纹理和深度遮挡的程序化渲染。所有图片都是受控合成材料，不是自然照片。学习图和查询图采用确定性但不同的渲染参数，身份一致不要求像素相同；布局变化可能同时改变图片数量和视觉输入规模。')
    counts=[]
    levelnames={'L0':'单物体属性识别','L1':'基础绑定与普通诱饵','L2':'严格三元组绑定','L3':'数量与每图布局','L4':'空间与共同出现关系','L5':'三维、视角与遮挡'}
    for level,name in levelnames.items():
        ns=[sum(v['observed'] for key,v in baseline['splits'][split]['by_level_question'].items() if key.startswith(level+'/')) for split in ('dev','test')]
        counts.append([level+' '+name,f'{ns[0]:,}',f'{ns[1]:,}','Qwen 全部完成'])
    add(table(['v2 任务','dev 题数','test 题数','完成范围'],counts))
    add(table(['补充实验','Qwen 正式 test','LLaVA 正式 test','比较原则'],[
        ['二维校准＋L2 的 Q1/Q2/Q3','1,680（来自原基准）','1,680','完全相同 trial_id'],
        ['六物体 0–3 梯度 Q1/Q3','3,840','3,840','完全相同 trial_id'],
        ['六物体梯度 Q2','1,920','未运行','仅报告 Qwen，不能声称双模型复现']]))
    add('原基准与梯度各有 320 个随机组，按整组分为 80 个 dev 和 240 个 test。两套数据使用不同生成种子，是不同实验中的组；同组的世界、查询形式和条件不跨 dev/test。Qwen 共完成 75,600 道正式 test（69,840＋5,760），但不会把不同题型混成一个总准确率。LLaVA 正式同题比较只覆盖上表的 5,520 题，不代表它已完成全部 v2。四物体 1–3 梯度代码已实现，但本报告没有该设计的正式运行结果。')
    add('原基准生成种子为 20261022，梯度为 20261028，推理固定种子为 20261022。Qwen 原基准正式 test 有 436 条无效回答（0.62%），均按错误计分；这不属于缺失题或程序失败。LLaVA 正式范围的无效回答为 28 条，全部来自二维校准，其余二元判断均能按规则解析。')
    add('LLaVA 最初启动过全量任务，观察到选择题频繁产生不符合格式的长篇描述后，为加快获得核心结果而停止全量作业。后续范围只按任务类型选择，保留完整的 240 个 test 组，没有按某道题答对与否筛选。核心结果复用 85 条已完成的同题回复，并补齐剩余题目；原始全量阶段记录保留，未完成任务不报告正式完整成绩。这是观察结果后的范围调整，应如实披露。')
    add('## 3. 输入、输出与评分协议')
    fig('14_protocol','图 1｜从题目生成到同题统计的完整流程。标签只在评分阶段使用。')
    add('应用层为每题创建一条独立的用户多模态消息，依次放入规则、带序号的学习图片、学习结束文字、查询内容。前一题的模型回答不会带入后一题。Qwen 使用原生聊天模板；LLaVA 使用 USER/ASSISTANT 包装并保留图文交错顺序。')
    add('共同规则的实际英文原文：\n\n```text\n'+cases[0]['input']['instruction']+'\n```')
    add('结构示例：\n\n```text\nUSER\n  [任务规则]\n  Study image 1: [学习图片 1]\n  Study image 2: [学习图片 2]\n  ...\n  End of study. Answer the following query.\n  [查询文字和／或查询图片]\nASSISTANT\n  [模型生成的回答]\n```')
    add(table(['问法','查询输入','规定输出'],[
        ['CAL 校准','单张学习图后要求识别颜色、纹理、形状','恰好含三个指定字段的 JSON'],
        ['Q1 图片判断','候选图片＋是否完整出现的问题','YES / NO'],
        ['Q2 文字判断','候选的三属性文字描述，无候选图；学习仍是图片','YES / NO'],
        ['Q3 图文判断','候选图片＋三属性文字描述','YES / NO'],
        ['Q4 同图二选一','一张包含 A、B 候选的图片','A / B'],
        ['Q5 分图二选一','A、B 各自一张候选图片','A / B'],
        ['Q6 四选一','一张包含 A–D 候选的图片','A / B / C / D'],
        ['Q7 纹理回忆','给颜色和形状，询问该物体的纹理','SOLID / STRIPES / DOTS / GRID'],
        ['Q8 关系判断','描述两个物体及待判断的空间／共现关系','YES / NO']]))
    add('例如 Q1 的真实查询文字为：\n\n```text\nDid this complete object occur in the study set? Output only YES or NO.\n```\n\nQ2 在此前加上候选属性句，例如 `The candidate is a red solid square.`；Q3 同时提供候选图和同样的属性句。标准答案、题目 ID、私有物体属性列表和“强重组”等类型标签不会额外输入模型。报告中的中文解释只供读者阅读。')
    add('评分先去除首尾空白，对非 JSON 回答统一大小写；因此 `Yes` 可以解析为 YES，但 `YES because ...`、`A.`、越界选项或不合要求的 JSON 均计为无效，并按错误计入准确率。校准要求颜色、纹理、形状三个字段同时符合词表且全部正确。没有在看到模型结果后放宽解析规则，也没有把解释性长回复事后猜测成选项。')
    add(table(['指标','分母与含义','阅读要点'],[
        ['准确率','对应题目中严格判定正确的比例','无效输出计错，不跨题型混合'],
        ['目标命中率','真实目标中正确回答 YES 的比例','应与负例表现同时看'],
        ['负例误认率','完整候选未出现时，错误回答 YES 的比例','越低越好；不包括无效回答'],
        ['负例正确拒绝率','完整候选未出现时，正确回答 NO 的比例','越高越好；有无效输出时不等于 1 减误认率'],
        ['成对正确率','同一对应正负世界的两道题均答对的比例','可识别恒定回答造成的虚高单题成绩'],
        ['同题模型差','LLaVA 正确率减 Qwen 正确率','按完全相同 trial_id 配对，单位为百分点']]))
    add('主要置信区间使用 1,000 次按 cluster_id 的 bootstrap，保留同组内问法、世界和条件的关联；模型差和梯度差使用配对统计。独立统计单位是 240 个随机组，而不是把所有相关题目当作独立样本。全对或全错时经验区间可能退化为一个点，不表示总体没有不确定性。区间未作多重比较校正，边界差异不解释为普遍规律。已有汇总与同题子集重新汇总的 bootstrap 组遍历顺序可能不同，区间端点存在小幅 Monte Carlo 差异；各图和对应导出表使用同一份统计对象。')
    add('## 4. 模型、运行设置与比较边界')
    add(table(['设置','Qwen3-VL-8B-Instruct','LLaVA-1.5-7B'],[
        ['权重','本地 Qwen3-VL-8B-Instruct','llava-hf/llava-1.5-7b-hf；固定 commit'],
        ['生成','do_sample=False；最多 80 个新 token','相同'],
        ['数值与注意力','dtype=auto；SDPA','FP16；SDPA'],
        ['图片处理','原生动态分辨率，max_pixels=802816','先按 CLIP 均值补方，再原生缩放至 336×336'],
        ['多图顺序','规则、学习图、结束提示、查询','语义顺序一致，使用 LLaVA 的文本包装'],
        ['硬件','单张 NVIDIA RTX 3090','单张 NVIDIA RTX 3090'],
        ['依赖','torch 2.10.0+cu128；transformers 5.19.0','相同推理环境'],
        ['可比范围','原 v2 全套＋梯度 Q1/Q2/Q3','二维校准、严格 Q1/Q2/Q3、梯度 Q1/Q3']]))
    add('LLaVA 权重 commit 为 `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。它的原生上下文为 4,096 token；预检发现原全量基准中 4,480 道 dev＋test 的“8 个学习物体、每图 1 个”题超限，故不会通过截断原题来凑完整结果。本次精简的 5,520 道 test 均在支持范围内，没有上下文排除题。补方步骤用于防止中心裁剪切掉宽图上的物体，但有效物体分辨率与 Qwen 不同。')
    add('LLaVA 官方文档指出，多图输入技术上可行，但该模型没有专门针对同一提示中的多图进行训练。因此这里比较的是两个模型各自完整推理系统在同一受控题目上的表现，不能把差距全部归因于“记忆容量”或“绑定机制”。[LLaVA 模型卡](https://huggingface.co/llava-hf/llava-1.5-7b-hf)、[Transformers LLaVA 文档](https://huggingface.co/docs/transformers/model_doc/llava)。')
    add('## 5. Qwen 全套实验：目的、设计与结果')
    add('这一部分涵盖 Qwen 已完成的全部原基准。各实验的题目可能从不同查询维度重复使用相同随机组，不能把下面每个实验标题理解为互不重叠的数据集。尤其“查询方式”是跨 L1/L2/L3/L5 的分析维度。')
    fig('01_design','图 2｜真实严格正负世界。上下两题的查询物体相同；单属性与两两属性出现次数匹配，完整三元组是否出现不同。')
    old=Path('runs/bindmem_v2/qwen8b_report/visual_report.md').read_text()
    body=old[old.index('## 实验 1'):old.index('## 逐任务成功率')]
    body=body.replace('## 实验','### 实验').replace('](visual_figures/','](figures/').replace('73.12%','73.13%')
    # Number all figures by their order in this report.
    body=re.sub(r'图 ([2-8])｜',lambda m:f'图 {int(m.group(1))+1}｜',body).replace('图 1 展示','本章图 2 展示')
    add(body)
    add('### 全部问法的 Qwen 成绩总表')
    add(table(['问法','L1 基础绑定','L2 严格绑定','L3 多物体','L5 三维扩展'],[
        [QN[q]]+[pct(s['by_level_question'][f'{level}/{q}']['accuracy']['value']) for level in ['L1','L2','L3','L5']] for q in QN]))
    add('L2 每格 480 题，L3 每格 3,360 题，L5 每格 1,920 题。L1 的 Q1–Q3 各 5,760 题，Q4–Q6 各 2,880 题，Q7 为 720 题。基础绑定中的普通重组只保证候选属性分别熟悉，不等同于所有候选属性对均出现过的强重组；不能用普通负例的高拒绝率替代严格任务表现。完整分条件正确数、分母和区间见附表与 CSV。')
    add('## 6. 两个模型的正式同题比较')
    add('比较使用相同的 240 道二维校准题，以及相同的严格绑定 Q1/Q2/Q3 各 480 题，合计 1,680 题。没有将 LLaVA 精简集与 Qwen 全套汇总直接对比。')
    fig('09_model_comparison','图 10｜正式同题准确率。左侧为三属性联合识别；右侧为严格绑定判断。误差线为按组 95% 区间。')
    add(table(['任务','题数','Qwen 准确率 [95% CI]','LLaVA 准确率 [95% CI]'],[
        [('二维校准' if i==0 else QN[['Q1','Q2','Q3'][i-1]]),core['by_condition'][key]['a']['accuracy']['n'],measure(core['by_condition'][key]['a']['accuracy'],True),measure(core['by_condition'][key]['b']['accuracy'],True)] for i,key in enumerate(ck)]))
    add('LLaVA 二维联合识别为 161/240（67.08%）。其中 28 条输出无效，另有 51 条格式有效但至少一个属性错误；仅在有效回答中计算也只有 161/212（75.94%），这一诊断值不替代主评分。Qwen 同一批二维校准题为 240/240。因而双模型差距从单物体识别和输出遵循阶段就已存在，不能直接把后续差距解释为纯粹的绑定差。')
    fig('10_response_bias','图 11｜把目标命中和强重组误认拆开。蓝色为 Qwen，橙色为 LLaVA；左图越高越好，右图越低越好。')
    add(table(['问法','Qwen 目标命中','Qwen 强重组正确拒绝','LLaVA 目标命中','LLaVA 强重组正确拒绝'],[
        [QN[q]]+[measure(core['by_condition'][f'L2/{q}/2d/n4/per1'][side][metric],True) for side,metric in [('a','target_hit_rate'),('a','negative_correct_rejection'),('b','target_hit_rate'),('b','negative_correct_rejection')]] for q in ['Q1','Q2','Q3']]))
    add('图片判断中，LLaVA 对 480 题全部回答 YES：240 个目标全对、240 个强重组负例全错，因此准确率恰好 50%。图文判断则几乎全答 YES，目标命中与误认均为 239/240，准确率也为 50%。文字判断表现为较强的接受倾向；相对地，Qwen 文字判断几乎全部拒绝，只有 2/240 个目标被命中。相似的约 50% 准确率背后，是方向相反的回答偏向。')
    diffs=[]
    for key in ck:
        d=core['by_condition'][key]['delta_b_minus_a'];lo,hi=d['ci95']
        diffs.append([key.split('/')[1],f'{d["delta_b_minus_a"]*100:+.2f}',f'[{lo*100:+.2f}, {hi*100:+.2f}]'])
    add(table(['任务','LLaVA 减 Qwen（百分点）','配对差的 95% 区间'],diffs))
    add('图片与图文判断的差异区间均在 0 以下，说明在当前同题协议中 Qwen 的严格正确率较高；文字判断差异区间跨 0，不能把 LLaVA 数值上略高解释为清晰优势。二元准确率接近 50% 不自动等价于随机猜测；应优先结合命中、误认和真实回复解释。')
    add('## 7. 严格匹配的重组难度梯度')
    add('### 7.1 目的与组合约束')
    add('原 L2 只检验最强的三属性对重组，本实验进一步问：在单属性熟悉度相同时，候选的已见属性对从 0 增加至 3，误认是否随之改变？这里 k 表示已见属性对的“种类数”，三个可能的属性对是颜色—纹理、颜色—形状、纹理—形状。负例的完整三元组始终没有出现。')
    add('最初设想固定 4 个学习物体并保持候选属性频数相同，但无法同时实现 k=0 与 k=3：在三元组未出现的前提下，覆盖三种属性对要求每种候选属性至少出现两次，共至少 6 次；而 k=0 时每个物体最多包含一种候选属性，4 个物体最多容纳 4 次。因此完整梯度改为固定 6 个学习物体。这是明确的组合约束，并非事后挑选难度。')
    add(table(['控制项','正式六物体设计'],[
        ['学习规模','每题 6 个互不相同的物体；每张学习图 2 个；共 3 张学习图'],
        ['候选','同组各 k 使用相同候选三元组和相同查询图'],
        ['单属性频数','每维 3 个取值、各出现 2 次；所有一阶边际均匹配'],
        ['属性对','k=0、1、2、3；每个已见候选属性对恰好出现 1 次'],
        ['渲染','均为二维；固定布局规则、渲染方式及同组参数'],
        ['正负配比','每个 k、每个问法均为 1:1'],
        ['正例','同组各 k 重复同一正例输入，不作为独立正例证据'],
        ['正式比较','每个问法每个 k 为 480 题：240 目标＋240 负例']]))
    add('不同 k 下学习物体的联合组合和某些其他属性对关系必然改变；该设计控制上述明确列出的变量，不能声称只有一个抽象变量发生变化、所有像素或所有二阶统计均完全相同。它与原 L2 的“正负世界二阶边际也相同”的控制目的不同，分别报告。')
    fig('11_gradient_design','图 12｜来自第一组正式梯度数据的真实学习材料。各行候选相同，候选三个属性各出现两次；已见候选属性对种类数依次为 0、1、2、3，物体组合随之改变。')
    add('### 7.2 误认曲线与目标命中')
    fig('12_gradient_results','图 13｜正式梯度结果。左侧是负例误认率，右侧是正负合计准确率。两列回答不同问题，不能只读准确率。')
    gradient_table=[]
    for q in ['Q1','Q3']:
        for k in range(4):
            c=grad['by_condition'][f'G/{q}/n6/k{k}']
            gradient_table.append([QN[q],k,measure(c['a']['negative_false_acceptance'],True),measure(c['b']['negative_false_acceptance'],True),pct(c['a']['accuracy']['value']),pct(c['b']['accuracy']['value'])])
    add(table(['问法','已见属性对 k','Qwen 误认率 [95% CI]','LLaVA 误认率 [95% CI]','Qwen 准确率','LLaVA 准确率'],gradient_table))
    add('Qwen 图片查询的目标命中在所有 k 均为 230/240（95.83%），图文查询为 196/240（81.67%）。由于对应正例输入相同，这些不是四次独立复现。在目标命中保持相同的同时，负例误认随属性对数量上升：图片为 0/240、2/240、25/240、94/240；图文为 0/240、1/240、18/240、53/240。这个结果比单独比较普通负例与强负例更有力地支持“在一阶熟悉度匹配时，二阶组合熟悉度与误认相关”。它仍不构成模型内部机制的直接证据。')
    effect=[]
    for model in ['Qwen','LLaVA']:
        for q in ['Q1','Q3']:
            d=contrasts[f'{model}/{q}/k0_to_k3'];lo,hi=d['ci95'];effect.append([model,QN[q],f'{d["delta"]*100:+.2f}',f'[{lo*100:+.2f}, {hi*100:+.2f}]'])
    add(table(['模型','问法','k=3 减 k=0 的误认差（百分点）','配对 95% 区间'],effect))
    add('LLaVA 图片查询在全部 k 下均为全答 YES，负例误认率 100%；图文查询也接近全答 YES。这种饱和的接受偏向让当前协议下的梯度失去辨别力。不能据此说 LLaVA“不受重组难度影响”，更不能将饱和误认解释为记忆能力稳定。')
    add('### 7.3 Qwen 文字梯度的补充诊断')
    add('Qwen 还完成了梯度 Q2，但 LLaVA 精简阶段没有运行这一部分。Qwen 在四个 k 下都只认出 16/240 个目标（6.67%），负例全部正确拒绝，总准确率均为 53.33%。这再次表明文字查询的低误认主要伴随强烈拒绝偏向，不能当作“比图片更可靠的绑定记忆”。完整结果见 `tables/qwen_gradient_conditions.csv`。')
    shutil.copy2('runs/bindmem_gradient_n6/qwen8b/scored/test/rates.csv',OUT/'tables/qwen_gradient_conditions.csv')
    add('## 8. 真实强重组误认：输入、回复与证据')
    add('下列案例直接来自正式 test 的原始记录，图片不由报告作者重画。案例 1 是按 trial_id 字典顺序最早的“双模型都误认”的 L2/Q1 强负例；案例 2 是最早的“Qwen 正确拒绝而 LLaVA 误认”的同类负例。它们用于解释错误形态，不代表随机抽样的发生比例。发生比例以上面的全量负例统计为准。')
    fig('13_real_errors','图 14｜两个实际同题案例。每行前四张为学习图，最后一张为候选图；两行标准答案均为 NO，顶部标注实际回复。')
    for i,c in enumerate(cases,1):
        r=c['label'];study=[o for sc in r['study_scenes'] for o in sc]
        add(f'### 案例 {i}：{describe_zh(r["probe"])}')
        add(f'题目 ID：`{c["trial_id"]}`。查询是图片形式，没有用属性文字额外提示模型。候选完整三元组未出现，因此标准答案为 **NO**。')
        add(table(['学习物体序号','完整属性'],[[j+1,describe_zh(o)] for j,o in enumerate(study)]))
        witnesses=[]
        for a,b,name in [(0,1,'颜色＋纹理'),(0,2,'颜色＋形状'),(1,2,'纹理＋形状')]:
            indices=[j+1 for j,o in enumerate(study) if (o[a],o[b])==(r['probe'][a],r['probe'][b])]
            witnesses.append([name,'、'.join(map(str,indices))])
        add(table(['候选属性对','来源学习物体'],witnesses))
        add('模型原始输出：\n\n```text\nQwen: '+c['qwen_raw']+'\nLLaVA: '+c['llava_raw']+'\nGold: NO\n```')
        add('三个属性对都能在学习集合中找到来源，但它们分散在不同的物体上，不能推出完整三元组已经出现。回复 YES 是格式有效但语义错误的误认；它与输出一段描述导致的格式失败不同。')
    add('还提供各模型独立的案例网页与完整证据 JSON：包含原始提示、图片 SHA、来源预测文件、模型运行指纹和对应正例。全部案例都是受控合成图像上的真实模型回复，不应称为自然照片中的真实场景案例。')
    add('## 9. 综合讨论与限制')
    add(table(['观察','当前证据支持的解释','不能据此推出'],[
        ['Qwen 单物体与基础图片任务很高，严格重组仍误认','基础属性可识别不保证复杂组合能被区分','所有识别困难均已排除，或已定位到内部绑定机制'],
        ['Qwen 图片误认随 k 上升','匹配一阶熟悉度后，属性对熟悉度与误认相关','只存在唯一因果变量，或所有 VLM 都有同样曲线'],
        ['文字与图片的约 50% 分数可以来自相反偏向','必须拆开命中和误认，而非只看总分','约 50% 就等于随机猜测'],
        ['Qwen 数量／布局结果不单调','当前配置不支持简单“越多越差”','物体数量没有影响，或已估计独立记忆容量'],
        ['Qwen 查询遮挡后全答 NO','遮挡伴随识别／回答偏向问题','可以只归因于忘记学习内容'],
        ['LLaVA 识别和多图任务表现偏低','当前完整推理协议下弱于 Qwen 的若干任务','参数规模或模型家族本身就是差异的唯一原因']]))
    add('适用范围有六点限制。第一，只有两个具体 checkpoint，不能推广为所有视觉语言模型的排序。第二，图片是有限属性空间的合成物体，尚无自然图片验证。第三，学习图仍在当前上下文中，不能将结果等同于长期记忆能力。第四，模型的视觉预处理、分辨率、提示包装和多图训练适配程度不同。第五，LLaVA 的实验范围在观察到输出问题后进行了缩减，报告应区分正式核心比较与中止的探索性全量运行。第六，多条件检验与退化 bootstrap 区间限制了统计解释，置信区间不代表所有系统误差都被量化。')
    add('若继续扩展，优先考虑在相同完整协议下加入有明确多图训练能力的模型，并先预设任务范围及格式诊断标准；再增加自然图像或更丰富属性材料。可以另立实验研究输出约束或提示改变，但不能把改变后的 LLaVA 成绩与当前 Qwen 成绩直接混作同一协议。本文不把这些尚未运行的工作写成已完成结果。')
    add('## 10. 可复现材料、版本与结果索引')
    add(table(['材料','位置／用途'],[
        ['原基准','data/bindmem_v2：输入与标签分离，manifest 与图片哈希'],
        ['六物体梯度','data/bindmem_gradient_n6：候选频数、属性对计数及生成约束'],
        ['原实现','bindmem/：生成、渲染、验证、推理、严格解析和统计'],
        ['补充实现','bindmem_extra/：匹配梯度、LLaVA 适配、同题评分、案例导出'],
        ['完整结果','原六批 Qwen 运行及 runs/bindmem_gradient_n6/qwen8b'],
        ['双模型正式比较','runs/bindmem_priority/core_comparison 与 gradient_comparison'],
        ['本报告数据','tables/：原始精度 CSV、配对梯度差、图表数值'],
        ['本报告证据','evidence/：真实案例、实际梯度示例与来源指纹'],
        ['生成脚本','scripts/build_complete_project_report.py；只读取结果，不运行推理']]))
    add('模型推理环境无需变化。报告使用已有 base Python 的 Matplotlib/Markdown；PDF 排版库安装在独立临时目录 `/tmp/bindmem-report-deps`，不修改 `qwen3vl`。重新生成报告：\n\n```bash\n/ceph/home/shizhongyu/anaconda3/bin/python scripts/build_complete_project_report.py\n```\n\n若临时 PDF 依赖目录已被清理，仅为报告补装：\n\n```bash\n/ceph/home/shizhongyu/anaconda3/bin/python -m pip install \\\n  --target /tmp/bindmem-report-deps reportlab pypdf\n```')
    add('原始图片、预测和模型权重不被报告脚本修改。`source_audit.json` 记录本次使用的数据文件与运行文件哈希、核对题数及图表指标数；`evidence/shared_error_cases.json` 和 `evidence/gradient_design.json` 可逐图追溯。模型卡与处理器文档用于说明模型使用条件，实验数值全部来自本地正式结果。')
    add('## 11. 结论')
    add('Qwen 在基础识别、简单图片绑定和左右／上下关系上较可靠，但强重组判断暴露出“识别熟悉属性”与“确认完整物体出现过”之间的差距；匹配的一阶频数下，属性对越熟悉，Qwen 图片／图文查询的误认越高。查询形式能改变结果，也能引入明显的接受或拒绝偏向。LLaVA 在当前多图协议下表现出近乎恒定的接受偏向，并存在较弱的基础属性联合识别，尚不能据此将模型差距归因于单一绑定机制。项目最核心的结论是：评测绑定能力需要严格匹配的负例、目标命中与误认的联合报告，以及可追溯的真实输入输出。')
    add('## 附录导航')
    add('- [Qwen 逐任务完整分数附表](appendix_qwen_tasks.md)：含目标命中、各类诱饵、数量／布局、关系、三维及成对正确率。\n- [实际输入输出详解](appendix_input_output.md)：17 个真实题目示例及评分依据。\n- 实际输入输出网页版 `input_output_examples.html` 可在本地生成；GitHub 提供上述 Markdown 和图片。\n- [Qwen 真实误认案例](cases_qwen.html)、[LLaVA 严格绑定案例](cases_llava_core.html)、[LLaVA 梯度案例](cases_llava_gradient.html)。\n- [读者简版](brief.md)、[图表目录](figures/)、[完整精度表格目录](tables/)。')
    report='\n\n'.join(parts)+'\n'
    # Portable appendices retain the actual input images and unmodified case evidence.
    old=ROOT/'runs/bindmem_v2/qwen8b_report'
    for folder in ['io_examples','task_figures']:
        shutil.copytree(old/folder,OUT/folder,dirs_exist_ok=True)
    task_md=(old/'task_success_report.md').read_text().replace('`task_success_rates.csv`','`tables/qwen_task_success_rates.csv`').replace('`input_output_report.html`','`input_output_examples.html`')
    (OUT/'appendix_qwen_tasks.md').write_text(task_md)
    io_md=(old/'input_output_report.md').read_text()
    (OUT/'appendix_input_output.md').write_text(io_md)
    shutil.copy2(old/'input_output_report.html',OUT/'input_output_examples.html')
    for name,path in [('qwen',old/'strong_error_cases'),('llava_core',ROOT/'runs/bindmem_priority/llava7b_core/strong_error_cases'),('llava_gradient',ROOT/'runs/bindmem_priority/llava7b_gradient/strong_error_cases')]:
        shutil.copy2(path/'cases.html',OUT/f'cases_{name}.html')
        shutil.copy2(path/'cases.json',OUT/'evidence'/f'cases_{name}.json')
        source(path/'cases.json')
    source(old/'io_examples/cases.json')
    (OUT/'report.md').write_text(report)
    # A genuinely shorter reader's version, with enough protocol to interpret the numbers.
    brief='''# 模型能看见属性，是否能认出完整物体？

**BindMem 项目图文简报 · 2026-10-29**

## 做了什么，为什么做

本项目用可控制颜色、纹理和形状的合成物体，评测视觉语言模型能否区分“某些属性见过”与“这个完整物体见过”。每题把学习图片和查询放进同一条用户消息，模型独立作答，学习图仍在上下文中。这是上下文内的行为评测，不是跨会话长期记忆，也不训练模型或做探针实验。

原基准包含单物体识别、基础绑定、严格重组、数量与布局、空间关系、三维／视角／遮挡六类任务，查询包括图片、文字、图文、选择和属性回忆。Qwen3-VL-8B 完成原基准 69,840 道正式测试和新增梯度 5,760 道正式测试；LLaVA-1.5-7B 完成其中 5,520 道完全相同题目的核心比较。开发集结果不混入这些成绩。

## 关键设计：完整物体没有出现，但属性对都出现过

最难的强重组负例要求候选的颜色＋纹理、颜色＋形状、纹理＋形状都在学习集合中出现过，但三种属性从未组成同一个物体。因此，正确答案仍然是 NO。仅凭“看起来熟悉”回答 YES，就是一次重组误认。

![两个模型的真实强重组案例](figures/13_real_errors.png)

每行前四张是学习图，最后一张是候选。第一行两个模型都误认；第二行 Qwen 正确拒绝、LLaVA 仍误认。图中回复来自实际预测文件；这些是为解释现象选出的案例，错误发生比例以全量统计为准。

## 哪些表现好，哪些表现不好

Qwen 的基础能力较强：单物体三属性联合识别为 99.90%，基础图片判断为 99.25%，左右与上下关系均为 100%。困难集中在严格重组、某些查询形式和视觉变化：严格图片判断准确率降至 71.46%，前后关系为 79.38%，共同出现关系为 78.96%。遮挡查询中的图片／图文判断全部回答 NO，不能解释成“成功避免了误认”。

![相同题目的双模型比较](figures/09_model_comparison.png)

二维单物体校准中，Qwen 为 100%，LLaVA 为 67.08%；后者 240 题有 28 个无效输出、51 个格式有效但属性错误的回复。严格绑定的 480 道图片题中，Qwen 为 71.46%，LLaVA 为 50%；图文题分别为 78.54% 和 50%。条形图误差线为按随机组重采样的 95% 置信区间。

总准确率不足以判断模型是否记住了物体。以下每格分母均为 240，目标与负例各占一半：

'''
    brief+=table(['模型与问法','真实目标命中率（越高越好）','强重组误认率（越低越好）'],[
        [model+'／'+QN[q],pct(core['by_condition'][f'L2/{q}/2d/n4/per1'][side]['target_hit_rate']['value']),pct(core['by_condition'][f'L2/{q}/2d/n4/per1'][side]['negative_false_acceptance']['value'])]
        for side,model in [('a','Qwen'),('b','LLaVA')] for q in ['Q1','Q2','Q3']])
    brief+='''

Qwen 的严格图片题能认出 99.17% 的真实目标，但也接受了 56.25% 的强重组负例。Qwen 的文字题则几乎全答 NO，目标命中仅 0.83%。LLaVA 的图片题全部答 YES，命中和误认同时为 100%。因此，约 50% 的准确率可以来自完全相反的回答偏向，不能一概称为随机猜测。

## 新增梯度：属性对越熟悉，误认是否越多

我们固定学习物体数、候选单属性频数、正负比例和渲染方式，将负例已出现的属性对数量设为 0、1、2、3。四个物体无法在这些约束下覆盖完整梯度，因此正式实验采用六个物体、每图两个：候选的每种属性均出现两次，每种已见属性对出现一次。候选图片保持相同；学习物体的组合关系随条件改变，并非整套图片像素相同。

![严格匹配梯度结果](figures/12_gradient_results.png)

Qwen 图片查询的误认率随已见属性对数量上升，依次为 **0.00%、0.83%、10.42%、39.17%**；图文查询为 **0.00%、0.42%、7.50%、22.08%**。每种问法的对应正例输入完全相同，目标命中分别保持在 95.83% 和 81.67%。这说明单属性频数匹配后，属性对熟悉度仍与误认相关，但尚不是模型内部机制的直接证明。

LLaVA 在图片梯度中全答 YES，图文梯度也几乎全答 YES。饱和的接受偏向使当前协议下的梯度无法区分其难度变化，不能解释为“难度不会影响它”。

## 怎样使用这些结论

本项目支持的结论是：**能识别单个属性，不保证能正确判断完整组合；评测必须同时报告目标命中与重组误认，并展示真实输入输出。** Qwen 在基础任务上可靠，但在两两属性都熟悉的负例上仍容易误认；LLaVA 在当前协议下存在明显接受偏向。

比较范围有明确限制：只测试两个 checkpoint、合成图像和当前上下文；两个模型的原生分辨率、提示包装和多图训练适配不同。LLaVA 并未专门训练同一提示中的多图输入，因此结果比较的是完整推理协议，不能归因于单一绑定机制。[官方模型文档](https://huggingface.co/docs/transformers/model_doc/llava)。LLaVA 的精简范围是在观察到全量选择题格式问题后决定的，保留全部 240 个正式随机组，没有按答对与否筛题。置信区间按随机组进行 1,000 次 bootstrap，未作多重比较校正。

[阅读全文](report.md)可查看实验目的、实际英文提示、完整逐任务成绩、同题置信区间与误认证据；[输入输出附录](appendix_input_output.md)含 17 个真实题目，[逐任务附表](appendix_qwen_tasks.md)列出各条件正确数／总题数。
'''
    (OUT/'brief.md').write_text(brief)
    def html_document(md,title):
        converter=markdown.Markdown(extensions=['tables','fenced_code','toc'])
        body=converter.convert(md)
        def embed(m):
            p=OUT/html.unescape(m.group(1))
            mime='image/svg+xml' if p.suffix=='.svg' else 'image/png'
            return 'src="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
        body=re.sub(r'src="([^"]+)"',embed,body)
        for name in ['report','brief','appendix_qwen_tasks','appendix_input_output']:
            body=body.replace(f'href="{name}.md"',f'href="{name}.html"')
        style='''body{margin:0;background:#edf2f6;color:#223543;font:17px/1.85 system-ui,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}main{max-width:1050px;margin:28px auto;padding:48px 6%;background:white;border-radius:14px;box-shadow:0 8px 40px #173e5d12}h1{font-size:34px;line-height:1.45;color:#163f5d}h2{margin-top:2em;border-bottom:2px solid #d6e5ef;padding-bottom:.25em;color:#163f5d}h3{color:#24699c;margin-top:1.6em}p{margin:.9em 0}a{color:#1a659b}img{display:block;max-width:100%;height:auto;margin:24px auto}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:20px 0}th,td{border:1px solid #d2dfe7;padding:9px 11px;text-align:left;vertical-align:top}th{background:#e7f0f6}tr:nth-child(even){background:#f6f9fb}pre{background:#eef3f7;padding:18px;border-radius:7px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}code{font-family:ui-monospace,monospace;font-size:.88em;color:#31566d}details{border:1px solid #cfdee8;padding:16px 24px;border-radius:8px;background:#f7fafc}.toc ul{margin:3px 0}.table-scroll{overflow-x:auto}em{color:#5b6b76}footer{margin-top:3em;border-top:1px solid #ccd9e2;padding-top:1em;font-size:14px;color:#587081}@media(max-width:700px){main{padding:24px 5%;margin:0;border-radius:0}h1{font-size:27px}table{min-width:650px}body{font-size:16px}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0}details{display:none}h2,h3{break-after:avoid}img,tr{break-inside:avoid}}'''
        body=body.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
        return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+style+'</style></head><body><main><details><summary>目录：展开选择章节</summary>'+converter.toc+'</details>'+body+'<footer>BindMem · 2026-10-29 · 正文图片已全部内嵌；附录为随包提供的独立文件。</footer></main></body></html>'
    for name,body,title in [('report',report,'BindMem 完整项目报告'),('brief',brief,'BindMem 图文简报'),('appendix_qwen_tasks',task_md,'Qwen 逐任务成功率'),('appendix_input_output',io_md,'BindMem 实际输入输出')]:
        (OUT/f'{name}.html').write_text(html_document(body,title))
    print('Verified results and generated Markdown, HTML and figures.',flush=True)
    render_pdf(report,OUT/'report.pdf',OUT,'BindMem 完整项目报告 · Qwen 与 LLaVA')
    render_pdf(brief,OUT/'brief.pdf',OUT,'BindMem 图文简报')
    with PdfPages(OUT/'figure_atlas.pdf') as pdf:
        for p in sorted((OUT/'figures').glob('*.png')):
            im=Image.open(p);w,h=im.size
            page,ax=plt.subplots(figsize=(13.5,max(3.5,13.5*h/w+.55)))
            ax.imshow(im);ax.axis('off');page.subplots_adjust(left=.02,right=.98,top=.98,bottom=.065)
            page.text(.5,.018,p.stem,ha='center',fontsize=10,color='#526C7F')
            pdf.savefig(page);plt.close(page)
    for p in [Path(__file__),ROOT/'scripts/report_pdf.py',ROOT/'scripts/build_bindmem_visual_report.py']:
        source(p.relative_to(ROOT))
    audit_doc=dict(generated_at=datetime.now(timezone.utc).isoformat(),report_date='2026-10-29',
        verified_test_rows=dict(qwen_v2=69840,qwen_gradient=5760,llava_core=1680,llava_gradient=3840),
        qwen_v2_invalid=436,paired_core=core['shared'],paired_gradient=grad['shared'],
        comparison_test_clusters=240,figure_metric_rows=len(audit),
        ci_policy='cluster bootstrap; 1000 replicates; paired contrasts; source objects retained; no multiplicity adjustment',
        sources=sources)
    write_json(OUT/'source_audit.json',audit_doc)
    index='''# BindMem 完整报告交付包

日期：2026-10-29。正式结果：Qwen 原基准 test 69,840 题、梯度 test 5,760 题；LLaVA 同题 test 5,520 题。

| 阅读用途 | 文件 |
| --- | --- |
| 完整内容、可编辑 | [完整 Markdown](report.md) |
| 浏览器阅读、单文件分享正文 | 本地导出 `report.html`，正文图片全部内嵌；大型附件不上传 GitHub |
| 打印、提交、全文搜索 | 本地导出 `report.pdf`，包含完整正文；PDF 不上传 GitHub |
| 面向不了解项目的读者 | [图文简版 MD](brief.md)、[HTML](brief.html)；PDF 为本地导出 `brief.pdf` |
| 单独使用图表 | [14 张 PNG/SVG 图](figures/)；图集 PDF 为本地导出 `figure_atlas.pdf` |
| 检查每项分数 | [逐任务附表 MD](appendix_qwen_tasks.md)、[HTML](appendix_qwen_tasks.html) |
| 查看真实输入输出 | [17 例说明 MD](appendix_input_output.md)；大型 HTML 为本地导出 `appendix_input_output.html` |
| 查看强重组错误 | [Qwen](cases_qwen.html)、[LLaVA 核心](cases_llava_core.html)、[LLaVA 梯度](cases_llava_gradient.html) |
| 核对数值与来源 | [完整精度 CSV/JSON](tables/)、[案例证据](evidence/)、[来源文件哈希](source_audit.json) |
| 一次性打包分享 | 本地导出 `report_bundle.zip`；不上传 GitHub |

Markdown 的图片使用相对路径，请保留整个目录结构。每个 HTML 的正文图片均已内嵌，可单独打开；链接到其他报告或证据时需保留对应附件。CSV 保存原始精度，正文按百分比保留两位；旧汇总与同题重新汇总的 bootstrap 随机组遍历顺序可能不同，置信区间有少量 Monte Carlo 差异。

重新生成：在仓库根目录运行 ` /ceph/home/shizhongyu/anaconda3/bin/python scripts/build_complete_project_report.py `。PDF 依赖仅位于 `/tmp/bindmem-report-deps`；若被清理，可用同一 Python 执行 `-m pip install --target /tmp/bindmem-report-deps reportlab pypdf`。不改变模型推理环境，不需要 GPU，不运行模型。

本交付包含图表与案例证据，不包含模型权重和全量原始数据。原始运行、生成／推理脚本及数据在项目仓库，完整报告第 10 节提供索引。
'''
    (OUT/'README.md').write_text(index)
    prefix=OUT.relative_to(ROOT).as_posix()+'/'
    root_report=re.sub(r'(!?\[[^\]]*\]\()([^\s)]+)(\))',lambda m:m.group(1)+(m.group(2) if m.group(2).startswith(('http://','https://','#')) else prefix+m.group(2))+m.group(3),report)
    (ROOT/'PROJECT_REPORT_COMPLETE.md').write_text(root_report)
    with zipfile.ZipFile(OUT/'report_bundle.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,Path(OUT.name)/p.relative_to(OUT))
    print(json.dumps(dict(output=str(OUT),files=sum(p.is_file() for p in OUT.rglob('*')),figures=len(list((OUT/'figures').glob('*.png'))),audited_sources=len(sources)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
