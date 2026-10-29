#!/usr/bin/env python3
"""Build a Chinese experiment-by-experiment report from verified behavior results.
Run with /ceph/home/shizhongyu/anaconda3/bin/python (existing matplotlib + markdown).
Does not modify the inference environment, input data, or model predictions.
"""
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
from collections import Counter

os.environ.setdefault('MPLCONFIGDIR', '/tmp/bindmem-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/bindmem-font-cache')
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
import markdown
from bindmem_report_io import build_io_section
from bindmem_report_tasks import build_task_section

FONT = '/usr/share/fonts/google-droid/DroidSansFallback.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
    'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 14,
    'axes.labelsize': 11, 'figure.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white'})
COLORS = ['#24699C', '#D67A28', '#249382', '#815A9D', '#B34B64', '#6B7789', '#B39731']
QN = {'Q1':'图片判断','Q2':'文字判断','Q3':'图文判断','Q4':'同图二选一',
      'Q5':'分图二选一','Q6':'四选一','Q7':'属性回忆'}
LEVELS = {'L1':'基础绑定','L2':'严格绑定','L3':'多物体绑定','L5':'三维扩展'}
MODES = ['2d','3d_fixed','3d_cross','3d_occluded']
MN = ['二维','固定视角三维','跨视角三维','查询图遮挡']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',default='runs/bindmem_v2/qwen8b_report')
    parser.add_argument('--data',default='data/bindmem_v2')
    args=parser.parse_args()
    root=Path(args.report_dir); data=Path(args.data)
    result=json.loads((root/'summary.json').read_text());test=result['splits']['test']
    assert test['complete'] and test['observed']==69840
    assert result['splits']['dev']['observed']==23280
    cells=test['by_level_question'];conditions=test['by_condition']
    paired=test['paired_formats'];design=test['paired_design']
    dest=root/'visual_figures';dest.mkdir(exist_ok=True)
    plotted=[]; images={}; pdf=PdfPages(root/'实验结果图集.pdf')

    def audit(key,m):
        assert m['value'] is not None and 0<=m['value']<=1
        lo,hi=m['ci95'];assert 0<=lo<=hi<=1
        plotted.append([key,m['n'],m['value'],lo,hi])
    def err(m):
        v=m['value']*100;lo,hi=np.array(m['ci95'])*100
        # Bootstrap percentile intervals may not bracket the estimate in extreme cases.
        return v,np.array([[max(0,v-lo)],[max(0,hi-v)]])
    def bar(ax,x,m,color,key,width=.65,label=True):
        audit(key,m);v,e=err(m)
        ax.bar(x,v,width,color=color,zorder=3)
        ax.errorbar(x,v,yerr=e,fmt='none',ecolor='#263645',capsize=3,lw=1,zorder=4)
        if label:ax.annotate(f'{v:.1f}%',(x,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
    def percent_axis(ax,ylim=(0,110),chance=False):
        ax.set_ylim(*ylim);ax.set_ylabel('准确率（%）');ax.grid(axis='y',alpha=.2,zorder=0)
        ax.set_axisbelow(True)
        if chance:ax.axhline(50,color='#8C939A',linestyle='--',lw=1)
    def save(fig,name):
        fig.savefig(dest/f'{name}.svg',bbox_inches='tight')
        fig.savefig(dest/f'{name}.png',dpi=180,bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight')
        images[name]=dest/f'{name}.png'
        plt.close(fig)
    def point_series(ax,xx,metrics,color,label,keys):
        yy=np.array([m['value']*100 for m in metrics]);lo=np.array([m['ci95'][0]*100 for m in metrics]);hi=np.array([m['ci95'][1]*100 for m in metrics])
        for k,m in zip(keys,metrics):audit(k,m)
        ax.errorbar(xx,yy,yerr=[np.maximum(0,yy-lo),np.maximum(0,hi-yy)],color=color,marker='o',capsize=3,lw=2,label=label)
    def forest(ax,items,title):
        ax.axvline(0,color='#7F8992',lw=1,linestyle='--')
        for y,(label,key,v,color) in enumerate(items):
            d=v['delta_b_minus_a']*100;lo,hi=np.array(v['ci95'])*100
            ax.plot([lo,hi],[y,y],color=color,lw=2)
            ax.plot(d,y,'o',color=color)
            ax.text(.99,y-.19,f'{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=9)
            plotted.append([key,v['n_pairs'],v['delta_b_minus_a'],v['ci95'][0],v['ci95'][1]])
        ax.set_yticks(range(len(items)),[it[0] for it in items]);ax.invert_yaxis();ax.set_ylim(len(items)-.45,-.6)
        ax.set_title(title);ax.set_xlabel('准确率差（百分点）');ax.grid(axis='x',alpha=.2)
    def matrix(ax,values,rows,cols,title):
        im=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(len(cols)),cols);ax.set_yticks(range(len(rows)),rows);ax.set_title(title)
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>=65 else '#1B2731',fontsize=12)
        return im

    # Actual first test pair; selected by dataset order, not correctness.
    labels=[];pair_id=None
    for line in (data/'labels/test.jsonl').open():
        r=json.loads(line)
        if r['level']=='L2' and r['question']=='Q1':
            if pair_id is None:pair_id=r['pair_id']
            if r['pair_id']==pair_id:labels.append(r)
    assert len(labels)==2
    labels.sort(key=lambda r:r['answer']!='YES')
    ids={r['trial_id'] for r in labels};inputs={}
    for line in (data/'inputs/test.jsonl').open():
        r=json.loads(line)
        if r['trial_id'] in ids:inputs[r['trial_id']]=r
    a,b=[[tuple(x) for scene in r['study_scenes'] for x in scene] for r in labels]
    for dims in [(0,),(1,),(2,),(0,1),(0,2),(1,2)]:
        assert Counter(tuple(o[i] for i in dims) for o in a)==Counter(tuple(o[i] for i in dims) for o in b)
    qpaths=[[q['image'] for q in inputs[r['trial_id']]['query'] if q['type']=='image'] for r in labels]
    assert qpaths[0]==qpaths[1]
    fig,axes=plt.subplots(2,5,figsize=(13,5.3),layout='constrained')
    for row,r in enumerate(labels):
        inp=inputs[r['trial_id']];paths=inp['study_images']+qpaths[row]
        for col,path in enumerate(paths):
            axes[row,col].imshow(Image.open(data/path));axes[row,col].axis('off')
            axes[row,col].set_title(f'学习图片 {col+1}' if col<4 else '相同的查询图片',fontsize=11)
        axes[row,0].text(-.12,.5,'正确答案：'+('见过' if r['answer']=='YES' else '没见过'),transform=axes[row,0].transAxes,rotation=90,ha='center',va='center',fontsize=12,color=COLORS[row*2])
    fig.suptitle('实际题目示例：查询不变，学习图片中的完整组合改变',fontsize=16)
    save(fig,'01_design')
    (dest/'example_provenance.json').write_text(json.dumps({'trial_ids':[r['trial_id'] for r in labels],'pair_id':pair_id,'selection':'First L2/Q1 test pair in dataset order; not selected by model answer','single_and_pairwise_marginals_verified':True},indent=2))

    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for i,mode in enumerate(MODES):
        k=f'L0/CAL/{mode}/n1/per1';bar(ax,i,conditions[k]['accuracy'],COLORS[0],k)
    ax.set_xticks(range(4),MN);percent_axis(ax);ax.set_title('实验 1｜颜色、纹理、形状同时答对的比例')
    save(fig,'02_calibration')

    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
    types=['unrelated','near1','near2','critical'];xx=np.arange(4);w=.23
    for qi,q in enumerate(['Q1','Q2','Q3']):
        for i,typ in enumerate(types):
            m=cells['L1/'+q]['lure_false_acceptance'][typ]
            bar(axs[0],xx[i]+(qi-1)*w,m,COLORS[qi],f'L1/{q}/false_accept/{typ}',width=w*.9)
        axs[0].plot([],[],color=COLORS[qi],lw=7,label=QN[q])
        bar(axs[1],qi,cells['L1/'+q]['target_hit_rate'],COLORS[qi],f'L1/{q}/target_hit')
    axs[0].set_xticks(xx,['全新物体','仅 1 类属性见过','仅 2 类属性见过','属性重组'])
    axs[0].set_ylim(0,7.5);axs[0].set_ylabel('没见过却回答“见过”的比例（%）');axs[0].grid(axis='y',alpha=.2);axs[0].legend(frameon=False)
    axs[0].set_title('诱饵错误接受率：越低越好')
    axs[1].set_xticks(range(3),[QN[q] for q in ['Q1','Q2','Q3']]);percent_axis(axs[1]);axs[1].set_ylabel('见过且回答“见过”的比例（%）');axs[1].set_title('真实目标命中率：越高越好')
    save(fig,'03_basic')

    fig,axs=plt.subplots(1,3,figsize=(13.5,4.4),layout='constrained')
    for i,q in enumerate(['Q1','Q2','Q3']):
        v=cells['L2/'+q]
        bar(axs[0],i,v['accuracy'],COLORS[i],f'L2/{q}/accuracy')
        bar(axs[1],i-.17,v['target_hit_rate'],COLORS[0],f'L2/{q}/target_hit',width=.28)
        bar(axs[1],i+.17,v['lure_false_acceptance']['pairwise_complete'],COLORS[1],f'L2/{q}/false_accept',width=.28)
        bar(axs[2],i,v['pair_accuracy'],COLORS[i],f'L2/{q}/pair_accuracy')
    for i,ax in enumerate(axs):
        ax.set_xticks(range(3),['图片','文字','图文']);percent_axis(ax,chance=i==0)
    axs[0].set_title('单题准确率（每种问法 480 题）')
    axs[1].set_title('真实目标与重组诱饵（各 240 题）');axs[1].set_ylabel('回答“见过”的比例（%）')
    axs[1].plot([],[],color=COLORS[0],lw=7,label='真实目标');axs[1].plot([],[],color=COLORS[1],lw=7,label='重组诱饵');axs[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
    axs[2].set_title('正负两题都答对（240 对）');axs[2].set_ylabel('整对答对的比例（%）')
    save(fig,'04_strict')

    fig=plt.figure(figsize=(13,8),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.15,1])
    ax=fig.add_subplot(gs[0,:]);lev=list(LEVELS);qs=list(QN)
    vals=np.array([[cells[f'{l}/{q}']['accuracy']['value']*100 for q in qs] for l in lev])
    for l in lev:
        for q in qs:audit(f'{l}/{q}/accuracy',cells[f'{l}/{q}']['accuracy'])
    im=matrix(ax,vals,list(LEVELS.values()),[f'{q}\n{QN[q]}' for q in qs],'实验 4｜各问法准确率（%）；不同题型不合并排名')
    fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label='准确率（%）')
    for j,(a,b,title) in enumerate([('Q1','Q3','增加文字：图文判断 减 图片判断'),('Q4','Q5','改变候选布局：分图 减 同图')]):
        ax=fig.add_subplot(gs[1,j]);items=[(LEVELS[l],f'{l}/{a}_vs_{b}',paired[f'{l}/{a}_vs_{b}'],COLORS[i]) for i,l in enumerate(lev)]
        forest(ax,items,title);ax.set_xlim((-6,14) if j==0 else (-4,12))
    save(fig,'05_formats')

    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for j,q in enumerate(['Q1','Q2','Q3']):
        ax=axs[0,j]
        for n,color in [(4,COLORS[0]),(8,COLORS[1])]:
            pers=[1,2,4] if n==4 else [1,2,4,8];keys=[f'L3/{q}/2d/n{n}/per{per}' for per in pers]
            point_series(ax,np.arange(len(pers)),[conditions[k]['accuracy'] for k in keys],color,f'总数 {n}',keys)
        ax.set_xticks(range(4),[1,2,4,8]);ax.set_xlabel('每张学习图中的物体数');percent_axis(ax,ylim=(40,95),chance=True);ax.set_title(QN[q]+'：改变每图数量');ax.legend(frameon=False)
        items=[]
        for i,per in enumerate([1,2,4]):
            k=f'L3/{q}/per{per}/n4_vs_n8';items.append((f'每图 {per} 个',k,design[k],COLORS[i]))
        forest(axs[1,j],items,'固定每图数量：总数 8 减 总数 4');axs[1,j].set_xlim(-18,18)
    save(fig,'06_load')

    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
    keys=['L4/Q8/2d/n2/per2/left','L4/Q8/2d/n2/per2/above','L4/Q8/3d_fixed/n2/per2/front','L4/Q8/2d/n4/per2/cooccur']
    for i,k in enumerate(keys):bar(ax,i,conditions[k]['accuracy'],COLORS[i],k)
    ax.set_xticks(range(4),['左右\n2 个物体','上下\n2 个物体','深度前后\n2 个物体','同图共现\n4 个物体']);percent_axis(ax,chance=True);ax.set_title('实验 6｜物体身份不变，只改变关系（每类 480 题）')
    save(fig,'07_spatial')

    fig=plt.figure(figsize=(13,8.5),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05])
    ax=fig.add_subplot(gs[0,0])
    for q,color in [('Q1',COLORS[0]),('Q3',COLORS[2])]:
        keys=[f'L5/{q}/{m}/n4/per1' for m in MODES]
        point_series(ax,np.arange(4),[conditions[k]['accuracy'] for k in keys],color,QN[q],keys)
    ax.set_xticks(range(4),['二维','固定三维','跨视角','查询遮挡']);percent_axis(ax,ylim=(40,90),chance=True);ax.set_title('图片相关判断（每条件 480 题）');ax.legend(frameon=False)
    ax=fig.add_subplot(gs[0,1]);items=[]
    for i,q in enumerate(['Q1','Q3']):
        for mode,name in [('3d_cross','跨视角'),('3d_occluded','遮挡')]:
            k=f'L5/{q}/3d_fixed_vs_{mode}';items.append((QN[q]+' / '+name,k,design[k],COLORS[i*2]))
    forest(ax,items,'相对固定视角三维的准确率变化');ax.set_xlim(-33,20)
    ax=fig.add_subplot(gs[1,:]);vals=np.array([[conditions[f'L5/{q}/{m}/n4/per1']['accuracy']['value']*100 for q in qs] for m in MODES])
    for m in MODES:
        for q in qs:audit(f'L5/{q}/{m}/n4/per1',conditions[f'L5/{q}/{m}/n4/per1']['accuracy'])
    im=matrix(ax,vals,MN,[f'{q}\n{QN[q]}' for q in qs],'完整模式结果（%）：Q2 / Q7 不含查询图，三种三维条件是重复文本控制')
    fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label='准确率（%）');save(fig,'08_3d')
    pdf.close()
    with (dest/'figure_data.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric_or_comparison','n','estimate_fraction','ci_low','ci_high']);w.writerows(plotted)

    # Report text and HTML are generated below from the same verified statistics.
    def pct(x):return f'{x*100:.2f}%'
    def score(k):return pct(cells[k]['accuracy']['value'])
    def cscore(k):return pct(conditions[k]['accuracy']['value'])
    def delta(k,family=paired):
        v=family[k];lo,hi=np.array(v['ci95'])*100
        return f'{v["delta_b_minus_a"]*100:+.2f} 个百分点（95% 区间 {lo:+.2f} 至 {hi:+.2f}）'
    parts=[]
    def add(text):parts.append(text.strip())
    def chart(name,caption):add(f'![{caption}](visual_figures/{name}.png)\n\n*{caption}*')
    def protocol(purpose,how,n):
        add(f'**实验目的：** {purpose}\n\n**怎么做：** {how}\n\n**正式测试规模：** {n}')
    add('# 看清特征之后，模型能否判断完整物体？\n\nQwen3-VL-8B 物体绑定行为评测 · 实验设计与可视化结果 · 2026-10-26')
    add('## 项目概述\n\n本报告回答一个具体问题：当颜色、纹理和形状都能被识别时，模型是否仍会把“见过这些特征”误当成“见过这个完整物体”？例如见过红色条纹圆形和蓝色圆点星形，不代表见过红色条纹星形。正确对应同一物体的属性，称为“物体属性绑定”。下面按实验目的、操作方法和结果图逐项说明。所有成绩均来自正式测试集，不包含探针实验。')
    add('本报告对应 **BindMem v2**。此前 `PROJECT_REPORT.md` 的 864 题实验属于旧版本，属性空间、任务和题目结构均不同，保留为历史记录，不与本次结果合并。当前研究不包含独立的上下文距离或额外视觉干扰图实验。')
    add('## 数据集设计与制作\n\n程序根据属性组合生成图片，物体身份由三个属性共同定义。二维图由程序绘制；三维图使用带纹理的挤出网格、正交投影和深度遮挡，而非自然照片。学习图和查询图采用不同的确定性渲染参数，同一身份无需像素完全一致。单物体尺寸受控，多物体画布随布局增大，不能将所有条件称为固定单图尺寸。')
    add('| 属性 | 本版取值 | 数量 |\n|---|---|---:|\n| 颜色 | 红、黄、蓝、绿、紫、橙 | 6 |\n| 纹理 | 纯色、条纹、圆点、网格 | 4 |\n| 形状 | 圆、三角形、正方形、星形、五边形、六边形 | 6 |')
    counts=['| 任务 | 开发集题数 | 正式测试题数 |','|---|---:|---:|']
    names={'L0':'属性识别','L1':'基础绑定','L2':'严格绑定','L3':'多物体绑定','L4':'空间关系','L5':'三维扩展'}
    for level,name in names.items():
        ns=[sum(v['observed'] for k,v in result['splits'][split]['by_level_question'].items() if k.startswith(level+'/')) for split in ('dev','test')]
        counts.append(f'| {name}（{level}） | {ns[0]:,} | {ns[1]:,} |')
    counts.append('| 合计 | 23,280 | 69,840 |');add('\n'.join(counts))
    add('320 个独立随机组按组切分：80 组用于开发检查，240 组用于正式测试。同组的正负世界、问法和布局不跨切分。正负判断配平，选择题位置按组平衡。图像文件名不包含属性语义，标准答案与输入分离；文字或图文查询中必要的候选属性描述会正常提供。生成后验证图像完整性、答案唯一性以及严格配对的一阶/二阶属性频数。')
    add('## 研究怎样展开\n\n先检查基础识别，再比较不同诱饵，随后用严格配对任务检验完整组合；在相同任务内改变问法、数量和图片布局，最后扩展到空间关系与三维条件。各实验检验不同问题，不能把层级编号当成必然递增的难度，也不计算跨任务总分。')
    add('| 实验 | 想回答的问题 | 主要改变的因素 |\n|---|---|---|\n| 1. 属性识别 | 后续错误是否只是基础属性识别失败？ | 单物体的二维/三维呈现 |\n| 2. 基础绑定 | 模型是否更容易误认“熟悉属性的新组合”？ | 全新、部分熟悉、重组诱饵 |\n| 3. 严格绑定 | 一阶、二阶特征统计相同时，还能认出完整组合吗？ | 学习集合的三元组合 |\n| 4. 查询方式 | 图片、文字、候选布局和回答形式是否改变表现？ | 查询信息与输出方式 |\n| 5. 数量与布局 | 总物体数和每图物体数分别有什么影响？ | 总数 4/8；每图 1/2/4/8 |\n| 6. 空间关系 | 能否区分同一批物体之间的不同关系？ | 位置、深度与共同出现关系 |\n| 7. 三维与遮挡 | 换视角或遮住查询图后表现如何？ | 查询图的三维视角/遮挡 |')
    add('查询方式是跨任务的实验维度：下文“实验 4”复用已有基础、严格、多物体和三维题目，不是额外增加的一批试题。')
    add('## Benchmark 测试协议与运行设置\n\n1. 每题创建独立输入上下文，发送任务规则。\n2. 按预定顺序发送学习图片。\n3. 发送查询图片、文字描述或候选图，具体取决于问法。\n4. 要求模型给出规定格式的回答。\n5. 保存原始回答，再按独立标签评分。\n\n学习图片仍在当前输入中，所以这里不测跨会话的长期记忆；模型参数不更新。没有把标签、答案、题型名称或题目 ID 发给模型。')
    m=result['runs'][0]['manifest']
    add('| 项目 | 设置 |\n|---|---|\n| 模型 | Qwen3-VL-8B-Instruct，本地权重 |\n| 硬件 | 单张 NVIDIA RTX 3090，Slurm 计算节点 |\n| 推理 | Transformers AutoModelForImageTextToText，SDPA |\n| 解码 | do_sample=False，最多 '+str(m['max_new_tokens'])+' 个输出 token |\n| 图像配置 | max_pixels='+str(m['max_pixels'])+'；实际图片数量/尺寸随实验变化 |\n| 环境 | torch '+m['packages']['torch']+'；transformers '+m['packages']['transformers']+' |\n| 重复性 | 固定随机种子 '+str(m['seed'])+'，记录数据/权重/代码/依赖指纹 |')
    add('判断题只接受 YES/NO；二选一接受 A/B；四选一接受 A–D；回忆接受指定纹理词表；校准需输出规定属性字段。非法选项、词表外回答及不符合格式的回复均记错，不做事后宽松重解析。任务中断可续跑，运行错误会重试；模型本身的无效回答则保留为错误。')
    add(f'**数据与评分：** 6 种颜色 × 4 种纹理 × 6 种形状，共 144 种组合；全部 93,120 题已完成。开发集 23,280 题，正式测试集 **69,840 题、240 个独立随机组**。同组会生成多个问法和条件，置信区间按组重采样 1,000 次，保留这种关联。正式测试中 {test["invalid"]} 条无效回答（{test["invalid"]/test["observed"]:.2%}）按错误计分。')
    add('**怎样读图：** 普通误差线表示按组计算的 95% 置信区间；差值图中，点在 0 右侧表示后一个条件更好，横线跨过 0 表示当前数据未显示清晰差异。差值直接来自配对比较，不能用两组单独误差线是否重叠来替代。图中百分比通常保留一位小数；完整精度见数据表。区间未做多重比较校正，不将边界结果解释为普遍规律。')
    add('| 指标 | 分母与含义 | 理想方向 |\n|---|---|---|\n| 准确率 | 所有对应题中答对的比例；无效回答也计入分母 | 越高越好 |\n| 目标命中率 | 真正出现过的候选中，回答“见过”的比例 | 越高越好 |\n| 诱饵错误接受率 | 从未出现的候选中，误答“见过”的比例 | 越低越好 |\n| 成对正确率 | 同一严格正负对中，两题都答对的比例 | 越高越好 |')
    add('## 实际题目长什么样\n\n下面使用数据集中第一组正式严格绑定图片作为例子，按文件顺序选取，没有按模型对错挑选。两行查询图片完全一样，但学习图片不同，所以正确答案相反。报告程序再次核对了两行中单个属性和两两属性组合的出现次数完全相同。')
    chart('01_design','图 1｜真实严格配对题目。每行左侧四张为学习图，右侧为查询图；上下两题分别独立输入模型。')
    add('## 实验 1：先确认模型看得清属性')
    protocol('检查颜色、纹理和形状识别是否存在明显基础问题，为解释绑定错误提供参照。','每次只给一个物体，要求同时输出颜色、纹理和形状；覆盖二维、固定视角三维、另一三维视角和遮挡三维。','共 960 题，每种呈现 240 题。')
    chart('02_calibration','图 2｜三个属性同时答对才记为正确。纵轴从 0 开始，顶部数值展示接近满分的细小差异。')
    add(f'**结果：** 联合属性识别为 **{score("L0/CAL")}**；前三种呈现均为 100%，遮挡为 99.58%。遮挡条件下有一条回答不符合允许的属性词表，严格评分记为错误。\n\n**说明：** 简单单物体属性识别整体可靠，但这并不保证多物体学习后的查询同样容易，也不能排除后续任务中的视觉识别问题。')
    add('## 实验 2：基础绑定与不同诱饵')
    protocol('检验模型是否把“属性分别见过”误当成“完整物体见过”，并区分拒绝所有候选与真正识别。','每组学习 2、3 或 4 个物体，每张图放一个。判断题比较四类没见过的候选：所有属性全新、仅一类属性见过、仅两类属性见过、三类属性都见过但组合新。每类负例都配有真实出现过的正例。','图片、文字、图文三种判断各 5,760 题；其中目标 2,880 题，每类诱饵 720 题。另有二选一、四选一与属性回忆，见实验 4。')
    chart('03_basic','图 3｜左：诱饵错误接受率，越低越好；右：真实目标命中率，越高越好。两图纵轴范围不同，不能直接比较柱子的视觉高度。')
    add('**结果：** 图片判断对全新物体和只熟悉一类属性的诱饵均无误认，但对属性重组诱饵的误认率为 **4.44%**。文字判断对所有诱饵的误认率都是 0%，同时却只认出 **6.63%** 的真实目标。\n\n**说明：** 低误认率必须结合目标命中率解释。文字查询在本任务下主要呈现“很少回答见过”的倾向，不能据此判断绑定能力更强。')
    add('## 实验 3：严格配对，检验完整三元组合')
    protocol('减少单个特征或两两特征熟悉程度的作用，要求模型区分完整颜色—纹理—形状组合。','构造两组各 4 个物体的学习集合：单个属性及所有属性对的出现次数一致，同一个查询物体只在其中一组出现。两题的查询图片/文字相同，正确答案一正一负。图 1 展示一组真实例子。','每种问法 480 题，来自 240 对；图片、文字、图文判断各有目标 240 题、严格重组诱饵 240 题。')
    chart('04_strict','图 4｜同一实验的三种读法。左：单题准确率，虚线为均衡判断题的 50% 参照；中：目标命中与诱饵误认；右：同一对正负题都正确的比例。')
    add(f'**结果：** 图片判断准确率为 **{score("L2/Q1")}**，能认出 240 个真实目标中的 238 个，却将 **56.25%** 的严格诱饵误认成见过；正负两题同时答对的比例只有 **43.33%**。文字判断仅认出 240 个目标中的 2 个，负例全部答对，得到 **{score("L2/Q2")}** 的总准确率。\n\n**说明：** “约 50%”不意味着随机猜测；这里文字回答几乎总是拒绝。图片判断则更愿意接受候选，却对复杂重组有明显误认。')
    add('## 实验 4：同样的学习材料，换一种问法')
    protocol('检查输入信息、候选布局和回答形式对表现的影响，避免只用一个提问模板代表模型能力。','比较图片/文字/图文判断、同图/分图二选一、四选一和属性回忆。图片、文字、图文判断按相同查询语义配对；同图与分图二选一保持候选身份、顺序和学习内容一致。四选一、回忆属于补充任务，不与判断题混合计算总分。','严格绑定各问法 480 题，多物体绑定各 3,360 题，三维扩展各 1,920 题。基础绑定的三种判断各 5,760 题，三种选择各 2,880 题，回忆 720 题。')
    chart('05_formats','图 5｜上：各任务准确率总览，热图只展示数值，不能忽略不同任务的线索和猜测概率。下：真正可配对的问法差异及 95% 区间；标注单位为百分点。')
    add(f'**结果：** 在严格绑定中，图文判断相对图片判断提高 **{delta("L2/Q1_vs_Q3")}**；但基础绑定下降，多物体绑定的差值区间跨过 0。分图二选一在严格绑定中提高 **{delta("L2/Q4_vs_Q5")}**，多物体绑定提高 **{delta("L3/Q4_vs_Q5")}**；三维汇总差异不明确。')
    add('**说明：** 没有一种问法在所有实验里都最好。严格任务中的四选一额外加入了两个在两组学习集合中均未出现的候选，四个候选并非同等熟悉。属性回忆先给颜色和形状，再问纹理，线索与输出空间都不同。因此，回忆分数高不能直接解释为“比判断更强的同一种记忆”。')
    add('## 实验 5：分开研究总数量与每图数量')
    protocol('检验总物体数和每图摆放数量各自对应什么变化，而非预设“越多越差”。','严格配对任务保持查询核心不变；总数为 8 时，在原有 4 个核心物体外加入 4 个两种世界共用的物体。总数 4 比较每图 1/2/4 个；总数 8 比较每图 1/2/4/8 个。固定总数比较布局，固定每图数量比较总数。','7 个数量条件，每条件、每问法 480 题；每种问法合计 3,360 题。')
    chart('06_load','图 6｜上排固定总数，观察每图数量变化；纵轴为 40%–95% 以便比较。下排固定每图数量，直接比较总数 8 减总数 4 的配对差异。')
    add('**结果：** 图片判断在总数为 8 时，每图 1/2/4/8 个的准确率为 **76.25% / 81.88% / 79.17% / 73.12%**，并不单调。每图 4 个时，将总数从 4 增至 8，图片判断反而增加 '+delta('L3/Q1/per4/n4_vs_n8',design)+'。\n\n**说明：** 本数据不支持简单的“数量越多，准确率越低”。布局变化还会改变图片数和视觉输入规模；这些因素没有完全分离，不能将差异解释成独立的记忆容量效应。')
    add('## 实验 6：物体不变，关系改变')
    protocol('检查模型能否利用物体之间的相对位置、深度和共同出现信息。','每对题保持物体身份相同，通过交换位置或调整分组，使同一个关系问题的正确答案发生变化。左右、上下、深度前后使用 2 个物体；共现使用 4 个物体、每图 2 个，改变哪两个共同出现。','每类关系 480 题、240 对；共 1,920 题。')
    chart('07_spatial','图 7｜四种关系判断的准确率及 95% 区间。虚线为均衡 Yes/No 的 50% 参照；不同关系的视觉材料和物体数量并不完全一致。')
    add('**结果：** 左右和上下均为 **100%**，深度前后为 **79.38%**，同图共现为 **78.96%**。\n\n**说明：** 模型在这些简单左右/上下场景中表现可靠，但复杂关系并未同样达到满分。由于深度使用三维材料、共现使用更多物体，不能将四类分数差全部归因于抽象“关系类型”。')
    add('## 实验 7：三维、跨视角和查询图遮挡')
    protocol('检查二维结果能否延伸到受控三维物体，以及改变查询图的视角或可见性如何影响回答。','保持同组物体的属性语义和严格配对设计，比较二维、固定视角三维、跨视角三维、查询图部分遮挡。三维物体由二维轮廓挤出成网格并做深度遮挡渲染；跨视角/遮挡操作施加在查询图上。','每种模式、每种问法 480 题；4 种模式 × 7 种问法，共 13,440 题。')
    chart('08_3d','图 8｜上左：图片与图文判断随模式变化，纵轴为 40%–90%；上右：以固定视角三维为基准的配对差异。下：所有问法分模式结果；Q2/Q7 的三种三维条件没有查询图变化，不是独立视觉实验。')
    add('**结果：** 图片判断由固定视角三维的 **66.88%** 降至查询遮挡的 **50.00%**，差值为 '+delta('L5/Q1/3d_fixed_vs_3d_occluded',design)+'。图文判断对应从 **74.79%** 降至 **50.00%**。跨视角本身没有呈现预设的稳定下降：图片判断略升，图文判断差值区间跨过 0。\n\n**说明：** 遮挡可能降低查询物体本身的可识别性，所以下降不能直接等同于忘记学习内容。文字判断与属性回忆没有查询图片，其三维模式下重复输入的结果不能被当成额外独立证据。')
    task_section,task_assets=build_task_section(root)
    add(task_section)
    io_section,io_assets=build_io_section(data,root)
    add(io_section)
    add('## 结果边界与可复查材料\n\n目前只测试了 Qwen3-VL-8B 一个模型，图片为程序生成的受控物体；结论不能直接推广到所有视觉语言模型或自然照片。准确率、命中率、误认率和成对正确率回答不同问题，不能互相替代。本报告只讨论行为证据，不解释内部机制。')
    add('样本中 100% 只表示当前题目全部答对，不保证未见样本没有错误；样本全对时，重采样区间也可能退化为 100%–100%。正式结果没有把无效回答删去，因此选择题成绩同时反映选对候选和遵守输出格式的表现。')
    add('**已交付：** 全量行为实验与评分；按实验组织的设计说明；八张结果/设计图；分任务、分条件、配对统计和无效回答原文。其他模型可以后续使用相同协议验证，目前不在本报告结论范围内。')
    add('**数据来源：** `summary.json`；全精度表格见 `accuracy.csv`、`conditions.csv`、`paired_comparisons.csv`；异常输出见 `invalid_responses.csv`。图表使用的数值和区间另存为 `visual_figures/figure_data.csv`，示例题来源见 `visual_figures/example_provenance.json`。独立图集为 `实验结果图集.pdf`。')
    md='\n\n'.join(parts)+'\n';(root/'visual_report.md').write_text(md)
    project_root=Path(__file__).resolve().parents[1]
    relative_figures=os.path.relpath(dest.resolve(),project_root)
    relative_io=os.path.relpath((root/'io_examples').resolve(),project_root)
    relative_tasks=os.path.relpath((root/'task_figures').resolve(),project_root)
    (project_root/'PROJECT_REPORT_V2.md').write_text(md.replace('](visual_figures/',f']({relative_figures}/').replace('](io_examples/',f']({relative_io}/').replace('](task_figures/',f']({relative_tasks}/'))
    body=markdown.markdown(md,extensions=['tables','fenced_code','toc'])
    for name,path in images.items():
        body=body.replace(f'src="visual_figures/{name}.png"',f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"')
    for relative,path in {**io_assets,**task_assets}.items():
        body=body.replace(f'src="{relative}"',f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"')
    css='''body{margin:0;background:#eef2f5;color:#253340;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:28px auto;padding:38px 46px;background:white}h1{font-size:30px;line-height:1.4;color:#153d5b}h2{font-size:23px;color:#153d5b;margin-top:42px;border-bottom:2px solid #d5e3ed;padding-bottom:8px}p{margin:15px 0}strong{color:#174766}img{max-width:100%;height:auto;display:block;margin:24px auto 10px}em{font-size:14px;color:#576675;font-style:normal}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:10px 12px;border:1px solid #d5dfe7;text-align:left}th{background:#eef4f8}code{font-size:14px;overflow-wrap:anywhere}@media(max-width:700px){main{margin:0;padding:18px}body{font-size:16px}table{font-size:13px}}@media print{body{background:white;font-size:10.5pt}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}img,tr{break-inside:avoid}p{orphans:3;widows:3}thead{display:table-header-group}}'''
    css+='pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:14px 18px;border-left:3px solid #729bb8;line-height:1.6}h3{margin-top:34px;color:#174766}pre code{font-family:monospace;font-size:14px}'
    (root/'visual_report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>物体绑定评测：实验设计与可视化结果</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
    io_md='# 物体绑定实验：实际输入、模型输出与评分说明\n\n本报告解释一题从输入到评分的全过程，使用本次 Qwen3-VL-8B 正式测试的真实记录；英文文本是实际提示，中文文字是读者说明。\n\n'+io_section
    (root/'input_output_report.md').write_text(io_md)
    (project_root/'INPUT_OUTPUT_REPORT.md').write_text(io_md.replace('](io_examples/',f']({relative_io}/'))
    io_body=markdown.markdown(io_md,extensions=['tables','fenced_code'])
    for relative,path in io_assets.items():
        io_body=io_body.replace(f'src="{relative}"',f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"')
    (root/'input_output_report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>物体绑定实验：实际输入与输出</title><style>'+css+'</style></head><body><main>'+io_body+'</main></body></html>')
    task_md='# 物体绑定实验：逐任务成功率与强重组诱饵\n\n模型：Qwen3-VL-8B-Instruct。数据：bindmem_v2 正式测试集，69,840 题、240 个随机组。\n\n'+task_section
    (root/'task_success_report.md').write_text(task_md)
    (project_root/'TASK_SUCCESS_REPORT.md').write_text(task_md.replace('](task_figures/',f']({relative_tasks}/'))
    task_body=markdown.markdown(task_md,extensions=['tables','fenced_code'])
    for relative,path in task_assets.items():
        task_body=task_body.replace(f'src="{relative}"',f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"')
    (root/'task_success_report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>物体绑定实验：逐任务成功率</title><style>'+css+'</style></head><body><main>'+task_body+'</main></body></html>')
    print('Created',root/'visual_report.html','and',len(images),'figures; plotted rows',len(plotted))


if __name__=='__main__':main()
