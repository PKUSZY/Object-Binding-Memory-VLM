"""Build actual input/output examples for the behavior report (no inference)."""
import base64
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bindmem.common import parse_answer, sha_file
from bindmem.run import make_messages

NAMES={'CAL':'属性校准：输出三个属性','Q1':'图片查询：判断是否见过',
       'Q2':'文字查询：判断是否见过','Q3':'图片＋文字：判断是否见过',
       'Q4':'同一张图中的二选一','Q5':'两张候选图中的二选一',
       'Q6':'四选一','Q7':'给颜色和形状，回忆纹理','Q8':'判断物体关系'}
KINDS={'attributes':'一个 JSON 对象，恰好包含 color、texture、shape，值必须来自指定词表。',
       'binary':'只回答 YES 或 NO。','choice2':'只回答 A 或 B。',
       'choice4':'只回答 A、B、C 或 D。','recall':'只回答 SOLID、STRIPES、DOTS 或 GRID。'}
ZH={'red':'红色','yellow':'黄色','blue':'蓝色','green':'绿色','purple':'紫色','orange':'橙色',
    'solid':'纯色','stripes':'条纹','dots':'圆点','grid':'网格',
    'circle':'圆形','triangle':'三角形','square':'正方形','star':'星形',
    'pentagon':'五边形','hexagon':'六边形'}


def describe_zh(obj):
    return '、'.join(ZH[x] for x in obj)
NOTES={
 'CAL':'学习阶段仅给一张单物体图，查询阶段没有额外图片；目标是识别当前物体的三种属性。',
 'Q1':'查询阶段给一张候选物体图，文字仅问它是否作为完整物体出现过；并未直接用文字给出候选属性。',
 'Q2':'学习阶段仍然是图片；只有最终查询变成文字。它不是纯文本任务，也没有额外查询图。',
 'Q3':'候选图片与文字描述指向同一个物体；与 Q1 相比多了属性文字，与 Q2 相比多了候选图。',
 'Q4':'候选 A、B 画在同一张图中；字母本身位于图片上。只有一个候选属于学习集合。',
 'Q5':'候选 A、B 分别提供为两张图片；每张图前附 Candidate A 或 Candidate B。候选和顺序与对应 Q4 一致。',
 'Q6':'A–D 四个候选画在同一张图中。严格绑定任务中，两个额外候选在两种学习世界中都未出现。',
 'Q7':'颜色和形状作为定位线索，问题不直接给出纹理，也不提供查询图片；生成器保证线索唯一对应一个学习物体。',
 'Q8':'使用文字描述要比较的物体和关系；查询阶段没有额外候选图，证据来自前面的学习图。'}


def build_io_section(data,root):
    data=Path(data);root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    labels=[json.loads(x) for x in (data/'labels/test.jsonl').open()]
    first_group=next(r['cluster_id'] for r in labels if r['level']=='L2')
    def pick(fn):return next(r for r in labels if fn(r))
    chosen=[('属性识别',pick(lambda r:r['question']=='CAL' and r['mode']=='2d'))]
    for q in [f'Q{i}' for i in range(1,8)]:
        chosen.append((NAMES[q],pick(lambda r:r['cluster_id']==first_group and r['level']=='L2' and r['question']==q and r['world']==0)))
    for q in ['Q1','Q2']:
        chosen.append(('严格负例：'+NAMES[q],pick(lambda r:r['cluster_id']==first_group and r['level']=='L2' and r['question']==q and r['world']==1)))
    for relation,name in [('left','左右'),('above','上下'),('front','深度前后'),('cooccur','同图共现')]:
        chosen.append(('关系判断：'+name,pick(lambda r:r['level']=='L4' and r['relation']==relation and r['world']==0)))
    chosen.append(('多物体输入：8 个物体、每图 4 个',pick(lambda r:r['level']=='L3' and r['question']=='Q1' and r['total_objects']==8 and r['objects_per_image']==4 and r['world']==0)))
    chosen.append(('三维输入：查询图部分遮挡',pick(lambda r:r['level']=='L5' and r['question']=='Q1' and r['mode']=='3d_occluded' and r['world']==0)))
    predictions={};locations={}
    label_by_id={r['trial_id']:r for r in labels}
    for entry in summary['runs']:
        run=Path(entry['path'])
        for line in (run/'predictions.jsonl').open():
            r=json.loads(line);tid=r['trial_id']
            if tid not in label_by_id or r.get('error'):continue
            if tid in predictions:raise ValueError('Duplicate successful trial')
            predictions[tid]=r;locations[tid]=str(run/'predictions.jsonl')
    # Explicit diagnostic selection, separate from first-in-order examples.
    invalid=pick(lambda r:r['question'] in ['Q4','Q5'] and predictions[r['trial_id']]['parsed_answer'] is None)
    chosen.append(('无效回答诊断：二选一却输出范围外字母',invalid))
    wanted={r['trial_id'] for _,r in chosen}
    inputs={}
    for line in (data/'inputs/test.jsonl').open():
        r=json.loads(line)
        if r['trial_id'] in wanted:inputs[r['trial_id']]=r
    assert set(inputs)==wanted
    dest=root/'io_examples';dest.mkdir(exist_ok=True)
    assets={};panels={};parts=[];traces=[];seen_study={}
    def add(s):parts.append(s.strip())
    def fence(s,lang='text'):return f'```{lang}\n{s}\n```'
    def display(v):return json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else str(v)
    def panel(paths,titles,caption):
        key=(tuple(paths),tuple(titles))
        if key in panels:return panels[key]
        name='input_'+hashlib.sha256(json.dumps(key).encode()).hexdigest()[:10]
        n=len(paths);cols=min(4,n);rows=math.ceil(n/cols)
        fig,axs=plt.subplots(rows,cols,figsize=(max(3.2,cols*2.8),rows*2.8+.3),squeeze=False,layout='constrained')
        for ax in axs.flat:ax.axis('off')
        for ax,path,title in zip(axs.flat,paths,titles):
            ax.imshow(Image.open(data/path));ax.set_title(title,fontsize=11)
        # PNG is the original image displayed inside a labeled report panel, not an altered stimulus.
        fig.savefig(dest/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
        relative=f'io_examples/{name}.png';assets[relative]=dest/f'{name}.png'
        panels[key]=relative
        return relative
    def use_panel(paths,titles,caption):
        relative=panel(paths,titles,caption)
        add(f'![{caption}]({relative})')

    add('## 模型具体收到了什么，又输出了什么？')
    add('本节直接读取正式测试的 `inputs/test.jsonl`、独立的 `labels/test.jsonl` 和各批 `predictions.jsonl`。下面的英文是实际发送文本，图片来自实际输入，模型回复是原始记录，不是示意答案。常规例子按数据文件顺序选取；Q1–Q7 使用同一严格学习集合，便于横向理解。末尾的无效回答例子按诊断目的单独选取，不代表随机抽样。')
    add('### 一道题在程序中的实际结构')
    instruction=inputs[chosen[1][1]['trial_id']]['instruction']
    add('应用层只构造**一条 `role=user` 的多模态消息**，内容按下列顺序排列；并不是先让模型回答学习阶段，再发下一轮问题。处理器随后把文本和图片转换成模型输入，并添加开始生成回答的模板标记。')
    add(fence('user.content = [\n  任务规则文本,\n  "Study image 1:", 学习图片1,\n  "Study image 2:", 学习图片2,\n  ……本题的其余学习图片……,\n  "End of study. Answer the following query.",\n  本题查询内容（文字和/或图片，按下面实例所列顺序）\n]'))
    add('每题开头实际发送的共同规则：')
    add(fence(instruction))
    add('中文含义：记住学习物体；颜色、纹理、形状必须同时属于同一个物体才算相同身份；位置、大小和视角不改变身份；三维物体的形状按正面轮廓理解。这里的中文解释仅供读者阅读，实际提示使用上方英文。')
    add('每题学习图结束后，实际追加：')
    add(fence('End of study. Answer the following query.'))
    add('**标准答案不是输入。** 本文后面显示的标签、物体属性列表、trial ID、对错判断和错误类型，均属于读者解释或评测器记录，没有额外发送给模型。文字查询里明确写出的候选属性则属于正常问题内容。')
    add('| 任务 | 学习阶段 | 查询阶段 | 允许输出 |\n|---|---|---|---|\n| CAL 属性校准 | 1 张单物体图 | 识别三属性的文字指令 | 指定字段和词表的 JSON |\n| Q1 图片判断 | 学习图片 | 候选图片＋Yes/No 问题 | YES / NO |\n| Q2 文字判断 | 学习图片 | 候选属性文字＋Yes/No 问题 | YES / NO |\n| Q3 图文判断 | 学习图片 | 候选图片＋候选属性文字＋问题 | YES / NO |\n| Q4 同图二选一 | 学习图片 | 选择指令＋含 A/B 的一张候选图 | A / B |\n| Q5 分图二选一 | 学习图片 | 选择指令＋A 标签和图＋B 标签和图 | A / B |\n| Q6 四选一 | 学习图片 | 选择指令＋含 A–D 的一张候选图 | A / B / C / D |\n| Q7 属性回忆 | 学习图片 | 给颜色和形状，询问纹理 | SOLID / STRIPES / DOTS / GRID |\n| Q8 关系判断 | 关系场景图 | 指定物体与关系的文字问题 | YES / NO |')
    for idx,(title,label) in enumerate(chosen,1):
        tid=label['trial_id'];item=inputs[tid];raw=predictions[tid]
        parsed=parse_answer(raw['raw_response'],item['response_kind'])
        assert parsed==raw['parsed_answer']
        valid=parsed is not None;correct=valid and parsed==label['answer']
        messages=make_messages(item,data,summary['runs'][0]['manifest']['max_pixels'])
        trace={'case':idx,'title':title,'selection':'first invalid choice trial in test order' if idx==len(chosen) else 'first matching test trial in dataset order',
               'trial_id':tid,'input_record':item,'messages':messages,'raw_prediction':raw,
               'private_label_not_sent_to_model':label,'score':{'valid':valid,'correct':correct},
               'source':{'input':'inputs/test.jsonl','label':'labels/test.jsonl','prediction':locations[tid]},
               'images':{path:sha_file(data/path) for path in item['study_images']+[b['image'] for b in item['query'] if b['type']=='image']}}
        traces.append(trace)
        add(f'### 输入输出案例 {idx}：{title}')
        add(f'任务为 `{label["level"]}/{label["question"]}`。{NOTES[label["question"]]}')
        study=tuple(item['study_images'])
        if study in seen_study:
            add(f'**输入 A · 学习阶段：** 与案例 {seen_study[study]} 的学习图片完全相同，顺序也相同，共 {len(study)} 张；本题仍会实际发送全部图片，并非省略学习输入。')
        else:
            seen_study[study]=idx
            add(f'**输入 A · 学习阶段：** 共 {len(study)} 张图片，合计 {label["total_objects"]} 个物体，每张 {label["objects_per_image"]} 个；下图按实际输入顺序排列。每张图前有 `Study image 序号:` 文本。')
            use_panel(list(study),[f'Study image {i+1}' for i in range(len(study))],f'案例 {idx} 的实际学习图片')
        add('**输入 B · 查询阶段：** 以下按真正发送顺序展示；此前已发送共同规则、学习图与学习结束提示。')
        for bi,b in enumerate(item['query'],1):
            if b['type']=='text':
                add(f'第 {bi} 个查询内容块，文字原文：\n\n'+fence(b['text']))
            else:
                add(f'第 {bi} 个查询内容块，图片：')
                use_panel([b['image']],['Query image'],f'案例 {idx} 的查询图片（第 {bi} 块）')
        if label['question'] in ['Q1','Q2','Q3']:
            translation=f'是否在学习集合中出现过“{describe_zh(label["probe"])}”这个完整物体？'
        elif label['question']=='Q7':
            translation=f'学习集合中，{ZH[label["cue"][0]]}、{ZH[label["cue"][1]]}的物体是什么纹理？'
        elif label['question'] in ['Q4','Q5','Q6']:
            translation='这些候选中，哪一个曾作为完整物体出现在学习集合里？只回答它的字母。'
        elif label['question']=='CAL':
            translation='识别这张学习图中唯一物体的颜色、纹理、形状，并按指定 JSON 字段和词表输出。'
        else:
            a,b=map(describe_zh,label['probe'])
            relation={'left':'是否在第二个物体左侧','above':'是否在第二个物体上方',
                      'front':'是否比第二个物体更靠近相机','cooccur':'是否曾与第二个物体出现在同一张学习图中'}[label['relation']]
            translation=f'第一个物体为“{a}”，第二个为“{b}”：第一个物体{relation}？'
        add('**问题的中文解释（仅供读者，不是额外模型输入）：** '+translation)
        add('**输出要求：** '+KINDS[item['response_kind']])
        add('**模型实际输出：**\n\n'+fence(raw['raw_response']))
        add('**评测器标准答案（不发送给模型）：**\n\n'+fence(display(label['answer']),'json' if isinstance(label['answer'],dict) else 'text'))
        if not valid:
            explanation='无效回答，按错误计分。该输出不属于本题允许的答案集合；程序不会把越界字母猜测成另一个选项。'
        elif correct:
            explanation='有效且正确：解析后的输出与标准答案完全一致。'
        elif item['response_kind']=='binary':
            explanation='有效但错误：'+('真实目标被拒绝，属于漏认。' if label['answer']=='YES' else '未出现的候选被接受，属于误认。')
        else:explanation='有效但错误：输出格式合法，但选项或属性与标准答案不一致。'
        add('**如何评分：** '+explanation)
        if label['question'] in ['Q4','Q5','Q6']:
            objects=[o for scene in label['study_scenes'] for o in scene]
            explanation=['**评分依据（来自独立标签，不作为额外输入）：**','',
                         '| 选项 | 候选完整属性 | 是否属于学习集合 |',
                         '|---|---|---|']
            for ci,candidate in enumerate(label['candidates']):
                explanation.append(f'| {chr(65+ci)} | {describe_zh(candidate)} | '+('是' if candidate in objects else '否')+' |')
            add('\n'.join(explanation))
            add('仅一个候选完整出现。模型应输出对应字母，不要求输出属性描述。')
        elif label['question'] in ['Q1','Q2','Q3']:
            objects=[o for scene in label['study_scenes'] for o in scene]
            included=label['probe'] in objects
            assert included==(label['answer']=='YES')
            add('**评分依据：** 独立标签检查查询物体的完整三元组'+('存在于' if included else '不存在于')+'本题学习集合中；属性分别出现不等于完整组合出现。')
        if label['question']=='Q7':
            add('评分依据是学习集合中与颜色、形状线索匹配的物体纹理，而不是把查询候选的完整三元组直接当作答案。')
        add(f'追溯记录：`{tid}`；原始回复位于 `{locations[tid]}`。完整消息、标签和评分另存于 `io_examples/cases.json` 的第 {idx} 个案例。')
    add('### 如何从单题回答得到报告中的指标')
    add('单题正确率把合法且与标签一致的回答计为正确，其他模型回答计错。目标命中率仅统计标签为 YES 的目标题；诱饵误认率仅统计相应标签为 NO 的诱饵题。严格配对的两题分别独立输入模型，只有正负两题都答对，整对才算正确。这里展示的例子用来解释流程；总体结论仍来自全部正式测试题，而不是这几个例子。')
    add('解析规则会去掉首尾空白，并将非 JSON 答案转为大写后与允许词表匹配；不会从长解释中抽取一个看起来像答案的词。校准 JSON 必须恰好包含指定字段，且三个值均在允许词表中。运行异常不作为模型判断，失败题会重试；合法性错误则保留并按错计分。')
    section='\n\n'.join(parts)+'\n'
    (dest/'cases.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n')
    return section,assets
