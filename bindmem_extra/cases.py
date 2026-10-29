"""Export actual strong-recombination false acceptances with images and provenance."""
import argparse
import base64
import html
import json
from collections import defaultdict
from pathlib import Path

from bindmem.common import ATTRS, parse_answer, read_jsonl, sha_file, write_json
from bindmem.run import make_messages
from .gradient import PAIRS, pair_counts
from .score import load_rows


def export_cases(data,runs,output,limit=3,levels=('L2',),questions=('Q1','Q3')):
    data,out=Path(data),Path(output);out.mkdir(parents=True,exist_ok=True)
    inputs={r['trial_id']:r for r in read_jsonl(data/'inputs/test.jsonl')}
    assets={r['path']:r['sha256'] for r in read_jsonl(data/'assets.jsonl')}
    raw_hashes={str(run):sha_file(Path(run)/'predictions.jsonl') for run in runs}
    groups=defaultdict(list);all_records={};seen=set()
    for run in runs:
        rows,_=load_rows(data,run,'test',allow_partial=True)
        raw={r['trial_id']:r for r in read_jsonl(Path(run)/'predictions.jsonl')}
        for r in rows:
            if r['trial_id'] in seen:raise ValueError('Overlapping trial IDs; export one model at a time')
            seen.add(r['trial_id']);all_records[r['trial_id']]=(r,raw.get(r['trial_id']),str(run))
            if r['level'] in levels and r['question'] in questions and r['probe_type']=='pairwise_complete' and r['answer']=='NO':
                groups[(r['level'],r['question'])].append((r,raw.get(r['trial_id']),str(run)))
    cases=[];denominators=[]
    for key,rr in sorted(groups.items()):
        observed=[entry for entry in rr if entry[0]['observed']]
        failures=[entry for entry in observed if entry[0]['prediction']=='YES']
        denominators.append(dict(level=key[0],question=key[1],expected=len(rr),observed=len(observed),false_acceptances=len(failures)))
        for r,raw,run in sorted(failures,key=lambda entry:entry[0]['trial_id'])[:limit]:
            study=[tuple(o) for scene in r['study_scenes'] for o in scene];probe=tuple(r['probe'])
            assert probe not in study and all(pair_counts(study,probe))
            assert not raw['error'] and parse_answer(raw['raw_response'],'binary')=='YES'
            witnesses={f'{ATTRS[a]} + {ATTRS[b]}':[i+1 for i,o in enumerate(study) if (o[a],o[b])==(probe[a],probe[b])] for a,b in PAIRS}
            item=inputs[r['trial_id']]
            paths=item['study_images']+[b['image'] for b in item['query'] if b['type']=='image']
            for path in paths:
                if sha_file(data/path)!=assets[path]:raise ValueError(f'Changed case image: {path}')
            paired=[]
            if r.get('pair_id'):
                paired=[dict(label=x,raw_response=y['raw_response'],run=z) for x,y,z in all_records.values()
                        if x.get('pair_id')==r['pair_id'] and x['question']==r['question'] and x['answer']=='YES' and x['observed']]
            cases.append(dict(trial_id=r['trial_id'],input=item,label=r,prediction=raw,run=run,
                run_manifest_sha256=sha_file(Path(run)/'manifest.json'),
                predictions_file_sha256=raw_hashes[run],
                input_file_sha256=sha_file(data/'inputs/test.jsonl'),label_file_sha256=sha_file(data/'labels/test.jsonl'),
                dataset_sha256=sha_file(data/'manifest.json'),pair_witnesses=witnesses,paired_positive=paired,
                messages=make_messages(item,data,802816),images={p:sha_file(data/p) for p in paths}))
    document=dict(selection_rule='First trial IDs in lexical order among observed NO->YES strong-recombination errors, separately per level/question; at most limit each. No synthetic replies.',
        limit_per_condition=limit,denominators=denominators,cases=cases)
    write_json(out/'cases.json',document)
    def esc(x):return html.escape(str(x))
    parts=['<h1>真实强重组误认案例</h1>',
        '<p>这些回复来自已保存的模型运行记录。仅选“完整三元组未见过、三种属性对均见过，却回答 YES”的错误；每个条件按 trial_id 排序取前若干条，不按视觉效果挑选。它们是程序生成任务中的真实模型错误，不是自然照片案例。</p>',
        '<p>分母包括全部相应负例；未完成题不计入误认率，单独列出。完整输入、标签、回复和文件指纹见 cases.json。</p>']
    for d in denominators:
        parts.append(f'<p>{esc(d["level"])}/{esc(d["question"])}：误认 {d["false_acceptances"]}/{d["observed"]} 道已完成强重组负例；预期 {d["expected"]} 道。</p>')
    def img(path):
        return '<img src="data:image/png;base64,'+base64.b64encode((data/path).read_bytes()).decode()+'">'
    for c in cases:
        r=c['label'];item=c['input']
        parts.append(f'<h2>{esc(r["level"])}/{esc(r["question"])} · {esc(c["trial_id"])}</h2>')
        parts.append(f'<p><strong>标准答案：NO；模型原始回答：{esc(c["prediction"]["raw_response"])}</strong></p>')
        parts.append('<p>候选：'+esc(' / '.join(r['probe']))+'。下列属性解释仅供报告读者阅读，Q1 没有向模型提供这些文字标签。</p>')
        parts.append('<h3>实际输入顺序</h3><pre>'+esc(item['instruction'])+'</pre>')
        for i,path in enumerate(item['study_images'],1):
            parts.append(f'<p>Study image {i}:</p>'+img(path))
        parts.append('<pre>End of study. Answer the following query.</pre>')
        for block in item['query']:
            parts.append(img(block['image']) if block['type']=='image' else '<pre>'+esc(block['text'])+'</pre>')
        parts.append('<h3>为什么这是强重组误认</h3><ol>')
        for scene in r['study_scenes']:
            for o in scene:parts.append('<li>'+esc(' / '.join(o))+'</li>')
        parts.append('</ol><ul>')
        for pair,indices in c['pair_witnesses'].items():parts.append('<li>'+esc(pair)+' 出现在学习物体 '+esc(', '.join(map(str,indices)))+'</li>')
        parts.append('</ul><p>三种属性对都有来源，但没有任何一个学习物体同时具备候选的三个属性。</p>')
        for pair in c['paired_positive']:
            parts.append('<p>对应正例的真实回复：'+esc(pair['raw_response'])+'；正例标准答案 YES，trial_id：'+esc(pair['label']['trial_id'])+'。正例使用另一套学习材料，不是上面这套。</p>')
        parts.append('<p>来源：'+esc(c['run'])+'/predictions.jsonl</p>')
    if not cases:parts.append('<p>当前选择中没有符合条件的已观察误认案例，不生成或替换模型回复。</p>')
    css='body{font:17px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#243544}img{max-width:100%;max-height:360px;margin:10px}h2{border-top:2px solid #ccc;padding-top:30px}pre{white-space:pre-wrap;background:#eef3f7;padding:12px}strong{color:#a33}'
    (out/'cases.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实强重组误认案例</title><style>'+css+'</style><body>'+''.join(parts)+'</body></html>')
    print(f'Exported {len(cases)} actual errors -> {out}/cases.html')
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/bindmem_v2');p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=3)
    p.add_argument('--levels',nargs='+',default=['L2']);p.add_argument('--questions',nargs='+',default=['Q1','Q3'])
    args=p.parse_args()
    if args.limit<1:p.error('limit must be positive')
    export_cases(args.data,args.runs,args.output,args.limit,args.levels,args.questions)


if __name__=='__main__':main()
