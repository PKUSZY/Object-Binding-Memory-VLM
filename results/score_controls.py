#!/usr/bin/env python3
import argparse, json
from collections import Counter
from pathlib import Path

def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--raw',default='results/control_benchmark/raw.jsonl'); p.add_argument('--out',default='results/control_benchmark/summary.json'); a=p.parse_args()
 rows=read(a.raw); labels={x['trial_id']:x for typ in ('pairwise','strong_recombination') for x in read(f'results/control_benchmark/labels/{typ}.jsonl')}; out=[]
 for r in rows:
  l=labels[r['trial_id']]; out.append({**r,**l,'correct':r.get('valid') and r.get('parsed_answer')==l['answer']})
 summary={'n':len(out),'valid':sum(x['valid'] for x in out),'correct':sum(x['correct'] for x in out),'by_control':{}}
 for kind in sorted({x['control'] for x in out}):
  z=[x for x in out if x['control']==kind]; summary['by_control'][kind]={'n':len(z),'valid':sum(x['valid'] for x in z),'accuracy':sum(x['correct'] for x in z)/len(z)}
 Path(a.out).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
