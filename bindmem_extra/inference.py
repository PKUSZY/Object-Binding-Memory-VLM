"""LLaVA-1.5 and Qwen3-VL behavioral runner with immutable selection/resume records."""
import argparse
import fcntl
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from PIL import Image
from bindmem.common import digest, parse_answer, read_jsonl, sha_file, write_json, write_jsonl
from bindmem.run import checkpoint_identity, make_messages, recover_rows

# Previous runner differed only in selection/reuse orchestration, not model inputs or generation.
REUSABLE_RUNNER_SHA256 = '27580f46bf749fe7998d5260a79474861ce6cc395962f91bbdcb4bbd973f4094'


def reuse_completed(run, manifest, ids):
    run = Path(run)
    old = json.loads((run/'manifest.json').read_text())
    fields = ('dataset_sha256','checkpoint','processor_files','backend','max_new_tokens','max_pixels',
              'seed','do_sample','attention','dtype','capture','image_policy','prompt_policy','packages')
    for field in fields:
        if old.get(field) != manifest.get(field):
            raise ValueError(f'Cannot reuse predictions: incompatible {field}')
    if set(old['source']) != set(manifest['source']):
        raise ValueError('Cannot reuse predictions: source file set differs')
    for name, sha in old['source'].items():
        if sha == manifest['source'][name]: continue
        if Path(name).name == 'inference.py' and sha == REUSABLE_RUNNER_SHA256: continue
        raise ValueError(f'Cannot reuse changed inference dependency: {name}')
    raw = (run/'predictions.jsonl').read_bytes()
    latest = {}
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try: r = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines)-1 and not line.endswith(b'\n'): break
            raise ValueError('Corrupt source predictions; source has not been modified')
        if r['trial_id'] in latest and not latest[r['trial_id']].get('error'):
            raise ValueError('Duplicate successful source prediction')
        latest[r['trial_id']] = r
    return [dict(r, reused_from=str(run.resolve())) for tid,r in latest.items()
            if tid in ids and not r.get('error')]


def llava_prompt(item, root):
    blocks = make_messages(item, Path(root), 802816)[0]['content']
    # Official USER/ASSISTANT format, preserving text/image interleaving and order.
    text = '\n'.join(b['text'] if b['type'] == 'text' else '<image>' for b in blocks)
    return 'USER: ' + text + '\nASSISTANT:', [b['image'] for b in blocks if b['type'] == 'image']


def pad_square(im, color):
    im = im.convert('RGB')
    side = max(im.size)
    padded = Image.new('RGB', (side, side), color)
    padded.paste(im, ((side-im.width)//2, (side-im.height)//2))
    return padded


def configure_llava(processor, config):
    if config.model_type != 'llava':
        raise ValueError('This adapter is for llava-1.5, not llava_next or onevision')
    processor.patch_size = config.vision_config.patch_size
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1  # CLIP has a CLS token; default feature selection removes it.
    processor.tokenizer.padding_side = 'left'
    crop = processor.image_processor.crop_size
    return (crop['height']//processor.patch_size)*(crop['width']//processor.patch_size) + 1 - int(config.vision_feature_select_strategy == 'default')


def llava_inputs(item, root, processor):
    prompt, paths = llava_prompt(item, root)
    color = tuple(round(x*255) for x in processor.image_processor.image_mean)
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(pad_square(im, color))
    # Padding before native CLIP preprocessing retains the whole scene, including wide choice panels.
    return processor(text=[prompt], images=images, return_tensors='pt', truncation=False), prompt


def select_items(items, root, backend, processor, config, max_new_tokens, smoke=False):
    selected, excluded = [], []
    cache = {}; seen = set()
    image_tokens = configure_llava(processor, config) if backend == 'llava' else None
    context = config.text_config.max_position_embeddings
    for item in items:
        tokens = None
        if backend == 'llava':
            prompt, paths = llava_prompt(item, root)
            if prompt not in cache:
                expanded = prompt.replace('<image>', processor.image_token * image_tokens)
                cache[prompt] = len(processor.tokenizer(expanded, add_special_tokens=True, truncation=False)['input_ids'])
            tokens = cache[prompt]
            if tokens + max_new_tokens > context:
                excluded.append(dict(trial_id=item['trial_id'], split=item['split'], level=item['level'], question=item['question'],
                    reason='native_context_limit', input_tokens=tokens, max_new_tokens=max_new_tokens, context_limit=context))
                continue
        if smoke:
            # Cover all question formats and study-image counts in each level; dev only.
            key = (item['level'], item['question'], len(item['study_images']))
            if key in seen: continue
            seen.add(key)
        selected.append(dict(trial_id=item['trial_id'], input_tokens=tokens))
    return selected, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2')
    p.add_argument('--model', default='models/llava-1.5-7b-hf')
    p.add_argument('--backend', choices=['llava','qwen'], default='llava')
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['dev','test','both'], default='both')
    p.add_argument('--levels', nargs='+'); p.add_argument('--questions', nargs='+')
    p.add_argument('--limit', type=int)
    p.add_argument('--max-new-tokens', type=int, default=80)
    p.add_argument('--max-pixels', type=int, default=802816)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--context-policy', choices=['error','exclude'], default='error')
    p.add_argument('--preflight', action='store_true', help='CPU token/image checks, no weights or predictions')
    p.add_argument('--smoke', action='store_true', help='Dev representative subset; must use a separate output directory')
    p.add_argument('--trial-ids', help='JSON list of explicitly selected trial IDs from this dataset/split')
    p.add_argument('--reuse-from', help='Copy compatible completed predictions into this NEW run; source is never changed')
    p.add_argument('--initialize-only', action='store_true', help='CPU: freeze selection and import compatible records without loading weights')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    if args.max_new_tokens < 1: p.error('--max-new-tokens must be positive')
    if args.smoke and args.split != 'dev': p.error('--smoke requires --split dev')
    root, out = Path(args.data), Path(args.output)
    dm = json.loads((root/'manifest.json').read_text())
    splits = ['dev','test'] if args.split == 'both' else [args.split]
    for split in splits:
        f = f'inputs/{split}.jsonl'
        if sha_file(root/f) != dm['files'][f]: raise ValueError(f'Changed input: {f}')
    items = [r for split in splits for r in read_jsonl(root/'inputs'/f'{split}.jsonl')
             if (not args.levels or r['level'] in args.levels) and (not args.questions or r['question'] in args.questions)]
    if args.trial_ids:
        chosen=json.loads(Path(args.trial_ids).read_text())
        if not isinstance(chosen,list) or not all(isinstance(t,str) for t in chosen) or len(set(chosen))!=len(chosen):
            p.error('--trial-ids must contain a JSON list of unique strings')
        chosen=set(chosen)
        if not chosen <= {r['trial_id'] for r in items}:p.error('Trial selection contains IDs outside the selected dataset/split/filters')
        items=[r for r in items if r['trial_id'] in chosen]
    if not items: p.error('No matching trials')
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if args.backend == 'qwen' and config.model_type != 'qwen3_vl': raise ValueError('Expected Qwen3-VL checkpoint')
    selected, excluded = select_items(items, root, args.backend, processor, config, args.max_new_tokens, args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.lock').open('w')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another process is using this output directory')
    plan = dict(selected=selected, excluded=excluded, smoke=args.smoke,
                requested=len(items), context_policy=args.context_policy,
                context_limit=config.text_config.max_position_embeddings)
    ids = {r['trial_id'] for r in selected}
    print(json.dumps(dict(requested=len(items), selected=len(selected), excluded=len(excluded),
        excluded_by_task=dict(Counter(f'{r["level"]}/{r["question"]}' for r in excluded))), indent=2), flush=True)
    if args.preflight:
        # Check a real processed input for every format/image-count combination, not only token estimates.
        checked = set()
        estimates = {r['trial_id']: r['input_tokens'] for r in selected}
        for item in items:
            if item['trial_id'] not in ids: continue
            key = (item['level'], item['question'], len(item['study_images']))
            if key in checked: continue
            checked.add(key)
            if args.backend == 'llava':
                encoded, _ = llava_inputs(item, root, processor)
                assert encoded['input_ids'].shape[1] == estimates[item['trial_id']], 'Preflight token estimate disagrees with processor'
            else:
                for block in make_messages(item,root,args.max_pixels)[0]['content']:
                    if block['type']=='image' and not Path(block['image']).is_file(): raise FileNotFoundError(block['image'])
        write_json(out/'preflight.json', dict(plan, checked_formats=len(checked), backend=args.backend,
            dataset_sha256=sha_file(root/'manifest.json'), model=str(Path(args.model).resolve())))
        if excluded and args.context_policy == 'error': raise SystemExit('Context overflow; inspect preflight.json, then explicitly choose --context-policy exclude if appropriate')
        return
    if excluded and args.context_policy == 'error': raise SystemExit('Context overflow; run --preflight to inspect; no predictions written')
    if not selected: raise SystemExit('No supported trials')
    import numpy as np
    import torch
    source_files = [*Path(__file__).parent.glob('*.py'), *Path('bindmem').glob('*.py')]
    model_path=Path(args.model)
    manifest = dict(version='2.0.0', runner='bindmem_extra-1', backend=args.backend,
        dataset_sha256=sha_file(root/'manifest.json'), checkpoint=checkpoint_identity(args.model),
        processor_files={f.name:sha_file(f) for pattern in ('*.json','*.jinja','*.model','*.txt') for f in model_path.glob(pattern)},
        split=args.split, levels=args.levels, questions=args.questions, selection_sha256=digest(plan),
        max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels, seed=args.seed,
        do_sample=False, attention='sdpa', dtype='float16' if args.backend=='llava' else 'auto', capture=False,
        image_policy='pad_to_square_clip_mean_then_native_336' if args.backend=='llava' else 'native_qwen_dynamic',
        prompt_policy='USER/ASSISTANT_interleaved_images' if args.backend=='llava' else 'native_chat_template',
        packages={k:version(k) for k in ['torch','transformers','pillow','numpy','accelerate']},
        source={str(f):sha_file(f) for f in source_files})
    imported=[]
    if args.reuse_from:
        source_run=Path(args.reuse_from)
        if source_run.resolve()==out.resolve():p.error('--reuse-from must be a different output directory')
        source_lock=(source_run/'.lock').open('a')
        try:fcntl.flock(source_lock,fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Source run is still writing; stop it before importing')
        imported=reuse_completed(source_run,manifest,ids)
        manifest['reuse_source']=dict(path=str(source_run.resolve()),
            manifest_sha256=sha_file(source_run/'manifest.json'),predictions_sha256=sha_file(source_run/'predictions.jsonl'))
    mpath=out/'manifest.json'
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise SystemExit('Resume manifest mismatch; use a new output directory')
    previous=recover_rows(out/'predictions.jsonl')
    if previous and not mpath.exists(): raise SystemExit('Predictions without manifest; use a new directory')
    if mpath.exists() and (not (out/'selection.json').is_file() or digest(json.loads((out/'selection.json').read_text())) != manifest['selection_sha256']):
        raise SystemExit('Missing or changed selection.json')
    write_json(out/'selection.json',plan); write_json(mpath,manifest)
    latest={r['trial_id']:r for r in previous}
    new_imports=[r for r in imported if r['trial_id'] not in latest]
    if new_imports:
        with (out/'predictions.jsonl').open('a') as stream:
            for r in new_imports:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        latest.update({r['trial_id']:r for r in new_imports})
    print(f'Reused {len(new_imports)} completed records this invocation',flush=True)
    pending=[r for r in items if r['trial_id'] in ids and (r['trial_id'] not in latest or latest[r['trial_id']].get('error'))]
    if args.limit: pending=pending[:args.limit]
    print(f'Pending this invocation: {len(pending)}',flush=True)
    if args.initialize_only or not pending: return
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable: inference must run on a GPU compute node')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    from transformers import AutoModelForImageTextToText
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,
        dtype=torch.float16 if args.backend=='llava' else 'auto',device_map={'':'cuda:0'},attn_implementation='sdpa')
    model.eval(); model.requires_grad_(False)
    write_json(out/'hardware.json',dict(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,model_config=model.config.to_dict()))
    if args.backend=='qwen':
        from qwen_vl_utils import process_vision_info
    estimates={r['trial_id']:r['input_tokens'] for r in selected}
    with (out/'predictions.jsonl').open('a') as stream:
        for index,item in enumerate(pending,1):
            start=time.monotonic();inputs=generated=None
            rec=dict(trial_id=item['trial_id'],timestamp=datetime.now(timezone.utc).isoformat(),raw_response=None,parsed_answer=None,error=None,activation_file=None)
            try:
                if args.backend=='llava':
                    inputs,prompt=llava_inputs(item,root,processor)
                    assert inputs['input_ids'].shape[1]==estimates[item['trial_id']]
                    inputs=inputs.to(model.device,torch.float16)
                else:
                    messages=make_messages(item,root,args.max_pixels)
                    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    ims,vids=process_vision_info(messages)
                    inputs=processor(text=[prompt],images=ims,videos=vids,padding=True,return_tensors='pt').to(model.device)
                nt=inputs['input_ids'].shape[1]
                if nt+args.max_new_tokens > config.text_config.max_position_embeddings:
                    raise ValueError('Context overflow; inputs are never truncated')
                with torch.inference_mode():
                    generated=model.generate(**inputs,do_sample=False,max_new_tokens=args.max_new_tokens)
                raw=processor.batch_decode(generated[:,nt:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
                rec.update(raw_response=raw,parsed_answer=parse_answer(raw,item['response_kind']),input_tokens=nt,prompt_sha256=digest(prompt))
                if 'image_grid_thw' in inputs:rec['image_grid_thw']=inputs['image_grid_thw'].tolist()
            except Exception as exc:
                rec['error']=f'{type(exc).__name__}: {exc}'
            finally:
                del inputs,generated
                if rec['error']:torch.cuda.empty_cache()
            rec['latency_ms']=round((time.monotonic()-start)*1000)
            stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            print(f'[{index}/{len(pending)}] {item["level"]}/{item["question"]} {rec["error"] or rec["raw_response"]}',flush=True)
            if rec['error']:raise SystemExit('Failed trial recorded; fix cause and resubmit to retry. No further trials attempted.')


if __name__=='__main__':main()
