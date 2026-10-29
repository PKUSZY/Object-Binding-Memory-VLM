"""Local deterministic VLM inference, resumable outputs and pre-answer activations."""
import argparse
import fcntl
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .common import VERSION, digest, parse_answer, read_jsonl, sha_file, write_json


def make_messages(item, root, max_pixels):
    content = [{'type': 'text', 'text': item['instruction']}]
    for i, p in enumerate(item['study_images']):
        content += [{'type': 'text', 'text': f'Study image {i+1}:'},
                    {'type': 'image', 'image': str((root/p).resolve()), 'max_pixels': max_pixels}]
    content += [{'type': 'text', 'text': 'End of study. Answer the following query.'}]
    for block in item['query']:
        block = dict(block)
        if block['type'] == 'image':
            block['image'] = str((root/block['image']).resolve())
            block['max_pixels'] = max_pixels
        content.append(block)
    return [{'role': 'user', 'content': content}]


def selected_layers(n, spec):
    if spec == 'all': return list(range(n))
    if spec == 'auto': return sorted({0, n//2, n-1})
    result = [int(x) for x in spec.split(',')]
    if not result or min(result) < 0 or max(result) >= n:
        raise ValueError(f'Layer indices must be in [0,{n-1}]')
    return sorted(set(result))


class Capture:
    """Hooks copy only pooled vectors, never retain full hidden-state stacks.

    Supports the installed Qwen3-VL architecture. Other architectures can run
    behavioral inference but must explicitly implement their own probe adapter.
    Decoder hooks observe block output (before final norm and any subsequent
    deepstack addition). Last prompt token is before any generated answer.
    """
    def __init__(self, model, inputs, n_study, layers):
        import torch
        self.values, self.handles = {}, []
        base = getattr(model, 'model', None)
        if not (hasattr(base, 'visual') and hasattr(base, 'language_model') and
                hasattr(base.visual, 'blocks') and hasattr(base.language_model, 'layers')):
            raise ValueError('Activation capture currently requires Qwen3-VL')
        counts = inputs['image_grid_thw'].prod(-1)
        n_patches = int(counts[:n_study].sum())
        merge = base.visual.spatial_merge_size**2
        n_tokens = n_patches // merge
        positions = torch.where(inputs['input_ids'][0] == model.config.image_token_id)[0][:n_tokens]
        if len(positions) != n_tokens:
            raise ValueError('Could not locate all study image tokens')
        seq_len = inputs['input_ids'].shape[1]

        def vision_hook(index):
            def hook(module, args, output):
                key = f'vision.{index}.study'
                if key in self.values: return
                h = output[0] if isinstance(output, tuple) else output
                self.values[key] = h[:n_patches].float().mean(0).detach().cpu().numpy()
            return hook

        def decoder_hook(index):
            def hook(module, args, output):
                key = f'decoder.{index}.study'
                if key in self.values: return
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] != seq_len: return
                self.values[key] = h[0, positions.to(h.device)].float().mean(0).detach().cpu().numpy()
                self.values[f'decoder.{index}.answer'] = h[0, -1].float().detach().cpu().numpy()
            return hook
        for i in selected_layers(len(base.visual.blocks), layers):
            self.handles.append(base.visual.blocks[i].register_forward_hook(vision_hook(i)))
        for i in selected_layers(len(base.language_model.layers), layers):
            self.handles.append(base.language_model.layers[i].register_forward_hook(decoder_hook(i)))

    def close(self):
        for h in self.handles: h.remove()


def recover_rows(path):
    """Recover only a torn final append; reject corruption anywhere else."""
    if not path.exists(): return []
    data = path.read_bytes()
    lines, rows, offset = data.splitlines(keepends=True), [], 0
    for i, line in enumerate(lines):
        try: row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError(f'Corrupted JSONL at line {i+1}: {path}')
            with path.open('r+b') as f: f.truncate(offset)
            print('Recovered interrupted final JSONL append', flush=True)
            break
        rows.append(row)
        offset += len(line)
    if rows and data and not data.endswith(b'\n') and offset == len(data):
        with path.open('ab') as f: f.write(b'\n')
    return rows


def checkpoint_identity(model):
    path = Path(model)
    if not path.is_dir(): return {'hub_id': model}
    files = sorted(path.glob('*.safetensors')) + sorted(path.glob('*.bin'))
    return {'path': str(path.resolve()), 'config_sha256': sha_file(path/'config.json'),
            'weights': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/bindmem_v2')
    p.add_argument('--model', default='models/Qwen3-VL-8B-Instruct')
    p.add_argument('--revision', default=None, help='Pin a Hub commit for remote checkpoints')
    p.add_argument('--output', default='runs/bindmem_v2/qwen3vl8b')
    p.add_argument('--split', choices=['dev','test','both'], default='both')
    p.add_argument('--levels', nargs='+', default=None)
    p.add_argument('--questions', nargs='+', default=None)
    p.add_argument('--limit', type=int, default=None, help='Maximum pending trials this invocation')
    p.add_argument('--max-new-tokens', type=int, default=80)
    p.add_argument('--max-pixels', type=int, default=802816)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--capture', action='store_true', help='L0 and Q2 L2/L3/L5 only; no query-image leakage')
    p.add_argument('--layers', default='auto', help='auto (first/middle/last), all, or comma-separated indices')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--trust-remote-code', action='store_true')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    root, out = Path(args.data), Path(args.output)
    splits = ['dev','test'] if args.split == 'both' else [args.split]
    dataset_manifest = json.loads((root/'manifest.json').read_text())
    for split in splits:
        name = f'inputs/{split}.jsonl'
        if sha_file(root/name) != dataset_manifest['files'][name]:
            raise SystemExit(f'Input file fingerprint mismatch: {name}')
    items = [r for s in splits for r in read_jsonl(root/'inputs'/f'{s}.jsonl')
             if (not args.levels or r['level'] in args.levels) and
                (not args.questions or r['question'] in args.questions)]
    if not items: p.error('No trials match filters')
    if args.dry_run:
        # No labels, no weights and no output records are loaded on this path.
        for item in items:
            for b in make_messages(item, root, args.max_pixels)[0]['content']:
                if b['type'] == 'image' and not Path(b['image']).is_file():
                    raise FileNotFoundError(b['image'])
        print(json.dumps({'selected': len(items), 'example': make_messages(items[0], root, args.max_pixels)}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.lock').open('w')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another process is writing this output directory')
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable; run this command on a GPU compute node')
    manifest = dict(version=VERSION, dataset_sha256=sha_file(root/'manifest.json'),
                    checkpoint=checkpoint_identity(args.model), revision=args.revision,
                    split=args.split, levels=args.levels, questions=args.questions,
                    max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels,
                    seed=args.seed, capture=args.capture, layers=args.layers, trust_remote_code=args.trust_remote_code,
                    do_sample=False, dtype='auto', attention='sdpa',
                    packages={k: version(k) for k in ['torch','transformers','qwen-vl-utils','pillow','numpy']},
                    source={f.name: sha_file(f) for f in Path(__file__).parent.glob('*.py')})
    mpath = out/'manifest.json'
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        raise SystemExit('Resume manifest mismatch (dataset/model/settings/code/env). Use a new --output directory.')
    previous = recover_rows(out/'predictions.jsonl')
    if previous and not mpath.exists(): raise SystemExit('Existing predictions lack a manifest; use a new output directory')
    write_json(mpath, manifest)
    latest = {r['trial_id']: r for r in previous}
    pending = [r for r in items if r['trial_id'] not in latest or latest[r['trial_id']].get('error')]
    if args.limit: pending = pending[:args.limit]
    print(f'Selected {len(items)}, pending this invocation {len(pending)}', flush=True)
    if not pending: return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=args.revision,
                trust_remote_code=args.trust_remote_code, dtype='auto', device_map='auto', attn_implementation='sdpa')
    model.eval()
    for param in model.parameters(): param.requires_grad_(False)
    write_json(out/'hardware.json', dict(gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                                        cuda=torch.version.cuda, model_config=model.config.to_dict()))
    errors = 0
    with (out/'predictions.jsonl').open('a', encoding='utf-8') as stream:
        for index, item in enumerate(pending, 1):
            start = time.monotonic()
            capture, inputs, generated = None, None, None
            record = dict(trial_id=item['trial_id'], timestamp=datetime.now(timezone.utc).isoformat(),
                          raw_response=None, parsed_answer=None, error=None, activation_file=None)
            try:
                messages = make_messages(item, root, args.max_pixels)
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ims, vids = process_vision_info(messages)
                inputs = processor(text=[prompt], images=ims, videos=vids, padding=True, return_tensors='pt').to(model.device)
                eligible = item['level'] == 'L0' or (item['question'] == 'Q2' and item['level'] in ('L2','L3','L5'))
                if args.capture and eligible:
                    capture = Capture(model, inputs, len(item['study_images']), args.layers)
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
                raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0].strip()
                record.update(raw_response=raw, parsed_answer=parse_answer(raw, item['response_kind']),
                              input_tokens=inputs['input_ids'].shape[1],
                              image_grid_thw=inputs['image_grid_thw'].tolist())
                if capture:
                    if not capture.values or any(not np.isfinite(v).all() for v in capture.values.values()):
                        raise RuntimeError('Activation hooks captured empty or non-finite values')
                    dest = out/'activations'/f'{item["trial_id"]}.npz'
                    dest.parent.mkdir(exist_ok=True)
                    with dest.with_suffix('.tmp').open('wb') as f:
                        np.savez_compressed(f, **capture.values)
                    dest.with_suffix('.tmp').replace(dest)
                    record['activation_file'] = str(dest.relative_to(out))
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                errors += 1
            finally:
                if capture: capture.close()
                del inputs, generated
                if record['error']: torch.cuda.empty_cache()
            record['latency_ms'] = round((time.monotonic()-start)*1000)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            print(f'[{index}/{len(pending)}] {item["level"]} {item["question"]} {record["error"] or record["raw_response"]}', flush=True)
            if record['error']:
                # A systemic failure must not silently produce thousands of error rows.
                raise SystemExit('Trial failed; record saved. Fix the cause and rerun to retry it.')
    print(f'Completed invocation; error count {errors}')


if __name__ == '__main__': main()
