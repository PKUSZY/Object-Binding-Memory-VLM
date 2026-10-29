#!/usr/bin/env python3
"""Download a pinned HF checkpoint (or only its small processor files)."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[1]/'.cache/huggingface'))
from huggingface_hub import HfApi, snapshot_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='llava-hf/llava-1.5-7b-hf')
    p.add_argument('--revision',default='main')
    p.add_argument('--output',default='models/llava-1.5-7b-hf')
    p.add_argument('--metadata-only',action='store_true')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pin=out/'download_manifest.json'
    if pin.exists():
        previous=json.loads(pin.read_text())
        if previous['repo']!=args.repo or (args.revision!='main' and args.revision!=previous['requested_revision'] and args.revision!=previous['commit']):
            raise SystemExit('Checkpoint identity changed; choose a new output directory')
        commit=previous['commit']
    else:
        commit=HfApi().model_info(args.repo,revision=args.revision).sha
        pin.write_text(json.dumps(dict(repo=args.repo,requested_revision=args.revision,commit=commit),indent=2)+'\n')
    snapshot_download(args.repo,revision=commit,local_dir=out,
                      allow_patterns=['*.json','*.jinja','*.model','*.txt'] if args.metadata_only else
                      ['*.json','*.jinja','*.model','*.txt','*.safetensors'],max_workers=4)
    from transformers import AutoConfig, AutoProcessor
    config=AutoConfig.from_pretrained(out,local_files_only=True)
    processor=AutoProcessor.from_pretrained(out,local_files_only=True,use_fast=True)
    print('Pinned commit:',commit,'model:',config.model_type,'processor:',type(processor).__name__)
    print('Native context limit:',config.text_config.max_position_embeddings)
    print('Metadata only; weights NOT downloaded.' if args.metadata_only else 'Download complete; run GPU smoke checks before formal inference.')


if __name__=='__main__':main()
