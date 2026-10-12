#!/usr/bin/env python3
"""Download Qwen3-VL weights into a local directory for offline testing."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token; prefer HF_TOKEN environment variable")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface_hub. Run: python -m pip install -U huggingface_hub") from exc

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    token = args.token or os.environ.get("HF_TOKEN")
    print(f"Downloading {args.model}@{args.revision}")
    print(f"Local directory: {output}")
    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(output),
        token=token,
        resume_download=True,
    )
    required = ["config.json", "processor_config.json"]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise SystemExit(f"Download finished but expected files are missing: {', '.join(missing)}")
    print(f"Download complete: {output}")


if __name__ == "__main__":
    main()
