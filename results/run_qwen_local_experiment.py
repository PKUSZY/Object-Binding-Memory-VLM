#!/usr/bin/env python3
"""Run the formal object-binding experiment with a local Qwen3-VL model."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "binding_memory_integrated/core"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_yes_no(text):
    if not text:
        return None
    value = text.strip().upper()
    return value if value in {"YES", "NO"} else None


def load_model(model_path, trust_remote_code=False):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass
    from qwen_vl_utils import process_vision_info

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible to PyTorch. Run nvidia-smi and check the GPU job allocation.")
    kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    processor_kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
        processor_kwargs["trust_remote_code"] = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)
    model = ModelClass.from_pretrained(str(model_path), **kwargs)
    model.eval()
    return torch, processor, model, process_vision_info


def make_messages(item):
    content = [{"type": "text", "text": item["instruction"]}]
    for image in item["study_images"]:
        content.append({"type": "image", "image": str((CORE / image).resolve())})
    if item["distractor_images"]:
        content.append({"type": "text", "text": "下面是干扰图片，它们不计入学习集合。"})
        for image in item["distractor_images"]:
            content.append({"type": "image", "image": str((CORE / image).resolve())})
    content.extend([
        {"type": "text", "text": "现在观察最后一张测试图片，只回答 YES 或 NO。"},
        {"type": "image", "image": str((CORE / item["test_image"]).resolve())},
    ])
    return [{"role": "user", "content": content}]


def infer(item, processor, model, process_vision_info, torch, max_new_tokens):
    messages = make_messages(item)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()
    del inputs, generated, new_tokens
    return text, round((time.monotonic() - start) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--output-dir", default="results/experiment_raw")
    parser.add_argument("--progress-log", default=None, help="Append structured start/end events to this JSONL file")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    split_names = ["dev", "test"] if args.split == "both" else [args.split]
    items = []
    for split in split_names:
        for item in read_jsonl(CORE / "inputs" / f"{split}.jsonl"):
            items.append((split, item))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_log) if args.progress_log else output_dir / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    for split in split_names:
        output = output_dir / f"{split}.jsonl"
        if output.exists():
            done[split] = {x["trial_id"] for x in read_jsonl(output)}
        else:
            done[split] = set()
    pending = [(split, item) for split, item in items if item["trial_id"] not in done[split]]
    print(f"Splits: {','.join(split_names)}; total: {len(items)}; pending: {len(pending)}")
    torch, processor, model, process_vision_info = load_model(Path(args.model), args.trust_remote_code)
    streams = {}
    progress = progress_path.open("a", encoding="utf-8")

    def log_event(event, **fields):
        row = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        progress.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress.flush()

    log_event("run_start", model=args.model, splits=split_names, total=len(items), pending=len(pending))
    try:
        for split in split_names:
            streams[split] = (output_dir / f"{split}.jsonl").open("a", encoding="utf-8")
        for index, (split, item) in enumerate(pending, 1):
            log_event("trial_start", index=index, total_pending=len(pending), split=split,
                      trial_id=item["trial_id"])
            raw, latency, error = None, None, None
            try:
                raw, latency = infer(item, processor, model, process_vision_info, torch, args.max_new_tokens)
                answer = parse_yes_no(raw)
            except Exception as exc:
                answer = None
                error = f"{type(exc).__name__}: {exc}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record = {
                "trial_id": item["trial_id"], "split": split, "model": args.model,
                "model_version": args.model, "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_response": raw, "parsed_answer": answer, "valid": answer is not None,
                "latency_ms": latency, "error": error,
            }
            streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            streams[split].flush()
            log_event("trial_end", index=index, total_pending=len(pending), split=split,
                      trial_id=item["trial_id"], raw_response=raw, parsed_answer=answer,
                      valid=answer is not None, latency_ms=latency, error=error)
            print(f"[{index}/{len(pending)}] {split} {item['trial_id']} {answer or 'INVALID/ERROR'} {latency or '-'} ms", flush=True)
            if args.pause:
                time.sleep(args.pause)
    finally:
        for stream in streams.values():
            stream.close()
        log_event("run_end", total=len(items), completed=len(pending))
        progress.close()
    print(f"Finished. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
