#!/usr/bin/env python3
"""Run the paired image-study/text-probe track with local Qwen3-VL."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from run_qwen_local_experiment import load_model, parse_yes_no


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "results/text_query_benchmark"
CORE = ROOT / "binding_memory_integrated/core"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_messages(item):
    content = [{"type": "text", "text": item["instruction"]}]
    content.extend({"type": "image", "image": str((CORE / image).resolve())}
                   for image in item["study_images"])
    if item["distractor_images"]:
        content.append({"type": "text", "text": "下面是干扰图片，它们不计入学习集合。"})
        content.extend({"type": "image", "image": str((CORE / image).resolve())}
                       for image in item["distractor_images"])
    content.append({"type": "text", "text": item["text_query"]})
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
    return text, round((time.monotonic() - start) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--output-dir", default="results/text_query_benchmark/raw")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    splits = ("dev", "test") if args.split == "both" else (args.split,)
    inputs = {split: read_jsonl(TRACK / "inputs" / f"{split}.jsonl") for split in splits}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {}
    for split in splits:
        path = output_dir / f"{split}.jsonl"
        existing[split] = {row["trial_id"] for row in read_jsonl(path)} if path.exists() else set()
    pending = {split: [item for item in inputs[split] if item["trial_id"] not in existing[split]]
               for split in splits}
    total = sum(map(len, inputs.values()))
    pending_total = sum(map(len, pending.values()))
    print(f"Splits: {','.join(splits)}; total: {total}; pending: {pending_total}")

    torch, processor, model, process_vision_info = load_model(Path(args.model), args.trust_remote_code)
    streams = {split: (output_dir / f"{split}.jsonl").open("a", encoding="utf-8") for split in splits}
    progress_path = output_dir / "progress.jsonl"
    try:
        index = 0
        for split in splits:
            for item in pending[split]:
                index += 1
                raw, latency, error = None, None, None
                try:
                    raw, latency = infer(item, processor, model, process_vision_info,
                                         torch, args.max_new_tokens)
                    answer = parse_yes_no(raw)
                except Exception as exc:
                    answer = None
                    error = f"{type(exc).__name__}: {exc}"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                record = {
                    "trial_id": item["trial_id"], "split": split,
                    "condition": "image_study_text_probe", "model": args.model,
                    "model_version": args.model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw_response": raw, "parsed_answer": answer,
                    "valid": answer in {"YES", "NO"},
                    "latency_ms": latency, "error": error,
                }
                streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                streams[split].flush()
                with progress_path.open("a", encoding="utf-8") as progress:
                    progress.write(json.dumps({"index": index, "total_pending": pending_total,
                                               "split": split, "trial_id": item["trial_id"],
                                               "answer": answer, "latency_ms": latency,
                                               "error": error}, ensure_ascii=False) + "\n")
                print(f"[{index}/{pending_total}] {split} {item['trial_id']} "
                      f"{answer or 'INVALID/ERROR'} {latency or '-'} ms", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    print(f"Finished. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
