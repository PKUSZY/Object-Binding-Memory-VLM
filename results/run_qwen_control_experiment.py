#!/usr/bin/env python3
"""Run pairwise and strong-recombination controls with local Qwen3-VL."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from run_qwen_local_experiment import load_model, parse_yes_no


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "results/control_benchmark"
CORE = ROOT / "binding_memory_integrated/core"


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def messages(item, kind):
    if kind == "pairwise":
        return [{"role": "user", "content": [
            {"type": "text", "text": item["instruction"]},
            {"type": "image", "image": str((CORE / item["study_image"]).resolve())},
            {"type": "image", "image": str((CORE / item["test_image"]).resolve())},
        ]}]
    content = [{"type": "text", "text": item["instruction"]}]
    content += [{"type": "image", "image": str((CORE / x).resolve())} for x in item["study_images"]]
    content += [{"type": "text", "text": "现在观察测试图片，只回答 YES 或 NO。"},
                {"type": "image", "image": str((CORE / item["test_image"]).resolve())}]
    return [{"role": "user", "content": content}]


def infer(item, kind, processor, model, process_vision_info, torch):
    msg = messages(item, kind)
    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msg)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    new_tokens = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text, round((time.monotonic() - start) * 1000)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="models/Qwen3-VL-8B-Instruct"); p.add_argument("--output", default="results/control_benchmark/raw.jsonl"); p.add_argument("--trust-remote-code", action="store_true"); args = p.parse_args()
    controls = [("pairwise", CONTROL / "inputs/pairwise.jsonl"), ("strong_recombination", CONTROL / "inputs/strong_recombination.jsonl")]
    rows = [(kind, x) for kind, path in controls for x in read_jsonl(path)]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = {x["trial_id"] for x in read_jsonl(out)} if out.exists() else set()
    rows = [(kind, x) for kind, x in rows if x["trial_id"] not in done]
    torch, processor, model, process_vision_info = load_model(Path(args.model), args.trust_remote_code)
    with out.open("a", encoding="utf-8") as f:
        for i, (kind, item) in enumerate(rows, 1):
            raw, latency, error = None, None, None
            try: raw, latency = infer(item, kind, processor, model, process_vision_info, torch); answer = parse_yes_no(raw)
            except Exception as exc: answer = None; error = f"{type(exc).__name__}: {exc}"
            record = {"trial_id": item["trial_id"], "control": kind, "model": args.model, "timestamp": datetime.now(timezone.utc).isoformat(), "raw_response": raw, "parsed_answer": answer, "valid": answer in {"YES", "NO"}, "latency_ms": latency, "error": error}
            f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{i}/{len(rows)}] {kind} {item['trial_id']} {answer or 'INVALID/ERROR'} {latency or '-'} ms", flush=True)
    print(f"Saved {out}")


if __name__ == "__main__": main()
