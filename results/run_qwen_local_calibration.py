#!/usr/bin/env python3
"""Run local Qwen3-VL image attribute calibration and save resumable JSONL."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "binding_memory_integrated/core/calibration/inputs.jsonl"
IMAGE_ROOT = ROOT / "binding_memory_integrated/core"
PROMPT = (
    "Identify the object's color, shape, and surface texture. "
    "Return exactly one JSON object with keys color, shape, texture. "
    "Use these values only: colors red, blue, green, yellow, purple, orange; "
    "shapes circle, square, triangle, hexagon, star, diamond; "
    "textures dots, diagonal_stripes, grid, checkerboard, waves, chevrons. "
    "Do not include explanations or markdown."
)
VALID = {
    "color": {"red", "blue", "green", "yellow", "purple", "orange"},
    "shape": {"circle", "square", "triangle", "hexagon", "star", "diamond"},
    "texture": {"dots", "diagonal_stripes", "grid", "checkerboard", "waves", "chevrons"},
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_prediction(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    value = {field: str(value.get(field, "")).strip().lower() for field in VALID}
    return value if all(value[field] in VALID[field] for field in VALID) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", default="results/calibration_review/qwen_local_pilot_raw.jsonl")
    parser.add_argument("--limit", type=int, default=12, help="Evenly spread sample; use 432 for full calibration")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable only if required by the chosen model revision")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise SystemExit(f"Missing inference dependency: {exc}. Follow results/QWEN_LOCAL_SETUP.md first.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible to PyTorch. Check nvidia-smi and install the CUDA-enabled PyTorch wheel.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch: {torch.__version__}; model: {args.model}")

    items = read_jsonl(INPUTS)
    if not 1 <= args.limit <= len(items):
        parser.error(f"--limit must be between 1 and {len(items)}")
    if args.limit == len(items):
        selected = items
    else:
        indices = sorted({round(i * (len(items) - 1) / max(args.limit - 1, 1)) for i in range(args.limit)})
        selected = [items[i] for i in indices]

    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    processor_kwargs = {}
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        processor_kwargs["trust_remote_code"] = True
    print("Loading processor and model; the first run downloads model files.")
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    model = ModelClass.from_pretrained(args.model, **model_kwargs)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {row["calibration_id"] for row in read_jsonl(output)}
    pending = [x for x in selected if x["calibration_id"] not in done]
    print(f"Selected {len(selected)}; already complete {len(done)}; pending {len(pending)}")

    with output.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(pending, 1):
            image_path = (IMAGE_ROOT / item["image"]).resolve()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            start = time.monotonic()
            error, raw_text, prediction = None, None, None
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                new_tokens = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0].strip()
                prediction = parse_prediction(raw_text)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency = round((time.monotonic() - start) * 1000)
            record = {
                "calibration_id": item["calibration_id"], "model": args.model, "model_version": args.model,
                "timestamp": datetime.now(timezone.utc).isoformat(), "raw_response": raw_text,
                "predicted_color": prediction["color"] if prediction else None,
                "predicted_shape": prediction["shape"] if prediction else None,
                "predicted_texture": prediction["texture"] if prediction else None,
                "valid": prediction is not None, "latency_ms": latency, "error": error,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(pending)}] {item['calibration_id']} {'OK' if prediction else 'INVALID/ERROR'} ({latency} ms)")
            if args.pause:
                time.sleep(args.pause)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
