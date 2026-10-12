#!/usr/bin/env python3
"""Run image attribute calibration against an OpenAI-compatible Qwen endpoint.

The calibration ID and answer labels remain local. Each image is sent as its own
request with only the fixed instruction and image bytes.
"""

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
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


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_answer(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    normalized = {k: str(value.get(k, "")).strip().lower() for k in VALID}
    return normalized if all(normalized[k] in VALID[k] for k in VALID) else None


def request_one(endpoint, token, model, image_path, timeout, retries):
    raw = image_path.read_bytes()
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
            return text, parse_answer(text), round((time.monotonic() - start) * 1000), None
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            err = f"HTTP {exc.code}: {detail}"
            if exc.code < 500 or attempt >= retries:
                break
            time.sleep(min(2 ** attempt, 8))
    return None, None, None, err


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("QWEN_BASE_URL"), help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--output", default="results/calibration_review/qwen_pilot_raw.jsonl")
    parser.add_argument("--limit", type=int, default=12, help="Spread sample size; use 432 for the full calibration")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between requests")
    args = parser.parse_args()

    token = os.environ.get("QWEN_API_KEY")
    if not args.endpoint:
        parser.error("Set QWEN_BASE_URL or pass --endpoint")
    if not token:
        parser.error("Set QWEN_API_KEY in the environment; do not place it in files or command history")
    all_inputs = jsonl(INPUTS)
    if not 1 <= args.limit <= len(all_inputs):
        parser.error(f"--limit must be between 1 and {len(all_inputs)}")
    if args.limit == len(all_inputs):
        selected = all_inputs
    else:
        # Evenly spread pilot items across the complete calibration ordering.
        indices = sorted({round(i * (len(all_inputs) - 1) / max(args.limit - 1, 1)) for i in range(args.limit)})
        selected = [all_inputs[i] for i in indices]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {row["calibration_id"] for row in jsonl(output)}
    pending = [x for x in selected if x["calibration_id"] not in done]
    print(f"Model: {args.model}; selected: {len(selected)}; already complete: {len(done)}; pending: {len(pending)}")

    with output.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(pending, 1):
            image_path = IMAGE_ROOT / item["image"]
            start = time.monotonic()
            text, prediction, latency, error = request_one(
                args.endpoint, token, args.model, image_path, args.timeout, args.retries
            )
            record = {
                "calibration_id": item["calibration_id"],
                "model": args.model,
                "model_version": args.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_response": text,
                "predicted_color": prediction["color"] if prediction else None,
                "predicted_shape": prediction["shape"] if prediction else None,
                "predicted_texture": prediction["texture"] if prediction else None,
                "valid": prediction is not None,
                "latency_ms": latency if latency is not None else round((time.monotonic() - start) * 1000),
                "error": error,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(pending)}] {item['calibration_id']} {'OK' if prediction else 'ERROR'}")
            if args.pause:
                time.sleep(args.pause)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
