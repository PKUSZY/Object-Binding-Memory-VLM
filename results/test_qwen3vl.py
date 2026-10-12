#!/usr/bin/env python3
"""Load a local Qwen3-VL model and run one image inference smoke test."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "binding_memory_integrated/core/calibration/images"
PROMPT = (
    "Identify the object's color, shape, and surface texture. "
    "Return exactly one JSON object with keys color, shape, texture. "
    "Use these values only: colors red, blue, green, yellow, purple, orange; "
    "shapes circle, square, triangle, hexagon, star, diamond; "
    "textures dots, diagonal_stripes, grid, checkerboard, waves, chevrons. "
    "Do not include explanations or markdown."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--trust-remote-code", action="store_true")
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
        raise SystemExit("Missing dependency. Follow results/QWEN_LOCAL_SETUP.md first.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible to PyTorch. Run nvidia-smi and install CUDA-enabled PyTorch.")
    image = Path(args.image).expanduser() if args.image else next(DEFAULT_IMAGE.glob("*.png"))
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")
    model_path = Path(args.model).expanduser()
    kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    print(f"Loading model from {model_path}")
    processor = AutoProcessor.from_pretrained(str(model_path), **({"trust_remote_code": True} if args.trust_remote_code else {}))
    model = ModelClass.from_pretrained(str(model_path), **kwargs)
    model.eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image.resolve())},
        {"type": "text", "text": PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_tokens = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()
    print(json.dumps({"model": str(model_path), "image": str(image), "response": text}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
