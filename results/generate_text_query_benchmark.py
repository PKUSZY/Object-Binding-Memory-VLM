#!/usr/bin/env python3
"""Build a text-probe track paired with the image-probe benchmark."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "binding_memory_integrated/core"
OUT = ROOT / "results/text_query_benchmark"

COLOR_ZH = {
    "red": "红色", "blue": "蓝色", "green": "绿色",
    "yellow": "黄色", "purple": "紫色", "orange": "橙色",
}
SHAPE_ZH = {
    "circle": "圆形", "square": "方形", "triangle": "三角形",
    "hexagon": "六边形", "star": "星形", "diamond": "菱形",
}
TEXTURE_ZH = {
    "dots": "点状纹理", "diagonal_stripes": "斜向条纹纹理",
    "grid": "网格纹理", "checkerboard": "棋盘格纹理",
    "waves": "波纹纹理", "chevrons": "人字纹",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    trials = read_jsonl(CORE / "trials.jsonl")
    if len(trials) != 864 or {trial["split"] for trial in trials} != {"dev", "test"}:
        raise SystemExit(f"Unexpected source trial set: {len(trials)} trials")
    inputs = {"dev": [], "test": []}
    labels = {"dev": [], "test": []}
    for trial in trials:
        obj = trial["probe_object"]
        query = (
            f"问题：在刚才展示的学习图片中，是否出现过一个"
            f"{COLOR_ZH[obj['color']]}{SHAPE_ZH[obj['shape']]}、"
            f"表面具有{TEXTURE_ZH[obj['texture']]}的物体？\n"
            "仅回答 YES 或 NO。"
        )
        split = trial["split"]
        inputs[split].append({
            "trial_id": trial["trial_id"],
            "instruction": (
                "请依次观察并记住学习图片中的物体。一个物体由颜色、形状和表面纹理共同定义。"
                "干扰图片不计入学习集合。位置、大小和纹理细节的位置变化不改变物体身份。"
                "之后我会用文字询问一个属性组合是否曾在某一张学习图片中完整出现。"
                "请根据学习图片回答，且只输出 YES 或 NO。"
            ),
            "study_images": trial["study_images"],
            "distractor_images": trial["distractor_images"],
            "text_query": query,
        })
        labels[split].append({
            "trial_id": trial["trial_id"],
            "group_id": trial["group_id"],
            "answer": trial["answer"],
            "probe_type": trial["probe_type"],
            "structural_difficulty": trial["structural_difficulty"],
            "seed": trial["seed"],
            "study_count": trial["study_count"],
            "distractor_count": trial["distractor_count"],
        })

    for split in ("dev", "test"):
        write_jsonl(OUT / "inputs" / f"{split}.jsonl", inputs[split])
        write_jsonl(OUT / "labels" / f"{split}.jsonl", labels[split])
    metadata = {
        "condition": "image_study_text_probe",
        "total_trials": sum(map(len, inputs.values())),
        "by_split": {split: len(rows) for split, rows in inputs.items()},
        "probe_images_in_inputs": False,
        "paired_with": "binding_memory_integrated/core",
        "note": "Study and distractor stimuli match the image-probe track; only the final probe modality changes. Labels are stored separately.",
    }
    if any("test_image" in row or "probe_object" in row for rows in inputs.values() for row in rows):
        raise SystemExit("Probe input unexpectedly contains a test image or structured probe attributes")
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
