#!/usr/bin/env python3
"""Generate independent perception and pairwise-binding control inputs.

The controls use existing opaque PNG stimuli and write model inputs plus a
private label file. They do not alter the 864-trial main benchmark.
"""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "binding_memory_integrated/core"
ATTRS = ("color", "shape", "texture")


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(value):
    return hashlib.sha256(("control-v1|" + value).encode()).hexdigest()[:24]


def opaque_path(obj):
    key = "|".join(obj[a] for a in ATTRS)
    return f"images/{digest(key)}.png"


def find_image(obj, image_index):
    """Resolve a control object's existing opaque image by matching groups/trials."""
    key = tuple(obj[a] for a in ATTRS)
    return image_index.get(key)


def pair_sets(study):
    return [
        {(o["color"], o["shape"]) for o in study},
        {(o["color"], o["texture"]) for o in study},
        {(o["shape"], o["texture"]) for o in study},
    ]


def strong_candidates(study):
    colors, shapes, textures = ({o[a] for o in study} for a in ATTRS)
    seen = {tuple(o[a] for a in ATTRS) for o in study}
    pairs = pair_sets(study)
    return [dict(zip(ATTRS, triple)) for triple in itertools.product(colors, shapes, textures)
            if triple not in seen and all(pair in pairs[i] for i, pair in enumerate([
                (triple[0], triple[1]), (triple[0], triple[2]), (triple[1], triple[2])]))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/control_benchmark")
    args = parser.parse_args()
    out = Path(args.output)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    groups = read_jsonl(CORE / "groups.jsonl")
    image_index = {}
    for candidate in read_jsonl(CORE / "trials.jsonl"):
        image_index[tuple(candidate["probe_object"][a] for a in ATTRS)] = candidate["probe_image"]
    for group in groups:
        for image_obj, image in zip(group["study_objects"], group["study_images"]):
            image_index[tuple(image_obj[a] for a in ATTRS)] = image
    # One canonical no-distractor group per seed/study count.
    canonical = {(g["seed"], g["study_count"]): g for g in groups if g["distractor_count"] == 0}
    pair_inputs, pair_labels, strong_inputs, strong_labels = [], [], [], []
    for (seed, n), g in sorted(canonical.items()):
        study = g["study_objects"]
        # Direct two-image identity control: one exact match and one mismatch
        # for each study object, with mismatch changing exactly one attribute.
        for i, obj in enumerate(study):
            same_id = f"pair_s{seed:02d}_n{n:02d}_{i:02d}_same"
            same = {"trial_id": same_id, "instruction": "比较两张物体图片。仅当颜色、形状和纹理三项都相同，回答 YES；否则回答 NO。只输出 YES 或 NO。",
                    "study_image": find_image(obj, image_index), "test_image": find_image(obj, image_index)}
            pair_inputs.append(same); pair_labels.append({"trial_id": same_id, "answer": "YES", "relation": "identical", "changed_attribute": None, "seed": seed, "study_count": n})
            for attr in ATTRS:
                alt = dict(obj)
                alt_values = sorted({o[attr] for o in study} - {obj[attr]})
                if not alt_values: continue
                alt[attr] = alt_values[0]
                trial_id = f"pair_s{seed:02d}_n{n:02d}_{i:02d}_{attr}"
                alt_image = find_image(alt, image_index)
                if alt_image is None:
                    continue
                pair_inputs.append({"trial_id": trial_id, "instruction": same["instruction"], "study_image": find_image(obj, image_index), "test_image": alt_image})
                pair_labels.append({"trial_id": trial_id, "answer": "NO", "relation": "one_attribute_changed", "changed_attribute": attr, "seed": seed, "study_count": n})
        candidates = strong_candidates(study)
        if candidates:
            rng = random.Random(seed * 10007 + n)
            obj = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
            trial_id = f"strong_s{seed:02d}_n{n:02d}"
            instruction = "请依次观察学习图片，再观察测试图片。测试物体的颜色、形状和纹理是否作为完整组合出现在某一张学习图片中？只回答 YES 或 NO。"
            strong_inputs.append({"trial_id": trial_id, "instruction": instruction, "study_images": g["study_images"], "distractor_images": [], "test_image": find_image(obj, image_index)})
            strong_labels.append({"trial_id": trial_id, "answer": "NO", "probe_type": "strong_pairwise_recombination", "seed": seed, "study_count": n, "pairwise_seen_count": 3})
    (out / "inputs" / "pairwise.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pair_inputs) + "\n")
    (out / "labels" / "pairwise.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pair_labels) + "\n")
    (out / "inputs" / "strong_recombination.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in strong_inputs) + "\n")
    (out / "labels" / "strong_recombination.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in strong_labels) + "\n")
    metadata = {"version": "1.0", "pairwise_trials": len(pair_inputs), "strong_recombination_trials": len(strong_inputs),
                "strong_conditions": {"study_counts": sorted({x["study_count"] for x in strong_labels}), "pairwise_seen_count": 3,
                                      "full_triple_seen": False},
                "note": "Strong condition is impossible for the original 4-image groups; it is reported only for groups where all three attribute pairs are present but the full triple is absent."}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
