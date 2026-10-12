"""Structural tiers based on novelty cues, not measured model difficulty."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEME = {
    "version": "1.0",
    "kind": "structural_not_empirical",
    "empirically_calibrated": False,
    "basis": "availability_of_novel_attribute_rejection_cues",
    "levels": {
        "easy": {"rank": 1, "label": "L1 简单", "description": "三个属性全新，有三个可用于拒绝的线索。"},
        "medium": {"rank": 2, "label": "L2 中等", "description": "当前数据恰有一个新属性，需要找到这个拒绝线索。"},
        "hard": {"rank": 3, "label": "L3 困难", "description": "三个属性都见过，必须核对是否属于同一物体。"},
        "control": {"rank": None, "label": "Target 对照", "description": "完整组合出现过，作为各档共享的正例对照，不参与难度排序。"},
    },
    "notes": [
        "这是新异线索可用性的先验分层，不是 VLM 实测难度排名。",
        "2+1 与 1+1+1 同属 hard；二者的实际难度次序需分别测量。",
        "各档诱饵均为 NO；评测须同时报告共享 target 命中率，不能只报诱饵准确率。",
        "共享 target 不构成新增独立样本；不确定性应按 episode 重采样。",
    ],
}


def annotations(seen_feature_count, nearest_hamming_distance):
    novel = 3 - seen_feature_count
    if nearest_hamming_distance == 0:
        level = "control"
    elif novel == 3:
        level = "easy"
    elif novel > 0:
        level = "medium"
    else:
        level = "hard"
    return {"structural_difficulty": level, "novel_feature_count": novel}


def group_index(split, trials):
    return {
        "scheme_version": SCHEME["version"], "split": split,
        "shared_target_ids": [t["trial_id"] for t in trials if t["structural_difficulty"] == "control"],
        "tiers": {level: {
            "lure_ids": [t["trial_id"] for t in trials if t["structural_difficulty"] == level],
            "lure_types": sorted({t["test_type"] for t in trials if t["structural_difficulty"] == level}),
        } for level in ("easy", "medium", "hard")},
    }


def write_index(root, split, trials):
    directory = Path(root) / "difficulty"
    directory.mkdir(exist_ok=True)
    (directory / f"{split}.json").write_text(json.dumps(group_index(split, trials), ensure_ascii=False, indent=2) + "\n")


def annotate_existing(root):
    # Add metadata to a previous preview without regenerating images or inputs.
    from generate import write_jsonl
    root = Path(root).resolve()
    meta = json.loads((root / "dataset.json").read_text())
    if meta.get("generator") != "binding_dataset/generate.py":
        raise ValueError("Not a binding_dataset dataset")
    summary = json.loads((root / "summary.json").read_text())
    protected = sorted(p for directory in ("images", "inputs", "labels") for p in (root / directory).glob("*") if p.is_file())

    def fingerprint():
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

    before = fingerprint()
    for split in ("dev", "test"):
        episodes_path = root / "episodes" / f"{split}.jsonl"
        manifest_path = root / "manifests" / f"{split}.jsonl"
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        trials = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        for test in [t for ep in episodes for t in ep["tests"]] + trials:
            test.update(annotations(test["seen_feature_count"], test["nearest_hamming_distance"]))
        write_jsonl(episodes_path, episodes)
        write_jsonl(manifest_path, trials)
        write_index(root, split, trials)
        summary["splits"][split]["structural_difficulty_counts"] = dict(Counter(t["structural_difficulty"] for t in trials))
    meta["difficulty_scheme"] = SCHEME
    summary["difficulty_scheme_version"] = SCHEME["version"]
    for name, value in (("dataset.json", meta), ("summary.json", summary)):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    after = fingerprint()
    if before != after:
        raise RuntimeError("Images or model inputs/answer keys changed during annotation")
    report = {"scheme_version": SCHEME["version"], "protected_files": len(protected),
              "images_inputs_labels_unchanged": True, "before_sha256": before, "after_sha256": after,
              "annotator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note": "Original generator_sha256 retained as provenance of original image generation."}
    (root / "difficulty_annotation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "v0.1")
    print(json.dumps(annotate_existing(parser.parse_args().data), indent=2))
