#!/usr/bin/env python3
"""Score the image-study/text-probe track independently."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "results/text_query_benchmark"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(rows):
    n = len(rows)
    valid = sum(row["valid"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {"n": n, "valid": valid, "invalid": n - valid, "correct": correct,
            "accuracy": correct / n if n else None,
            "valid_rate": valid / n if n else None,
            "accuracy_on_valid": correct / valid if valid else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="results/text_query_benchmark/raw")
    parser.add_argument("--out-dir", default="results/text_query_benchmark/scored")
    args = parser.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for split in ("dev", "test"):
        raw = read_jsonl(raw_dir / f"{split}.jsonl")
        labels = read_jsonl(TRACK / "labels" / f"{split}.jsonl")
        label_map = {row["trial_id"]: row for row in labels}
        raw_map = {row["trial_id"]: row for row in raw}
        if len(raw_map) != len(raw):
            raise SystemExit(f"{split}: duplicate trial_id in raw output")
        if set(raw_map) != set(label_map):
            raise SystemExit(f"{split}: raw/label mismatch: {len(set(label_map)-set(raw_map))} missing, "
                             f"{len(set(raw_map)-set(label_map))} unknown")
        for trial_id, result in raw_map.items():
            label = label_map[trial_id]
            valid = result.get("parsed_answer") in {"YES", "NO"} and result.get("valid") is True
            scored.append({**result, **label, "valid": valid,
                           "correct": valid and result["parsed_answer"] == label["answer"]})

    scored.sort(key=lambda row: row["trial_id"])
    with (out_dir / "scored.jsonl").open("w", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_split = {split: [row for row in scored if row["split"] == split]
                for split in ("dev", "test")}
    by_type = defaultdict(list)
    by_condition = defaultdict(list)
    for row in scored:
        by_type[row["probe_type"]].append(row)
        by_condition[f"n{row['study_count']:02d}_d{row['distractor_count']}"].append(row)
    summary = {
        "condition": "image_study_text_probe",
        "total_trials": len(scored),
        "overall": metric(scored),
        "by_split": {split: metric(rows) for split, rows in by_split.items()},
        "by_probe_type": {key: metric(rows) for key, rows in sorted(by_type.items())},
        "by_condition": {key: metric(rows) for key, rows in sorted(by_condition.items())},
    }
    for key, rows in by_type.items():
        if key == "target":
            summary["by_probe_type"][key]["target_hit_rate"] = sum(
                row["parsed_answer"] == "YES" for row in rows if row["valid"]) / max(1, sum(row["valid"] for row in rows))
        else:
            summary["by_probe_type"][key]["false_accept_rate"] = sum(
                row["parsed_answer"] == "YES" for row in rows if row["valid"]) / max(1, sum(row["valid"] for row in rows))
            summary["by_probe_type"][key]["correct_reject_rate"] = sum(
                row["parsed_answer"] == "NO" for row in rows if row["valid"]) / max(1, sum(row["valid"] for row in rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
