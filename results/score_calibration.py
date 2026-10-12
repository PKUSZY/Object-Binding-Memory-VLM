#!/usr/bin/env python3
"""Score manually normalized VLM calibration responses against the private key."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = {
    "color": {"red", "blue", "green", "yellow", "purple", "orange"},
    "shape": {"circle", "square", "triangle", "hexagon", "star", "diamond"},
    "texture": {"dots", "diagonal_stripes", "grid", "checkerboard", "waves", "chevrons"},
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/calibration_review/calibration_raw.jsonl")
    parser.add_argument("--labels", default="binding_memory_integrated/core/calibration/labels.jsonl")
    parser.add_argument("--out", default="results/calibration_review")
    parser.add_argument("--allow-subset", action="store_true", help="Score a pilot subset without requiring all 432 IDs")
    args = parser.parse_args()

    raw = read_jsonl(args.raw)
    labels = read_jsonl(args.labels)
    label_map = {x["calibration_id"]: x for x in labels}
    raw_ids = [x.get("calibration_id") for x in raw]
    if len(raw_ids) != len(set(raw_ids)):
        raise SystemExit("Duplicate calibration_id in raw responses")
    extra = set(raw_ids) - set(label_map)
    missing = set(label_map) - set(raw_ids)
    if extra or (missing and not args.allow_subset):
        raise SystemExit(f"ID mismatch: {len(missing)} missing, {len(extra)} unknown")
    if not raw:
        raise SystemExit("No raw calibration responses to score")

    attr_stats = {a: Counter() for a in FIELDS}
    confusions = {a: defaultdict(Counter) for a in FIELDS}
    scored = []
    invalid = 0
    for row in raw:
        truth = label_map[row["calibration_id"]]["attributes"]
        predicted = {a: row.get(f"predicted_{a}") for a in FIELDS}
        valid = row.get("valid") is True and all(predicted[a] in FIELDS[a] for a in FIELDS)
        invalid += not valid
        all_correct = valid and all(predicted[a] == truth[a] for a in FIELDS)
        for attr in FIELDS:
            attr_stats[attr]["n"] += 1
            if valid:
                attr_stats[attr]["valid_n"] += 1
                correct = predicted[attr] == truth[attr]
                attr_stats[attr]["correct"] += correct
                confusions[attr][truth[attr]][predicted[attr]] += 1
            else:
                attr_stats[attr]["correct"] += 0
                confusions[attr][truth[attr]]["INVALID"] += 1
        scored.append({**row, "true_color": truth["color"], "true_shape": truth["shape"],
                       "true_texture": truth["texture"], "all_attributes_correct": all_correct,
                       "scored_valid": valid})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "calibration_scored.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {"items": len(raw), "full_calibration_items": len(label_map), "coverage": len(raw) / len(label_map),
              "invalid_items": int(invalid), "invalid_rate": invalid / len(raw),
              "joint_accuracy": sum(x["all_attributes_correct"] for x in scored) / len(scored) if scored else None,
              "attribute_accuracy": {}}
    for attr, counts in attr_stats.items():
        report["attribute_accuracy"][attr] = {
            "correct_of_all_items": counts["correct"] / counts["n"] if counts["n"] else None,
            "correct_of_valid_items": counts["correct"] / counts["valid_n"] if counts["valid_n"] else None,
            "valid_items": counts["valid_n"], "total_items": counts["n"],
            "confusion": {truth: dict(preds) for truth, preds in confusions[attr].items()},
        }
    (out / "calibration_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (out / "calibration_mistakes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["calibration_id", "true_color", "predicted_color", "true_shape", "predicted_shape",
                         "true_texture", "predicted_texture", "raw_response", "valid"])
        for row in scored:
            if not row["all_attributes_correct"]:
                writer.writerow([row["calibration_id"], row["true_color"], row.get("predicted_color"),
                                 row["true_shape"], row.get("predicted_shape"), row["true_texture"],
                                 row.get("predicted_texture"), row.get("raw_response"), row.get("valid")])
    print(json.dumps({k: report[k] for k in ("items", "invalid_items", "invalid_rate", "joint_accuracy", "attribute_accuracy")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
