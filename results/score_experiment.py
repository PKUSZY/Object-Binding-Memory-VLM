#!/usr/bin/env python3
"""Join formal Qwen answers with private labels and write summary metrics."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "binding_memory_integrated/core"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(rows):
    n = len(rows)
    valid = sum(r["valid"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    return {"n": n, "valid": valid, "invalid": n - valid,
            "correct": correct, "accuracy": correct / n if n else None,
            "valid_rate": valid / n if n else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="results/experiment_raw")
    parser.add_argument("--out-dir", default="results/experiment_scored")
    args = parser.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for split in ("dev", "test"):
        raw = read_jsonl(raw_dir / f"{split}.jsonl")
        labels = read_jsonl(CORE / "labels" / f"{split}.jsonl")
        label_map = {r["trial_id"]: r for r in labels}
        raw_map = {r["trial_id"]: r for r in raw}
        if len(raw_map) != len(raw):
            raise SystemExit(f"{split}: duplicate trial_id in raw output")
        missing = set(label_map) - set(raw_map)
        extra = set(raw_map) - set(label_map)
        if missing or extra:
            raise SystemExit(f"{split}: {len(missing)} missing and {len(extra)} unknown trial IDs")
        for trial_id, answer in raw_map.items():
            label = label_map[trial_id]
            parsed = answer.get("parsed_answer")
            valid = parsed in {"YES", "NO"} and answer.get("valid") is True
            scored.append({**answer, "answer": label["answer"], "probe_type": label["probe_type"],
                           "structural_difficulty": label["structural_difficulty"],
                           "seed": label["seed"], "study_count": label["study_count"],
                           "distractor_count": label["distractor_count"],
                           "valid": valid, "correct": valid and parsed == label["answer"]})

    with (out_dir / "scored.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(scored, key=lambda x: x["trial_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = {
        "overall": scored,
        "split": {s: [r for r in scored if r["split"] == s] for s in ("dev", "test")},
        "probe_type": defaultdict(list),
        "condition": defaultdict(list),
    }
    for row in scored:
        groups["probe_type"][row["probe_type"]].append(row)
        groups["condition"][f"n{row['study_count']:02d}_d{row['distractor_count']}"].append(row)
    summary = {"total_trials": len(scored), "overall": metric(groups["overall"]),
               "by_split": {k: metric(v) for k, v in groups["split"].items()},
               "by_probe_type": {k: metric(v) for k, v in sorted(groups["probe_type"].items())},
               "by_condition": {k: metric(v) for k, v in sorted(groups["condition"].items())}}
    for name, rows in sorted(groups["probe_type"].items()):
        if name == "target":
            summary["by_probe_type"][name]["target_hit_rate"] = sum(r["parsed_answer"] == "YES" for r in rows) / len(rows)
        else:
            summary["by_probe_type"][name]["false_accept_rate"] = sum(r["parsed_answer"] == "YES" for r in rows) / len(rows)
            summary["by_probe_type"][name]["correct_reject_rate"] = sum(r["parsed_answer"] == "NO" for r in rows) / len(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
