"""Check whether the full baseline is actually generated and visually reviewed."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/ip_const_0.6")
    args = p.parse_args()
    out = Path(args.run_dir)
    run = read_json(out / "run.json")
    cfg = run["config"]
    rows = read_json(cfg["photos_manifest"])["images"]
    expected = {(int(r["id"]), int(s)) for r in rows for s in cfg["seeds"]}
    present, flagged, issues = set(), [], []
    for f in out.glob("*_s*.json"):
        r = read_json(f)
        key = (int(r["photo_id"]), int(r["seed"]))
        try:
            valid = r["fingerprint"] == run["fingerprint"] and sha256(r["output_path"]) == r["image_sha256"]
        except FileNotFoundError:
            valid = False
        if valid:
            present.add(key)
            if r["nsfw_flagged"]:
                flagged.append(key)
        else:
            issues.append(str(f))
    reviewed = set()
    review_path = out / "visual_review.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["reviewed"].strip().lower() in {"1", "true", "yes", "да"}:
                    reviewed.add((int(row["photo_id"]), int(row["seed"])))
    missing = sorted(expected - present)
    unreviewed = sorted(expected - reviewed)
    summary_path = out / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    ready = (not missing and not unreviewed and not issues and cfg["method"] == "ip_adapter"
             and 20 <= len(rows) <= 30)
    status = {
        "baseline_execution_and_review_complete": ready,
        "expected_images": len(expected), "present_images": len(present & expected),
        "missing_pairs": missing, "unreviewed_pairs": unreviewed, "invalid_files": issues,
        "flagged_pairs": flagged,
        "metrics_status": summary.get("metrics_status", "NOT_COMPUTED"),
        "note": "This checks file coverage and recorded human review, not aesthetic quality or grading. "
                "Flagged outputs must be discussed; do not silently omit or regenerate them."}
    write_json(out / "completion_audit.json", status)
    print("Baseline generated and reviewed:" if ready else "Baseline NOT yet complete:", status)


if __name__ == "__main__":
    main()
