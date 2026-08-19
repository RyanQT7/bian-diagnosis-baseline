#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
from bian_baseline import LABELS, score_frame

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, dtype=str)
    required = ["case_id", "diagnosis_root_cause", "true_label"]
    if list(frame.columns) != required:
        raise ValueError(f"expected columns {required}, got {list(frame.columns)}")
    if frame["case_id"].duplicated().any():
        raise ValueError("duplicate case_id")
    if set(frame["diagnosis_root_cause"]) - set(LABELS) or set(frame["true_label"]) - set(LABELS):
        raise ValueError("labels must be remote/local/fiber")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(score_frame(frame), encoding="utf-8")
    print(args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
