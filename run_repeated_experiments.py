from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


METRICS = [
    "object_memory_bytes",
    "static_memory_bytes",
    "memory_saved_percent",
    "object_predict_ms",
    "static_predict_ms",
    "time_saved_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--model", choices=["tree", "forest"], default="forest")
    parser.add_argument("--trees", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--csv", default="data/mushroom.csv")
    parser.add_argument("--label", default="label")
    parser.add_argument("--out-dir", default="results/repeated")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run_one(args: argparse.Namespace, run_idx: int, seed: int, report_path: Path) -> dict[str, str]:
    cmd = [
        args.python,
        "experiment.py",
        "--model",
        args.model,
        "--csv",
        args.csv,
        "--label",
        args.label,
        "--max-depth",
        str(args.max_depth),
        "--seed",
        str(seed),
        "--report",
        str(report_path),
    ]
    if args.model == "forest":
        cmd.extend(["--trees", str(args.trees)])

    subprocess.run(cmd, check=True)
    with report_path.open(newline="") as f:
        row = next(csv.DictReader(f))
    row["run"] = str(run_idx)
    row["seed"] = str(seed)
    return row


def write_summary_csv(out_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["run", "seed"] + [name for name in rows[0] if name not in {"run", "seed"}]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_idx in range(1, args.runs + 1):
        seed = args.seed_start + run_idx - 1
        report_path = out_dir / f"{args.model}_run_{run_idx:02d}.csv"
        print(f"run {run_idx}/{args.runs}: seed={seed}")
        rows.append(run_one(args, run_idx, seed, report_path))

    summary_csv = out_dir / f"mushroom_{args.model}_10_runs.csv"
    write_summary_csv(summary_csv, rows)

    manifest = {
        "model": args.model,
        "runs": args.runs,
        "dataset": args.csv,
        "summary_csv": str(summary_csv),
        "metrics": METRICS,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"wrote raw run data: {summary_csv}")
    print(f"wrote manifest:     {manifest_path}")


if __name__ == "__main__":
    main()
