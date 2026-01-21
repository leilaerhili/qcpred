#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build hop sweep (B1) from executions.jsonl tree.")
    ap.add_argument("--family", type=str, default="random")
    ap.add_argument("--results-root", type=str, default="", help="Override results root (default: data/raw/results/<family>)")
    ap.add_argument("--processed-dir", type=str, default="data/processed")
    ap.add_argument("--max-qubits", type=int, default=10)
    ap.add_argument("--combine-out", type=str, default="data/processed/hop_sweep_b1.jsonl")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    family = args.family.strip()

    results_root = Path(args.results_root).resolve() if args.results_root else (Path("data/raw/results") / family)
    if not results_root.exists():
        print(f"No results root found: {results_root}")
        return 1

    exec_files = sorted(results_root.glob("*" + "/lam_*" + "/executions.jsonl"))
    if not exec_files:
        print(f"No executions.jsonl found under: {results_root}/*/lam_*/executions.jsonl")
        return 1

    processed_dir = Path(args.processed_dir).resolve()
    tmp_training = processed_dir / "training_rows.jsonl"
    tmp_with_hop = processed_dir / "training_rows_with_hop_tmp.jsonl"
    combine_out = Path(args.combine_out).resolve()

    # Fresh combined output
    combine_out.parent.mkdir(parents=True, exist_ok=True)
    if combine_out.exists():
        combine_out.unlink()

    total_ok = 0

    for exec_path in exec_files:
        # Build training rows from this executions file
        run(["python", "scripts/build_training_rows_from_executions.py", "--executions", str(exec_path), "--out", str(tmp_training)])

        # Compute hop
        run([
            "python",
            "scripts/compute_hop.py",
            "--in",
            str(tmp_training),
            "--out",
            str(tmp_with_hop),
            "--max-qubits",
            str(int(args.max_qubits)),
        ])

        # Append ok rows to combined file
        ok = 0
        with tmp_with_hop.open("r", encoding="utf-8") as f_in, combine_out.open("a", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                # quick filter: include only ok hop rows (compute_hop sets hop_meta.status)
                if '"hop_meta"' in line and '"status": "ok"' in line:
                    f_out.write(line)
                    ok += 1
        total_ok += ok
        print(f"[ok rows] {ok} from {exec_path}")

    print(f"\nWrote combined B1 hop sweep rows: n_ok={total_ok} -> {combine_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
