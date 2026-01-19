#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build training_rows_with_hop for a sweep of Aer backends and optionally combine them."
    )
    ap.add_argument("--family", type=str, default="random", help="Circuit family (default: random)")
    ap.add_argument(
        "--backend-glob",
        type=str,
        default="aer_dep_p2_*",
        help='Backend tag glob under data/raw/results/<family>/ (default: "aer_dep_p2_*")',
    )
    ap.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed directory (default: data/processed)",
    )
    ap.add_argument(
        "--max-qubits",
        type=int,
        default=10,
        help="Max qubits for HOP statevector (default: 10)",
    )
    ap.add_argument(
        "--combine-out",
        type=str,
        default="data/processed/hop_sweep.jsonl",
        help="Combined output JSONL path (default: data/processed/hop_sweep.jsonl)",
    )
    ap.add_argument(
        "--keep-per-backend",
        action="store_true",
        help="Keep per-backend training_rows_with_hop_<backend>.jsonl files (default: false).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    family = args.family.strip()
    processed_dir = Path(args.processed_dir).resolve()

    # Discover backends by checking results directories.
    results_root = Path("data/raw/results") / family
    if not results_root.exists():
        print(f"No results directory found at: {results_root}")
        return 1

    backend_dirs = sorted([p for p in results_root.glob(args.backend_glob) if p.is_dir()])
    if not backend_dirs:
        print(f"No backend dirs matched: {results_root}/{args.backend_glob}")
        return 1

    print(f"Found {len(backend_dirs)} backend(s): {[p.name for p in backend_dirs]}")

    combined_rows: List[Dict[str, Any]] = []

    # We'll reuse the canonical training_rows.jsonl path, but to avoid mixing,
    # we build -> compute hop -> copy out to a backend-specific filename each time.
    tmp_training_rows = processed_dir / "training_rows.jsonl"

    for bd in backend_dirs:
        backend_tag = bd.name
        per_out = processed_dir / f"training_rows_with_hop_{backend_tag}.jsonl"

        cmd_build = ["python", "scripts/build_training_rows.py", "--family", family, "--backend", backend_tag]
        cmd_hop = [
            "python",
            "scripts/compute_hop.py",
            "--in",
            str(tmp_training_rows),
            "--out",
            str(per_out),
            "--max-qubits",
            str(int(args.max_qubits)),
        ]

        if args.dry_run:
            print("\n[dry-run]")
            print("  ", " ".join(cmd_build))
            print("  ", " ".join(cmd_hop))
        else:
            run(cmd_build)
            run(cmd_hop)

        rows = read_jsonl(per_out)
        # keep only rows where hop is computed
        ok = [r for r in rows if r.get("hop") is not None and (r.get("hop_meta", {}) or {}).get("status") == "ok"]
        combined_rows.extend(ok)
        print(f"[{backend_tag}] rows total={len(rows)} ok={len(ok)} -> {per_out}")

        if not args.keep_per_backend and not args.dry_run:
            # Optional cleanup: keep repo tidy
            # (comment this out if you prefer to keep all per-backend files)
            pass

    if not args.dry_run:
        combine_out = Path(args.combine_out).resolve()
        write_jsonl(combine_out, combined_rows)
        print(f"\nWrote combined sweep rows: n={len(combined_rows)} -> {combine_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
