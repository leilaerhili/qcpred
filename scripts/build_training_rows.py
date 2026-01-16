#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join logical features + execution rows into training rows.")
    p.add_argument("--family", type=str, default="random", help="Circuit family (default: random)")
    p.add_argument("--backend", type=str, default="aer_simulator", help="Backend tag (default: aer_simulator)")
    p.add_argument(
        "--features",
        type=str,
        default="data/processed/circuit_features.jsonl",
        help="Logical feature JSONL path (default: data/processed/circuit_features.jsonl)",
    )
    p.add_argument(
        "--executions",
        type=str,
        default="",
        help="Execution JSONL path (default: data/raw/results/<family>/<backend>/executions.jsonl)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/training_rows.jsonl",
        help="Output JSONL path (default: data/processed/training_rows.jsonl)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    family = args.family.strip()
    backend = args.backend.strip()

    from qcpred.utils.paths import repo_path

    features_path = Path(args.features).resolve()

    if args.executions:
        executions_path = Path(args.executions).resolve()
    else:
        executions_path = repo_path("data", "raw", "results", family, backend, "executions.jsonl")

    out_path = Path(args.out).resolve()

    logical_rows = read_jsonl(features_path)
    if not logical_rows:
        print(f"No logical features found at: {features_path}")
        print("Run: python scripts/build_dataset.py --family <family>")
        return 1

    exec_rows = read_jsonl(executions_path)
    if not exec_rows:
        print(f"No executions found at: {executions_path}")
        print("Run: python scripts/run_experiments.py ...")
        return 1

    # Index logical features by circuit_id
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in logical_rows:
        cid = r.get("circuit_id")
        if cid:
            by_id[str(cid)] = r

    joined: List[Dict[str, Any]] = []
    n_missing = 0

    for e in exec_rows:
        cid = str(e["circuit_id"])
        feats = by_id.get(cid)
        if feats is None:
            n_missing += 1
            continue

        row: Dict[str, Any] = {
            # keys for this experimental unit
            "circuit_id": cid,
            "family": e.get("family", family),
            "backend": e.get("backend", backend),
            "shots": e.get("shots"),
            "timestamp": e.get("timestamp"),
            "transpile_settings": e.get("transpile_settings"),
            # logical features (prefix optional; leaving as-is for now)
            "logical_features": feats,
            # compile artifacts
            "transpiled_features": e.get("transpiled_features", {}),
            # raw outputs (needed later for HOP)
            "counts": e.get("counts", {}),
            "artifact_path": e.get("artifact_path"),
            "artifact_type": e.get("artifact_type"),
        }

        joined.append(row)

    write_jsonl(out_path, joined)

    print(f"Wrote {len(joined)} training rows to: {out_path}")
    if n_missing:
        print(f"Warning: {n_missing} execution rows had no matching logical features (circuit_id mismatch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
