#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def index_circuit_features(path: Path) -> Dict[str, Dict[str, Any]]:
    feats: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(path):
        cid = r.get("circuit_id")
        if cid:
            feats[str(cid)] = r
    return feats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build training_rows.jsonl directly from executions.jsonl.")
    ap.add_argument("--executions", type=str, required=True, help="Path to executions.jsonl")
    ap.add_argument("--out", type=str, default="data/processed/training_rows.jsonl", help="Output training_rows.jsonl")
    ap.add_argument(
        "--circuit-features",
        type=str,
        default="data/processed/circuit_features.jsonl",
        help="Path to circuit_features.jsonl for logical_features join",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    exec_path = Path(args.executions).resolve()
    out_path = Path(args.out).resolve()
    cf_path = Path(args.circuit_features).resolve()

    ex = read_jsonl(exec_path)
    if not ex:
        print(f"No executions found at: {exec_path}")
        return 1

    cf_index = index_circuit_features(cf_path) if cf_path.exists() else {}

    out_rows: List[Dict[str, Any]] = []
    miss = 0

    for r in ex:
        cid = str(r.get("circuit_id"))
        logical = cf_index.get(cid)
        if logical is None:
            miss += 1
            logical = {"circuit_id": cid}

        row: Dict[str, Any] = {
            "artifact_path": r.get("artifact_path"),
            "artifact_type": r.get("artifact_type", "qasm"),
            "backend": r.get("backend"),
            "circuit_id": cid,
            "counts": r.get("counts"),
            "family": r.get("family"),
            "shots": r.get("shots"),
            "timestamp": r.get("timestamp"),
            "transpile_settings": r.get("transpile_settings", {}),
            "transpiled_features": r.get("transpiled_features", {}),
            "logical_features": logical,
        }

        # Pass through B1 fields if present
        if "backend_profile_id" in r:
            row["backend_profile_id"] = r["backend_profile_id"]
        if "lambda" in r:
            row["lambda"] = r["lambda"]
        if "backend_features" in r:
            row["backend_features"] = r["backend_features"]

        out_rows.append(row)

    write_jsonl(out_path, out_rows)
    print(f"Wrote {len(out_rows)} training rows -> {out_path}")
    if miss:
        print(f"[warn] missing circuit_features for {miss} rows (filled minimal logical_features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
