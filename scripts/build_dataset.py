#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build processed circuit feature dataset from raw circuits manifest.")
    p.add_argument("--out", type=str, default="data/processed/circuit_features.jsonl", help="Output path (default: data/processed/circuit_features.jsonl)")
    return p.parse_args()


def main() -> int:
    from qcpred.datasets.locations import raw_circuits_dir, raw_circuits_manifest_path
    from qcpred.features.circuit import extract_circuit_features_from_record, features_to_dict

    circuits_root = raw_circuits_dir()
    manifest_path = raw_circuits_manifest_path()

    if not manifest_path.exists():
        print(f"No manifest found at: {manifest_path}")
        return 1

    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("Manifest is empty; nothing to do.")
        return 0

    out_path = Path(parse_args().out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("w", encoding="utf-8") as out:
        for r in records:
            feats = extract_circuit_features_from_record(r, circuits_root=circuits_root)
            out.write(json.dumps(features_to_dict(feats), sort_keys=True) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} feature rows to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
