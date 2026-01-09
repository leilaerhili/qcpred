#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect generated circuits manifest.")
    p.add_argument("--n", type=int, default=5, help="Number of sample records to print (default: 5).")
    p.add_argument("--show-paths", action="store_true", help="Print artifact paths for first N records.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from qcpred.datasets.locations import raw_circuits_manifest_path, raw_circuits_dir

    manifest_path = raw_circuits_manifest_path()
    circuits_root = raw_circuits_dir()

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
        print(f"Manifest exists but is empty: {manifest_path}")
        return 0

    print(f"Circuits root: {circuits_root}")
    print(f"Manifest:      {manifest_path}")
    print(f"Total records: {len(records)}")

    # Summaries
    reps = Counter(r.get("representation", "unknown") for r in records)
    types = Counter(r.get("artifact_type", "unknown") for r in records)
    families = Counter(r.get("family", "unknown") for r in records)

    def get_int(field: str) -> List[int]:
        return [int(r[field]) for r in records if field in r and isinstance(r[field], (int, float, str)) and str(r[field]).isdigit()]

    n_qubits = get_int("n_qubits")
    depth_target = get_int("depth_target")

    print("\nBy representation:")
    for k, v in reps.items():
        print(f"  {k}: {v}")

    print("\nBy artifact_type:")
    for k, v in types.items():
        print(f"  {k}: {v}")

    print("\nBy family:")
    for k, v in families.items():
        print(f"  {k}: {v}")

    if n_qubits:
        print(f"\nQubits:        min={min(n_qubits)}  max={max(n_qubits)}")
    if depth_target:
        print(f"Depth target:  min={min(depth_target)}  max={max(depth_target)}")

    # Validate artifact existence for first few
    print(f"\nValidating first {min(args.n, len(records))} artifact paths...")
    for r in records[: args.n]:
        rel = r.get("artifact_path")
        cid = r.get("circuit_id", "unknown")
        ok = "?"
        if isinstance(rel, str):
            ok = "OK" if (circuits_root / rel).exists() else "MISSING"
        print(f"  {cid}  ->  {rel}  [{ok}]")

    # Optional: print sample records
    print(f"\nSample records (first {min(args.n, len(records))}):")
    for r in records[: args.n]:
        if args.show_paths:
            # print short line instead of full record
            print(json.dumps({k: r.get(k) for k in ["circuit_id", "family", "n_qubits", "depth_target", "seed", "representation", "artifact_type", "artifact_path"]}, indent=2))
        else:
            print(json.dumps(r, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
