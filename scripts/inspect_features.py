#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect processed circuit feature dataset.")
    p.add_argument(
        "--path",
        type=str,
        default="data/processed/circuit_features.jsonl",
        help="Path to feature JSONL (default: data/processed/circuit_features.jsonl)",
    )
    p.add_argument("--n", type=int, default=10, help="Number of sample rows to print (default: 10).")
    p.add_argument("--sort-by", type=str, default="", help="Optional numeric column to sort by (e.g., cx_count).")
    p.add_argument("--desc", action="store_true", help="Sort descending if --sort-by is set.")
    return p.parse_args()


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _summarize_numeric(values: List[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    values.sort()
    n = len(values)
    if n == 0:
        return {}
    mean = sum(values) / n
    return {
        "count": float(n),
        "min": values[0],
        "p25": values[int(0.25 * (n - 1))],
        "median": values[int(0.50 * (n - 1))],
        "p75": values[int(0.75 * (n - 1))],
        "max": values[-1],
        "mean": mean,
    }


def main() -> int:
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Feature file not found: {path}")
        print("Tip: run `python scripts/build_dataset.py` first.")
        return 1

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print(f"File exists but is empty: {path}")
        return 0

    print(f"Feature file: {path}")
    print(f"Rows: {len(rows)}")

    # Columns overview
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    cols = sorted(all_keys)
    print("\nColumns:")
    print("  " + ", ".join(cols))

    # Basic missingness for a few important columns
    key_cols = ["n_qubits", "depth_target", "op_count", "cx_count", "oneq_count", "twoq_ratio", "representation"]
    print("\nMissingness (selected columns):")
    for k in key_cols:
        missing = sum(1 for r in rows if k not in r or r[k] is None)
        print(f"  {k:>14}: {missing} missing ({missing/len(rows):.1%})")

    # Numeric summaries
    numeric_cols = ["n_qubits", "depth_target", "op_count", "oneq_count", "cx_count", "twoq_ratio"]
    print("\nNumeric summaries:")
    for k in numeric_cols:
        vals = [r.get(k) for r in rows]
        vals = [v for v in vals if _is_number(v)]
        summ = _summarize_numeric(vals)
        if not summ:
            print(f"  {k:>14}: (no numeric data)")
            continue
        print(
            f"  {k:>14}: "
            f"min={summ['min']:.3g}  p25={summ['p25']:.3g}  med={summ['median']:.3g}  "
            f"p75={summ['p75']:.3g}  max={summ['max']:.3g}  mean={summ['mean']:.3g}"
        )

    # Representation counts
    rep_counts: Dict[str, int] = {}
    for r in rows:
        rep = str(r.get("representation", "unknown"))
        rep_counts[rep] = rep_counts.get(rep, 0) + 1
    print("\nRepresentation counts:")
    for k, v in sorted(rep_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

    # Optional sorting
    if args.sort_by:
        sort_key = args.sort_by
        def keyfn(r: Dict[str, Any]) -> float:
            v = r.get(sort_key)
            return float(v) if _is_number(v) else float("-inf")
        rows = sorted(rows, key=keyfn, reverse=args.desc)

    # Print samples
    print(f"\nSample rows (first {min(args.n, len(rows))}):")
    for r in rows[: args.n]:
        # keep sample print compact
        compact = {k: r.get(k) for k in ["circuit_id", "family", "representation", "n_qubits", "depth_target", "op_count", "cx_count", "twoq_ratio"]}
        print(json.dumps(compact, sort_keys=True))

    # Quick heuristic checks (helpful early warnings)
    # - If stub circuits: op_count should usually equal depth_target
    # - cx_count should not be always 0 unless you intentionally generated only 1Q gates
    stub_rows = [r for r in rows if str(r.get("representation")) == "stub"]
    if stub_rows:
        mismatches = sum(
            1 for r in stub_rows
            if _is_number(r.get("op_count")) and _is_number(r.get("depth_target"))
            and int(r["op_count"]) != int(r["depth_target"])
        )
        print(f"\nHeuristic check (stub): op_count == depth_target mismatches: {mismatches}/{len(stub_rows)}")

    cx_all_zero = all((_is_number(r.get("cx_count")) and float(r.get("cx_count")) == 0.0) for r in rows)
    if cx_all_zero:
        print("\nWarning: cx_count is 0 for ALL rows. If unintended, check circuit generator probabilities.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
