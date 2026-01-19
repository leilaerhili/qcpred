#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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


def parse_p2_from_backend_tag(tag: str) -> float:
    """
    Accepts tags like:
      - aer_dep_p2_0p05
      - aer_dep_p2_0.05
    and returns float p2.
    """
    m = re.search(r"p2_([0-9]+(?:p[0-9]+)?(?:\.[0-9]+)?)", tag)
    if not m:
        raise ValueError(f"Could not parse p2 from backend tag: {tag}")
    s = m.group(1)
    s = s.replace("p", ".")
    return float(s)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute robustness targets from a hop sweep (A2).")
    ap.add_argument(
        "--in",
        dest="inp",
        type=str,
        default="data/processed/hop_sweep.jsonl",
        help="Input combined hop sweep JSONL (default: data/processed/hop_sweep.jsonl)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/processed/robustness_targets.jsonl",
        help="Output JSONL of per-circuit targets (default: data/processed/robustness_targets.jsonl)",
    )
    ap.add_argument(
        "--degree",
        type=int,
        default=2,
        choices=[1, 2],
        help="Polynomial degree to fit hop(p2): 1=linear, 2=quadratic (default: 2)",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum distinct p2 points needed per circuit (default: 3)",
    )
    ap.add_argument(
        "--backend-field",
        type=str,
        default="backend",
        help="Field name that contains backend tag (default: backend)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.inp).resolve()
    out_path = Path(args.out).resolve()

    rows = read_jsonl(in_path)
    if not rows:
        print(f"No rows found at: {in_path}")
        return 1

    # Group by circuit_id
    by_cid: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = r.get("circuit_id")
        hop = r.get("hop")
        if not cid or hop is None:
            continue
        by_cid.setdefault(str(cid), []).append(r)

    out_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0

    for cid, rr in by_cid.items():
        # Extract (p2, hop) points, dedupe by p2 with averaging if needed.
        pts: Dict[float, List[float]] = {}
        for r in rr:
            tag = r.get(args.backend_field, "")
            try:
                p2 = parse_p2_from_backend_tag(str(tag))
            except Exception:
                continue
            pts.setdefault(float(p2), []).append(float(r["hop"]))

        xs = sorted(pts.keys())
        if len(xs) < int(args.min_points):
            n_skip += 1
            continue

        x = np.array(xs, dtype=float)
        y = np.array([float(np.mean(pts[v])) for v in xs], dtype=float)

        # Fit polynomial
        deg = int(args.degree)
        coefs = np.polyfit(x, y, deg=deg)  # highest power first

        if deg == 1:
            # y = a*x + b
            a, b = coefs
            hop0 = float(b)
            hop1 = float(a)
            hop2 = 0.0
        else:
            # y = a*x^2 + b*x + c
            a, b, c = coefs
            hop0 = float(c)
            hop1 = float(b)
            hop2 = float(a)

        # Carry over features from any row (they’re same circuit)
        # Prefer logical/transpiled features for training the final predictor.
        sample = rr[0]
        out_rows.append(
            {
                "circuit_id": cid,
                "family": sample.get("family"),
                "logical_features": sample.get("logical_features", {}),
                "transpiled_features": sample.get("transpiled_features", {}),
                "robustness": {
                    "hop0": hop0,
                    "hop1": hop1,
                    "hop2": hop2,
                    "degree": deg,
                    "n_points": int(len(x)),
                    "p2_min": float(x.min()),
                    "p2_max": float(x.max()),
                },
            }
        )
        n_ok += 1

    write_jsonl(out_path, out_rows)
    print(f"Wrote {len(out_rows)} circuits -> {out_path}")
    print(f"ok={n_ok} skipped={n_skip} (skipped = insufficient distinct p2 points or parse issues)")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
