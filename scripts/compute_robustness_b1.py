#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute robustness targets per (circuit, backend_profile) from hop sweep (B1).")
    ap.add_argument("--in", dest="inp", type=str, default="data/processed/hop_sweep_b1.jsonl")
    ap.add_argument("--out", type=str, default="data/processed/robustness_targets_b1.jsonl")
    ap.add_argument("--degree", type=int, default=2, choices=[1, 2])
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--lambda-field", type=str, default="lambda")
    ap.add_argument("--backend-profile-field", type=str, default="backend_profile_id")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.inp).resolve()
    out_path = Path(args.out).resolve()

    rows = read_jsonl(in_path)
    if not rows:
        print(f"No rows found at: {in_path}")
        return 1

    # group by (circuit_id, backend_profile_id)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        cid = r.get("circuit_id")
        bpid = r.get(args.backend_profile_field)
        hop = r.get("hop")
        lam = r.get(args.lambda_field)
        if cid is None or bpid is None or hop is None or lam is None:
            continue
        by_key.setdefault((str(cid), str(bpid)), []).append(r)

    out_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0

    for (cid, bpid), rr in by_key.items():
        pts: Dict[float, List[float]] = {}
        for r in rr:
            try:
                lam = float(r[args.lambda_field])
                hop = float(r["hop"])
            except Exception:
                continue
            pts.setdefault(lam, []).append(hop)

        xs = sorted(pts.keys())
        if len(xs) < int(args.min_points):
            n_skip += 1
            continue

        x = np.array(xs, dtype=float)
        y = np.array([float(np.mean(pts[v])) for v in xs], dtype=float)

        deg = int(args.degree)
        coefs = np.polyfit(x, y, deg=deg)

        if deg == 1:
            a1, a0 = coefs
            hop0, hop1, hop2 = float(a0), float(a1), 0.0
        else:
            a2, a1, a0 = coefs
            hop0, hop1, hop2 = float(a0), float(a1), float(a2)

        sample = rr[0]
        out_rows.append({
            "circuit_id": cid,
            "backend_profile_id": bpid,
            "family": sample.get("family"),
            "logical_features": sample.get("logical_features", {}),
            "transpiled_features": sample.get("transpiled_features", {}),
            "backend_features": sample.get("backend_features", {}),
            "robustness": {
                "hop0": hop0,
                "hop1": hop1,
                "hop2": hop2,
                "degree": deg,
                "n_points": int(len(x)),
                "lambda_min": float(x.min()),
                "lambda_max": float(x.max()),
            },
        })
        n_ok += 1

    write_jsonl(out_path, out_rows)
    print(f"Wrote {len(out_rows)} (circuit, backend_profile) rows -> {out_path}")
    print(f"ok={n_ok} skipped={n_skip}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
