#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
    p = argparse.ArgumentParser(description="Compute HOP label for each training row.")
    p.add_argument(
        "--in",
        dest="inp",
        type=str,
        default="data/processed/training_rows.jsonl",
        help="Input training rows JSONL (default: data/processed/training_rows.jsonl)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/training_rows_with_hop.jsonl",
        help="Output JSONL (default: data/processed/training_rows_with_hop.jsonl)",
    )
    p.add_argument(
        "--max-qubits",
        type=int,
        default=10,
        help="Safety limit for statevector simulation (default: 10)",
    )
    return p.parse_args()


def bitstring_probs_from_statevector(qc) -> Dict[str, float]:
    """
    Returns probabilities keyed by bitstring like '0101' (Qiskit ordering).
    Uses Statevector.from_instruction for ideal simulation.
    """
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()  # dict: bitstring -> probability
    # Ensure plain Python floats
    return {k: float(v) for k, v in probs.items()}


def heavy_set_from_probs(probs: Dict[str, float], n: int | None = None) -> Tuple[set[str], float]:
    """
    QV-style heavy set: the top half of bitstrings by ideal probability.
    Returns (heavy_set, threshold_prob) where threshold_prob is the prob of the last included item.
    """
    if n is None:
        if probs:
            n = len(next(iter(probs.keys())))
        else:
            n = 0

    total_states = 1 << n
    if total_states == 0:
        return set(), 0.0

    # Build full list including zero-prob states so ties are handled consistently
    items: List[Tuple[str, float]] = []
    for i in range(total_states):
        b = format(i, f"0{n}b")
        items.append((b, float(probs.get(b, 0.0))))

    # Sort by prob desc, then bitstring to make tie-breaking deterministic
    items.sort(key=lambda t: (-t[1], t[0]))

    k = total_states // 2  # heavy set size = 2^(n-1)
    heavy_items = items[:k]
    heavy = {b for b, _p in heavy_items}
    threshold = float(heavy_items[-1][1]) if heavy_items else 0.0
    return heavy, threshold



def hop_from_counts(counts: Dict[str, int], heavy: set[str]) -> float:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    heavy_hits = sum(int(v) for k, v in counts.items() if k in heavy)
    return float(heavy_hits / total)


def main() -> int:
    args = parse_args()

    in_path = Path(args.inp).resolve()
    out_path = Path(args.out).resolve()

    rows = read_jsonl(in_path)
    if not rows:
        print(f"No rows found at: {in_path}")
        return 1

    from qcpred.datasets.locations import raw_circuits_family_dir
    from qiskit.qasm2 import load as qasm2_load

    out_rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0

    for r in rows:
        family = r.get("family", "random")
        lf = r.get("logical_features", {})
        n_qubits = int(lf.get("n_qubits", 0))

        if n_qubits > int(args.max_qubits):
            r2 = dict(r)
            r2["hop"] = None
            r2["hop_meta"] = {"status": "skipped_max_qubits", "n_qubits": n_qubits}
            out_rows.append(r2)
            n_skip += 1
            continue

        artifact_type = r.get("artifact_type")
        artifact_path = r.get("artifact_path")
        if artifact_type != "qasm" or not artifact_path:
            r2 = dict(r)
            r2["hop"] = None
            r2["hop_meta"] = {"status": "skipped_no_qasm"}
            out_rows.append(r2)
            n_skip += 1
            continue

        circuits_root = raw_circuits_family_dir(str(family))
        qasm_path = circuits_root / str(artifact_path)

        if not qasm_path.exists():
            r2 = dict(r)
            r2["hop"] = None
            r2["hop_meta"] = {"status": "missing_qasm", "qasm_path": str(qasm_path)}
            out_rows.append(r2)
            n_skip += 1
            continue

        try:
            qc = qasm2_load(str(qasm_path))
            # Ensure measurements exist; Statevector can't handle classical-only ops reliably
            # If circuit has measurements, strip them for ideal statevector simulation.
            qc_nom = qc.remove_final_measurements(inplace=False)

            probs = bitstring_probs_from_statevector(qc_nom)
            n = int(lf.get("n_qubits", 0))
            heavy, thresh = heavy_set_from_probs(probs, n=n)
            hop = hop_from_counts(r.get("counts", {}), heavy)

            r2 = dict(r)
            r2["hop"] = hop
            r2["hop_meta"] = {
                "status": "ok",
                "heavy_set_size": len(heavy),
                "threshold_prob": thresh,
            }

            out_rows.append(r2)
            n_ok += 1

        except Exception as e:
            r2 = dict(r)
            r2["hop"] = None
            r2["hop_meta"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            out_rows.append(r2)
            n_skip += 1

    write_jsonl(out_path, out_rows)
    print(f"Wrote {len(out_rows)} rows to: {out_path}")
    print(f"HOP computed: {n_ok}, skipped/errors: {n_skip}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
