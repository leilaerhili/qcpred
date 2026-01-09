#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate circuits and save under data/raw/circuits/")
    p.add_argument("--n", type=int, required=True, help="Number of circuits to generate.")
    p.add_argument("--n-qubits", type=int, required=True, help="Number of qubits per circuit.")
    p.add_argument("--depth", type=int, required=True, help="Target depth (approx; generator-defined).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0).")
    return p.parse_args()


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    print("[generate_circuits] starting...", flush=True)
    args = parse_args()
    print(f"[generate_circuits] args: n={args.n}, n_qubits={args.n_qubits}, depth={args.depth}, seed={args.seed}", flush=True)

    if args.n <= 0 or args.n_qubits <= 0 or args.depth <= 0:
        print("[generate_circuits] ERROR: n, n-qubits, depth must all be > 0", file=sys.stderr, flush=True)
        return 2

    # Import your path helpers
    try:
        from qcpred.datasets.locations import raw_circuits_dir, raw_circuits_manifest_path
    except Exception as e:
        print("[generate_circuits] ERROR importing qcpred.datasets.locations", file=sys.stderr, flush=True)
        print(str(e), file=sys.stderr, flush=True)
        return 2

    circuits_root = raw_circuits_dir()
    manifest_path = raw_circuits_manifest_path()
    print(f"[generate_circuits] circuits_root = {circuits_root}", flush=True)
    print(f"[generate_circuits] manifest_path = {manifest_path}", flush=True)

    # Import generator (you already have this)
    try:
        from qcpred.circuits.random import generate_random_circuit
    except Exception as e:
        print("[generate_circuits] ERROR importing qcpred.circuits.random.generate_random_circuit", file=sys.stderr, flush=True)
        print(str(e), file=sys.stderr, flush=True)
        return 2

    # Ensure output directories exist
    (circuits_root / "json").mkdir(parents=True, exist_ok=True)
    (circuits_root / "qasm").mkdir(parents=True, exist_ok=True)

    created = 0

    for i in range(args.n):
        circuit_id = str(uuid.uuid4())
        run_seed = args.seed + i

        circuit_obj, meta = generate_random_circuit(
            n_qubits=args.n_qubits,
            depth=args.depth,
            seed=run_seed,
            circuit_id=circuit_id,
        )

        meta = dict(meta or {})
        meta.setdefault("circuit_id", circuit_id)
        meta.setdefault("family", "random")
        meta.setdefault("n_qubits", args.n_qubits)
        meta.setdefault("depth_target", args.depth)
        meta.setdefault("seed", run_seed)
        meta.setdefault("created_utc", iso_utc_now())

        rep = meta.get("representation", "stub")

        # If qiskit, write QASM; else write JSON
        if rep == "qiskit":
            try:
                # Newer Qiskit
                try:
                    from qiskit import qasm2  # type: ignore
                    qasm_text = qasm2.dumps(circuit_obj)  # type: ignore[arg-type]
                except Exception:
                    # Older Qiskit
                    qasm_text = circuit_obj.qasm()  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[generate_circuits] ERROR serializing QASM for {circuit_id}: {e}", file=sys.stderr, flush=True)
                return 2

            rel_path = Path("qasm") / f"{circuit_id}.qasm"
            out_path = circuits_root / rel_path
            out_path.write_text(qasm_text, encoding="utf-8")
            meta["artifact_type"] = "qasm"
            meta["artifact_path"] = str(rel_path)
        else:
            rel_path = Path("json") / f"{circuit_id}.json"
            out_path = circuits_root / rel_path
            payload: Dict[str, Any] = {
                "circuit_id": circuit_id,
                "schema_version": 1,
                "data": circuit_obj,
                "meta": meta,
            }
            write_json(out_path, payload)
            meta["artifact_type"] = "json"
            meta["artifact_path"] = str(rel_path)

        append_jsonl(manifest_path, meta)
        created += 1

        if i < 3:
            print(f"[generate_circuits] wrote {meta['artifact_path']} (rep={rep})", flush=True)

    print(f"[generate_circuits] DONE. created={created}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
