#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from qcpred.features.transpiled import extract_transpiled_features, features_to_dict as tf_to_dict

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run circuits on a backend (start: Aer) and record counts.")
    p.add_argument("--family", type=str, default="random", help="Circuit family (default: random)")
    p.add_argument("--backend", type=str, default="aer_simulator", help="Backend name tag (default: aer_simulator)")
    p.add_argument("--shots", type=int, default=2048, help="Shots per circuit (default: 2048)")
    p.add_argument("--max-circuits", type=int, default=0, help="If >0, limit number of circuits to run")
    p.add_argument("--seed", type=int, default=0, help="Seed for simulator/transpiler (default: 0)")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSONL path. Default: data/raw/results/<family>/<backend>/executions.jsonl",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    family = args.family.strip()
    backend_tag = args.backend.strip()

    from qcpred.datasets.locations import raw_circuits_family_dir, raw_circuits_manifest_path_for
    from qcpred.utils.paths import repo_path

    circuits_root = raw_circuits_family_dir(family)
    manifest_path = raw_circuits_manifest_path_for(family)

    if not manifest_path.exists():
        print(f"No manifest found at: {manifest_path}")
        return 1

    records = read_jsonl(manifest_path)
    if not records:
        print("Manifest is empty; nothing to do.")
        return 0

    # Default output location
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = repo_path("data", "raw", "results", family, backend_tag, "executions.jsonl")

    # Filter to QASM records for now (we can add JSON/stub support later)
    qasm_records = [r for r in records if r.get("artifact_type") == "qasm"]
    if not qasm_records:
        print("No QASM circuits found in manifest. Generate QASM circuits first.")
        return 1

    if args.max_circuits and args.max_circuits > 0:
        qasm_records = qasm_records[: int(args.max_circuits)]

    # Qiskit imports
    from qiskit import QuantumCircuit, transpile
    from qiskit.qasm2 import load as qasm2_load
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=int(args.seed))
    transpile_settings: Dict[str, Any] = {
        "optimization_level": 1,
        "seed_transpiler": int(args.seed),
    }

    n_ok = 0
    n_fail = 0

    for r in qasm_records:
        circuit_id = r["circuit_id"]
        artifact_rel = r["artifact_path"]
        artifact_path = circuits_root / artifact_rel

        if not artifact_path.exists():
            print(f"[skip] Missing artifact for circuit_id={circuit_id}: {artifact_path}")
            n_fail += 1
            continue

        try:
            # Load QASM2 -> QuantumCircuit
            qc: QuantumCircuit = qasm2_load(str(artifact_path))

            # Transpile for Aer
            tqc = transpile(
                qc,
                sim,
                optimization_level=int(transpile_settings["optimization_level"]),
                seed_transpiler=int(transpile_settings["seed_transpiler"]),
            )

            tf = extract_transpiled_features(tqc)

            # Run
            job = sim.run(tqc, shots=int(args.shots))
            result = job.result()
            counts = result.get_counts()

            exec_record: Dict[str, Any] = {
                "circuit_id": circuit_id,
                "family": r.get("family", family),
                "backend": backend_tag,
                "shots": int(args.shots),
                "timestamp": utc_now_iso(),
                "transpile_settings": transpile_settings,
                "counts": counts,
                # helpful provenance pointers:
                "artifact_path": str(artifact_rel),
                "artifact_type": "qasm",
                "transpiled_features": tf_to_dict(tf),
            }
            append_jsonl(out_path, exec_record)
            n_ok += 1

        except Exception as e:
            print(f"[fail] circuit_id={circuit_id}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"Wrote {n_ok} execution rows to: {out_path}")
    print(f"Failures/skips: {n_fail}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
