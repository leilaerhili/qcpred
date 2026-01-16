#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate circuits and write raw artifacts + manifest entries.")
    p.add_argument("--family", type=str, default="random", help="Circuit family to generate (default: random)")
    p.add_argument("--n", type=int, required=True, help="Number of circuits to generate")
    p.add_argument("--n-qubits", type=int, required=True, help="Number of qubits")
    p.add_argument("--depth", type=int, required=True, help="Target depth (generator-defined meaning)")
    p.add_argument("--seed", type=int, default=0, help="Base seed (default: 0)")
    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    family = args.family.strip()

    # Centralized paths (family-aware)
    from qcpred.datasets.locations import raw_circuits_family_dir, raw_circuits_manifest_path_for

    circuits_root = raw_circuits_family_dir(family)
    manifest_path = raw_circuits_manifest_path_for(family)

    # Ensure the family is registered (explicit import for now)
    if family == "random":
        import qcpred.circuits.random  # noqa: F401
    else:
        raise SystemExit(f"Unknown --family '{family}'. Only 'random' is available right now.")

    from qcpred.circuits.registry import get_generator

    gen = get_generator(family)

    qasm_dir = circuits_root / "qasm"
    json_dir = circuits_root / "json"

    n_written = 0
    for i in range(args.n):
        circuit_id = str(uuid.uuid4())
        seed_i = int(args.seed) + i

        circuit_obj, meta = gen(
            n_qubits=int(args.n_qubits),
            depth=int(args.depth),
            seed=seed_i,
            circuit_id=circuit_id,
        )

        # Default to JSON artifact
        artifact_type = "json"
        artifact_relpath = f"json/{circuit_id}.json"

        try:
            # If qiskit is installed and the object looks like a Qiskit circuit, write QASM2.
            from qiskit import QuantumCircuit  # type: ignore

            if isinstance(circuit_obj, QuantumCircuit):
                from qiskit.qasm2 import dump as qasm2_dump  # type: ignore

                artifact_type = "qasm"
                artifact_relpath = f"qasm/{circuit_id}.qasm"
                ensure_dir(qasm_dir)
                with (qasm_dir / f"{circuit_id}.qasm").open("w", encoding="utf-8") as f:
                    qasm2_dump(circuit_obj, f)
            else:
                ensure_dir(json_dir)
                write_json(json_dir / f"{circuit_id}.json", {"data": circuit_obj, "meta": meta})

        except Exception:
            # Either qiskit isn't available, QASM export failed, or we weren't given a Qiskit circuit.
            # Fall back to JSON only if it's JSON-serializable; otherwise fail loudly.
            try:
                ensure_dir(json_dir)
                write_json(json_dir / f"{circuit_id}.json", {"data": circuit_obj, "meta": meta})
            except TypeError as e:
                raise SystemExit(
                    "Failed to export circuit as QASM and it is not JSON-serializable. "
                    "This likely means a Qiskit QuantumCircuit was produced but QASM export failed.\n"
                    f"Original error: {e}"
                )

        record: Dict[str, Any] = {
            "circuit_id": circuit_id,
            "family": meta.get("family", family),
            "representation": meta.get("representation", "unknown"),
            "n_qubits": int(meta.get("n_qubits", args.n_qubits)),
            "depth_target": int(meta.get("depth_target", args.depth)),
            "seed": seed_i,
            "artifact_type": artifact_type,
            "artifact_path": artifact_relpath,
            "created_at": utc_now_iso(),
        }
        append_jsonl(manifest_path, record)
        n_written += 1

    print(f"Wrote {n_written} circuits for family='{family}'")
    print(f"Artifacts root: {circuits_root}")
    print(f"Manifest:       {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
