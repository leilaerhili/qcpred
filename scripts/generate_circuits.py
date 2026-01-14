#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


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


def resolve_family_paths(repo_root: Path, family: str) -> Tuple[Path, Path]:
    """
    Returns:
      circuits_family_root: data/raw/circuits/<family>/
      manifest_path:        data/raw/circuits/<family>/manifest.jsonl
    """
    circuits_family_root = repo_root / "data" / "raw" / "circuits" / family
    manifest_path = circuits_family_root / "manifest.jsonl"
    return circuits_family_root, manifest_path


def main() -> int:
    args = parse_args()

    # Resolve repo root via your existing helper
    from qcpred.utils.paths import find_repo_root

    repo_root = find_repo_root()

    family = args.family.strip()

    # For now, explicitly import known families to ensure registration happens.
    # Later, you can make this dynamic (importlib) once you have more families.
    if family == "random":
        import qcpred.circuits.random  # noqa: F401
    else:
        raise SystemExit(f"Unknown --family '{family}'. Only 'random' is available right now.")

    from qcpred.circuits.registry import get_generator

    gen = get_generator(family)

    circuits_root, manifest_path = resolve_family_paths(repo_root, family)

    # Choose artifact format: prefer QASM if qiskit is available and generator returns a QuantumCircuit
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

        # Decide how to store the artifact
        artifact_type = "json"
        artifact_relpath = f"json/{circuit_id}.json"

        # If qiskit is installed and the object looks like a Qiskit circuit, write QASM
        try:
            from qiskit import QuantumCircuit  # type: ignore

            if isinstance(circuit_obj, QuantumCircuit):
                artifact_type = "qasm"
                artifact_relpath = f"qasm/{circuit_id}.qasm"
                ensure_dir(qasm_dir)
                (qasm_dir / f"{circuit_id}.qasm").write_text(circuit_obj.qasm(), encoding="utf-8")
            else:
                ensure_dir(json_dir)
                write_json(json_dir / f"{circuit_id}.json", {"data": circuit_obj, "meta": meta})
        except Exception:
            # No qiskit, or not a Qiskit object -> fall back to JSON
            ensure_dir(json_dir)
            write_json(json_dir / f"{circuit_id}.json", {"data": circuit_obj, "meta": meta})

        # Manifest record = join key + generation params + artifact pointer
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
