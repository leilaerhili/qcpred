from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CircuitFeatures:
    circuit_id: str
    family: str
    representation: str
    n_qubits: int
    depth_target: int

    # Derived from stored representation
    op_count: int
    oneq_count: int
    cx_count: int

    # Handy ratios
    twoq_ratio: float


def _count_from_stub(stub: Dict[str, Any]) -> Dict[str, int]:
    ops = stub.get("ops", [])
    op_count = len(ops)

    cx_count = 0
    oneq_count = 0
    for op in ops:
        gate = op.get("gate")
        if gate == "cx":
            cx_count += 1
        else:
            oneq_count += 1

    return {
        "op_count": op_count,
        "oneq_count": oneq_count,
        "cx_count": cx_count,
    }


def _load_stub_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Our generator script stores either:
    # - {"data": <stub>, "meta": {...}}
    # or possibly just a stub (if you change format later).
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def extract_circuit_features_from_record(record: Dict[str, Any], circuits_root: Path) -> CircuitFeatures:
    """
    record is one JSON object from manifest.jsonl.
    circuits_root is data/raw/circuits/
    """
    circuit_id = record["circuit_id"]
    family = record.get("family", "unknown")
    representation = record.get("representation", "unknown")
    n_qubits = int(record.get("n_qubits", 0))
    depth_target = int(record.get("depth_target", 0))

    artifact_path = record.get("artifact_path")
    artifact_type = record.get("artifact_type")

    op_count = oneq_count = cx_count = 0

    if isinstance(artifact_path, str):
        full_path = circuits_root / artifact_path

        if artifact_type == "json" or full_path.suffix.lower() == ".json":
            stub = _load_stub_json(full_path)
            counts = _count_from_stub(stub)
            op_count = counts["op_count"]
            oneq_count = counts["oneq_count"]
            cx_count = counts["cx_count"]

        elif artifact_type == "qasm" or full_path.suffix.lower() == ".qasm":
            # QASM feature extraction requires Qiskit to parse reliably.
            # We'll compute basic counts if Qiskit is available; otherwise, leave zeros.
            try:
                from qiskit import QuantumCircuit  # type: ignore
                qc = QuantumCircuit.from_qasm_file(str(full_path))
                op_count = sum(qc.count_ops().values())
                cx_count = int(qc.count_ops().get("cx", 0))
                oneq_count = op_count - cx_count
            except Exception:
                # Keep zeros if qiskit isn't available / parsing fails
                pass

    twoq_ratio = (cx_count / op_count) if op_count else 0.0

    return CircuitFeatures(
        circuit_id=circuit_id,
        family=family,
        representation=representation,
        n_qubits=n_qubits,
        depth_target=depth_target,
        op_count=op_count,
        oneq_count=oneq_count,
        cx_count=cx_count,
        twoq_ratio=float(twoq_ratio),
    )


def features_to_dict(f: CircuitFeatures) -> Dict[str, Any]:
    return asdict(f)
