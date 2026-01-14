from __future__ import annotations

import random
from typing import Any, Dict, Tuple
from qcpred.circuits.registry import register

def _qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except Exception:
        return False


def _generate_stub_circuit(
    n_qubits: int,
    depth: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Fallback representation when Qiskit is not available.
    This is JSON-serializable and keeps development unblocked locally.
    """
    gates = ["h", "x", "y", "z", "cx"]

    ops = []
    for _ in range(depth):
        gate = rng.choice(gates)
        if gate == "cx" and n_qubits >= 2:
            q1, q2 = rng.sample(range(n_qubits), 2)
            ops.append({"gate": "cx", "qubits": [q1, q2]})
        else:
            q = rng.randrange(n_qubits)
            ops.append({"gate": gate, "qubits": [q]})

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "ops": ops,
    }


def _generate_qiskit_circuit(
    n_qubits: int,
    depth: int,
    rng: random.Random,
):
    """
    Generate a random QuantumCircuit using Qiskit.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)

    one_q_gates = ["h", "x", "y", "z"]
    for _ in range(depth):
        if rng.random() < 0.7 or n_qubits < 2:
            gate = rng.choice(one_q_gates)
            q = rng.randrange(n_qubits)
            getattr(qc, gate)(q)
        else:
            q1, q2 = rng.sample(range(n_qubits), 2)
            qc.cx(q1, q2)

    return qc


def generate_random_circuit(
    *,
    n_qubits: int,
    depth: int,
    seed: int,
    circuit_id: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Generate a random circuit and return (circuit_object, metadata).

    - If Qiskit is available, circuit_object is a QuantumCircuit
    - Otherwise, circuit_object is a JSON-serializable dict
    """
    rng = random.Random(seed)

    meta: Dict[str, Any] = {
        "circuit_id": circuit_id,
        "family": "random",
        "n_qubits": n_qubits,
        "depth_target": depth,
        "seed": seed,
    }

    if _qiskit_available():
        circuit = _generate_qiskit_circuit(
            n_qubits=n_qubits,
            depth=depth,
            rng=rng,
        )
        meta["representation"] = "qiskit"
    else:
        circuit = _generate_stub_circuit(
            n_qubits=n_qubits,
            depth=depth,
            rng=rng,
        )
        meta["representation"] = "stub"

    return circuit, meta



register("random", generate_random_circuit)
