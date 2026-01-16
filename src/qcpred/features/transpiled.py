from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TranspiledFeatures:
    op_count_post: int
    cx_count_post: int
    swap_count_post: int
    depth_post: int
    twoq_ratio_post: float


def extract_transpiled_features(qc) -> TranspiledFeatures:
    """
    Extract simple, high-ROI features from a transpiled Qiskit QuantumCircuit.
    """
    # Qiskit circuit depth
    depth_post = int(qc.depth())

    # Count ops by name
    ops = qc.count_ops()  # returns dict-like: {"cx": 12, "u3": 40, ...}
    op_count_post = int(sum(int(v) for v in ops.values()))

    cx_count_post = int(ops.get("cx", 0))
    swap_count_post = int(ops.get("swap", 0))

    # For now, treat cx+swap as "two-qubit" (both are 2Q gates)
    twoq_count = cx_count_post + swap_count_post
    twoq_ratio_post = float(twoq_count / op_count_post) if op_count_post > 0 else 0.0

    return TranspiledFeatures(
        op_count_post=op_count_post,
        cx_count_post=cx_count_post,
        swap_count_post=swap_count_post,
        depth_post=depth_post,
        twoq_ratio_post=twoq_ratio_post,
    )


def features_to_dict(f: TranspiledFeatures) -> Dict[str, Any]:
    return {
        "op_count_post": f.op_count_post,
        "cx_count_post": f.cx_count_post,
        "swap_count_post": f.swap_count_post,
        "depth_post": f.depth_post,
        "twoq_ratio_post": f.twoq_ratio_post,
    }
