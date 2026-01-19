#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def parse_p2_grid(s: str) -> List[float]:
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("Empty --p2-grid")
    return out


def backend_tag_for_p2(prefix: str, p2: float) -> str:
    # 0.05 -> "0p05" (keep it deterministic and filename-friendly)
    s = f"{p2:.6f}".rstrip("0").rstrip(".")
    s = s.replace(".", "p")
    return f"{prefix}_p2_{s}"


def build_noise_simulator(seed: int, noise: str, p1: float, p2: float, readout: float):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_mode = noise.strip()

    if noise_mode == "depolarizing":
        nm = NoiseModel()

        e1 = depolarizing_error(float(p1), 1)
        e2 = depolarizing_error(float(p2), 2)

        # Common basis gates post-transpile; harmless if a gate isn't present
        nm.add_all_qubit_quantum_error(e1, ["x", "y", "z", "h", "sx", "rz", "rx", "ry", "id"])
        nm.add_all_qubit_quantum_error(e2, ["cx", "cz", "swap"])

        ro = float(readout)
        if ro > 0:
            nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))

        return AerSimulator(seed_simulator=int(seed), noise_model=nm)

    return AerSimulator(seed_simulator=int(seed))


def run_one_backend(
    *,
    family: str,
    backend_tag: str,
    qasm_records: List[Dict[str, Any]],
    circuits_root: Path,
    out_path: Path,
    shots: int,
    seed: int,
    noise: str,
    p1: float,
    p2: float,
    readout: float,
) -> tuple[int, int]:
    """
    Runs qasm_records on Aer, writes execution rows to out_path.
    Returns (n_ok, n_fail).
    """
    # Qiskit imports
    from qiskit import QuantumCircuit, transpile
    from qiskit.qasm2 import load as qasm2_load

    sim = build_noise_simulator(seed=seed, noise=noise, p1=p1, p2=p2, readout=readout)

    transpile_settings: Dict[str, Any] = {
        "optimization_level": 1,
        "seed_transpiler": int(seed),
        "noise": noise,
        "p1": float(p1),
        "p2": float(p2),
        "readout": float(readout),
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
            qc: QuantumCircuit = qasm2_load(str(artifact_path))

            tqc = transpile(
                qc,
                sim,
                optimization_level=int(transpile_settings["optimization_level"]),
                seed_transpiler=int(transpile_settings["seed_transpiler"]),
            )

            tf = extract_transpiled_features(tqc)

            job = sim.run(tqc, shots=int(shots))
            result = job.result()
            counts = result.get_counts()

            exec_record: Dict[str, Any] = {
                "circuit_id": circuit_id,
                "family": r.get("family", family),
                "backend": backend_tag,
                "shots": int(shots),
                "timestamp": utc_now_iso(),
                "transpile_settings": transpile_settings,
                "counts": counts,
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
    return n_ok, n_fail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run circuits on Aer and record counts (optionally with a noise model).")
    p.add_argument("--family", type=str, default="random", help="Circuit family (default: random)")

    # Tag used for output folder naming (single-backend mode)
    p.add_argument("--backend", type=str, default="aer_simulator", help="Backend tag for output folder naming")

    p.add_argument("--shots", type=int, default=2048, help="Shots per circuit (default: 2048)")
    p.add_argument("--max-circuits", type=int, default=0, help="If >0, limit number of circuits to run")
    p.add_argument("--seed", type=int, default=0, help="Seed for simulator/transpiler (default: 0)")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSONL path. Default: data/raw/results/<family>/<backend>/executions.jsonl",
    )

    # DIY noise model knobs
    p.add_argument(
        "--noise",
        type=str,
        default="none",
        choices=["none", "depolarizing"],
        help="Noise model to apply in Aer (default: none)",
    )
    p.add_argument("--p1", type=float, default=0.001, help="1Q depolarizing error prob (default: 0.001)")
    p.add_argument("--p2", type=float, default=0.01, help="2Q depolarizing error prob (default: 0.01)")
    p.add_argument("--readout", type=float, default=0.02, help="Readout error prob (default: 0.02)")

    # NEW: sweep mode (optional)
    p.add_argument(
        "--p2-grid",
        type=str,
        default="",
        help='Comma-separated p2 values to sweep (e.g. "0.01,0.03,0.05,0.08,0.10"). '
             "If set, runs one backend per p2 value.",
    )
    p.add_argument(
        "--p1-ratio",
        type=float,
        default=50.0,
        help="If sweeping, tie p1 = p2 / p1_ratio (default: 50).",
    )
    p.add_argument(
        "--backend-prefix",
        type=str,
        default="aer_dep",
        help='If sweeping, backend tags become "<prefix>_p2_<value>" (default: aer_dep).',
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    family = args.family.strip()

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

    # Filter to QASM records for now
    qasm_records = [r for r in records if r.get("artifact_type") == "qasm"]
    if not qasm_records:
        print("No QASM circuits found in manifest. Generate QASM circuits first.")
        return 1

    if args.max_circuits and args.max_circuits > 0:
        qasm_records = qasm_records[: int(args.max_circuits)]

    # Sweep mode
    if args.p2_grid.strip():
        p2_vals = parse_p2_grid(args.p2_grid)
        p1_ratio = float(args.p1_ratio)
        if p1_ratio <= 0:
            print("[error] --p1-ratio must be > 0")
            return 2

        # Sweeps are only meaningful with depolarizing noise
        if args.noise.strip() != "depolarizing":
            print("[info] --p2-grid set, forcing --noise depolarizing")
        noise = "depolarizing"

        total_ok = 0
        total_fail = 0

        for p2 in p2_vals:
            p1 = float(p2) / p1_ratio
            backend_tag = backend_tag_for_p2(args.backend_prefix.strip(), float(p2))
            out_path = repo_path("data", "raw", "results", family, backend_tag, "executions.jsonl")

            print(f"\n=== sweep backend={backend_tag}  p2={p2}  p1={p1}  readout={float(args.readout)} ===")

            n_ok, n_fail = run_one_backend(
                family=family,
                backend_tag=backend_tag,
                qasm_records=qasm_records,
                circuits_root=circuits_root,
                out_path=out_path,
                shots=int(args.shots),
                seed=int(args.seed),
                noise=noise,
                p1=float(p1),
                p2=float(p2),
                readout=float(args.readout),
            )
            total_ok += n_ok
            total_fail += n_fail

        print(f"\n[sweep done] total_ok={total_ok} total_fail={total_fail}")
        return 0 if total_ok > 0 else 1

    # Single-backend mode (original behavior)
    backend_tag = args.backend.strip()
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = repo_path("data", "raw", "results", family, backend_tag, "executions.jsonl")

    n_ok, _ = run_one_backend(
        family=family,
        backend_tag=backend_tag,
        qasm_records=qasm_records,
        circuits_root=circuits_root,
        out_path=out_path,
        shots=int(args.shots),
        seed=int(args.seed),
        noise=args.noise.strip(),
        p1=float(args.p1),
        p2=float(args.p2),
        readout=float(args.readout),
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
