from __future__ import annotations

from pathlib import Path
from qcpred.utils.paths import repo_path


def raw_circuits_dir() -> Path:
    return repo_path("data", "raw", "circuits")


def raw_circuits_qasm_dir() -> Path:
    return raw_circuits_dir() / "qasm"


def raw_circuits_manifest_path() -> Path:
    return raw_circuits_dir() / "manifest.jsonl"
