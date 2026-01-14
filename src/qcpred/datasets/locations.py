from __future__ import annotations

from pathlib import Path
from qcpred.utils.paths import repo_path

DEFAULT_FAMILY = "random"


# ---------- New, family-aware API (use this going forward) ----------

def raw_circuits_family_dir(family: str) -> Path:
    family = family.strip()
    if not family:
        raise ValueError("family must be a non-empty string")
    return repo_path("data", "raw", "circuits", family)


def raw_circuits_manifest_path_for(family: str) -> Path:
    return raw_circuits_family_dir(family) / "manifest.jsonl"


def raw_circuits_qasm_dir_for(family: str) -> Path:
    return raw_circuits_family_dir(family) / "qasm"


def raw_circuits_json_dir_for(family: str) -> Path:
    return raw_circuits_family_dir(family) / "json"



# ---------- Legacy API (temporary wrappers) ----------

def raw_circuits_dir() -> Path:
    return raw_circuits_family_dir(DEFAULT_FAMILY)


def raw_circuits_qasm_dir() -> Path:
    return raw_circuits_qasm_dir_for(DEFAULT_FAMILY)


def raw_circuits_manifest_path() -> Path:
    return raw_circuits_manifest_path_for(DEFAULT_FAMILY)
