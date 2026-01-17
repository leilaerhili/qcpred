#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def is_id_like(key: str) -> bool:
    k = key.lower()
    return (
        k.endswith("circuit_id")
        or k.endswith("__circuit_id")
        or k == "circuit_id"
        or k.endswith("artifact_path")
        or k.endswith("__artifact_path")
        or k.endswith("timestamp")
        or k.endswith("__timestamp")
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run feature ablations for HOP prediction (with group split by circuit_id).")
    p.add_argument(
        "--glob",
        type=str,
        default="data/processed/training_rows_with_hop*.jsonl",
        help="Glob for input JSONL files (default: data/processed/training_rows_with_hop*.jsonl)",
    )
    p.add_argument("--test-size", type=float, default=0.3, help="Test fraction (default: 0.3)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--model", choices=["ridge", "rf", "both"], default="both", help="Which model(s) to run")
    p.add_argument(
        "--run-sanity",
        action="store_true",
        help="Run backend-only baseline + shuffled-y sanity check (recommended).",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def flatten(r: Dict[str, Any], which: str) -> Tuple[Dict[str, Any], float, str]:
    """
    which in:
      - logical_only
      - transpiled_only
      - logical_plus_transpiled
      - plus_settings
    Returns: (x_dict, y, circuit_id)
    """
    y = r.get("hop", None)
    if y is None:
        raise ValueError("Missing 'hop' in row.")
    circuit_id = r.get("circuit_id")
    if not circuit_id:
        raise ValueError("Missing 'circuit_id' in row.")

    x: Dict[str, Any] = {}

    # Always include backend tag as categorical (it encodes your noise config naming)
    # x["backend"] = r.get("backend")

    if which in ("logical_only", "logical_plus_transpiled", "plus_settings"):
        lf = r.get("logical_features", {}) or {}
        for k, v in lf.items():
            fk = f"logical__{k}"
            if is_id_like(fk):
                continue
            x[fk] = v

    if which in ("transpiled_only", "logical_plus_transpiled", "plus_settings"):
        tf = r.get("transpiled_features", {}) or {}
        for k, v in tf.items():
            fk = f"transpiled__{k}"
            if is_id_like(fk):
                continue
            x[fk] = v

    if which == "plus_settings":
        ts = r.get("transpile_settings", {}) or {}
        # noise experiment knobs
        for k in ["noise", "p1", "p2", "readout", "optimization_level"]:
            if k in ts:
                x[f"settings__{k}"] = ts.get(k)

    return x, float(y), str(circuit_id)


def dicts_to_matrix(dict_rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for d in dict_rows for k in d.keys()})
    X = np.empty((len(dict_rows), len(keys)), dtype=object)
    for i, d in enumerate(dict_rows):
        for j, k in enumerate(keys):
            X[i, j] = d.get(k, None)
    return X, keys


def build_preprocessor_ridge(feature_names: List[str]) -> ColumnTransformer:
    # Categorical: backend + family/representation + noise string if present
    cat_indices = []
    for i, name in enumerate(feature_names):
        #if name in ("backend",):
        #    cat_indices.append(i)
        if name.endswith("__family") or name.endswith("__representation"):
            cat_indices.append(i)
        if name == "settings__noise":
            cat_indices.append(i)

    cat_indices = sorted(set(cat_indices))
    num_indices = [i for i in range(len(feature_names)) if i not in cat_indices]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_indices),
            ("cat", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_indices),
        ],
        remainder="drop",
    )


def build_preprocessor_rf(feature_names: List[str]) -> ColumnTransformer:
    # Same categorical selection logic
    cat_indices = []
    for i, name in enumerate(feature_names):
    #    if name in ("backend",):
    #        cat_indices.append(i)
        if name.endswith("__family") or name.endswith("__representation"):
            cat_indices.append(i)
        if name == "settings__noise":
            cat_indices.append(i)

    cat_indices = sorted(set(cat_indices))
    num_indices = [i for i in range(len(feature_names)) if i not in cat_indices]

    # RF: impute numeric but DO NOT scale
    # Also: make onehot dense so RF sees a normal numeric matrix.
    # (Dense is fine at your current feature scale.)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
            ]), num_indices),
            ("cat", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_indices),
        ],
        remainder="drop",
    )



def eval_model(name: str, model: Pipeline, X_train, X_test, y_train, y_test) -> Tuple[float, float]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"    {name:>5}: MAE={mae:.4f}  R2={r2:.4f}")
    return mae, r2


def group_split_indices(groups: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def backend_only_baseline(backends: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    # backend is a single categorical feature
    Xb = backends.reshape(-1, 1).astype(object)

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [0],
            )
        ],
        remainder="drop",
    )
    model = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])

    print("[backend_only baseline]")
    eval_model("ridge", model, Xb[train_idx], Xb[test_idx], y[train_idx], y[test_idx])
    print()


def shuffled_y_sanity(pre: ColumnTransformer, X_train, X_test, y_train, y_test, seed: int) -> None:
    # Shuffle y_train; performance should collapse (R2 ~ 0 or negative)
    rng = np.random.default_rng(int(seed))
    y_shuf = y_train.copy()
    rng.shuffle(y_shuf)

    ridge = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])
    print("[sanity shuffled-y]")
    eval_model("ridge", ridge, X_train, X_test, y_shuf, y_test)
    print()


def main() -> int:
    args = parse_args()

    paths = [Path(p) for p in glob.glob(args.glob)]
    if not paths:
        print(f"No files matched glob: {args.glob}")
        return 1

    rows: List[Dict[str, Any]] = []
    for p in sorted(paths):
        rows.extend(read_jsonl(p))

    if len(rows) < 20:
        print(f"[warn] Very small dataset (n={len(rows)}). Results will be noisy, but pipeline sanity is still useful.")

    # Dataset-level diagnostics
    circuit_ids = [r.get("circuit_id") for r in rows if r.get("circuit_id")]
    backends = [r.get("backend") for r in rows]
    unique_c = len(set(circuit_ids))
    unique_b = sorted({b for b in backends if b is not None})
    print(f"Loaded {len(rows)} rows from {len(paths)} file(s).")
    print(f"unique circuit_id: {unique_c}")
    if unique_c > 0:
        # rows per circuit
        counts = {}
        for cid in circuit_ids:
            counts[cid] = counts.get(cid, 0) + 1
        per = sorted(counts.values())
        p50 = per[len(per) // 2]
        print(f"rows/circuit: min={per[0]} median={p50} max={per[-1]}")
    print(f"unique backends: {len(unique_b)} {unique_b[:10]}{' ...' if len(unique_b) > 10 else ''}")
    print()

    ablations = ["logical_only", "transpiled_only", "logical_plus_transpiled", "plus_settings"]
    print("Running ablations:", ", ".join(ablations))
    print()

    # We will compute a group split ONCE using circuit_id groups, and reuse across ablations.
    # Build "groups" aligned to rows order:
    groups = np.array([str(r.get("circuit_id")) for r in rows], dtype=object)
    train_idx, test_idx = group_split_indices(groups, test_size=float(args.test_size), seed=int(args.seed))

    # Verify no group overlap
    train_c = set(groups[train_idx])
    test_c = set(groups[test_idx])
    print(f"group split: train_rows={len(train_idx)} test_rows={len(test_idx)}")
    print(f"train_circuits={len(train_c)} test_circuits={len(test_c)} overlap={len(train_c & test_c)}")
    if len(train_c & test_c) != 0:
        print("[error] circuit_id overlap detected in group split (should be 0).")
        return 2
    print()

    # Backend-only baseline (uses same split)
    if args.run_sanity:
        backend_arr = np.array([r.get("backend") for r in rows], dtype=object)
        # Need y for baseline; use hop from rows
        y_all = np.array([float(r["hop"]) for r in rows], dtype=float)
        backend_only_baseline(backend_arr, y_all, train_idx, test_idx)

    for which in ablations:
        X_dicts: List[Dict[str, Any]] = []
        y_list: List[float] = []

        for r in rows:
            x, y, _cid = flatten(r, which)
            X_dicts.append(x)
            y_list.append(y)

        X_obj, feature_names = dicts_to_matrix(X_dicts)
        # quick check: how many non-numeric values ended up in numeric cols?
        num_like = []
        for name in feature_names:
            if name in ("backend", "settings__noise") or name.endswith("__family") or name.endswith("__representation"):
                continue
            num_like.append(name)

        print(f"    features: total={len(feature_names)} numeric_like={len(num_like)}")

        y = np.array(y_list, dtype=float)

        pre_ridge = build_preprocessor_ridge(feature_names)
        pre_rf = build_preprocessor_rf(feature_names)

        X_train = X_obj[train_idx]
        X_test = X_obj[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"[{which}] n_train={len(y_train)} n_test={len(y_test)}")

        if args.model in ("ridge", "both"):
            ridge = Pipeline(steps=[("pre", pre_ridge), ("model", Ridge(alpha=1.0))])
            eval_model("ridge", ridge, X_train, X_test, y_train, y_test)

        if args.model in ("rf", "both"):
            rf = Pipeline(
                steps=[
                    ("pre", pre_rf),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            random_state=int(args.seed),
                            min_samples_leaf=5,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

            eval_model("rf", rf, X_train, X_test, y_train, y_test)

        if args.run_sanity and which == "logical_only":
            # run shuffled-y sanity once (logical_only is fine)
            shuffled_y_sanity(pre_ridge, X_train, X_test, y_train, y_test, seed=int(args.seed))


        print()

    print("Interpretation guide:")
    print("- If transpiled_only >> logical_only: compilation dominates (expected on NISQ).")
    print("- If logical_plus_transpiled > transpiled_only: logical structure adds predictive signal.")
    print("- If plus_settings improves a lot: your label is strongly driven by noise parameters (expected in Aer-noise phase).")
    print()
    print("Notes:")
    print("- This script uses GroupShuffleSplit by circuit_id, so the same circuit never appears in both train and test.")
    print("- Use --run-sanity to print backend-only baseline + shuffled-y leak check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
