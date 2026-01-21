#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train robustness predictor from circuit+backend features (B1).")
    p.add_argument("--in", dest="inp", type=str, default="data/processed/robustness_targets_b1.jsonl")
    p.add_argument("--target", choices=["hop0", "hop1", "hop2"], default="hop2")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["ridge", "rf", "both"], default="both")
    p.add_argument("--run-sanity", action="store_true")
    p.add_argument("--split-by", choices=["circuit", "backend"], default="circuit")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_id_like(key: str) -> bool:
    k = key.lower()
    return (
        k.endswith("circuit_id")
        or k.endswith("__circuit_id")
        or k == "circuit_id"
        or k.endswith("backend_profile_id")
        or k.endswith("__backend_profile_id")
        or k.endswith("artifact_path")
        or k.endswith("__artifact_path")
        or k.endswith("timestamp")
        or k.endswith("__timestamp")
    )


def flatten_features(r: Dict[str, Any]) -> Dict[str, Any]:
    x: Dict[str, Any] = {}

    for k, v in (r.get("logical_features", {}) or {}).items():
        fk = f"logical__{k}"
        if not is_id_like(fk):
            x[fk] = v

    for k, v in (r.get("transpiled_features", {}) or {}).items():
        fk = f"transpiled__{k}"
        if not is_id_like(fk):
            x[fk] = v

    for k, v in (r.get("backend_features", {}) or {}).items():
        fk = f"backend__{k}"
        if not is_id_like(fk):
            x[fk] = v

    return x


def dicts_to_matrix(dict_rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for d in dict_rows for k in d.keys()})
    X = np.empty((len(dict_rows), len(keys)), dtype=object)
    for i, d in enumerate(dict_rows):
        for j, k in enumerate(keys):
            X[i, j] = d.get(k, None)
    return X, keys


def build_preprocessor(feature_names: List[str], for_rf: bool) -> ColumnTransformer:
    cat_indices = []
    for i, name in enumerate(feature_names):
        if name.endswith("__family") or name.endswith("__representation") or name.endswith("backend__noise_model"):
            cat_indices.append(i)

    cat_indices = sorted(set(cat_indices))
    num_indices = [i for i in range(len(feature_names)) if i not in cat_indices]

    if for_rf:
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), num_indices),
                ("cat", Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), cat_indices),
            ],
            remainder="drop",
        )

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


def eval_model(name: str, model: Pipeline, X_train, X_test, y_train, y_test) -> None:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"  {name:>5}: MAE={mae:.4f}  R2={r2:.4f}")


def main() -> int:
    args = parse_args()
    path = Path(args.inp)

    rows = read_jsonl(path)
    if not rows:
        print(f"No rows found at: {path}")
        return 1

    X_dicts = [flatten_features(r) for r in rows]
    X_obj, feature_names = dicts_to_matrix(X_dicts)

    y = np.array([float(r["robustness"][args.target]) for r in rows], dtype=float)

    if args.split_by == "circuit":
        groups = np.array([str(r["circuit_id"]) for r in rows], dtype=object)
        split_name = "circuit_id"
    else:
        groups = np.array([str(r["backend_profile_id"]) for r in rows], dtype=object)
        split_name = "backend_profile_id"

    print(f"Loaded {len(rows)} (circuit, backend) rows")
    print(f"target={args.target}  y min/mean/max = {y.min():.4g} {y.mean():.4g} {y.max():.4g}")
    print(f"grouping by {split_name}: unique={len(set(groups))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.seed))
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), y, groups=groups))

    X_train, X_test = X_obj[train_idx], X_obj[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if args.run_sanity:
        rng = np.random.default_rng(int(args.seed))
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        pre = build_preprocessor(feature_names, for_rf=False)
        ridge = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])
        print("[sanity shuffled-y]")
        eval_model("ridge", ridge, X_train, X_test, y_shuf, y_test)
        print()

    print("[models]")
    if args.model in ("ridge", "both"):
        pre = build_preprocessor(feature_names, for_rf=False)
        ridge = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])
        eval_model("ridge", ridge, X_train, X_test, y_train, y_test)

    if args.model in ("rf", "both"):
        pre = build_preprocessor(feature_names, for_rf=True)
        rf = Pipeline(steps=[("pre", pre), ("model", RandomForestRegressor(
            n_estimators=500,
            random_state=int(args.seed),
            min_samples_leaf=5,
            n_jobs=-1,
        ))])
        eval_model("rf", rf, X_train, X_test, y_train, y_test)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
