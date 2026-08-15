from __future__ import annotations

import hashlib
import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data"
PROTOCOL = ROOT / "02_protocol"
OUT = ROOT / "03_models"


def load_config() -> dict[str, Any]:
    return json.loads((PROTOCOL / "model_config_locked.json").read_text(encoding="utf-8"))


def load_feature_names() -> list[str]:
    manifest = pd.read_csv(PROTOCOL / "feature_manifest.csv")
    features = manifest.loc[manifest["role"].eq("model_feature"), "column"].astype(str).tolist()
    if not features:
        raise ValueError("Feature manifest is empty")
    return features


def load_samples(columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = DATA / "supervised_samples_h1_h3_h5.csv.gz"
    return pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)


def load_development_samples(columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = DATA / "supervised_development_locked.csv.gz"
    frame = pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)
    if "partition" in frame and not frame["partition"].eq("development").all():
        raise ValueError("Locked development artifact contains non-development rows")
    if "target_year" in frame and frame["target_year"].max() > 2018:
        raise ValueError("Locked development artifact contains post-2018 targets")
    return frame


def load_final_test_samples(columns: Iterable[str] | None = None) -> pd.DataFrame:
    path = DATA / "supervised_final_tests_sealed.csv.gz"
    frame = pd.read_csv(path, usecols=list(columns) if columns is not None else None, low_memory=False)
    allowed = {"test_temporal_seen_country", "test_spatiotemporal_unseen_country"}
    if "partition" in frame and not set(frame["partition"]).issubset(allowed):
        raise ValueError("Sealed final-test artifact contains an unexpected partition")
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_task_seed(master_seed: int, outcome: str, horizon: int, family: str = "") -> int:
    text = f"{master_seed}|{outcome}|{horizon}|{family}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clip_prediction(prediction: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(prediction, dtype=float), 0.0)


def original_from_log_prediction(log_prediction: np.ndarray) -> np.ndarray:
    safe = np.clip(np.asarray(log_prediction, dtype=float), -20.0, 20.0)
    return clip_prediction(np.expm1(safe))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_prediction(y_pred)
    residual = y_pred - y_true
    absolute = np.abs(residual)
    squared = residual**2
    log_residual = np.log1p(y_pred) - np.log1p(y_true)
    denom = float(np.sum(np.abs(y_true)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.ptp(y_true) <= np.finfo(float).eps or np.ptp(y_pred) <= np.finfo(float).eps:
        rho = math.nan
    else:
        rho = spearmanr(y_true, y_pred).statistic
    return {
        "rmsle": float(np.sqrt(np.mean(log_residual**2))),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "wape": float(np.sum(absolute) / denom) if denom > 0 else math.nan,
        "median_absolute_error": float(np.median(absolute)),
        "r2": float(1.0 - np.sum(squared) / total) if total > 0 else math.nan,
        "spearman": float(rho) if np.isfinite(rho) else math.nan,
    }


def get_cv_masks(
    frame: pd.DataFrame, fold: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    train = frame["target_year"].le(int(fold["train_target_year_max"])).to_numpy()
    valid = frame["target_year"].isin([int(x) for x in fold["validation_target_years"]]).to_numpy()
    if not train.any() or not valid.any():
        raise ValueError(f"Empty fold: {fold}")
    if frame.loc[train, "target_year"].max() >= frame.loc[valid, "target_year"].min():
        raise ValueError(f"Temporal ordering violation in fold: {fold}")
    return train, valid


def _cartesian(spec: dict[str, list[Any]], ignore: set[str] | None = None) -> list[dict[str, Any]]:
    ignore = ignore or set()
    keys = [key for key, value in spec.items() if key not in ignore and isinstance(value, list)]
    return [dict(zip(keys, values)) for values in product(*(spec[key] for key in keys))]


def _sample_grid(grid: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(grid) <= n:
        return grid
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(grid), size=n, replace=False))
    return [grid[int(index)] for index in chosen]


def parameter_candidates(family: str, config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    spec = config["tabular_models"][family]
    if family == "ridge":
        return [{"alpha": value} for value in spec["alpha"]]
    if family == "elastic_net":
        return _cartesian(spec)
    limit = int(spec.get("max_random_configs", 10**9))
    grid = _cartesian(
        spec,
        ignore={"max_random_configs", "early_stopping_rounds"},
    )
    return _sample_grid(grid, limit, seed)


def model_complexity_key(family: str, params: dict[str, Any]) -> tuple[Any, ...]:
    if family == "ridge":
        return (-float(params["alpha"]),)
    if family == "elastic_net":
        return (-float(params["alpha"]), -float(params["l1_ratio"]))
    if family in {"random_forest", "extra_trees"}:
        depth = params.get("max_depth")
        depth_score = 10**6 if depth is None else int(depth)
        max_features = params.get("max_features")
        feature_score = 1.0 if max_features == "sqrt" else float(max_features)
        return (
            depth_score,
            -int(params.get("min_samples_leaf", 1)),
            feature_score,
        )
    if family == "hist_gradient_boosting":
        return (
            int(params["max_leaf_nodes"]),
            -int(params["min_samples_leaf"]),
            -float(params["l2_regularization"]),
            -float(params["learning_rate"]),
        )
    if family == "xgboost":
        return (
            int(params["max_depth"]),
            -float(params["min_child_weight"]),
            -float(params["reg_lambda"]),
            -float(params["reg_alpha"]),
            -float(params["subsample"]),
            -float(params["colsample_bytree"]),
        )
    return tuple(sorted(params.items()))


def make_estimator(
    family: str,
    params: dict[str, Any],
    seed: int,
    n_jobs: int = 8,
) -> RegressorMixin:
    if family == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    Ridge(
                        alpha=float(params["alpha"]),
                        solver="lsqr",
                        tol=1e-10,
                        max_iter=10000,
                    ),
                ),
            ]
        )
    if family == "elastic_net":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(params["alpha"]),
                        l1_ratio=float(params["l1_ratio"]),
                        max_iter=30000,
                        tol=1e-7,
                        selection="cyclic",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=n_jobs,
            bootstrap=True,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=n_jobs,
            bootstrap=False,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            max_iter=int(params["max_iter"]),
            early_stopping=False,
            random_state=seed,
        )
    if family == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            early_stopping_rounds=50,
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise KeyError(f"Unknown family: {family}")


def fit_predict_log_model(
    estimator: RegressorMixin,
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None = None,
) -> np.ndarray:
    y_train_log = np.log1p(np.asarray(y_train, dtype=float))
    if family == "xgboost":
        if y_valid is None:
            # XGBoost early stopping is disabled for final all-development fitting.
            estimator.set_params(early_stopping_rounds=None)
            estimator.fit(x_train, y_train_log, verbose=False)
        else:
            estimator.fit(
                x_train,
                y_train_log,
                eval_set=[(x_valid, np.log1p(np.asarray(y_valid, dtype=float)))],
                verbose=False,
            )
    else:
        estimator.fit(x_train, y_train_log)
    return original_from_log_prediction(estimator.predict(x_valid))


def deterministic_gzip_options() -> dict[str, Any]:
    return {"method": "gzip", "compresslevel": 6, "mtime": 0}
