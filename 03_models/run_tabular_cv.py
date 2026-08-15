from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from model_utils import (
    OUT,
    deterministic_gzip_options,
    fit_predict_log_model,
    get_cv_masks,
    load_config,
    load_feature_names,
    load_development_samples,
    make_estimator,
    model_complexity_key,
    original_from_log_prediction,
    parameter_candidates,
    regression_metrics,
    sha256,
    stable_task_seed,
)


FAMILIES = [
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
]
BASELINES = ["persistence", "logtrend5", "logtrend10", "pooled_ridge"]
IDENTITY = [
    "sample_id",
    "location_id",
    "location_name",
    "origin_year",
    "target_year",
    "horizon",
    "sdi_quintile_2023_assignment_only",
    "partition",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_id(family: str, params: dict[str, Any]) -> str:
    payload = f"{family}|{canonical_json(params)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def baseline_columns(outcome: str) -> dict[str, str]:
    return {
        "persistence": f"baseline_persistence_{outcome}",
        "logtrend5": f"baseline_logtrend5_{outcome}",
        "logtrend10": f"baseline_logtrend10_{outcome}",
    }


def pooled_baseline_features(outcome: str) -> list[str]:
    prefix = f"bmi_{outcome}_rate"
    return [
        "origin_year",
        f"{prefix}__current",
        f"{prefix}__log_slope5",
        f"{prefix}__log_slope10",
        "sdi__current",
    ]


def pooled_baseline_matrix(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    """Features for the prespecified pooled regularized log-linear trend.

    The target is fitted in log1p space, so the current burden level must also be
    represented in log1p space. Local slopes are already log-scale quantities.
    """
    prefix = f"bmi_{outcome}_rate"
    return np.column_stack(
        [
            (frame["origin_year"].to_numpy(dtype=float) - 2000.0) / 10.0,
            np.log1p(frame[f"{prefix}__current"].to_numpy(dtype=float)),
            frame[f"{prefix}__log_slope5"].to_numpy(dtype=float),
            frame[f"{prefix}__log_slope10"].to_numpy(dtype=float),
            frame["sdi__current"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)


def outer_fold_prediction(
    family: str,
    params: dict[str, Any],
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    target_year: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    """Fit without exposing the outer validation window to early stopping.

    XGBoost uses the final two years within the outer training window as an
    inner temporal stopping set, then is refit on the complete outer training
    window for the selected number of boosting rounds.
    """
    if family != "xgboost":
        estimator = make_estimator(family, params, seed, n_jobs=8)
        prediction = fit_predict_log_model(
            estimator,
            family,
            x[train_mask],
            y[train_mask],
            x[valid_mask],
            y[valid_mask],
        )
        return prediction, None

    outer_train_max = int(target_year[train_mask].max())
    inner_valid = train_mask & (target_year >= outer_train_max - 1)
    inner_train = train_mask & (target_year < outer_train_max - 1)
    if inner_train.sum() == 0 or inner_valid.sum() == 0:
        raise ValueError("XGBoost inner temporal early-stopping split is empty")

    stopping_model = make_estimator("xgboost", params, seed, n_jobs=8)
    stopping_model.fit(
        x[inner_train],
        np.log1p(y[inner_train]),
        eval_set=[(x[inner_valid], np.log1p(y[inner_valid]))],
        verbose=False,
    )
    effective_rounds = int(getattr(stopping_model, "best_iteration", params["n_estimators"] - 1)) + 1
    refit_params = dict(params)
    refit_params["n_estimators"] = effective_rounds
    refit_model = make_estimator("xgboost", refit_params, seed, n_jobs=8)
    refit_model.set_params(early_stopping_rounds=None)
    refit_model.fit(x[train_mask], np.log1p(y[train_mask]), verbose=False)
    prediction = original_from_log_prediction(refit_model.predict(x[valid_mask]))
    return prediction, effective_rounds


def candidate_rows_for_task_family(
    frame: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    outcome: str,
    horizon: int,
    family: str,
    config: dict[str, Any],
    smoke: bool,
) -> pd.DataFrame:
    seed = stable_task_seed(config["master_seed"], outcome, horizon, family)
    candidates = parameter_candidates(family, config, seed)
    if smoke:
        candidates = candidates[:1]
    x = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame[target_column].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(candidates):
        candidate_id = config_id(family, params)
        for fold in config["cross_validation"]:
            train_mask, valid_mask = get_cv_masks(frame, fold)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                prediction, effective_rounds = outer_fold_prediction(
                    family,
                    params,
                    seed + candidate_index,
                    x,
                    y,
                    frame["target_year"].to_numpy(dtype=int),
                    train_mask,
                    valid_mask,
                )
            metrics = regression_metrics(y[valid_mask], prediction)
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": horizon,
                    "family": family,
                    "config_id": candidate_id,
                    "candidate_index": candidate_index,
                    "params_json": canonical_json(params),
                    "fold": fold["fold"],
                    "n_train": int(train_mask.sum()),
                    "n_validation": int(valid_mask.sum()),
                    "effective_n_estimators": effective_rounds,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def search_pooled_ridge(
    frame: pd.DataFrame,
    target_column: str,
    outcome: str,
    horizon: int,
    config: dict[str, Any],
    smoke: bool,
) -> pd.DataFrame:
    family = "pooled_ridge"
    seed = stable_task_seed(config["master_seed"], outcome, horizon, family)
    candidates = [{"alpha": value} for value in config["tabular_models"]["ridge"]["alpha"]]
    if smoke:
        candidates = candidates[:1]
    x = pooled_baseline_matrix(frame, outcome)
    y = frame[target_column].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for candidate_index, params in enumerate(candidates):
        candidate_id = config_id(family, params)
        for fold in config["cross_validation"]:
            train_mask, valid_mask = get_cv_masks(frame, fold)
            estimator = make_estimator("ridge", params, seed + candidate_index, n_jobs=8)
            prediction = fit_predict_log_model(
                estimator,
                "ridge",
                x[train_mask],
                y[train_mask],
                x[valid_mask],
                y[valid_mask],
            )
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": horizon,
                    "family": family,
                    "config_id": candidate_id,
                    "candidate_index": candidate_index,
                    "params_json": canonical_json(params),
                    "fold": fold["fold"],
                    "n_train": int(train_mask.sum()),
                    "n_validation": int(valid_mask.sum()),
                    **regression_metrics(y[valid_mask], prediction),
                }
            )
    return pd.DataFrame(rows)


def select_config(search: pd.DataFrame, tolerance: float) -> pd.Series:
    numeric = [
        "rmsle",
        "mae",
        "rmse",
        "wape",
        "median_absolute_error",
        "r2",
        "spearman",
    ]
    summary = (
        search.groupby(
            ["outcome", "horizon", "family", "config_id", "candidate_index", "params_json"],
            as_index=False,
        )[numeric]
        .mean()
        .rename(columns={name: f"mean_{name}" for name in numeric})
    )
    minimum = float(summary["mean_rmsle"].min())
    eligible = summary.loc[summary["mean_rmsle"].le(minimum * (1.0 + tolerance))].copy()
    family = str(eligible["family"].iloc[0])
    complexity_family = "ridge" if family == "pooled_ridge" else family
    eligible["_complexity_key"] = eligible["params_json"].map(
        lambda text: model_complexity_key(complexity_family, json.loads(text))
    )
    eligible = eligible.sort_values(
        ["_complexity_key", "mean_rmsle", "config_id"], kind="mergesort"
    )
    selected = eligible.iloc[0].copy()
    selected["complexity_key_json"] = canonical_json(list(selected.pop("_complexity_key")))
    selected["minimum_mean_rmsle"] = minimum
    selected["selection_tolerance"] = tolerance
    return selected


def baseline_fold_metrics(
    frame: pd.DataFrame,
    target_column: str,
    outcome: str,
    horizon: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    y = frame[target_column].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for baseline, column in baseline_columns(outcome).items():
        prediction = frame[column].to_numpy(dtype=np.float64)
        for fold in config["cross_validation"]:
            _, valid_mask = get_cv_masks(frame, fold)
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": horizon,
                    "family": baseline,
                    "config_id": baseline,
                    "candidate_index": 0,
                    "params_json": "{}",
                    "fold": fold["fold"],
                    "n_train": 0,
                    "n_validation": int(valid_mask.sum()),
                    **regression_metrics(y[valid_mask], prediction[valid_mask]),
                }
            )
    return pd.DataFrame(rows)


def selected_oof_predictions(
    frame: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    outcome: str,
    horizon: int,
    selected_rows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    y = frame[target_column].to_numpy(dtype=np.float64)
    records: list[pd.DataFrame] = []

    for baseline, column in baseline_columns(outcome).items():
        for fold in config["cross_validation"]:
            _, valid_mask = get_cv_masks(frame, fold)
            part = frame.loc[valid_mask, IDENTITY].copy()
            part["outcome"] = outcome
            part["family"] = baseline
            part["fold"] = fold["fold"]
            part["observed"] = y[valid_mask]
            part["prediction"] = frame.loc[valid_mask, column].to_numpy(dtype=float)
            part["config_id"] = baseline
            records.append(part)

    for selected in selected_rows.itertuples(index=False):
        family = str(selected.family)
        params = json.loads(selected.params_json)
        # Replay the exact candidate seed used during hyperparameter search.
        # Omitting candidate_index would silently change stochastic OOF models
        # (trees and XGBoost) after configuration selection.
        seed = stable_task_seed(config["master_seed"], outcome, horizon, family) + int(
            selected.candidate_index
        )
        if family == "pooled_ridge":
            x = pooled_baseline_matrix(frame, outcome)
            estimator_family = "ridge"
        else:
            x = frame[feature_names].to_numpy(dtype=np.float32)
            estimator_family = family
        for fold in config["cross_validation"]:
            train_mask, valid_mask = get_cv_masks(frame, fold)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                prediction, _ = outer_fold_prediction(
                    estimator_family,
                    params,
                    seed,
                    x,
                    y,
                    frame["target_year"].to_numpy(dtype=int),
                    train_mask,
                    valid_mask,
                )
            part = frame.loc[valid_mask, IDENTITY].copy()
            part["outcome"] = outcome
            part["family"] = family
            part["fold"] = fold["fold"]
            part["observed"] = y[valid_mask]
            part["prediction"] = prediction
            part["config_id"] = selected.config_id
            records.append(part)

    output = pd.concat(records, ignore_index=True)
    output["log1p_residual"] = np.log1p(output["observed"]) - np.log1p(output["prediction"])
    return output.sort_values(
        ["outcome", "horizon", "family", "fold", "location_id", "target_year"],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_inputs(frame: pd.DataFrame, feature_names: list[str], config: dict[str, Any]) -> None:
    if frame[feature_names].isna().any().any():
        raise ValueError("Model feature matrix contains missing values")
    forbidden = {"location_id", "location_name", "country_split", "partition", "target_year"}
    overlap = forbidden.intersection(feature_names)
    if overlap:
        raise ValueError(f"Forbidden model features: {sorted(overlap)}")
    if not frame["partition"].eq(config["development_partition"]).all():
        raise ValueError("CV frame contains non-development rows")
    if frame["target_year"].max() > 2018:
        raise ValueError("CV frame has final-test target years")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked tabular expanding-window CV")
    parser.add_argument("--smoke", action="store_true", help="Run one task and one config per family")
    parser.add_argument("--force", action="store_true", help="Recompute completed search blocks")
    parser.add_argument("--family", choices=FAMILIES, help="Restrict full run to one family")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    feature_names = load_feature_names()
    needed = sorted(
        set(
            IDENTITY
            + feature_names
            + list(config["outcomes"].values())
            + [column for outcome in config["outcomes"] for column in baseline_columns(outcome).values()]
        )
    )
    all_samples = load_development_samples(needed)
    validate_inputs(all_samples, feature_names, config)

    work = OUT / ("smoke_v2" if args.smoke else "cv_search_blocks_v2")
    work.mkdir(parents=True, exist_ok=True)
    outcomes = ["daly"] if args.smoke else list(config["outcomes"])
    horizons = [1] if args.smoke else [int(value) for value in config["horizons"]]
    families = [args.family] if args.family else FAMILIES

    for outcome in outcomes:
        target_column = config["outcomes"][outcome]
        for horizon in horizons:
            frame = all_samples.loc[all_samples["horizon"].eq(horizon)].copy()
            task = f"{outcome}_h{horizon}"
            print(f"[task] {task}: {len(frame)} development samples", flush=True)
            baseline_path = work / f"{task}__transparent_baselines.csv"
            if args.force or not baseline_path.exists():
                baseline_fold_metrics(frame, target_column, outcome, horizon, config).to_csv(
                    baseline_path, index=False, float_format="%.12g"
                )
                print(f"[done] {task}: transparent baselines", flush=True)
            pooled_path = work / f"{task}__pooled_ridge.csv"
            if args.force or not pooled_path.exists():
                result = search_pooled_ridge(
                    frame, target_column, outcome, horizon, config, args.smoke
                )
                result.to_csv(pooled_path, index=False, float_format="%.12g")
                print(f"[done] {task}: pooled_ridge ({len(result) // 3} configs)", flush=True)
            for family in families:
                output_path = work / f"{task}__{family}.csv"
                if output_path.exists() and not args.force:
                    print(f"[skip] {task}: {family}", flush=True)
                    continue
                print(f"[run]  {task}: {family}", flush=True)
                result = candidate_rows_for_task_family(
                    frame,
                    feature_names,
                    target_column,
                    outcome,
                    horizon,
                    family,
                    config,
                    args.smoke,
                )
                result.to_csv(output_path, index=False, float_format="%.12g")
                print(f"[done] {task}: {family} ({len(result) // 3} configs)", flush=True)

    # Only immutable task-family blocks are inputs; aggregate outputs from a
    # previous run must never be recursively re-ingested.
    blocks = sorted(work.glob("*__*.csv"))
    if not blocks:
        raise RuntimeError("No CV blocks were generated")
    searches = pd.concat([pd.read_csv(path) for path in blocks], ignore_index=True)
    searches = searches.sort_values(
        ["outcome", "horizon", "family", "candidate_index", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    searches.to_csv(work / "tabular_cv_fold_metrics.csv", index=False, float_format="%.12g")

    selected: list[pd.Series] = []
    for _, group in searches.loc[searches["family"].isin(FAMILIES + ["pooled_ridge"])].groupby(
        ["outcome", "horizon", "family"], sort=True
    ):
        selected.append(select_config(group, float(config["selection_tie_relative_tolerance"])))
    selected_frame = pd.DataFrame(selected).sort_values(
        ["outcome", "horizon", "family"], kind="mergesort"
    )
    selected_frame.to_csv(work / "selected_tabular_configurations.csv", index=False, float_format="%.12g")

    predictions: list[pd.DataFrame] = []
    for outcome in outcomes:
        target_column = config["outcomes"][outcome]
        for horizon in horizons:
            frame = all_samples.loc[all_samples["horizon"].eq(horizon)].copy()
            chosen = selected_frame.loc[
                selected_frame["outcome"].eq(outcome)
                & selected_frame["horizon"].eq(horizon)
            ]
            predictions.append(
                selected_oof_predictions(
                    frame,
                    feature_names,
                    target_column,
                    outcome,
                    horizon,
                    chosen,
                    config,
                )
            )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_path = work / "selected_tabular_oof_predictions.csv.gz"
    prediction_frame.to_csv(
        prediction_path,
        index=False,
        float_format="%.12g",
        compression=deterministic_gzip_options(),
    )

    metadata = {
        "status": "smoke" if args.smoke else "complete",
        "dataset_sha256": sha256(OUT.parent / "01_data" / "supervised_development_locked.csv.gz"),
        "feature_manifest_sha256": sha256(OUT.parent / "02_protocol" / "feature_manifest.csv"),
        "model_config_sha256": sha256(OUT.parent / "02_protocol" / "model_config_locked.json"),
        "implementation_sha256": sha256(Path(__file__)),
        "model_utils_sha256": sha256(Path(__file__).with_name("model_utils.py")),
        "outcomes": outcomes,
        "horizons": horizons,
        "families": sorted(searches["family"].astype(str).unique().tolist()),
        "invoked_family_scope": families,
        "n_search_rows": int(len(searches)),
        "n_selected_configurations": int(len(selected_frame)),
        "n_oof_predictions": int(len(prediction_frame)),
        "outputs": {
            "fold_metrics_sha256": sha256(work / "tabular_cv_fold_metrics.csv"),
            "selected_configurations_sha256": sha256(work / "selected_tabular_configurations.csv"),
            "oof_predictions_sha256": sha256(prediction_path),
        },
    }
    (work / "tabular_cv_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
