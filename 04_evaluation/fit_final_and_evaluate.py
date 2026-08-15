from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "03_models"
sys.path.insert(0, str(MODELS))

from lock_model_selection import ADVANCED, TRANSPARENT  # noqa: E402
from model_utils import (  # noqa: E402
    DATA,
    deterministic_gzip_options,
    fit_predict_log_model,
    load_config,
    load_development_samples,
    load_feature_names,
    load_final_test_samples,
    make_estimator,
    regression_metrics,
    sha256,
    stable_task_seed,
)
from run_gru_cv import (  # noqa: E402
    GRURegressor,
    apply_x_scaler,
    build_sequence_tensor,
    fit_fixed_epochs,
    predict_network,
    x_scaler,
    y_scaler,
)
from run_tabular_cv import baseline_columns, pooled_baseline_matrix  # noqa: E402


OUT = ROOT / "04_evaluation"
FITTED = OUT / "fitted_models"
LOCK_PATH = MODELS / "selection_lock" / "model_selection_lock.json"
LOCK_HASH_PATH = MODELS / "selection_lock" / "model_selection_lock.sha256.txt"
TAB_OOF = MODELS / "cv_search_blocks_v2" / "selected_tabular_oof_predictions.csv.gz"
GRU_OOF = MODELS / "gru_cv_blocks_v2" / "selected_gru_oof_predictions.csv.gz"

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


def verify_selection_lock() -> tuple[dict[str, Any], str]:
    if not LOCK_PATH.exists() or not LOCK_HASH_PATH.exists():
        raise FileNotFoundError("A completed, hashed model-selection lock is required")
    expected = LOCK_HASH_PATH.read_text(encoding="ascii").split()[0]
    actual = sha256(LOCK_PATH)
    if actual != expected:
        raise ValueError("Model-selection lock hash mismatch")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not lock.get("locked") or not lock.get("sealed_final_test_not_loaded"):
        raise ValueError("Selection lock is not valid for final-test opening")
    return lock, actual


def task_lock(lock: dict[str, Any], outcome: str, horizon: int) -> dict[str, Any]:
    matches = [
        task
        for task in lock["tasks"]
        if task["outcome"] == outcome and int(task["horizon"]) == int(horizon)
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate selection lock for {outcome} h{horizon}")
    return matches[0]


def conformal_quantile(residual: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(residual, dtype=float))
    if len(values) == 0:
        raise ValueError("Empty conformal residual set")
    rank = min(int(np.ceil((len(values) + 1) * float(coverage))), len(values))
    return float(values[rank - 1])


def calibration_table(oof: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome, horizon, family), group in oof.groupby(
        ["outcome", "horizon", "family"], sort=True
    ):
        residual = np.abs(
            np.log1p(group["observed"].to_numpy(dtype=float))
            - np.log1p(group["prediction"].to_numpy(dtype=float))
        )
        for coverage in coverages:
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": int(horizon),
                    "family": family,
                    "nominal_coverage": float(coverage),
                    "n_oof": int(len(residual)),
                    "absolute_log1p_residual_quantile": conformal_quantile(residual, coverage),
                }
            )
    return pd.DataFrame(rows)


def add_intervals(predictions: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    for coverage in sorted(calibration["nominal_coverage"].unique()):
        label = int(round(coverage * 100))
        quantile = calibration.loc[calibration["nominal_coverage"].eq(coverage), [
            "outcome",
            "horizon",
            "family",
            "absolute_log1p_residual_quantile",
        ]].rename(columns={"absolute_log1p_residual_quantile": f"q{label}"})
        output = output.merge(
            quantile,
            on=["outcome", "horizon", "family"],
            how="left",
            validate="many_to_one",
        )
        log_prediction = np.log1p(output["prediction"].to_numpy(dtype=float))
        q = output[f"q{label}"].to_numpy(dtype=float)
        output[f"lower_{label}"] = np.maximum(np.expm1(log_prediction - q), 0.0)
        output[f"upper_{label}"] = np.expm1(log_prediction + q)
        output = output.drop(columns=f"q{label}")
    return output


def fit_gru_final(
    x_dev: np.ndarray,
    x_test: np.ndarray,
    y_dev: np.ndarray,
    outcome: str,
    signals: list[str],
    params: dict[str, Any],
    epochs: int,
    seed: int,
    checkpoint_path: Path,
) -> np.ndarray:
    target_channel = signals.index(f"bmi_{outcome}_rate")
    base_dev = x_dev[:, -1, target_channel].astype(float)
    base_test = x_test[:, -1, target_channel].astype(float)
    delta_dev = np.log1p(y_dev) - base_dev
    x_mean, x_scale = x_scaler(x_dev)
    y_mean, y_scale = y_scaler(delta_dev)
    model = fit_fixed_epochs(
        apply_x_scaler(x_dev, x_mean, x_scale),
        ((delta_dev - y_mean) / y_scale).astype(np.float32),
        params,
        len(signals),
        seed,
        epochs,
    )
    standardized = predict_network(
        model,
        apply_x_scaler(x_test, x_mean, x_scale),
        int(params["batch_size"]),
    )
    prediction = np.maximum(np.expm1(base_test + standardized * y_scale + y_mean), 0.0)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "epochs": int(epochs),
            "seed": int(seed),
            "signals": signals,
            "x_mean": x_mean,
            "x_scale": x_scale,
            "target_delta_mean": y_mean,
            "target_delta_scale": y_scale,
        },
        checkpoint_path,
    )
    return prediction


def model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        [
            "partition",
            "outcome",
            "horizon",
            "family",
            "selected_by_cv",
            "reference_transparent_by_cv",
        ],
        sort=True,
    ):
        partition, outcome, horizon, family, selected_by_cv, reference_transparent_by_cv = keys
        base = {
            "partition": partition,
            "outcome": outcome,
            "horizon": int(horizon),
            "family": family,
            "selected_by_cv": bool(selected_by_cv),
            "reference_transparent_by_cv": bool(reference_transparent_by_cv),
            "n": int(len(group)),
            **regression_metrics(group["observed"], group["prediction"]),
        }
        for coverage in [90, 95]:
            covered = (
                group["observed"].ge(group[f"lower_{coverage}"])
                & group["observed"].le(group[f"upper_{coverage}"])
            )
            width = group[f"upper_{coverage}"] - group[f"lower_{coverage}"]
            base[f"coverage_{coverage}"] = float(covered.mean())
            base[f"mean_width_{coverage}"] = float(width.mean())
            base[f"median_width_{coverage}"] = float(width.median())

        country_values: list[dict[str, float]] = []
        for _, country in group.groupby("location_id", sort=False):
            country_values.append(regression_metrics(country["observed"], country["prediction"]))
        for metric in ["rmsle", "mae", "rmse", "wape", "median_absolute_error", "r2", "spearman"]:
            base[f"country_macro_{metric}"] = float(
                np.nanmean([value[metric] for value in country_values])
            )
        rows.append(base)
    result = pd.DataFrame(rows)
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby(["partition", "outcome", "horizon"], sort=True):
        group = group.copy()
        persistence = float(group.loc[group["family"].eq("persistence"), "rmsle"].iloc[0])
        trend_rows = group.loc[group["reference_transparent_by_cv"]]
        if len(trend_rows) != 1:
            raise ValueError("Expected one CV-locked transparent reference per final-test task")
        trend = float(trend_rows["rmsle"].iloc[0])
        group["cv_locked_transparent_reference_rmsle"] = trend
        group["skill_vs_persistence"] = 1.0 - group["rmsle"] / persistence
        group["skill_vs_best_transparent"] = 1.0 - group["rmsle"] / trend
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the sealed final test after CV selection lock")
    parser.add_argument("--confirm-open-sealed-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_open_sealed_test:
        raise RuntimeError("Explicit --confirm-open-sealed-test flag is required")
    OUT.mkdir(parents=True, exist_ok=True)
    FITTED.mkdir(parents=True, exist_ok=True)
    lock, lock_hash = verify_selection_lock()
    config = load_config()
    features = load_feature_names()
    signals = [str(value) for value in config["sequence_model"]["sequence_signals"]]
    needed = sorted(
        set(
            IDENTITY
            + features
            + list(config["outcomes"].values())
            + [column for outcome in config["outcomes"] for column in baseline_columns(outcome).values()]
        )
    )
    development = load_development_samples(needed)

    # This is the unique point at which the sealed target artifact is opened.
    test_opened_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    final_test = load_final_test_samples(needed)
    expected_test_hash = lock["source_hashes"]["sealed_final_test_sha256_from_pre_model_manifest"]
    actual_test_hash = sha256(DATA / "supervised_final_tests_sealed.csv.gz")
    if actual_test_hash != expected_test_hash:
        raise ValueError("Sealed final-test hash differs from the pre-model manifest")
    if set(final_test["target_year"]) != set(config["final_test_target_years"]):
        raise ValueError("Final-test target-year lock violation")

    oof = pd.concat([pd.read_csv(TAB_OOF), pd.read_csv(GRU_OOF)], ignore_index=True)
    calibration = calibration_table(oof, [float(value) for value in config["conformal"]["nominal_coverages"]])
    calibration_path = OUT / "cross_conformal_quantiles.csv"
    calibration.to_csv(calibration_path, index=False, float_format="%.12g")

    combined_samples = pd.concat(
        [development[IDENTITY], final_test[IDENTITY]], ignore_index=True
    )
    full_panel = pd.read_csv(DATA / "analytic_panel_204x34.csv.gz", low_memory=False)
    all_sequences = build_sequence_tensor(
        combined_samples,
        full_panel,
        signals,
        int(config["sequence_model"]["sequence_length"]),
    )
    n_development = len(development)
    development_sequences = all_sequences[:n_development]
    test_sequences = all_sequences[n_development:]

    records: list[pd.DataFrame] = []
    object_manifest: list[dict[str, Any]] = []
    for outcome, target_column in config["outcomes"].items():
        for horizon in [int(value) for value in config["horizons"]]:
            locked = task_lock(lock, outcome, horizon)
            dev_mask = development["horizon"].eq(horizon).to_numpy()
            test_mask = final_test["horizon"].eq(horizon).to_numpy()
            dev = development.loc[dev_mask].reset_index(drop=True)
            test = final_test.loc[test_mask].reset_index(drop=True)
            x_dev = dev[features].to_numpy(dtype=np.float32)
            x_test = test[features].to_numpy(dtype=np.float32)
            y_dev = dev[target_column].to_numpy(dtype=float)
            y_test = test[target_column].to_numpy(dtype=float)
            if dev["target_year"].max() > 2018:
                raise ValueError("Final fit contains post-2018 target")

            for family in TRANSPARENT + ADVANCED:
                model_lock = locked["all_family_configurations"][family]
                params = dict(model_lock["params"])
                config_id = str(model_lock["config_id"])
                model_path: Path | None = None
                if family in {"persistence", "logtrend5", "logtrend10"}:
                    prediction = test[baseline_columns(outcome)[family]].to_numpy(dtype=float)
                elif family == "pooled_ridge":
                    estimator = make_estimator("ridge", params, config["master_seed"], n_jobs=8)
                    prediction = fit_predict_log_model(
                        estimator,
                        "ridge",
                        pooled_baseline_matrix(dev, outcome),
                        y_dev,
                        pooled_baseline_matrix(test, outcome),
                        None,
                    )
                    model_path = FITTED / f"{outcome}_h{horizon}__{family}.joblib"
                    joblib.dump(estimator, model_path, compress=3)
                elif family == "gru":
                    model_path = FITTED / f"{outcome}_h{horizon}__gru.pt"
                    seed = stable_task_seed(config["master_seed"], outcome, horizon, "gru_final")
                    prediction = fit_gru_final(
                        development_sequences[dev_mask],
                        test_sequences[test_mask],
                        y_dev,
                        outcome,
                        signals,
                        params,
                        int(model_lock["final_epochs"]),
                        seed,
                        model_path,
                    )
                else:
                    estimator_family = family
                    if family == "xgboost":
                        params["n_estimators"] = int(model_lock["final_n_estimators"])
                    estimator = make_estimator(
                        estimator_family,
                        params,
                        stable_task_seed(config["master_seed"], outcome, horizon, family),
                        n_jobs=8,
                    )
                    prediction = fit_predict_log_model(
                        estimator,
                        estimator_family,
                        x_dev,
                        y_dev,
                        x_test,
                        None,
                    )
                    model_path = FITTED / f"{outcome}_h{horizon}__{family}.joblib"
                    joblib.dump(estimator, model_path, compress=3)

                part = test[IDENTITY].copy()
                part["outcome"] = outcome
                part["family"] = family
                part["config_id"] = config_id
                part["selected_by_cv"] = family == locked["selected_family"]
                part["reference_transparent_by_cv"] = (
                    family == locked["reference_transparent_family"]
                )
                part["observed"] = y_test
                part["prediction"] = np.maximum(np.asarray(prediction, dtype=float), 0.0)
                if not np.isfinite(part[["observed", "prediction"]].to_numpy()).all():
                    raise ValueError(f"Non-finite final prediction for {outcome} h{horizon} {family}")
                records.append(part)
                if model_path is not None:
                    object_manifest.append(
                        {
                            "outcome": outcome,
                            "horizon": horizon,
                            "family": family,
                            "path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": sha256(model_path),
                        }
                    )
            print(f"[done] final fit/prediction: {outcome}_h{horizon}", flush=True)

    predictions = pd.concat(records, ignore_index=True)
    predictions = add_intervals(predictions, calibration)
    predictions = predictions.sort_values(
        ["partition", "outcome", "horizon", "family", "location_id", "target_year"],
        kind="mergesort",
    ).reset_index(drop=True)
    prediction_path = OUT / "final_test_predictions.csv.gz"
    predictions.to_csv(
        prediction_path,
        index=False,
        float_format="%.12g",
        compression=deterministic_gzip_options(),
    )
    metrics = model_metrics(predictions)
    metrics_path = OUT / "final_test_metrics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    object_manifest_frame = pd.DataFrame(object_manifest).sort_values(
        ["outcome", "horizon", "family"], kind="mergesort"
    )
    object_manifest_path = OUT / "fitted_model_manifest.csv"
    object_manifest_frame.to_csv(object_manifest_path, index=False)

    audit = {
        "status": "final_test_evaluated_without_reselection",
        "test_opened_utc": test_opened_utc,
        "selection_lock_sha256": lock_hash,
        "sealed_test_sha256": actual_test_hash,
        "sealed_test_rows": int(len(final_test)),
        "prediction_rows": int(len(predictions)),
        "implementation_sha256": sha256(Path(__file__)),
        "model_utils_sha256": sha256(MODELS / "model_utils.py"),
        "model_family_selection_changed_after_test": False,
        "preprocessing_or_early_stopping_used_test_targets": False,
        "outputs": {
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metrics_path),
            "conformal_quantiles_sha256": sha256(calibration_path),
            "fitted_model_manifest_sha256": sha256(object_manifest_path),
        },
    }
    (OUT / "final_test_opening_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
