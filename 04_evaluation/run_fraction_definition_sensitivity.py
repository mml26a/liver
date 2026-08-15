from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "03_models"
sys.path.insert(0, str(MODELS))
sys.path.insert(0, str(ROOT / "04_evaluation"))

from fit_final_and_evaluate import IDENTITY, fit_gru_final, task_lock  # noqa: E402
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
from run_gru_cv import build_sequence_tensor  # noqa: E402


OUT = ROOT / "04_evaluation"
LOCK_PATH = MODELS / "selection_lock" / "model_selection_lock.json"
FULL_PREDICTIONS = OUT / "final_test_predictions.csv.gz"
MODEL_DIR = OUT / "fraction_sensitivity_models"


def safe_log_growth(current: float, past: float, years: int) -> float:
    eps = 1e-12
    return float((np.log(current + eps) - np.log(past + eps)) / years)


def log_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.log(values.astype(float) + 1e-12), 1)[0])


def signal_features(series: pd.Series, origin_year: int, prefix: str) -> dict[str, float]:
    current = float(series.loc[origin_year])
    output: dict[str, float] = {f"{prefix}__current": current}
    for lag in [1, 3, 5, 10]:
        past = float(series.loc[origin_year - lag])
        output[f"{prefix}__lag{lag}"] = past
        output[f"{prefix}__delta{lag}"] = current - past
        output[f"{prefix}__log_growth{lag}"] = safe_log_growth(current, past, lag)
    for window in [3, 5, 10]:
        values = series.loc[origin_year - window + 1 : origin_year].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        output[f"{prefix}__mean{window}"] = mean
        output[f"{prefix}__std{window}"] = std
        output[f"{prefix}__cv{window}"] = std / max(abs(mean), 1e-12)
    for window in [5, 10]:
        values = series.loc[origin_year - window + 1 : origin_year].to_numpy(dtype=float)
        output[f"{prefix}__log_slope{window}"] = log_slope(values)
    return output


def replace_fraction_history(
    samples: pd.DataFrame,
    panel: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    output = samples[features].copy()
    histories = {
        int(location_id): group.set_index("year").sort_index()
        for location_id, group in panel.groupby("location_id", sort=False)
    }
    rows: list[dict[str, float]] = []
    for row in samples[["location_id", "origin_year"]].itertuples(index=False):
        history = histories[int(row.location_id)]
        values: dict[str, float] = {}
        values.update(
            signal_features(
                history["asr_ratio_daly"], int(row.origin_year), "gbd_daly_fraction"
            )
        )
        values.update(
            signal_features(
                history["asr_ratio_death"], int(row.origin_year), "gbd_death_fraction"
            )
        )
        rows.append(values)
    replacement = pd.DataFrame(rows, index=output.index)
    output.loc[:, replacement.columns] = replacement
    return output


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = load_config()
    features = load_feature_names()
    excluded_without_ratio_uncertainty = [
        "gbd_daly_fraction__current_rel_ui_width",
        "gbd_death_fraction__current_rel_ui_width",
        "derived__daly_fraction_definition_gap_pp",
        "derived__death_fraction_definition_gap_pp",
    ]
    retained = [feature for feature in features if feature not in excluded_without_ratio_uncertainty]
    needed = sorted(set(IDENTITY + features + list(config["outcomes"].values())))
    development = load_development_samples(needed)
    final_test = load_final_test_samples(needed)
    full_panel = pd.read_csv(DATA / "analytic_panel_204x34.csv.gz", low_memory=False)
    combined = pd.concat([development, final_test], ignore_index=True)
    alternative_features = replace_fraction_history(combined, full_panel, features)
    alt_dev_features = alternative_features.iloc[: len(development)].reset_index(drop=True)
    alt_test_features = alternative_features.iloc[len(development) :].reset_index(drop=True)

    signals = [str(value) for value in config["sequence_model"]["sequence_signals"]]
    alternative_panel = full_panel.copy()
    alternative_panel["gbd_daly_fraction"] = alternative_panel["asr_ratio_daly"]
    alternative_panel["gbd_death_fraction"] = alternative_panel["asr_ratio_death"]
    combined_identity = combined[IDENTITY].reset_index(drop=True)
    alternative_sequences = build_sequence_tensor(
        combined_identity,
        alternative_panel,
        signals,
        int(config["sequence_model"]["sequence_length"]),
    )
    alt_dev_sequences = alternative_sequences[: len(development)]
    alt_test_sequences = alternative_sequences[len(development) :]

    full_predictions = pd.read_csv(FULL_PREDICTIONS, low_memory=False)
    records: list[pd.DataFrame] = []
    objects: list[dict[str, Any]] = []
    for outcome, target_column in config["outcomes"].items():
        for horizon in [int(value) for value in config["horizons"]]:
            locked = task_lock(lock, outcome, horizon)
            dev_mask = development["horizon"].eq(horizon).to_numpy()
            test_mask = final_test["horizon"].eq(horizon).to_numpy()
            dev = development.loc[dev_mask].reset_index(drop=True)
            test = final_test.loc[test_mask].reset_index(drop=True)
            y_dev = dev[target_column].to_numpy(dtype=float)
            y_test = test[target_column].to_numpy(dtype=float)
            tree_family = str(locked["reference_tree_family"])

            for family in [tree_family, "gru"]:
                direct = full_predictions.loc[
                    full_predictions["outcome"].eq(outcome)
                    & full_predictions["horizon"].eq(horizon)
                    & full_predictions["family"].eq(family)
                ].copy()
                direct["fraction_definition"] = "direct_GBD_Percent"
                records.append(
                    direct[
                        IDENTITY
                        + ["outcome", "family", "observed", "prediction", "fraction_definition"]
                    ]
                )

                model_lock = locked["all_family_configurations"][family]
                params = dict(model_lock["params"])
                if family == "gru":
                    model_path = MODEL_DIR / f"{outcome}_h{horizon}__gru__asr_ratio.pt"
                    prediction = fit_gru_final(
                        alt_dev_sequences[dev_mask],
                        alt_test_sequences[test_mask],
                        y_dev,
                        outcome,
                        signals,
                        params,
                        int(model_lock["final_epochs"]),
                        stable_task_seed(config["master_seed"], outcome, horizon, "gru_asr_ratio"),
                        model_path,
                    )
                else:
                    if family == "xgboost":
                        params["n_estimators"] = int(model_lock["final_n_estimators"])
                    estimator = make_estimator(
                        family,
                        params,
                        stable_task_seed(config["master_seed"], outcome, horizon, "tree_asr_ratio"),
                        n_jobs=8,
                    )
                    prediction = fit_predict_log_model(
                        estimator,
                        family,
                        alt_dev_features.loc[dev_mask, retained].to_numpy(dtype=np.float32),
                        y_dev,
                        alt_test_features.loc[test_mask, retained].to_numpy(dtype=np.float32),
                        None,
                    )
                    model_path = MODEL_DIR / f"{outcome}_h{horizon}__{family}__asr_ratio.joblib"
                    joblib.dump(estimator, model_path, compress=3)
                objects.append(
                    {
                        "outcome": outcome,
                        "horizon": horizon,
                        "family": family,
                        "fraction_definition": "ASR_ratio_sensitivity",
                        "path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256(model_path),
                    }
                )
                part = test[IDENTITY].copy()
                part["outcome"] = outcome
                part["family"] = family
                part["observed"] = y_test
                part["prediction"] = prediction
                part["fraction_definition"] = "ASR_ratio_sensitivity"
                records.append(part)
            print(f"[done] fraction-definition sensitivity: {outcome}_h{horizon}", flush=True)

    predictions = pd.concat(records, ignore_index=True).sort_values(
        ["partition", "outcome", "horizon", "family", "fraction_definition", "sample_id"],
        kind="mergesort",
    )
    prediction_path = OUT / "fraction_definition_sensitivity_predictions.csv.gz"
    predictions.to_csv(
        prediction_path,
        index=False,
        float_format="%.12g",
        compression=deterministic_gzip_options(),
    )
    metric_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["partition", "outcome", "horizon", "family", "fraction_definition"], sort=True
    ):
        partition, outcome, horizon, family, definition = keys
        metric_rows.append(
            {
                "partition": partition,
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "fraction_definition": definition,
                "n": int(len(group)),
                **regression_metrics(group["observed"], group["prediction"]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    direct = metrics.loc[metrics["fraction_definition"].eq("direct_GBD_Percent"), [
        "partition",
        "outcome",
        "horizon",
        "family",
        "rmsle",
    ]].rename(columns={"rmsle": "direct_percent_rmsle"})
    metrics = metrics.merge(
        direct,
        on=["partition", "outcome", "horizon", "family"],
        how="left",
        validate="many_to_one",
    )
    metrics["delta_rmsle_vs_direct_percent"] = metrics["rmsle"] - metrics[
        "direct_percent_rmsle"
    ]
    metrics_path = OUT / "fraction_definition_sensitivity_metrics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    manifest_path = OUT / "fraction_sensitivity_model_manifest.csv"
    pd.DataFrame(objects).sort_values(
        ["outcome", "horizon", "family"], kind="mergesort"
    ).to_csv(manifest_path, index=False)
    audit = {
        "status": "complete",
        "primary_fraction_definition": "direct GBD Percent",
        "sensitivity_fraction_definition": "BMI-attributable ASR divided by overall liver-cancer ASR",
        "features_excluded_because_ratio_specific_uncertainty_is_unavailable": excluded_without_ratio_uncertainty,
        "hyperparameters_reused_without_test_tuning": True,
        "outputs": {
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metrics_path),
            "model_manifest_sha256": sha256(manifest_path),
        },
    }
    (OUT / "fraction_definition_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
