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

from fit_final_and_evaluate import IDENTITY, task_lock  # noqa: E402
from model_utils import (  # noqa: E402
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


OUT = ROOT / "04_evaluation"
LOCK_PATH = MODELS / "selection_lock" / "model_selection_lock.json"
FULL_PREDICTIONS = OUT / "final_test_predictions.csv.gz"
MODEL_DIR = OUT / "ablation_models"


def feature_matches(feature: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("__") and pattern in feature:
            return True
        if feature.startswith(pattern):
            return True
    return False


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = load_config()
    features = load_feature_names()
    needed = sorted(set(IDENTITY + features + list(config["outcomes"].values())))
    development = load_development_samples(needed)
    final_test = load_final_test_samples(needed)
    full = pd.read_csv(FULL_PREDICTIONS, low_memory=False)
    scenarios = config["feature_ablation_families"]
    records: list[pd.DataFrame] = []
    objects: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []

    for outcome, target_column in config["outcomes"].items():
        for horizon in [int(value) for value in config["horizons"]]:
            locked = task_lock(lock, outcome, horizon)
            family = str(locked["reference_tree_family"])
            model_lock = locked["all_family_configurations"][family]
            base_params = dict(model_lock["params"])
            if family == "xgboost":
                base_params["n_estimators"] = int(model_lock["final_n_estimators"])
            dev = development.loc[development["horizon"].eq(horizon)].reset_index(drop=True)
            test = final_test.loc[final_test["horizon"].eq(horizon)].reset_index(drop=True)
            y_dev = dev[target_column].to_numpy(dtype=float)
            y_test = test[target_column].to_numpy(dtype=float)

            full_part = full.loc[
                full["outcome"].eq(outcome)
                & full["horizon"].eq(horizon)
                & full["family"].eq(family)
            ].copy()
            full_part["ablation"] = "none_full_model"
            records.append(full_part[IDENTITY + ["outcome", "family", "observed", "prediction", "ablation"]])

            for scenario, patterns in scenarios.items():
                retained = [
                    feature for feature in features if not feature_matches(feature, list(patterns))
                ]
                removed = [feature for feature in features if feature not in retained]
                if not removed:
                    raise ValueError(f"Ablation {scenario} removed no features")
                estimator = make_estimator(
                    family,
                    base_params,
                    stable_task_seed(config["master_seed"], outcome, horizon, f"ablation_{scenario}"),
                    n_jobs=8,
                )
                prediction = fit_predict_log_model(
                    estimator,
                    family,
                    dev[retained].to_numpy(dtype=np.float32),
                    y_dev,
                    test[retained].to_numpy(dtype=np.float32),
                    None,
                )
                part = test[IDENTITY].copy()
                part["outcome"] = outcome
                part["family"] = family
                part["observed"] = y_test
                part["prediction"] = prediction
                part["ablation"] = f"remove_{scenario}"
                records.append(part)
                model_path = MODEL_DIR / f"{outcome}_h{horizon}__{family}__remove_{scenario}.joblib"
                joblib.dump(estimator, model_path, compress=3)
                objects.append(
                    {
                        "outcome": outcome,
                        "horizon": horizon,
                        "family": family,
                        "ablation": f"remove_{scenario}",
                        "path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256(model_path),
                    }
                )
                for feature in removed:
                    feature_records.append(
                        {
                            "outcome": outcome,
                            "horizon": horizon,
                            "family": family,
                            "ablation": f"remove_{scenario}",
                            "feature": feature,
                        }
                    )
            print(f"[done] tree feature ablations: {outcome}_h{horizon}", flush=True)

    predictions = pd.concat(records, ignore_index=True).sort_values(
        ["partition", "outcome", "horizon", "family", "ablation", "sample_id"],
        kind="mergesort",
    )
    prediction_path = OUT / "feature_ablation_predictions.csv.gz"
    predictions.to_csv(
        prediction_path,
        index=False,
        float_format="%.12g",
        compression=deterministic_gzip_options(),
    )
    metric_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["partition", "outcome", "horizon", "family", "ablation"], sort=True
    ):
        partition, outcome, horizon, family, ablation = keys
        metric_rows.append(
            {
                "partition": partition,
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "ablation": ablation,
                "n": int(len(group)),
                **regression_metrics(group["observed"], group["prediction"]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    full_rmsle = metrics.loc[metrics["ablation"].eq("none_full_model"), [
        "partition",
        "outcome",
        "horizon",
        "family",
        "rmsle",
    ]].rename(columns={"rmsle": "full_model_rmsle"})
    metrics = metrics.merge(
        full_rmsle,
        on=["partition", "outcome", "horizon", "family"],
        how="left",
        validate="many_to_one",
    )
    metrics["delta_rmsle_vs_full"] = metrics["rmsle"] - metrics["full_model_rmsle"]
    metric_path = OUT / "feature_ablation_metrics.csv"
    metrics.to_csv(metric_path, index=False, float_format="%.12g")
    removed_path = OUT / "feature_ablation_removed_features.csv"
    pd.DataFrame(feature_records).to_csv(removed_path, index=False)
    manifest_path = OUT / "ablation_model_manifest.csv"
    pd.DataFrame(objects).sort_values(
        ["outcome", "horizon", "ablation"], kind="mergesort"
    ).to_csv(manifest_path, index=False)

    audit = {
        "status": "complete",
        "model_family_per_task_locked_from_cv": True,
        "test_performance_not_used_to_choose_ablation_or_family": True,
        "outputs": {
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metric_path),
            "removed_features_sha256": sha256(removed_path),
            "model_manifest_sha256": sha256(manifest_path),
        },
    }
    (OUT / "feature_ablation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
