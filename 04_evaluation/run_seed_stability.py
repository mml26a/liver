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
)
from run_gru_cv import build_sequence_tensor  # noqa: E402


OUT = ROOT / "04_evaluation"
LOCK_PATH = MODELS / "selection_lock" / "model_selection_lock.json"
STABILITY_MODELS = OUT / "stability_models"


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("Model selection lock is required")
    STABILITY_MODELS.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = load_config()
    seeds = [int(value) for value in config["stability_seeds"]]
    features = load_feature_names()
    signals = [str(value) for value in config["sequence_model"]["sequence_signals"]]
    needed = sorted(set(IDENTITY + features + list(config["outcomes"].values())))
    development = load_development_samples(needed)
    final_test = load_final_test_samples(needed)

    combined = pd.concat([development[IDENTITY], final_test[IDENTITY]], ignore_index=True)
    panel = pd.read_csv(DATA / "analytic_panel_204x34.csv.gz", low_memory=False)
    sequences = build_sequence_tensor(
        combined,
        panel,
        signals,
        int(config["sequence_model"]["sequence_length"]),
    )
    n_dev = len(development)
    dev_sequences = sequences[:n_dev]
    test_sequences = sequences[n_dev:]

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
            x_dev = dev[features].to_numpy(dtype=np.float32)
            x_test = test[features].to_numpy(dtype=np.float32)

            tree_family = str(locked["reference_tree_family"])
            tree_lock = locked["all_family_configurations"][tree_family]
            tree_params = dict(tree_lock["params"])
            if tree_family == "xgboost":
                tree_params["n_estimators"] = int(tree_lock["final_n_estimators"])
            gru_lock = locked["all_family_configurations"]["gru"]
            gru_params = dict(gru_lock["params"])

            for seed in seeds:
                tree_model = make_estimator(tree_family, tree_params, seed, n_jobs=8)
                tree_prediction = fit_predict_log_model(
                    tree_model,
                    tree_family,
                    x_dev,
                    y_dev,
                    x_test,
                    None,
                )
                tree_path = (
                    STABILITY_MODELS
                    / f"{outcome}_h{horizon}__{tree_family}__seed{seed}.joblib"
                )
                joblib.dump(tree_model, tree_path, compress=3)
                objects.append(
                    {
                        "outcome": outcome,
                        "horizon": horizon,
                        "family": tree_family,
                        "seed": seed,
                        "path": str(tree_path.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256(tree_path),
                    }
                )

                gru_path = STABILITY_MODELS / f"{outcome}_h{horizon}__gru__seed{seed}.pt"
                gru_prediction = fit_gru_final(
                    dev_sequences[dev_mask],
                    test_sequences[test_mask],
                    y_dev,
                    outcome,
                    signals,
                    gru_params,
                    int(gru_lock["final_epochs"]),
                    seed,
                    gru_path,
                )
                objects.append(
                    {
                        "outcome": outcome,
                        "horizon": horizon,
                        "family": "gru",
                        "seed": seed,
                        "path": str(gru_path.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256(gru_path),
                    }
                )

                for family, prediction in [
                    (tree_family, tree_prediction),
                    ("gru", gru_prediction),
                ]:
                    part = test[IDENTITY].copy()
                    part["outcome"] = outcome
                    part["family"] = family
                    part["seed"] = seed
                    part["observed"] = y_test
                    part["prediction"] = prediction
                    records.append(part)
            print(f"[done] five-seed stability: {outcome}_h{horizon}", flush=True)

    predictions = pd.concat(records, ignore_index=True).sort_values(
        ["partition", "outcome", "horizon", "family", "seed", "sample_id"],
        kind="mergesort",
    )
    prediction_path = OUT / "five_seed_stability_predictions.csv.gz"
    predictions.to_csv(
        prediction_path,
        index=False,
        float_format="%.12g",
        compression=deterministic_gzip_options(),
    )

    metric_rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["partition", "outcome", "horizon", "family", "seed"], sort=True
    ):
        partition, outcome, horizon, family, seed = keys
        metric_rows.append(
            {
                "partition": partition,
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "seed": int(seed),
                "n": int(len(group)),
                **regression_metrics(group["observed"], group["prediction"]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics_path = OUT / "five_seed_stability_metrics.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    stability = (
        metrics.groupby(["partition", "outcome", "horizon", "family"], as_index=False)
        .agg(
            mean_rmsle=("rmsle", "mean"),
            sd_rmsle=("rmsle", "std"),
            min_rmsle=("rmsle", "min"),
            max_rmsle=("rmsle", "max"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
        )
    )
    sample_spread = (
        predictions.groupby(
            ["partition", "outcome", "horizon", "family", "sample_id"]
        )["prediction"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    sample_spread["prediction_seed_cv"] = sample_spread["std"] / sample_spread["mean"].clip(
        lower=1e-12
    )
    spread_summary = (
        sample_spread.groupby(["partition", "outcome", "horizon", "family"], as_index=False)[
            "prediction_seed_cv"
        ]
        .agg(median_prediction_seed_cv="median", p95_prediction_seed_cv=lambda x: x.quantile(0.95))
    )
    stability = stability.merge(
        spread_summary,
        on=["partition", "outcome", "horizon", "family"],
        how="left",
        validate="one_to_one",
    )
    stability_path = OUT / "five_seed_stability_summary.csv"
    stability.to_csv(stability_path, index=False, float_format="%.12g")

    manifest_path = OUT / "stability_model_manifest.csv"
    pd.DataFrame(objects).sort_values(
        ["outcome", "horizon", "family", "seed"], kind="mergesort"
    ).to_csv(manifest_path, index=False)
    audit = {
        "status": "complete",
        "seeds": seeds,
        "prediction_rows": int(len(predictions)),
        "fitted_objects": int(len(objects)),
        "outputs": {
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metrics_path),
            "summary_sha256": sha256(stability_path),
            "model_manifest_sha256": sha256(manifest_path),
        },
    }
    (OUT / "five_seed_stability_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
