from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "03_models"
sys.path.insert(0, str(MODELS))

from model_utils import DATA, load_development_samples, load_feature_names, regression_metrics, sha256  # noqa: E402


OUT = ROOT / "04_evaluation"
PREDICTIONS = OUT / "final_test_predictions.csv.gz"
LOCK = MODELS / "selection_lock" / "model_selection_lock.json"


def interval_metrics(group: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for level in [90, 95]:
        covered = group["observed"].between(group[f"lower_{level}"], group[f"upper_{level}"])
        width = group[f"upper_{level}"] - group[f"lower_{level}"]
        output[f"coverage_{level}"] = float(covered.mean())
        output[f"mean_width_{level}"] = float(width.mean())
        output[f"median_width_{level}"] = float(width.median())
    return output


def subgroup_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    sample_burden = frame[
        ["partition", "outcome", "horizon", "sample_id", "observed"]
    ].drop_duplicates()
    sample_burden["observed_burden_quartile"] = (
        sample_burden.groupby(["partition", "outcome", "horizon"])["observed"]
        .rank(method="first", pct=True)
        .map(lambda value: f"Q{min(4, int(np.ceil(value * 4)))}")
    )
    frame = frame.merge(
        sample_burden.drop(columns="observed"),
        on=["partition", "outcome", "horizon", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    frame["top_1pct_excluded"] = frame.groupby(
        ["partition", "outcome", "horizon"]
    )["observed"].transform(lambda values: values <= values.quantile(0.99))
    definitions = {
        "overall": pd.Series("All", index=frame.index),
        "sdi_quintile": frame["sdi_quintile_2023_assignment_only"],
        "target_year": frame["target_year"].astype(str),
        "observed_burden_quartile": frame["observed_burden_quartile"],
        "top_1pct_sensitivity": frame["top_1pct_excluded"].map(
            {True: "exclude_top_1pct", False: "top_1pct_only"}
        ),
    }
    rows: list[dict[str, Any]] = []
    for dimension, values in definitions.items():
        local = frame.assign(subgroup_value=values)
        for keys, group in local.groupby(
            ["partition", "outcome", "horizon", "family", "subgroup_value"], sort=True
        ):
            partition, outcome, horizon, family, subgroup_value = keys
            rows.append(
                {
                    "partition": partition,
                    "outcome": outcome,
                    "horizon": int(horizon),
                    "family": family,
                    "selected_by_cv": bool(group["selected_by_cv"].iloc[0]),
                    "reference_transparent_by_cv": bool(
                        group["reference_transparent_by_cv"].iloc[0]
                    ),
                    "subgroup_dimension": dimension,
                    "subgroup_value": subgroup_value,
                    "n": int(len(group)),
                    **regression_metrics(group["observed"], group["prediction"]),
                    **interval_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap(predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tasks = predictions[["partition", "outcome", "horizon"]].drop_duplicates()
    for task_index, task in tasks.reset_index(drop=True).iterrows():
        local = predictions.loc[
            predictions["partition"].eq(task["partition"])
            & predictions["outcome"].eq(task["outcome"])
            & predictions["horizon"].eq(task["horizon"])
        ]
        selected = local.loc[local["selected_by_cv"]].copy()
        reference = local.loc[local["reference_transparent_by_cv"]].copy()
        paired = selected.merge(
            reference[["sample_id", "prediction"]],
            on="sample_id",
            how="inner",
            suffixes=("_selected", "_reference"),
            validate="one_to_one",
        )
        if len(paired) != len(selected):
            raise ValueError("Selected/reference predictions are not paired")
        log_observed = np.log1p(paired["observed"].to_numpy(dtype=float))
        paired["sq_selected"] = (
            np.log1p(paired["prediction_selected"].to_numpy(dtype=float)) - log_observed
        ) ** 2
        paired["sq_reference"] = (
            np.log1p(paired["prediction_reference"].to_numpy(dtype=float)) - log_observed
        ) ** 2
        aggregate = paired.groupby("location_id", sort=True).agg(
            n=("sample_id", "size"),
            sum_sq_selected=("sq_selected", "sum"),
            sum_sq_reference=("sq_reference", "sum"),
            mean_sq_selected=("sq_selected", "mean"),
            mean_sq_reference=("sq_reference", "mean"),
        )
        array = aggregate.to_numpy(dtype=float)
        rng = np.random.default_rng(seed + int(task_index))
        boot_selected = np.empty(replicates, dtype=float)
        boot_reference = np.empty(replicates, dtype=float)
        boot_difference = np.empty(replicates, dtype=float)
        boot_skill = np.empty(replicates, dtype=float)
        boot_macro_difference = np.empty(replicates, dtype=float)
        n_countries = len(array)
        for replicate in range(replicates):
            chosen = rng.integers(0, n_countries, size=n_countries)
            sampled = array[chosen]
            selected_rmsle = float(np.sqrt(sampled[:, 1].sum() / sampled[:, 0].sum()))
            reference_rmsle = float(np.sqrt(sampled[:, 2].sum() / sampled[:, 0].sum()))
            boot_selected[replicate] = selected_rmsle
            boot_reference[replicate] = reference_rmsle
            boot_difference[replicate] = selected_rmsle - reference_rmsle
            boot_skill[replicate] = 1.0 - selected_rmsle / reference_rmsle
            boot_macro_difference[replicate] = float(
                np.mean(np.sqrt(sampled[:, 3]) - np.sqrt(sampled[:, 4]))
            )

        observed_selected = float(np.sqrt(aggregate["sum_sq_selected"].sum() / aggregate["n"].sum()))
        observed_reference = float(
            np.sqrt(aggregate["sum_sq_reference"].sum() / aggregate["n"].sum())
        )
        values = {
            "selected_rmsle": (boot_selected, observed_selected),
            "reference_rmsle": (boot_reference, observed_reference),
            "selected_minus_reference_rmsle": (
                boot_difference,
                observed_selected - observed_reference,
            ),
            "skill_vs_reference": (
                boot_skill,
                1.0 - observed_selected / observed_reference,
            ),
            "country_macro_selected_minus_reference_rmsle": (
                boot_macro_difference,
                float(
                    np.mean(
                        np.sqrt(aggregate["mean_sq_selected"])
                        - np.sqrt(aggregate["mean_sq_reference"])
                    )
                ),
            ),
        }
        for estimand, (distribution, observed) in values.items():
            rows.append(
                {
                    "partition": task["partition"],
                    "outcome": task["outcome"],
                    "horizon": int(task["horizon"]),
                    "selected_family": selected["family"].iloc[0],
                    "reference_family": reference["family"].iloc[0],
                    "estimand": estimand,
                    "estimate": observed,
                    "ci_lower": float(np.quantile(distribution, 0.025)),
                    "ci_upper": float(np.quantile(distribution, 0.975)),
                    "bootstrap_probability_below_zero": float(np.mean(distribution < 0.0)),
                    "n_countries": int(n_countries),
                    "replicates": int(replicates),
                }
            )
    return pd.DataFrame(rows)


def residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["partition", "outcome", "horizon", "family"], sort=True
    ):
        partition, outcome, horizon, family = keys
        observed = np.log1p(group["observed"].to_numpy(dtype=float))
        predicted = np.log1p(group["prediction"].to_numpy(dtype=float))
        slope, intercept = np.polyfit(predicted, observed, deg=1)
        residual = predicted - observed
        rows.append(
            {
                "partition": partition,
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "selected_by_cv": bool(group["selected_by_cv"].iloc[0]),
                "reference_transparent_by_cv": bool(
                    group["reference_transparent_by_cv"].iloc[0]
                ),
                "n": int(len(group)),
                "mean_log1p_bias_prediction_minus_observed": float(residual.mean()),
                "median_log1p_bias_prediction_minus_observed": float(np.median(residual)),
                "calibration_intercept_observed_on_predicted": float(intercept),
                "calibration_slope_observed_on_predicted": float(slope),
            }
        )
    return pd.DataFrame(rows)


def distribution_shift() -> pd.DataFrame:
    features = load_feature_names()
    required = ["partition", *features]
    development = load_development_samples(required)
    final_test = pd.read_csv(DATA / "supervised_final_tests_sealed.csv.gz", usecols=required)
    rows: list[dict[str, Any]] = []
    for partition, test in final_test.groupby("partition", sort=True):
        for feature in features:
            dev_values = development[feature].to_numpy(dtype=float)
            test_values = test[feature].to_numpy(dtype=float)
            pooled_sd = float(
                np.sqrt((np.var(dev_values, ddof=1) + np.var(test_values, ddof=1)) / 2.0)
            )
            rows.append(
                {
                    "partition": partition,
                    "feature": feature,
                    "development_mean": float(np.mean(dev_values)),
                    "test_mean": float(np.mean(test_values)),
                    "standardized_mean_difference": (
                        float((np.mean(test_values) - np.mean(dev_values)) / pooled_sd)
                        if pooled_sd > 0
                        else 0.0
                    ),
                    "ks_statistic": float(ks_2samp(dev_values, test_values).statistic),
                    "wasserstein_distance": float(wasserstein_distance(dev_values, test_values)),
                }
            )
    result = pd.DataFrame(rows)
    result["absolute_smd"] = result["standardized_mean_difference"].abs()
    return result.sort_values(
        ["partition", "absolute_smd"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    if not PREDICTIONS.exists() or not LOCK.exists():
        raise FileNotFoundError("Final predictions and selection lock are required")
    predictions = pd.read_csv(PREDICTIONS, low_memory=False)
    if predictions.duplicated(["partition", "outcome", "horizon", "family", "sample_id"]).any():
        raise ValueError("Duplicate final prediction key")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bootstrap_spec = json.loads(
        (ROOT / "02_protocol" / "model_config_locked.json").read_text(
            encoding="utf-8"
        )
    )["bootstrap"]

    subgroup = subgroup_table(predictions)
    subgroup_path = OUT / "subgroup_performance_and_coverage.csv"
    subgroup.to_csv(subgroup_path, index=False, float_format="%.12g")
    bootstrap = cluster_bootstrap(
        predictions,
        int(bootstrap_spec["replicates"]),
        int(bootstrap_spec["seed"]),
    )
    bootstrap_path = OUT / "country_cluster_bootstrap.csv"
    bootstrap.to_csv(bootstrap_path, index=False, float_format="%.12g")
    residual = residual_diagnostics(predictions)
    residual_path = OUT / "residual_calibration_diagnostics.csv"
    residual.to_csv(residual_path, index=False, float_format="%.12g")
    shift = distribution_shift()
    shift_path = OUT / "distribution_shift_diagnostics.csv"
    shift.to_csv(shift_path, index=False, float_format="%.12g")

    audit = {
        "status": "complete",
        "selection_lock_sha256": sha256(LOCK),
        "predictions_sha256": sha256(PREDICTIONS),
        "bootstrap_cluster": "location_id",
        "bootstrap_replicates": int(bootstrap_spec["replicates"]),
        "outputs": {
            "subgroup_sha256": sha256(subgroup_path),
            "bootstrap_sha256": sha256(bootstrap_path),
            "residual_diagnostics_sha256": sha256(residual_path),
            "distribution_shift_sha256": sha256(shift_path),
        },
    }
    (OUT / "robustness_analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
