from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data"


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else 0.0


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def describe(values: pd.Series) -> dict[str, float]:
    return {
        "n": int(values.notna().sum()),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def lag1_autocorrelation(frame: pd.DataFrame, column: str) -> pd.Series:
    values = {}
    for location_id, part in frame.groupby("location_id"):
        ordered = part.sort_values("year")[column]
        values[int(location_id)] = float(ordered.autocorr(lag=1))
    return pd.Series(values, name=column)


def main() -> None:
    panel = pd.read_csv(DATA / "analytic_panel_204x34.csv.gz", low_memory=False)
    split = pd.read_csv(DATA / "country_split_locked.csv", low_memory=False)
    samples = pd.read_csv(DATA / "supervised_samples_h1_h3_h5.csv.gz", low_memory=False)

    country = panel.loc[panel["year"].eq(2023)].drop(
        columns=["sdi_quintile_2023"]
    ).merge(
        split[["location_id", "country_split", "sdi_quintile_2023"]],
        on="location_id",
        how="left",
        validate="one_to_one",
    )
    variables = [
        "bmi_daly_rate",
        "bmi_death_rate",
        "gbd_daly_fraction",
        "gbd_death_fraction",
        "overall_daly_rate",
        "overall_death_rate",
        "sdi",
    ]
    log_transform = {
        "bmi_daly_rate",
        "bmi_death_rate",
        "overall_daly_rate",
        "overall_death_rate",
    }

    balance_rows = []
    for variable in variables:
        dev_raw = country.loc[country["country_split"].eq("development"), variable]
        hold_raw = country.loc[
            country["country_split"].eq("geographic_holdout"), variable
        ]
        dev = np.log1p(dev_raw.to_numpy()) if variable in log_transform else dev_raw.to_numpy()
        hold = (
            np.log1p(hold_raw.to_numpy())
            if variable in log_transform
            else hold_raw.to_numpy()
        )
        balance_rows.append(
            {
                "variable": variable,
                "comparison_scale": "log1p" if variable in log_transform else "raw",
                "development_n": len(dev),
                "holdout_n": len(hold),
                "development_median_raw": float(dev_raw.median()),
                "holdout_median_raw": float(hold_raw.median()),
                "standardized_mean_difference_holdout_minus_development": standardized_mean_difference(
                    dev, hold
                ),
                "ks_distance": ks_distance(dev, hold),
            }
        )
    balance = pd.DataFrame(balance_rows)

    top20 = country.nlargest(20, "bmi_daly_rate")[
        ["location_id", "location_name", "bmi_daly_rate", "country_split"]
    ].copy()
    top20_holdout_n = int(top20["country_split"].eq("geographic_holdout").sum())

    per_country_sample_counts = (
        samples.groupby(["partition", "horizon", "location_id"]).size().rename("rows").reset_index()
    )
    count_ranges = [
        {
            "partition": str(partition),
            "horizon": int(horizon),
            "countries": int(part["location_id"].nunique()),
            "min_rows_per_country": int(part["rows"].min()),
            "max_rows_per_country": int(part["rows"].max()),
        }
        for (partition, horizon), part in per_country_sample_counts.groupby(
            ["partition", "horizon"]
        )
    ]

    autocorrelation = {}
    for variable in variables:
        ac = lag1_autocorrelation(panel, variable)
        autocorrelation[variable] = describe(ac)

    target_distribution = {}
    for variable in ("bmi_daly_rate", "bmi_death_rate"):
        target_distribution[variable] = {
            "all_country_year_rows": describe(panel[variable]),
            "country_2023": describe(country[variable]),
            "skewness_all_country_year_rows": float(panel[variable].skew()),
            "max_to_median_2023": float(country[variable].max() / country[variable].median()),
        }

    max_abs_smd = float(
        balance["standardized_mean_difference_holdout_minus_development"].abs().max()
    )
    max_ks = float(balance["ks_distance"].max())
    result = {
        "split_balance": {
            "maximum_absolute_smd": max_abs_smd,
            "maximum_ks_distance": max_ks,
            "interpretation": (
                "The split was stratified only by SDI, never by outcome. Moderate outcome shift is "
                "retained as a transportability stress test rather than optimized away."
            ),
            "top20_2023_bmi_daly_rate": {
                "holdout_n": top20_holdout_n,
                "development_n": 20 - top20_holdout_n,
                "expected_holdout_at_20_percent": 4,
            },
        },
        "dependence_and_weighting": {
            "lag1_autocorrelation_by_signal": autocorrelation,
            "equal_sample_count_within_partition_horizon": bool(
                all(row["min_rows_per_country"] == row["max_rows_per_country"] for row in count_ranges)
            ),
            "per_country_sample_count_ranges": count_ranges,
            "required_inference_rule": (
                "Use country-clustered resampling or country-level macro aggregation; do not treat "
                "country-year rows as independent observations."
            ),
        },
        "target_distribution": target_distribution,
        "known_structural_limitations": [
            "GBD 2023 is a retrospectively harmonized vintage; historical point estimates may use information unavailable in real time.",
            "The geographic holdout evaluates parameter transportability to countries excluded from training while their own pre-origin history remains available; it is not a no-history cold-start test.",
            "Only lower/point/upper GBD estimates are available, not posterior draws; temporal correlation of source uncertainty cannot be reconstructed.",
            "All outcomes are ecological modeled estimates and cannot support patient-level inference or histology-specific HCC claims.",
        ],
        "gate_checks": {
            "all_sdi_quintiles_in_both_splits": bool(
                country.groupby(["sdi_quintile_2023", "country_split"]).size().gt(0).all()
            ),
            "top20_holdout_count_matches_expected": top20_holdout_n == 4,
            "maximum_absolute_smd_below_0_30": max_abs_smd < 0.30,
            "maximum_ks_distance_below_0_30": max_ks < 0.30,
            "equal_country_weighting_possible": bool(
                all(row["min_rows_per_country"] == row["max_rows_per_country"] for row in count_ranges)
            ),
        },
    }

    balance.to_csv(DATA / "split_balance_metrics.csv", index=False, float_format="%.12g")
    top20.to_csv(DATA / "split_top20_burden_audit.csv", index=False, float_format="%.12g")
    (DATA / "stage2_bias_dependence_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=scalar, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=scalar, allow_nan=False))


if __name__ == "__main__":
    main()
