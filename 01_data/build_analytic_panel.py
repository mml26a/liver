from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "01_data"
PROTOCOL = ROOT / "02_protocol"

SEED = 20260810
YEARS = list(range(1990, 2024))
HORIZONS = (1, 3, 5)
FIRST_ORIGIN_YEAR = 2000
FINAL_TEST_YEARS = set(range(2019, 2024))

MEASURE_MAP = {
    "DALYs (Disability-Adjusted Life Years)": "daly",
    "Deaths": "death",
}

CORE_SIGNALS = (
    "bmi_daly_rate",
    "bmi_death_rate",
    "gbd_daly_fraction",
    "gbd_death_fraction",
    "overall_daly_rate",
    "overall_death_rate",
    "sdi",
)

INTERVAL_SIGNALS = (
    "bmi_daly_rate",
    "bmi_death_rate",
    "gbd_daly_fraction",
    "gbd_death_fraction",
    "overall_daly_rate",
    "overall_death_rate",
    "sdi",
)


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_unique(frame: pd.DataFrame, key: Iterable[str], label: str) -> None:
    key = list(key)
    duplicates = frame.duplicated(key, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, key].head(10).to_dict("records")
        raise ValueError(f"{label} is not unique on {key}; examples={example}")


def extract_series(
    frame: pd.DataFrame,
    measure: str,
    metric: str,
    prefix: str,
) -> pd.DataFrame:
    selected = frame.loc[
        frame["measure_name"].eq(measure) & frame["metric_name"].eq(metric),
        ["location_id", "location_name", "year", "val", "lower", "upper"],
    ].copy()
    assert_unique(selected, ["location_id", "year"], prefix)
    return selected.rename(
        columns={
            "val": prefix,
            "lower": f"{prefix}_lower",
            "upper": f"{prefix}_upper",
        }
    )


def merge_country_series(parts: list[pd.DataFrame]) -> pd.DataFrame:
    first = parts[0]
    panel = first.copy()
    for part in parts[1:]:
        canonical = part.drop(columns="location_name")
        panel = panel.merge(
            canonical,
            on=["location_id", "year"],
            how="outer",
            validate="one_to_one",
        )
    return panel


def build_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    bmi_path = RAW / "gbd2023_BMI_HCC_country_1990_2023.csv"
    overall_path = RAW / "gbd2023_allHCC_country_1990_2023.csv"
    sdi_path = RAW / "gbd2023_SDI_values_1950_2023.csv"

    bmi = pd.read_csv(bmi_path, low_memory=False)
    overall = pd.read_csv(overall_path, low_memory=False)
    sdi_raw = pd.read_csv(sdi_path, low_memory=False)

    expected_common = {
        "sex_name": {"Both"},
        "age_name": {"Age-standardized"},
        "cause_name": {"Liver cancer"},
    }
    for label, frame in (("BMI", bmi), ("overall", overall)):
        for column, expected in expected_common.items():
            observed = set(frame[column].dropna().astype(str).unique())
            if observed != expected:
                raise ValueError(f"{label} {column}: expected {expected}, observed {observed}")

    if set(bmi["rei_name"].dropna().astype(str).unique()) != {"High body-mass index"}:
        raise ValueError("Unexpected BMI risk definition")

    parts = [
        extract_series(
            bmi,
            "DALYs (Disability-Adjusted Life Years)",
            "Rate",
            "bmi_daly_rate",
        ),
        extract_series(bmi, "Deaths", "Rate", "bmi_death_rate"),
        extract_series(
            bmi,
            "DALYs (Disability-Adjusted Life Years)",
            "Percent",
            "gbd_daly_fraction",
        ),
        extract_series(bmi, "Deaths", "Percent", "gbd_death_fraction"),
        extract_series(
            overall,
            "DALYs (Disability-Adjusted Life Years)",
            "Rate",
            "overall_daly_rate",
        ),
        extract_series(overall, "Deaths", "Rate", "overall_death_rate"),
    ]
    panel = merge_country_series(parts)

    country_ids = set(panel["location_id"].astype(int).unique())
    sdi = sdi_raw.loc[
        sdi_raw["location_id"].isin(country_ids)
        & sdi_raw["year_id"].between(1990, 2023),
        [
            "location_id",
            "location_name",
            "year_id",
            "mean_value",
            "lower_value",
            "upper_value",
        ],
    ].copy()
    sdi = sdi.rename(
        columns={
            "year_id": "year",
            "mean_value": "sdi",
            "lower_value": "sdi_lower",
            "upper_value": "sdi_upper",
        }
    )
    assert_unique(sdi, ["location_id", "year"], "country SDI")
    panel = panel.merge(
        sdi.drop(columns="location_name"),
        on=["location_id", "year"],
        how="left",
        validate="one_to_one",
    )

    panel = panel.sort_values(["location_id", "year"]).reset_index(drop=True)
    expected_rows = 204 * len(YEARS)
    if len(panel) != expected_rows:
        raise ValueError(f"Expected {expected_rows} panel rows, observed {len(panel)}")
    if panel["location_id"].nunique() != 204:
        raise ValueError("Panel does not contain exactly 204 locations")
    if set(panel["year"].astype(int).unique()) != set(YEARS):
        raise ValueError("Panel year coverage is not exactly 1990-2023")
    if panel.isna().any().any():
        missing = panel.isna().sum()
        raise ValueError(f"Unexpected panel missingness: {missing[missing > 0].to_dict()}")
    assert_unique(panel, ["location_id", "year"], "analytic panel")

    panel["asr_ratio_daly"] = panel["bmi_daly_rate"] / panel["overall_daly_rate"]
    panel["asr_ratio_death"] = panel["bmi_death_rate"] / panel["overall_death_rate"]
    panel["daly_fraction_definition_gap_pp"] = 100 * (
        panel["gbd_daly_fraction"] - panel["asr_ratio_daly"]
    )
    panel["death_fraction_definition_gap_pp"] = 100 * (
        panel["gbd_death_fraction"] - panel["asr_ratio_death"]
    )

    for signal in INTERVAL_SIGNALS:
        panel[f"{signal}_rel_ui_width"] = (
            panel[f"{signal}_upper"] - panel[f"{signal}_lower"]
        ) / panel[signal].clip(lower=1e-12)

    if (panel[[f"{name}_lower" for name in INTERVAL_SIGNALS]].to_numpy()
        > panel[list(INTERVAL_SIGNALS)].to_numpy()).any():
        raise ValueError("At least one lower uncertainty bound exceeds its point estimate")
    if (panel[list(INTERVAL_SIGNALS)].to_numpy()
        > panel[[f"{name}_upper" for name in INTERVAL_SIGNALS]].to_numpy()).any():
        raise ValueError("At least one point estimate exceeds its upper uncertainty bound")
    if (panel[["gbd_daly_fraction", "gbd_death_fraction"]].to_numpy() < 0).any() or (
        panel[["gbd_daly_fraction", "gbd_death_fraction"]].to_numpy() > 1
    ).any():
        raise ValueError("GBD Percent metric outside [0,1]")
    if (panel["bmi_daly_rate"] > panel["overall_daly_rate"]).any() or (
        panel["bmi_death_rate"] > panel["overall_death_rate"]
    ).any():
        raise ValueError("BMI-attributable rate exceeds overall liver-cancer rate")

    country_2023 = panel.loc[
        panel["year"].eq(2023), ["location_id", "location_name", "sdi"]
    ].copy()
    country_2023["sdi_quintile_2023"] = pd.qcut(
        country_2023["sdi"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    ).astype(str)
    quintile_map = country_2023.set_index("location_id")["sdi_quintile_2023"]
    panel["sdi_quintile_2023"] = panel["location_id"].map(quintile_map)

    audit = {
        "panel_rows": len(panel),
        "locations": panel["location_id"].nunique(),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "missing_cells": int(panel.isna().sum().sum()),
        "duplicate_location_year_rows": int(panel.duplicated(["location_id", "year"]).sum()),
        "bmi_rate_exceeds_overall_rows": int(
            (
                (panel["bmi_daly_rate"] > panel["overall_daly_rate"])
                | (panel["bmi_death_rate"] > panel["overall_death_rate"])
            ).sum()
        ),
        "fraction_definition_gap_pp": {
            "daly_median": float(panel["daly_fraction_definition_gap_pp"].median()),
            "daly_max_abs": float(panel["daly_fraction_definition_gap_pp"].abs().max()),
            "death_median": float(panel["death_fraction_definition_gap_pp"].median()),
            "death_max_abs": float(panel["death_fraction_definition_gap_pp"].abs().max()),
        },
        "sdi_quintile_counts": {
            str(key): int(value)
            for key, value in country_2023["sdi_quintile_2023"].value_counts().sort_index().items()
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (bmi_path, overall_path, sdi_path)
        },
    }
    return panel, audit


def allocate_holdout_quotas(counts: pd.Series, total_holdout: int) -> dict[str, int]:
    expected = counts * total_holdout / counts.sum()
    quotas = np.floor(expected).astype(int)
    remaining = total_holdout - int(quotas.sum())
    order = (expected - quotas).sort_values(ascending=False, kind="mergesort").index.tolist()
    for group in order[:remaining]:
        quotas.loc[group] += 1
    return {str(key): int(value) for key, value in quotas.items()}


def make_country_split(panel: pd.DataFrame) -> pd.DataFrame:
    country = panel.loc[
        panel["year"].eq(2023),
        ["location_id", "location_name", "sdi", "sdi_quintile_2023"],
    ].copy()
    country = country.sort_values("location_id").reset_index(drop=True)
    counts = country["sdi_quintile_2023"].value_counts().sort_index()
    total_holdout = round(0.20 * len(country))
    quotas = allocate_holdout_quotas(counts, total_holdout)

    rng = np.random.default_rng(SEED)
    holdout_ids: set[int] = set()
    for quintile in sorted(quotas):
        ids = country.loc[
            country["sdi_quintile_2023"].eq(quintile), "location_id"
        ].astype(int).to_numpy(copy=True)
        rng.shuffle(ids)
        holdout_ids.update(ids[: quotas[quintile]].tolist())

    country["country_split"] = np.where(
        country["location_id"].isin(holdout_ids),
        "geographic_holdout",
        "development",
    )
    country["split_seed"] = SEED
    country["split_stratifier"] = "2023 SDI quintile (assignment only; never a feature)"
    if country["country_split"].eq("geographic_holdout").sum() != total_holdout:
        raise ValueError("Incorrect geographic holdout size")
    return country


def safe_log_growth(current: float, past: float, years: int) -> float:
    eps = 1e-12
    return float((np.log(current + eps) - np.log(past + eps)) / years)


def log_slope(values: np.ndarray) -> float:
    eps = 1e-12
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.log(values.astype(float) + eps), 1)[0])


def make_features_for_origin(history: pd.DataFrame, origin_year: int) -> dict[str, float]:
    by_year = history.set_index("year")
    features: dict[str, float] = {}
    lags = (1, 3, 5, 10)
    windows = (3, 5, 10)

    for signal in CORE_SIGNALS:
        current = float(by_year.loc[origin_year, signal])
        features[f"{signal}__current"] = current
        for lag in lags:
            past = float(by_year.loc[origin_year - lag, signal])
            features[f"{signal}__lag{lag}"] = past
            features[f"{signal}__delta{lag}"] = current - past
            features[f"{signal}__log_growth{lag}"] = safe_log_growth(current, past, lag)

        for window in windows:
            values = by_year.loc[origin_year - window + 1 : origin_year, signal].to_numpy(dtype=float)
            if len(values) != window:
                raise ValueError(f"Incomplete {window}-year window for {signal} at {origin_year}")
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            features[f"{signal}__mean{window}"] = mean
            features[f"{signal}__std{window}"] = std
            features[f"{signal}__cv{window}"] = std / max(abs(mean), 1e-12)
        for window in (5, 10):
            values = by_year.loc[origin_year - window + 1 : origin_year, signal].to_numpy(dtype=float)
            features[f"{signal}__log_slope{window}"] = log_slope(values)

    for signal in INTERVAL_SIGNALS:
        features[f"{signal}__current_rel_ui_width"] = float(
            by_year.loc[origin_year, f"{signal}_rel_ui_width"]
        )

    features["derived__current_daly_rate_gap"] = float(
        by_year.loc[origin_year, "overall_daly_rate"]
        - by_year.loc[origin_year, "bmi_daly_rate"]
    )
    features["derived__current_death_rate_gap"] = float(
        by_year.loc[origin_year, "overall_death_rate"]
        - by_year.loc[origin_year, "bmi_death_rate"]
    )
    features["derived__daly_fraction_definition_gap_pp"] = float(
        by_year.loc[origin_year, "daly_fraction_definition_gap_pp"]
    )
    features["derived__death_fraction_definition_gap_pp"] = float(
        by_year.loc[origin_year, "death_fraction_definition_gap_pp"]
    )
    return features


def assign_partition(country_split: str, target_year: int) -> str:
    if country_split == "geographic_holdout":
        if target_year in FINAL_TEST_YEARS:
            return "test_spatiotemporal_unseen_country"
        return "reserved_unseen_country_history"
    if target_year in FINAL_TEST_YEARS:
        return "test_temporal_seen_country"
    return "development"


def assign_validation_block(target_year: int, partition: str) -> str:
    if partition != "development":
        return "not_development"
    if target_year <= 2012:
        return "pre_cv_training_pool"
    if target_year in (2013, 2014):
        return "cv1_validation"
    if target_year in (2015, 2016):
        return "cv2_validation"
    if target_year in (2017, 2018):
        return "cv3_validation"
    raise ValueError(f"Unexpected development target year: {target_year}")


def build_supervised_samples(
    panel: pd.DataFrame, country_split: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    split_map = country_split.set_index("location_id")["country_split"].to_dict()
    quintile_map = country_split.set_index("location_id")["sdi_quintile_2023"].to_dict()

    rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for location_id, country_history in panel.groupby("location_id", sort=True):
        country_history = country_history.sort_values("year").reset_index(drop=True)
        location_name = str(country_history["location_name"].iloc[0])
        by_year = country_history.set_index("year")
        split = str(split_map[int(location_id)])

        for horizon in HORIZONS:
            last_origin = 2023 - horizon
            for origin_year in range(FIRST_ORIGIN_YEAR, last_origin + 1):
                target_year = origin_year + horizon
                features = make_features_for_origin(country_history, origin_year)
                if feature_names is None:
                    feature_names = list(features)
                elif list(features) != feature_names:
                    raise ValueError("Feature order changed between samples")

                partition = assign_partition(split, target_year)
                target = by_year.loc[target_year]
                current = by_year.loc[origin_year]
                row: dict[str, Any] = {
                    "sample_id": f"loc{int(location_id)}_o{origin_year}_h{horizon}",
                    "location_id": int(location_id),
                    "location_name": location_name,
                    "origin_year": int(origin_year),
                    "target_year": int(target_year),
                    "horizon": int(horizon),
                    "country_split": split,
                    "sdi_quintile_2023_assignment_only": str(quintile_map[int(location_id)]),
                    "partition": partition,
                    "validation_block": assign_validation_block(target_year, partition),
                    "target_bmi_daly_rate": float(target["bmi_daly_rate"]),
                    "target_bmi_daly_rate_lower": float(target["bmi_daly_rate_lower"]),
                    "target_bmi_daly_rate_upper": float(target["bmi_daly_rate_upper"]),
                    "target_bmi_death_rate": float(target["bmi_death_rate"]),
                    "target_bmi_death_rate_lower": float(target["bmi_death_rate_lower"]),
                    "target_bmi_death_rate_upper": float(target["bmi_death_rate_upper"]),
                    "baseline_persistence_daly": float(current["bmi_daly_rate"]),
                    "baseline_persistence_death": float(current["bmi_death_rate"]),
                }
                row.update(features)
                for outcome, signal in (
                    ("daly", "bmi_daly_rate"),
                    ("death", "bmi_death_rate"),
                ):
                    for window in (5, 10):
                        slope = features[f"{signal}__log_slope{window}"]
                        prediction = float(current[signal] * np.exp(horizon * slope))
                        row[f"baseline_logtrend{window}_{outcome}"] = max(prediction, 0.0)
                rows.append(row)

    if feature_names is None:
        raise ValueError("No supervised samples were created")
    samples = pd.DataFrame(rows)
    assert_unique(samples, ["sample_id"], "supervised samples")
    if samples[feature_names].isna().any().any():
        missing = samples[feature_names].isna().sum()
        raise ValueError(f"Missing engineered features: {missing[missing > 0].to_dict()}")
    if any(name.startswith("target_") for name in feature_names):
        raise ValueError("Target leakage: target column appears in feature list")
    forbidden_features = {
        "location_id",
        "location_name",
        "country_split",
        "sdi_quintile_2023_assignment_only",
        "partition",
        "validation_block",
    }
    if forbidden_features.intersection(feature_names):
        raise ValueError("Identity/split metadata leaked into the feature list")

    development = samples.loc[samples["partition"].eq("development")]
    seen_test = samples.loc[samples["partition"].eq("test_temporal_seen_country")]
    unseen_test = samples.loc[
        samples["partition"].eq("test_spatiotemporal_unseen_country")
    ]
    if development["country_split"].ne("development").any():
        raise ValueError("Held-out country appears in development rows")
    if development["target_year"].max() > 2018:
        raise ValueError("Final temporal test years appear in development rows")
    if seen_test["target_year"].min() < 2019 or unseen_test["target_year"].min() < 2019:
        raise ValueError("Final test contains pre-2019 targets")

    audit = {
        "rows": len(samples),
        "features": len(feature_names),
        "horizon_counts": {
            str(key): int(value)
            for key, value in samples["horizon"].value_counts().sort_index().items()
        },
        "partition_counts": {
            str(key): int(value) for key, value in samples["partition"].value_counts().items()
        },
        "partition_horizon_counts": [
            {"partition": str(partition), "horizon": int(horizon), "rows": int(rows)}
            for (partition, horizon), rows in samples.groupby(["partition", "horizon"]).size().items()
        ],
        "development_target_year_range": [
            int(development["target_year"].min()),
            int(development["target_year"].max()),
        ],
        "seen_country_test_target_year_range": [
            int(seen_test["target_year"].min()),
            int(seen_test["target_year"].max()),
        ],
        "unseen_country_test_target_year_range": [
            int(unseen_test["target_year"].min()),
            int(unseen_test["target_year"].max()),
        ],
        "max_feature_source_year_minus_origin": 0,
        "minimum_required_history_years": 11,
        "country_identity_in_features": False,
        "2023_assignment_stratifier_in_features": False,
        "heldout_country_rows_in_development": 0,
        "post_2018_target_rows_in_development": 0,
        "preprocessing_applied_before_split": False,
    }
    return samples, feature_names, audit


def build_feature_manifest(feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in feature_names:
        if "__lag" in name:
            source_year_rule = "origin_year minus stated lag"
        elif any(token in name for token in ("__mean", "__std", "__cv", "__log_slope")):
            source_year_rule = "trailing window ending at origin_year"
        else:
            source_year_rule = "origin_year or earlier"
        rows.append(
            {
                "column": name,
                "role": "model_feature",
                "available_at_prediction_origin": True,
                "maximum_source_year_offset_from_origin": 0,
                "source_year_rule": source_year_rule,
                "requires_fold_fitted_preprocessing": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)

    panel, panel_audit = build_panel()
    country_split = make_country_split(panel)
    samples, feature_names, sample_audit = build_supervised_samples(panel, country_split)
    feature_manifest = build_feature_manifest(feature_names)

    panel_path = OUT / "analytic_panel_204x34.csv.gz"
    split_path = OUT / "country_split_locked.csv"
    samples_path = OUT / "supervised_samples_h1_h3_h5.csv.gz"
    feature_path = PROTOCOL / "feature_manifest.csv"
    audit_path = OUT / "stage2_data_leakage_audit.json"

    deterministic_gzip = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    panel.to_csv(
        panel_path,
        index=False,
        compression=deterministic_gzip,
        float_format="%.12g",
    )
    country_split.to_csv(split_path, index=False, float_format="%.12g")
    samples.to_csv(
        samples_path,
        index=False,
        compression=deterministic_gzip,
        float_format="%.12g",
    )
    feature_manifest.to_csv(feature_path, index=False)

    audit = {
        "stage": "Stage 2 data and leakage audit",
        "seed": SEED,
        "panel": panel_audit,
        "country_split": {
            "development_countries": int(
                country_split["country_split"].eq("development").sum()
            ),
            "geographic_holdout_countries": int(
                country_split["country_split"].eq("geographic_holdout").sum()
            ),
            "holdout_fraction": float(
                country_split["country_split"].eq("geographic_holdout").mean()
            ),
            "by_quintile": [
                {
                    "sdi_quintile_2023": str(quintile),
                    "country_split": str(split),
                    "countries": int(rows),
                }
                for (quintile, split), rows in country_split.groupby(
                    ["sdi_quintile_2023", "country_split"]
                ).size().items()
            ],
            "outcome_used_for_assignment": False,
        },
        "supervised_samples": sample_audit,
        "outputs_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (panel_path, split_path, samples_path, feature_path)
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=scalar, allow_nan=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "panel_path": str(panel_path),
                "panel_rows": len(panel),
                "countries": panel["location_id"].nunique(),
                "samples_path": str(samples_path),
                "sample_rows": len(samples),
                "features": len(feature_names),
                "country_split": audit["country_split"],
                "partition_counts": sample_audit["partition_counts"],
                "audit_path": str(audit_path),
            },
            ensure_ascii=False,
            indent=2,
            default=scalar,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
