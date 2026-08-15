from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "01_data"
BMI_GLOBAL = RAW / "gbd2023_BMI_HCC_global_alllevels.csv"
BMI_SDI = RAW / "gbd2023_BMI_HCC_SDI_1990_2023.csv"
OVERALL_GLOBAL = RAW / "gbd2023_allHCC_global_1990_2023.csv"
OUTPUT = OUT / "descriptive_global_sdi_panel.csv.gz"
AUDIT = OUT / "descriptive_aggregate_panel_audit.json"


MEASURES = {
    "DALYs (Disability-Adjusted Life Years)": "daly",
    "Deaths": "death",
}
SDI_ORDER = {
    "Low SDI": 1,
    "Low-middle SDI": 2,
    "Middle SDI": 3,
    "High-middle SDI": 4,
    "High SDI": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    required = {
        "measure_name",
        "location_id",
        "location_name",
        "sex_name",
        "age_name",
        "cause_name",
        "metric_name",
        "year",
        "val",
        "lower",
        "upper",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {scope}: {sorted(required - set(frame.columns))}")
    filtered = frame.loc[
        frame["measure_name"].isin(MEASURES)
        & frame["sex_name"].eq("Both")
        & frame["age_name"].eq("Age-standardized")
        & frame["cause_name"].eq("Liver cancer")
    ].copy()
    filtered["outcome"] = filtered["measure_name"].map(MEASURES)
    filtered["scope"] = scope
    return filtered


def main() -> None:
    bmi_global = normalize(pd.read_csv(BMI_GLOBAL), "global")
    bmi_sdi = normalize(pd.read_csv(BMI_SDI), "sdi_group")
    overall = normalize(pd.read_csv(OVERALL_GLOBAL), "global")
    bmi = pd.concat([bmi_global, bmi_sdi], ignore_index=True)

    keys = ["scope", "location_id", "location_name", "outcome", "year"]
    rate = bmi.loc[bmi["metric_name"].eq("Rate"), keys + ["val", "lower", "upper"]].rename(
        columns={"val": "bmi_rate", "lower": "bmi_rate_lower", "upper": "bmi_rate_upper"}
    )
    fraction = bmi.loc[
        bmi["metric_name"].eq("Percent"), keys + ["val", "lower", "upper"]
    ].rename(
        columns={
            "val": "direct_gbd_fraction",
            "lower": "direct_gbd_fraction_lower",
            "upper": "direct_gbd_fraction_upper",
        }
    )
    if rate.duplicated(keys).any() or fraction.duplicated(keys).any():
        raise ValueError("Duplicate BMI aggregate key")
    panel = rate.merge(fraction, on=keys, how="outer", validate="one_to_one")
    overall_rate = overall.loc[
        overall["metric_name"].eq("Rate"),
        ["location_id", "location_name", "outcome", "year", "val", "lower", "upper"],
    ].rename(
        columns={
            "val": "overall_rate",
            "lower": "overall_rate_lower",
            "upper": "overall_rate_upper",
        }
    )
    if overall_rate.duplicated(["location_id", "location_name", "outcome", "year"]).any():
        raise ValueError("Duplicate overall global key")
    panel = panel.merge(
        overall_rate,
        on=["location_id", "location_name", "outcome", "year"],
        how="left",
        validate="many_to_one",
    )
    panel["asr_ratio_fraction"] = panel["bmi_rate"] / panel["overall_rate"]
    panel["fraction_definition_gap_pp"] = 100 * (
        panel["direct_gbd_fraction"] - panel["asr_ratio_fraction"]
    )
    panel["sdi_group_order"] = panel["location_name"].map(SDI_ORDER)
    panel = panel.sort_values(
        ["scope", "location_name", "outcome", "year"], kind="mergesort"
    ).reset_index(drop=True)

    expected_rows = (1 + int(bmi_sdi["location_name"].nunique())) * 2 * 34
    if len(panel) != expected_rows:
        raise ValueError(f"Expected {expected_rows} aggregate rows, found {len(panel)}")
    if panel[["bmi_rate", "direct_gbd_fraction"]].isna().any().any():
        raise ValueError("Missing BMI aggregate measure")
    global_rows = panel["scope"].eq("global")
    if panel.loc[global_rows, "overall_rate"].isna().any():
        raise ValueError("Global overall rate is incomplete")
    if panel.loc[~global_rows, "overall_rate"].notna().any():
        raise ValueError("An SDI-group row unexpectedly matched a global overall rate")
    if not panel["year"].between(1990, 2023).all():
        raise ValueError("Unexpected year")
    if not panel["direct_gbd_fraction"].between(0, 1).all():
        raise ValueError("Direct attributable fraction is outside [0,1]")

    panel.to_csv(
        OUTPUT,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    global_2023 = panel.loc[global_rows & panel["year"].eq(2023)].set_index("outcome")
    audit = {
        "status": "complete",
        "rows": int(len(panel)),
        "locations": panel["location_name"].drop_duplicates().tolist(),
        "missing_expected_sdi_groups_not_imputed": sorted(
            set(SDI_ORDER) - set(bmi_sdi["location_name"].unique())
        ),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "global_2023": {
            outcome: {
                "bmi_rate": float(global_2023.loc[outcome, "bmi_rate"]),
                "direct_gbd_percent": float(100 * global_2023.loc[outcome, "direct_gbd_fraction"]),
                "asr_ratio_percent": float(100 * global_2023.loc[outcome, "asr_ratio_fraction"]),
            }
            for outcome in ["daly", "death"]
        },
        "source_hashes": {
            path.name: sha256(path) for path in [BMI_GLOBAL, BMI_SDI, OVERALL_GLOBAL]
        },
        "output_sha256": sha256(OUTPUT),
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
