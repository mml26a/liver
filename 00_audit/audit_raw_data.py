from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "00_audit"

RAW_FILES = [
    "gbd2023_allHCC_country_1990_2023.csv",
    "gbd2023_allHCC_global_1990_2023.csv",
    "gbd2023_BMI_HCC_country_1990_2023.csv",
    "gbd2023_BMI_HCC_global_alllevels.csv",
    "gbd2023_BMI_HCC_SDI_1990_2023.csv",
    "gbd2023_SDI_values_1950_2023.csv",
]


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sorted_unique(frame: pd.DataFrame, column: str, limit: int = 300) -> list[Any]:
    if column not in frame.columns:
        return []
    values = [json_value(value) for value in frame[column].dropna().unique().tolist()]
    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=str)
    return values[:limit]


def duplicate_key_summary(frame: pd.DataFrame, value_columns: set[str]) -> dict[str, Any]:
    key = [column for column in frame.columns if column not in value_columns]
    duplicate_mask = frame.duplicated(key, keep=False)
    return {
        "key": key,
        "duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_groups": int(frame.loc[duplicate_mask].groupby(key, dropna=False).ngroups)
        if duplicate_mask.any()
        else 0,
    }


def ui_summary(frame: pd.DataFrame, val: str, lower: str, upper: str) -> dict[str, Any]:
    for column in (val, lower, upper):
        if column not in frame.columns:
            return {"available": False}
    complete = frame[[val, lower, upper]].notna().all(axis=1)
    violation = complete & ((frame[lower] > frame[val]) | (frame[val] > frame[upper]))
    width = frame.loc[complete, upper] - frame.loc[complete, lower]
    return {
        "available": True,
        "complete_rows": int(complete.sum()),
        "missing_any_rows": int((~complete).sum()),
        "ordering_violation_rows": int(violation.sum()),
        "negative_width_rows": int((width < 0).sum()),
        "median_width": json_value(width.median()) if len(width) else None,
        "max_width": json_value(width.max()) if len(width) else None,
    }


def series_completeness(
    frame: pd.DataFrame, year_column: str, expected_start: int, expected_end: int
) -> dict[str, Any]:
    value_columns = {
        year_column,
        "val",
        "lower",
        "upper",
        "mean_value",
        "lower_value",
        "upper_value",
    }
    key = [column for column in frame.columns if column not in value_columns]
    expected_years = set(range(expected_start, expected_end + 1))
    counts = frame.groupby(key, dropna=False)[year_column].nunique()
    incomplete = counts[counts != len(expected_years)]
    examples = []
    if not incomplete.empty:
        for group_key, count in incomplete.head(12).items():
            group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
            mask = pd.Series(True, index=frame.index)
            for column, value in zip(key, group_key_tuple):
                mask &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
            observed = set(pd.to_numeric(frame.loc[mask, year_column], errors="coerce").dropna().astype(int))
            examples.append(
                {
                    "series": {column: json_value(value) for column, value in zip(key, group_key_tuple)},
                    "observed_year_count": int(count),
                    "missing_years": sorted(expected_years - observed),
                }
            )
    return {
        "series_key": key,
        "series_n": int(len(counts)),
        "expected_year_count": len(expected_years),
        "complete_series_n": int((counts == len(expected_years)).sum()),
        "incomplete_series_n": int(len(incomplete)),
        "min_observed_year_count": int(counts.min()) if len(counts) else 0,
        "max_observed_year_count": int(counts.max()) if len(counts) else 0,
        "incomplete_examples": examples,
    }


def name_id_consistency(frame: pd.DataFrame) -> dict[str, Any]:
    if not {"location_id", "location_name"}.issubset(frame.columns):
        return {}
    name_to_ids = frame.groupby("location_name")["location_id"].nunique()
    id_to_names = frame.groupby("location_id")["location_name"].nunique()
    bad_names = name_to_ids[name_to_ids > 1]
    bad_ids = id_to_names[id_to_names > 1]
    return {
        "names_with_multiple_ids_n": int(len(bad_names)),
        "ids_with_multiple_names_n": int(len(bad_ids)),
        "names_with_multiple_ids": {str(key): int(value) for key, value in bad_names.items()},
        "ids_with_multiple_names": {str(key): int(value) for key, value in bad_ids.items()},
    }


def summarize_gbd_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    year_column = "year" if "year" in frame.columns else "year_id"
    value_columns = {
        "val",
        "lower",
        "upper",
        "mean_value",
        "lower_value",
        "upper_value",
    }
    expected_start, expected_end = (1950, 2023) if year_column == "year_id" else (1990, 2023)

    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": frame.columns.tolist(),
        "column_count": int(frame.shape[1]),
        "year_column": year_column,
        "year_min": int(pd.to_numeric(frame[year_column], errors="coerce").min()),
        "year_max": int(pd.to_numeric(frame[year_column], errors="coerce").max()),
        "unique_locations": int(frame["location_name"].nunique())
        if "location_name" in frame.columns
        else None,
        "missing_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if int(count) > 0
        },
        "duplicate_key": duplicate_key_summary(frame, value_columns),
        "series_completeness": series_completeness(
            frame, year_column, expected_start, expected_end
        ),
        "location_name_id_consistency": name_id_consistency(frame),
    }

    for column in [
        "measure_name",
        "metric_name",
        "sex_name",
        "sex",
        "age_name",
        "age_group_name",
        "cause_name",
        "rei_name",
        "covariate_name_short",
        "location_name",
    ]:
        if column in frame.columns:
            summary[f"unique_{column}"] = sorted_unique(frame, column)

    if {"val", "lower", "upper"}.issubset(frame.columns):
        summary["uncertainty"] = ui_summary(frame, "val", "lower", "upper")
        summary["negative_val_rows"] = int((frame["val"] < 0).sum())
        if "metric_name" in frame.columns:
            percent = frame["metric_name"].eq("Percent")
            summary["percent_rows"] = int(percent.sum())
            summary["percent_outside_0_1_rows"] = int(
                (percent & ((frame["val"] < 0) | (frame["val"] > 1))).sum()
            )
            rate = frame["metric_name"].eq("Rate")
            summary["rate_rows"] = int(rate.sum())
            summary["negative_rate_rows"] = int((rate & (frame["val"] < 0)).sum())
    elif {"mean_value", "lower_value", "upper_value"}.issubset(frame.columns):
        summary["uncertainty"] = ui_summary(
            frame, "mean_value", "lower_value", "upper_value"
        )
        summary["mean_outside_0_1_rows"] = int(
            ((frame["mean_value"] < 0) | (frame["mean_value"] > 1)).sum()
        )

    return frame, summary


roster = pd.read_csv(PROJECT / "config" / "country_roster_204.csv")
roster_names = set(roster["location_name"].astype(str))

# The project roster uses two legacy/ASCII display names, whereas the GBD 2023
# exports use the current Unicode names.  These are label aliases, not extra or
# missing analytic locations; location_id remains stable in every source table.
ROSTER_TO_GBD_NAME = {
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Turkey": "Türkiye",
}
canonical_roster_names = {
    ROSTER_TO_GBD_NAME.get(name, name) for name in roster_names
}

frames: dict[str, pd.DataFrame] = {}
audit: dict[str, Any] = {
    "project": str(PROJECT.relative_to(ROOT)),
    "roster": {
        "rows": int(len(roster)),
        "unique_location_names": int(roster["location_name"].nunique()),
        "duplicate_rows": int(roster.duplicated("location_name").sum()),
    },
    "raw_files": {},
}

for filename in RAW_FILES:
    frame, summary = summarize_gbd_file(RAW / filename)
    frames[filename] = frame
    if "location_name" in frame.columns:
        location_names = set(frame["location_name"].astype(str))
        summary["roster_coverage"] = {
            "roster_present_n": len(canonical_roster_names & location_names),
            "roster_missing_n": len(canonical_roster_names - location_names),
            "roster_missing": sorted(canonical_roster_names - location_names),
            "non_roster_location_n": len(location_names - canonical_roster_names),
            "non_roster_locations": sorted(location_names - canonical_roster_names)[:300],
            "accepted_display_name_aliases": ROSTER_TO_GBD_NAME,
        }
    audit["raw_files"][filename] = summary


master_path = PROJECT / "data_clean" / "analytic_master_long.csv"
master = pd.read_csv(master_path, low_memory=False)
master_value_columns = {"val", "lower", "upper", "sdi", "population"}
master_summary: dict[str, Any] = {
    "rows": int(len(master)),
    "columns": master.columns.tolist(),
    "column_count": int(master.shape[1]),
    "year_min": int(master["year"].min()),
    "year_max": int(master["year"].max()),
    "unique_locations": int(master["location_name"].nunique()),
    "unique_location_ids": int(master["location_id"].nunique()),
    "unique_scopes": sorted_unique(master, "scope"),
    "unique_risk_groups": sorted_unique(master, "risk_group"),
    "unique_source_files": sorted_unique(master, "source_file"),
    "unique_measures": sorted_unique(master, "measure_name"),
    "unique_metrics": sorted_unique(master, "metric_name"),
    "unique_sexes": sorted_unique(master, "sex_name"),
    "unique_ages": sorted_unique(master, "age_name"),
    "unique_causes": sorted_unique(master, "cause_name"),
    "unique_risks": sorted_unique(master, "rei_name"),
    "missing_by_column": {
        column: int(count)
        for column, count in master.isna().sum().items()
        if int(count) > 0
    },
    "rows_by_scope_and_risk_group": [
        {"scope": str(scope), "risk_group": str(risk), "rows": int(rows)}
        for (scope, risk), rows in master.groupby(["scope", "risk_group"], dropna=False).size().items()
    ],
    "rows_by_source_file": {
        str(key): int(value) for key, value in master["source_file"].value_counts().items()
    },
    "duplicate_key": duplicate_key_summary(master, master_value_columns),
    "uncertainty": ui_summary(master, "val", "lower", "upper"),
    "location_name_id_consistency": name_id_consistency(master),
    "country_scope_roster_coverage": {},
}

country_master = master.loc[master["scope"].eq("country")]
country_master_names = set(country_master["location_name"].astype(str))
master_summary["country_scope_roster_coverage"] = {
    "roster_present_n": len(canonical_roster_names & country_master_names),
    "roster_missing_n": len(canonical_roster_names - country_master_names),
    "roster_missing": sorted(canonical_roster_names - country_master_names),
    "non_roster_location_n": len(country_master_names - canonical_roster_names),
    "non_roster_locations": sorted(country_master_names - canonical_roster_names),
    "accepted_display_name_aliases": ROSTER_TO_GBD_NAME,
}


def normalize_merge_key(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    copy = frame[columns].copy()
    for column in columns:
        if copy[column].dtype.kind in "if":
            copy[column] = copy[column].fillna(-999999)
        else:
            copy[column] = copy[column].fillna("__NA__")
    return copy


reconciliations = []
for source_file, master_part in master.groupby("source_file", dropna=False):
    source_file = str(source_file)
    if source_file not in frames:
        reconciliations.append(
            {
                "source_file": source_file,
                "status": "source_not_in_raw_file_list",
                "master_rows": int(len(master_part)),
            }
        )
        continue
    raw_part = frames[source_file].copy()
    if source_file == "gbd2023_BMI_HCC_global_alllevels.csv":
        raw_part = raw_part.loc[raw_part["location_name"].eq("Global")]

    candidate_key = [
        "measure_id",
        "location_id",
        "sex_id",
        "age_id",
        "cause_id",
        "rei_id",
        "metric_id",
        "year",
    ]
    key = [column for column in candidate_key if column in raw_part.columns and column in master_part.columns]
    raw_key = normalize_merge_key(raw_part, key)
    master_key = normalize_merge_key(master_part, key)
    raw_compare = pd.concat(
        [raw_key.reset_index(drop=True), raw_part[["val", "lower", "upper"]].reset_index(drop=True)],
        axis=1,
    )
    master_compare = pd.concat(
        [
            master_key.reset_index(drop=True),
            master_part[["val", "lower", "upper"]].reset_index(drop=True),
        ],
        axis=1,
    )
    merged = raw_compare.merge(
        master_compare,
        on=key,
        how="outer",
        suffixes=("_raw", "_master"),
        indicator=True,
    )
    both = merged["_merge"].eq("both")
    reconciliations.append(
        {
            "source_file": source_file,
            "status": "compared",
            "key": key,
            "raw_rows_after_scope_filter": int(len(raw_part)),
            "master_rows": int(len(master_part)),
            "matched_rows": int(both.sum()),
            "raw_only_rows": int(merged["_merge"].eq("left_only").sum()),
            "master_only_rows": int(merged["_merge"].eq("right_only").sum()),
            "max_abs_val_diff": json_value(
                np.nanmax(np.abs(merged.loc[both, "val_raw"] - merged.loc[both, "val_master"]))
            )
            if both.any()
            else None,
            "max_abs_lower_diff": json_value(
                np.nanmax(
                    np.abs(merged.loc[both, "lower_raw"] - merged.loc[both, "lower_master"])
                )
            )
            if both.any()
            else None,
            "max_abs_upper_diff": json_value(
                np.nanmax(
                    np.abs(merged.loc[both, "upper_raw"] - merged.loc[both, "upper_master"])
                )
            )
            if both.any()
            else None,
        }
    )

master_summary["source_reconciliation"] = reconciliations
audit["master_dataset"] = master_summary

audit["semantic_findings"] = {
    "cause_names_across_gbd_files": sorted(
        {
            str(value)
            for filename, frame in frames.items()
            if "cause_name" in frame.columns
            for value in frame["cause_name"].dropna().unique()
        }
    ),
    "risk_names_across_bmi_files": sorted(
        {
            str(value)
            for filename, frame in frames.items()
            if "rei_name" in frame.columns
            for value in frame["rei_name"].dropna().unique()
        }
    ),
    "hcc_histology_field_present": any(
        "histology" in column.lower()
        for frame in frames.values()
        for column in frame.columns
    ),
}

OUT.mkdir(parents=True, exist_ok=True)
output_path = OUT / "raw_data_audit.json"
output_path.write_text(
    json.dumps(audit, ensure_ascii=False, indent=2, default=json_value, allow_nan=False),
    encoding="utf-8",
)
print(output_path)
print(
    json.dumps(
        {
            "raw_file_rows": {
                filename: audit["raw_files"][filename]["rows"] for filename in RAW_FILES
            },
            "master_rows": audit["master_dataset"]["rows"],
            "master_unique_locations": audit["master_dataset"]["unique_locations"],
            "master_missing": audit["master_dataset"]["missing_by_column"],
            "semantic_findings": audit["semantic_findings"],
            "reconciliation": audit["master_dataset"]["source_reconciliation"],
        },
        ensure_ascii=False,
        indent=2,
        default=json_value,
        allow_nan=False,
    )
)
