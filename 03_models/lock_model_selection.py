from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_utils import OUT, load_config, regression_metrics, sha256


TABULAR = OUT / "cv_search_blocks_v2"
GRU = OUT / "gru_cv_blocks_v2"
LOCK_DIR = OUT / "selection_lock"
TRANSPARENT = ["persistence", "logtrend5", "logtrend10", "pooled_ridge"]
ADVANCED = [
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "gru",
]
SIMPLICITY_RANK = {
    "persistence": 0,
    "logtrend5": 1,
    "logtrend10": 2,
    "pooled_ridge": 3,
    "ridge": 4,
    "elastic_net": 5,
    "random_forest": 6,
    "extra_trees": 7,
    "hist_gradient_boosting": 8,
    "xgboost": 9,
    "gru": 10,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required completed CV artifact is missing: {path}")
    return path


def metric_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    keys = ["outcome", "horizon", "family"]
    for group_key, group in predictions.groupby([*keys, "fold"], sort=True):
        outcome, horizon, family, fold = group_key
        fold_rows.append(
            {
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "fold": fold,
                "n": int(len(group)),
                **regression_metrics(group["observed"], group["prediction"]),
            }
        )
    for group_key, group in predictions.groupby([*keys, "location_id"], sort=True):
        outcome, horizon, family, location_id = group_key
        country_rows.append(
            {
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "location_id": int(location_id),
                **regression_metrics(group["observed"], group["prediction"]),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    numeric = ["rmsle", "mae", "rmse", "wape", "median_absolute_error", "r2", "spearman"]
    comparison = (
        fold_metrics.groupby(keys, as_index=False)[numeric]
        .mean()
        .rename(columns={name: f"mean_fold_{name}" for name in numeric})
    )
    country = (
        pd.DataFrame(country_rows)
        .groupby(keys, as_index=False)[numeric]
        .mean()
        .rename(columns={name: f"country_macro_{name}" for name in numeric})
    )
    comparison = comparison.merge(country, on=keys, how="left", validate="one_to_one")
    comparison["complexity_rank"] = comparison["family"].map(SIMPLICITY_RANK).astype(int)
    comparison["rank_rmsle"] = comparison.groupby(["outcome", "horizon"])[
        "mean_fold_rmsle"
    ].rank(method="min")
    return fold_metrics, comparison


def add_skills(comparison: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in comparison.groupby(["outcome", "horizon"], sort=True):
        group = group.copy()
        persistence = float(group.loc[group["family"].eq("persistence"), "mean_fold_rmsle"].iloc[0])
        best_trend = float(
            group.loc[group["family"].isin(TRANSPARENT), "mean_fold_rmsle"].min()
        )
        group["best_transparent_rmsle"] = best_trend
        group["skill_vs_persistence"] = 1.0 - group["mean_fold_rmsle"] / persistence
        group["skill_vs_best_transparent"] = 1.0 - group["mean_fold_rmsle"] / best_trend
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def selected_params(
    outcome: str,
    horizon: int,
    family: str,
    tabular_selected: pd.DataFrame,
    gru_selected: pd.DataFrame,
    tabular_fold_search: pd.DataFrame,
    gru_fold_search: pd.DataFrame,
) -> dict[str, Any]:
    if family in {"persistence", "logtrend5", "logtrend10"}:
        return {"params": {}, "config_id": family}
    if family == "gru":
        row = gru_selected.loc[
            gru_selected["outcome"].eq(outcome) & gru_selected["horizon"].eq(horizon)
        ].iloc[0]
        fold = gru_fold_search.loc[
            gru_fold_search["outcome"].eq(outcome)
            & gru_fold_search["horizon"].eq(horizon)
            & gru_fold_search["config_id"].eq(row["config_id"])
        ]
        return {
            "params": json.loads(row["params_json"]),
            "config_id": str(row["config_id"]),
            "final_epochs": int(np.rint(np.median(fold["selected_epochs"]))),
            "outer_fold_selected_epochs": [int(value) for value in fold["selected_epochs"]],
        }
    row = tabular_selected.loc[
        tabular_selected["outcome"].eq(outcome)
        & tabular_selected["horizon"].eq(horizon)
        & tabular_selected["family"].eq(family)
    ].iloc[0]
    result: dict[str, Any] = {
        "params": json.loads(row["params_json"]),
        "config_id": str(row["config_id"]),
    }
    if family == "xgboost":
        fold = tabular_fold_search.loc[
            tabular_fold_search["outcome"].eq(outcome)
            & tabular_fold_search["horizon"].eq(horizon)
            & tabular_fold_search["family"].eq(family)
            & tabular_fold_search["config_id"].eq(row["config_id"])
        ]
        rounds = fold["effective_n_estimators"].dropna().astype(int)
        result["final_n_estimators"] = int(np.rint(np.median(rounds)))
        result["outer_fold_effective_n_estimators"] = rounds.tolist()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze CV-only model family selection")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / "model_selection_lock.json"
    if lock_path.exists() and not args.force:
        raise FileExistsError("Selection lock already exists; use --force only for an audited pre-test rebuild")

    config = load_config()
    tab_oof_path = require(TABULAR / "selected_tabular_oof_predictions.csv.gz")
    gru_oof_path = require(GRU / "selected_gru_oof_predictions.csv.gz")
    tab_selected_path = require(TABULAR / "selected_tabular_configurations.csv")
    gru_selected_path = require(GRU / "selected_gru_configurations.csv")
    tab_fold_path = require(TABULAR / "tabular_cv_fold_metrics.csv")
    gru_fold_path = require(GRU / "gru_cv_fold_metrics.csv")

    tab_oof = pd.read_csv(tab_oof_path, low_memory=False)
    gru_oof = pd.read_csv(gru_oof_path, low_memory=False)
    predictions = pd.concat([tab_oof, gru_oof], ignore_index=True)
    expected_families = set(TRANSPARENT + ADVANCED)
    if set(predictions["family"]) != expected_families:
        raise ValueError(
            f"Incomplete family set: expected {sorted(expected_families)}, got {sorted(set(predictions['family']))}"
        )
    duplicate_key = ["outcome", "horizon", "family", "sample_id"]
    if predictions.duplicated(duplicate_key).any():
        raise ValueError("Duplicate OOF prediction key")
    if not np.isfinite(predictions[["observed", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite OOF prediction")

    fold_metrics, comparison = metric_table(predictions)
    comparison = add_skills(comparison)
    comparison = comparison.sort_values(
        ["outcome", "horizon", "mean_fold_rmsle", "complexity_rank"], kind="mergesort"
    ).reset_index(drop=True)
    fold_path = LOCK_DIR / "cv_fold_metrics_selected_models.csv"
    comparison_path = LOCK_DIR / "cv_model_comparison.csv"
    fold_metrics.to_csv(fold_path, index=False, float_format="%.12g")
    comparison.to_csv(comparison_path, index=False, float_format="%.12g")

    tab_selected = pd.read_csv(tab_selected_path)
    gru_selected = pd.read_csv(gru_selected_path)
    tab_fold_search = pd.read_csv(tab_fold_path)
    gru_fold_search = pd.read_csv(gru_fold_path)
    tolerance = float(config["selection_tie_relative_tolerance"])
    tasks: list[dict[str, Any]] = []
    for (outcome, horizon), group in comparison.groupby(["outcome", "horizon"], sort=True):
        minimum = float(group["mean_fold_rmsle"].min())
        eligible = group.loc[group["mean_fold_rmsle"].le(minimum * (1.0 + tolerance))].copy()
        winner = eligible.sort_values(
            ["complexity_rank", "mean_fold_rmsle", "family"], kind="mergesort"
        ).iloc[0]
        transparent_group = group.loc[group["family"].isin(TRANSPARENT)].copy()
        transparent_minimum = float(transparent_group["mean_fold_rmsle"].min())
        transparent_eligible = transparent_group.loc[
            transparent_group["mean_fold_rmsle"].le(transparent_minimum * (1.0 + tolerance))
        ]
        reference_transparent = transparent_eligible.sort_values(
            ["complexity_rank", "mean_fold_rmsle", "family"], kind="mergesort"
        ).iloc[0]
        tree_families = [
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
            "xgboost",
        ]
        tree_group = group.loc[group["family"].isin(tree_families)].copy()
        tree_minimum = float(tree_group["mean_fold_rmsle"].min())
        tree_eligible = tree_group.loc[
            tree_group["mean_fold_rmsle"].le(tree_minimum * (1.0 + tolerance))
        ]
        reference_tree = tree_eligible.sort_values(
            ["complexity_rank", "mean_fold_rmsle", "family"], kind="mergesort"
        ).iloc[0]
        family_configs: dict[str, Any] = {}
        for family in TRANSPARENT + ADVANCED:
            family_configs[family] = selected_params(
                outcome,
                int(horizon),
                family,
                tab_selected,
                gru_selected,
                tab_fold_search,
                gru_fold_search,
            )
        tasks.append(
            {
                "outcome": outcome,
                "horizon": int(horizon),
                "selected_family": str(winner["family"]),
                "selected_cv_mean_fold_rmsle": float(winner["mean_fold_rmsle"]),
                "minimum_cv_mean_fold_rmsle": minimum,
                "selection_within_relative_tolerance": tolerance,
                "selected_skill_vs_persistence": float(winner["skill_vs_persistence"]),
                "selected_skill_vs_best_transparent": float(winner["skill_vs_best_transparent"]),
                "reference_transparent_family": str(reference_transparent["family"]),
                "reference_transparent_cv_mean_fold_rmsle": float(
                    reference_transparent["mean_fold_rmsle"]
                ),
                "reference_tree_family": str(reference_tree["family"]),
                "reference_tree_cv_mean_fold_rmsle": float(reference_tree["mean_fold_rmsle"]),
                "all_family_configurations": family_configs,
            }
        )

    partition_manifest = json.loads(
        (OUT.parent / "01_data" / "model_partition_manifest.json").read_text(encoding="utf-8")
    )
    lock = {
        "locked": True,
        "lock_created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_data_scope": "development-only expanding-window out-of-fold predictions through target year 2018",
        "sealed_final_test_not_loaded": True,
        "selection_metric": config["selection_metric"],
        "tie_rule": (
            "Among configurations/families within 1% relative RMSLE of the minimum, "
            "select the prespecified lower-complexity option."
        ),
        "family_simplicity_order": SIMPLICITY_RANK,
        "tasks": tasks,
        "source_hashes": {
            "development_samples_sha256": partition_manifest["development"]["sha256"],
            "development_sequence_panel_sha256": partition_manifest["development_sequence_panel"]["sha256"],
            "sealed_final_test_sha256_from_pre_model_manifest": partition_manifest["sealed_final_test"]["sha256"],
            "model_config_sha256": sha256(OUT.parent / "02_protocol" / "model_config_locked.json"),
            "tabular_oof_sha256": sha256(tab_oof_path),
            "gru_oof_sha256": sha256(gru_oof_path),
            "cv_model_comparison_sha256": sha256(comparison_path),
            "model_utils_sha256": sha256(OUT / "model_utils.py"),
            "tabular_runner_sha256": sha256(OUT / "run_tabular_cv.py"),
            "gru_runner_sha256": sha256(OUT / "run_gru_cv.py"),
            "selection_runner_sha256": sha256(Path(__file__)),
            "requirements_lock_sha256": sha256(
                OUT.parent / "02_protocol" / "requirements.lock.txt"
            ),
        },
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    lock_hash = sha256(lock_path)
    (LOCK_DIR / "model_selection_lock.sha256.txt").write_text(
        f"{lock_hash}  {lock_path.name}\n", encoding="ascii"
    )
    print(
        canonical_json(
            {
                "status": "locked",
                "lock_sha256": lock_hash,
                "selected": [
                    {
                        "outcome": row["outcome"],
                        "horizon": row["horizon"],
                        "family": row["selected_family"],
                        "cv_rmsle": row["selected_cv_mean_fold_rmsle"],
                    }
                    for row in tasks
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
