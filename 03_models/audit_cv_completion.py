from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_utils import load_config
from run_tabular_cv import select_config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "03_models"
PROTOCOL = ROOT / "02_protocol"
TABULAR = MODELS / "cv_search_blocks_v2"
GRU = MODELS / "gru_cv_blocks_v2"
ARCHIVE = TABULAR / "numerical_warning_archive_pre_lsqr"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(np.isfinite(frame[columns].to_numpy(dtype=float)).all())


def task_family_blocks() -> list[Path]:
    return sorted(
        path
        for path in TABULAR.glob("*__*.csv")
        if path.is_file()
        and path.name
        not in {
            "tabular_cv_fold_metrics.csv",
            "selected_tabular_configurations.csv",
        }
    )


def numerical_comparison(tolerance: float) -> pd.DataFrame:
    old_frames: list[pd.DataFrame] = []
    new_frames: list[pd.DataFrame] = []
    for old_path in sorted(ARCHIVE.glob("*__*.csv")):
        new_path = TABULAR / old_path.name
        if not new_path.exists():
            raise FileNotFoundError(new_path)
        old_frames.append(pd.read_csv(old_path))
        new_frames.append(pd.read_csv(new_path))
    old = pd.concat(old_frames, ignore_index=True)
    new = pd.concat(new_frames, ignore_index=True)
    keys = ["outcome", "horizon", "family", "config_id", "fold"]
    metric_columns = [
        "rmsle",
        "mae",
        "rmse",
        "wape",
        "median_absolute_error",
        "r2",
        "spearman",
    ]
    paired = old[keys + metric_columns].merge(
        new[keys + metric_columns],
        on=keys,
        how="outer",
        suffixes=("_pre_lsqr", "_lsqr"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Pre-LSQR and LSQR rows are not one-to-one")

    rows: list[dict[str, Any]] = []
    for keys_value, old_group in old.groupby(["outcome", "horizon", "family"], sort=True):
        outcome, horizon, family = keys_value
        new_group = new.loc[
            new["outcome"].eq(outcome)
            & new["horizon"].eq(horizon)
            & new["family"].eq(family)
        ]
        selected_old = select_config(old_group, tolerance)
        selected_new = select_config(new_group, tolerance)
        paired_task = paired.loc[
            paired["outcome"].eq(outcome)
            & paired["horizon"].eq(horizon)
            & paired["family"].eq(family)
        ]
        rows.append(
            {
                "outcome": outcome,
                "horizon": int(horizon),
                "family": family,
                "pre_lsqr_selected_config_id": str(selected_old["config_id"]),
                "lsqr_selected_config_id": str(selected_new["config_id"]),
                "selected_config_unchanged": bool(
                    selected_old["config_id"] == selected_new["config_id"]
                ),
                "pre_lsqr_selected_mean_rmsle": float(selected_old["mean_rmsle"]),
                "lsqr_selected_mean_rmsle": float(selected_new["mean_rmsle"]),
                "selected_mean_rmsle_change": float(
                    selected_new["mean_rmsle"] - selected_old["mean_rmsle"]
                ),
                "maximum_absolute_row_rmsle_change": float(
                    (
                        paired_task["rmsle_lsqr"]
                        - paired_task["rmsle_pre_lsqr"]
                    ).abs().max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["outcome", "horizon", "family"], kind="mergesort"
    )


def replay_rmsle_comparison(
    selected: pd.DataFrame, oof: pd.DataFrame
) -> pd.DataFrame:
    local = oof.copy()
    local["squared_log1p_error"] = (
        np.log1p(local["prediction"].to_numpy(dtype=float))
        - np.log1p(local["observed"].to_numpy(dtype=float))
    ) ** 2
    fold = (
        local.groupby(["outcome", "horizon", "family", "fold"], as_index=False)[
            "squared_log1p_error"
        ]
        .mean()
        .assign(replayed_fold_rmsle=lambda frame: np.sqrt(frame["squared_log1p_error"]))
    )
    replayed = (
        fold.groupby(["outcome", "horizon", "family"], as_index=False)[
            "replayed_fold_rmsle"
        ]
        .mean()
        .rename(columns={"replayed_fold_rmsle": "replayed_mean_rmsle"})
    )
    expected = selected[
        ["outcome", "horizon", "family", "config_id", "mean_rmsle"]
    ].copy()
    comparison = expected.merge(
        replayed,
        on=["outcome", "horizon", "family"],
        how="left",
        validate="one_to_one",
    )
    comparison["absolute_difference"] = (
        comparison["replayed_mean_rmsle"] - comparison["mean_rmsle"]
    ).abs()
    return comparison


def main() -> None:
    config = load_config()
    tolerance = float(config["selection_tie_relative_tolerance"])
    tab_metrics_path = TABULAR / "tabular_cv_fold_metrics.csv"
    tab_selected_path = TABULAR / "selected_tabular_configurations.csv"
    tab_oof_path = TABULAR / "selected_tabular_oof_predictions.csv.gz"
    gru_metrics_path = GRU / "gru_cv_fold_metrics.csv"
    gru_selected_path = GRU / "selected_gru_configurations.csv"
    gru_oof_path = GRU / "selected_gru_oof_predictions.csv.gz"

    tab_metrics = pd.read_csv(tab_metrics_path)
    tab_selected = pd.read_csv(tab_selected_path)
    tab_oof = pd.read_csv(tab_oof_path, low_memory=False)
    gru_metrics = pd.read_csv(gru_metrics_path)
    gru_selected = pd.read_csv(gru_selected_path)
    gru_oof = pd.read_csv(gru_oof_path, low_memory=False)
    tab_replay = replay_rmsle_comparison(tab_selected, tab_oof)
    gru_replay = replay_rmsle_comparison(
        gru_selected.assign(family="gru"), gru_oof
    )

    expected_tabular_families = {
        "persistence",
        "logtrend5",
        "logtrend10",
        "pooled_ridge",
        "ridge",
        "elastic_net",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
    }
    tabular_undefined_spearman = ~np.isfinite(
        tab_metrics["spearman"].to_numpy(dtype=float)
    )
    checks = {
        "task_family_blocks_48": len(task_family_blocks()) == 48,
        "tabular_search_rows_2412": len(tab_metrics) == 2412,
        "tabular_selected_configurations_42": len(tab_selected) == 42,
        "tabular_oof_rows_58680": len(tab_oof) == 58680,
        "gru_metric_rows_216": len(gru_metrics) == 216,
        "gru_selected_configurations_6": len(gru_selected) == 6,
        "gru_oof_rows_5868": len(gru_oof) == 5868,
        "tabular_expected_families": set(tab_oof["family"]) == expected_tabular_families,
        "gru_family_only": set(gru_oof["family"]) == {"gru"},
        "tabular_three_folds": tab_metrics.groupby(
            ["outcome", "horizon", "family", "config_id"]
        )["fold"].nunique().eq(3).all(),
        "gru_three_folds": gru_metrics.groupby(
            ["outcome", "horizon", "config_id"]
        )["fold"].nunique().eq(3).all(),
        "tabular_primary_metrics_finite": finite(
            tab_metrics,
            ["rmsle", "mae", "rmse", "wape", "median_absolute_error", "r2"],
        ),
        "undefined_spearman_only_degenerate_elastic_net": bool(
            tabular_undefined_spearman.any()
            and tab_metrics.loc[tabular_undefined_spearman, "family"]
            .eq("elastic_net")
            .all()
        ),
        "selected_tabular_mean_spearman_finite": finite(
            tab_selected, ["mean_spearman"]
        ),
        "gru_metrics_finite": finite(
            gru_metrics,
            ["rmsle", "mae", "rmse", "wape", "median_absolute_error", "r2", "spearman", "selected_epochs"],
        ),
        "tabular_oof_finite": finite(tab_oof, ["observed", "prediction"]),
        "gru_oof_finite": finite(gru_oof, ["observed", "prediction"]),
        "tabular_oof_unique": not tab_oof.duplicated(
            ["outcome", "horizon", "family", "sample_id"]
        ).any(),
        "gru_oof_unique": not gru_oof.duplicated(
            ["outcome", "horizon", "family", "sample_id"]
        ).any(),
        "tabular_selected_oof_exact_replay": bool(
            tab_replay["absolute_difference"].max() <= 1e-9
        ),
        "gru_selected_oof_exact_replay": bool(
            gru_replay["absolute_difference"].max() <= 1e-9
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"CV completion audit failed: {checks}")

    numerical = numerical_comparison(tolerance)
    numerical_path = PROTOCOL / "ridge_solver_sensitivity.csv"
    numerical.to_csv(numerical_path, index=False, float_format="%.12g")
    metadata = {
        "status": "PASS",
        "checks": checks,
        "counts": {
            "task_family_blocks": len(task_family_blocks()),
            "tabular_search_rows": len(tab_metrics),
            "tabular_selected_configurations": len(tab_selected),
            "tabular_oof_rows": len(tab_oof),
            "gru_metric_rows": len(gru_metrics),
            "gru_selected_configurations": len(gru_selected),
            "gru_oof_rows": len(gru_oof),
            "undefined_candidate_spearman_rows": int(
                tabular_undefined_spearman.sum()
            ),
        },
        "selected_oof_replay": {
            "tolerance": 1e-9,
            "tabular_maximum_absolute_mean_fold_rmsle_difference": float(
                tab_replay["absolute_difference"].max()
            ),
            "gru_maximum_absolute_mean_fold_rmsle_difference": float(
                gru_replay["absolute_difference"].max()
            ),
        },
        "undefined_spearman_note": (
            "Spearman correlation is mathematically undefined for 45 deliberately "
            "retained, over-regularised ElasticNet candidates with constant predictions. "
            "All primary metrics and every selected configuration are finite."
        ),
        "numerical_stability": {
            "solver_before": "Ridge(auto), which emitted ill-conditioned-matrix warnings",
            "solver_formal": "Ridge(lsqr, tol=1e-10, max_iter=10000)",
            "compared_task_families": int(len(numerical)),
            "selected_configuration_changes": int(
                (~numerical["selected_config_unchanged"]).sum()
            ),
            "maximum_absolute_selected_mean_rmsle_change": float(
                numerical["selected_mean_rmsle_change"].abs().max()
            ),
            "maximum_absolute_row_rmsle_change": float(
                numerical["maximum_absolute_row_rmsle_change"].max()
            ),
        },
        "sha256": {
            "tabular_fold_metrics": sha256(tab_metrics_path),
            "tabular_selected_configurations": sha256(tab_selected_path),
            "tabular_oof": sha256(tab_oof_path),
            "gru_fold_metrics": sha256(gru_metrics_path),
            "gru_selected_configurations": sha256(gru_selected_path),
            "gru_oof": sha256(gru_oof_path),
            "ridge_solver_sensitivity": sha256(numerical_path),
        },
    }
    audit_path = MODELS / "cv_completion_audit.json"
    audit_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    rows = [
        "# Numerical-stability addendum",
        "",
        "The initial development-only Ridge search emitted ill-conditioned-matrix warnings at very small regularisation values. Before any model-family lock or access to the sealed tests, the Ridge implementation was changed to the deterministic LSQR solver (`tol=1e-10`, `max_iter=10000`). The warning-version task blocks were preserved under `numerical_warning_archive_pre_lsqr`; only the LSQR outputs are eligible for selection.",
        "",
        f"**CV completion gate: PASS.** Tabular rows={len(tab_metrics):,}; tabular OOF={len(tab_oof):,}; GRU rows={len(gru_metrics):,}; GRU OOF={len(gru_oof):,}.",
        "",
        "| Outcome | Horizon | Family | Pre-LSQR config | LSQR config | Unchanged | Pre-LSQR RMSLE | LSQR RMSLE | Change |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in numerical.itertuples(index=False):
        rows.append(
            f"| {row.outcome} | {int(row.horizon)} | {row.family} | "
            f"{row.pre_lsqr_selected_config_id} | {row.lsqr_selected_config_id} | "
            f"{bool(row.selected_config_unchanged)} | "
            f"{row.pre_lsqr_selected_mean_rmsle:.6f} | "
            f"{row.lsqr_selected_mean_rmsle:.6f} | "
            f"{row.selected_mean_rmsle_change:+.2e} |"
        )
    (PROTOCOL / "NUMERICAL_STABILITY_ADDENDUM.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
