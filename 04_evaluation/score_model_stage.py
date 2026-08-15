from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data"
MODELS = ROOT / "03_models"
OUT = ROOT / "04_evaluation"
REPORT = OUT / "STAGE3_MODEL_EVALUATION_GATE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(path: Path, root: Path) -> tuple[int, int]:
    frame = pd.read_csv(path)
    valid = 0
    for row in frame.itertuples(index=False):
        artifact = root / str(row.path)
        if artifact.exists() and sha256(artifact) == str(row.sha256):
            valid += 1
    return valid, len(frame)


def main() -> None:
    rubric: list[dict[str, Any]] = []

    def add(item: str, maximum: float, earned: float, evidence: str) -> None:
        rubric.append(
            {
                "item": item,
                "maximum": float(maximum),
                "earned": float(max(0.0, min(maximum, earned))),
                "evidence": evidence,
            }
        )

    partition = load_json(DATA / "model_partition_manifest.json")
    partition_checks = dict(partition["checks"])
    partition_ok = (
        int(partition_checks.pop("sample_id_overlap")) == 0
        and all(bool(value) for value in partition_checks.values())
    )
    add(
        "Physical development/test isolation",
        6,
        6 if partition_ok else 0,
        f"development={partition['development']['rows']}; sealed test={partition['sealed_final_test']['rows']}; checks={partition['checks']}",
    )
    lock_path = MODELS / "selection_lock" / "model_selection_lock.json"
    lock_hash_path = MODELS / "selection_lock" / "model_selection_lock.sha256.txt"
    lock = load_json(lock_path)
    lock_hash_ok = sha256(lock_path) == lock_hash_path.read_text(encoding="ascii").split()[0]
    add(
        "Pre-test family/configuration lock",
        6,
        6 if lock_hash_ok and lock.get("sealed_final_test_not_loaded") else 0,
        f"lock_sha256={sha256(lock_path)}; sealed_final_test_not_loaded={lock.get('sealed_final_test_not_loaded')}",
    )

    comparison = pd.read_csv(MODELS / "selection_lock" / "cv_model_comparison.csv")
    complete_tasks = comparison[["outcome", "horizon"]].drop_duplicates().shape[0] == 6
    family_counts = comparison.groupby(["outcome", "horizon"])["family"].nunique()
    cv_complete = complete_tasks and family_counts.eq(11).all()
    add(
        "Complete CV comparison with strong baselines",
        8,
        8 if cv_complete else 0,
        f"rows={len(comparison)}; tasks={family_counts.size}; families_per_task={sorted(family_counts.unique())}",
    )
    tab_folds = pd.read_csv(MODELS / "cv_search_blocks_v2" / "tabular_cv_fold_metrics.csv")
    xgb = tab_folds.loc[tab_folds["family"].eq("xgboost")]
    gru_meta = load_json(MODELS / "gru_cv_blocks_v2" / "gru_cv_metadata.json")
    nested_ok = (
        not xgb["effective_n_estimators"].isna().any()
        and xgb.groupby(["outcome", "horizon", "config_id"])["fold"].nunique().eq(3).all()
        and gru_meta.get("nested_early_stopping") is True
    )
    add(
        "Nested temporal early stopping and three outer folds",
        7,
        7 if nested_ok else 0,
        f"xgboost_rows={len(xgb)}; xgboost_effective_rounds_complete={not xgb['effective_n_estimators'].isna().any()}; gru_nested={gru_meta.get('nested_early_stopping')}",
    )

    final_audit = load_json(OUT / "final_test_opening_audit.json")
    predictions = pd.read_csv(OUT / "final_test_predictions.csv.gz", low_memory=False)
    metrics = pd.read_csv(OUT / "final_test_metrics.csv")
    prediction_complete = (
        len(predictions) == 67320
        and metrics.groupby(["partition", "outcome", "horizon"])["family"].nunique().eq(11).all()
        and not predictions.duplicated(
            ["partition", "outcome", "horizon", "family", "sample_id"]
        ).any()
    )
    add(
        "Both locked final tests predicted for every task/family",
        7,
        7 if prediction_complete else 0,
        f"prediction_rows={len(predictions)}; metric_rows={len(metrics)}",
    )
    no_reselection = (
        final_audit.get("model_family_selection_changed_after_test") is False
        and final_audit.get("preprocessing_or_early_stopping_used_test_targets") is False
        and final_audit.get("selection_lock_sha256") == sha256(lock_path)
    )
    add(
        "No post-test reselection or target-informed preprocessing",
        5,
        5 if no_reselection else 0,
        str({key: final_audit.get(key) for key in ["model_family_selection_changed_after_test", "preprocessing_or_early_stopping_used_test_targets", "selection_lock_sha256"]}),
    )

    selected = metrics.loc[metrics["selected_by_cv"]].copy()
    persistence = metrics.loc[metrics["family"].eq("persistence"), [
        "partition",
        "outcome",
        "horizon",
        "rmsle",
    ]].rename(columns={"rmsle": "persistence_rmsle"})
    selected = selected.merge(
        persistence,
        on=["partition", "outcome", "horizon"],
        how="left",
        validate="one_to_one",
    )
    selected["improves_persistence"] = selected["rmsle"] < selected["persistence_rmsle"]
    improved_count = int(selected["improves_persistence"].sum())
    add(
        "Selected-model improvement over persistence across final tasks",
        10,
        10 * improved_count / 12,
        f"improved={improved_count}/12 task-partitions",
    )
    primary = selected.loc[selected["outcome"].eq("daly") & selected["horizon"].eq(5)]
    primary_ok = len(primary) == 2 and primary["improves_persistence"].all()
    add(
        "Primary DALY five-year improvement in both test partitions",
        5,
        5 if primary_ok else 0,
        primary[["partition", "family", "rmsle", "persistence_rmsle", "improves_persistence"]].to_dict("records"),
    )

    bootstrap = pd.read_csv(OUT / "country_cluster_bootstrap.csv")
    bootstrap_ok = (
        bootstrap["replicates"].eq(2000).all()
        and bootstrap[["ci_lower", "ci_upper", "estimate"]].notna().all().all()
        and bootstrap.groupby(["partition", "outcome", "horizon"])["estimand"].nunique().eq(5).all()
    )
    add(
        "Country-clustered bootstrap for paired model/reference differences",
        10,
        10 if bootstrap_ok else 0,
        f"rows={len(bootstrap)}; replicates={sorted(bootstrap['replicates'].unique())}",
    )

    interval_ok = metrics[["coverage_90", "coverage_95", "mean_width_90", "mean_width_95"]].notna().all().all()
    add(
        "90% and 95% predictive interval coverage and width",
        5,
        5 if interval_ok else 0,
        f"coverage90_range=({metrics['coverage_90'].min():.3f}, {metrics['coverage_90'].max():.3f}); coverage95_range=({metrics['coverage_95'].min():.3f}, {metrics['coverage_95'].max():.3f})",
    )
    subgroup = pd.read_csv(OUT / "subgroup_performance_and_coverage.csv")
    dimensions = set(subgroup["subgroup_dimension"])
    subgroup_ok = {
        "overall",
        "sdi_quintile",
        "target_year",
        "observed_burden_quartile",
        "top_1pct_sensitivity",
    }.issubset(dimensions)
    add(
        "Prespecified subgroup coverage and extreme-burden sensitivity",
        5,
        5 if subgroup_ok else 0,
        f"dimensions={sorted(dimensions)}; rows={len(subgroup)}",
    )

    stability_audit = load_json(OUT / "five_seed_stability_audit.json")
    stability_metrics = pd.read_csv(OUT / "five_seed_stability_metrics.csv")
    stability_ok = (
        set(stability_metrics["seed"]) == set(partition_seed for partition_seed in [20260810, 20260811, 20260812, 20260813, 20260814])
        and stability_metrics.groupby(["partition", "outcome", "horizon", "family"])["seed"].nunique().eq(5).all()
        and stability_audit.get("fitted_objects") == 60
    )
    add(
        "Five-seed tree and GRU stability with saved objects",
        8,
        8 if stability_ok else 0,
        f"metric_rows={len(stability_metrics)}; fitted_objects={stability_audit.get('fitted_objects')}",
    )

    robustness_files = {
        "feature ablation": OUT / "feature_ablation_metrics.csv",
        "fraction-definition sensitivity": OUT / "fraction_definition_sensitivity_metrics.csv",
        "distribution shift": OUT / "distribution_shift_diagnostics.csv",
        "SHAP": OUT / "shap_feature_importance.csv",
    }
    present = {name: path.exists() and path.stat().st_size > 0 for name, path in robustness_files.items()}
    add(
        "Ablation, definition, shift, and model-explanation analyses",
        8,
        2 * sum(present.values()),
        str(present),
    )

    final_valid, final_total = verify_manifest(OUT / "fitted_model_manifest.csv", ROOT)
    stability_valid, stability_total = verify_manifest(OUT / "stability_model_manifest.csv", ROOT)
    ablation_valid, ablation_total = verify_manifest(OUT / "ablation_model_manifest.csv", ROOT)
    fraction_valid, fraction_total = verify_manifest(
        OUT / "fraction_sensitivity_model_manifest.csv", ROOT
    )
    all_valid = (
        final_valid == final_total == 48
        and stability_valid == stability_total == 60
        and ablation_valid == ablation_total == 30
        and fraction_valid == fraction_total == 12
    )
    add(
        "Fitted-object preservation and hash verification",
        10,
        10 if all_valid else 0,
        f"final={final_valid}/{final_total}; stability={stability_valid}/{stability_total}; ablation={ablation_valid}/{ablation_total}; fraction={fraction_valid}/{fraction_total}",
    )

    score = sum(item["earned"] for item in rubric)
    maximum = sum(item["maximum"] for item in rubric)
    status = "PASS" if score >= 90 else "FAIL"
    performance_lines = [
        "| Partition | Outcome | Horizon | Selected family | Selected RMSLE | Persistence RMSLE | Improved |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in selected.sort_values(["partition", "outcome", "horizon"]).itertuples(index=False):
        performance_lines.append(
            f"| {row.partition} | {row.outcome} | {int(row.horizon)} | {row.family} | "
            f"{row.rmsle:.5f} | {row.persistence_rmsle:.5f} | {bool(row.improves_persistence)} |"
        )
    lines = [
        "# Stage 3 model and evaluation gate",
        "",
        f"**Score: {score:.1f}/{maximum:.0f} — {status} (required ≥90)**",
        "",
        "This score is generated from artifact counts, hashes, locked selections, and recorded metrics. It is a workflow-quality gate, not a probability of journal acceptance.",
        "",
        "| Criterion | Earned | Maximum | Evidence |",
        "|---|---:|---:|---|",
    ]
    for item in rubric:
        evidence = str(item["evidence"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['item']} | {item['earned']:.1f} | {item['maximum']:.1f} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Locked final-test performance versus persistence",
            "",
            *performance_lines,
            "",
            "## Gate decision",
            "",
            (
                "The figure stage is authorized because the objective model/evaluation score is at least 90."
                if status == "PASS"
                else "The figure stage is blocked. Address failed criteria without changing the sealed-test selection lock."
            ),
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"score": score, "maximum": maximum, "status": status, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
