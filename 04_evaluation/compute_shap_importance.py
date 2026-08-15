from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "03_models"
sys.path.insert(0, str(MODELS))
sys.path.insert(0, str(ROOT / "04_evaluation"))

from fit_final_and_evaluate import IDENTITY, task_lock  # noqa: E402
from model_utils import (  # noqa: E402
    deterministic_gzip_options,
    load_config,
    load_feature_names,
    load_final_test_samples,
    sha256,
    stable_task_seed,
)


OUT = ROOT / "04_evaluation"
LOCK_PATH = MODELS / "selection_lock" / "model_selection_lock.json"
FITTED = OUT / "fitted_models"


def feature_family(feature: str) -> str:
    if feature.startswith(("bmi_daly_rate__", "bmi_death_rate__")):
        return "BMI-attributable burden history"
    if feature.startswith(("gbd_daly_fraction__", "gbd_death_fraction__")):
        return "GBD attributable fraction history"
    if feature.startswith(("overall_daly_rate__", "overall_death_rate__")):
        return "overall liver-cancer burden history"
    if feature.startswith("sdi__"):
        return "sociodemographic index history"
    if "current_rel_ui_width" in feature:
        return "uncertainty width"
    if feature.startswith("derived__"):
        return "derived definition/gap features"
    return "other"


def balanced_explanation_sample(frame: pd.DataFrame, max_per_partition: int, seed: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for partition, group in frame.groupby("partition", sort=True):
        n = min(max_per_partition, len(group))
        pieces.append(group.sample(n=n, random_state=seed, replace=False))
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["partition", "location_id", "target_year"], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = load_config()
    features = load_feature_names()
    needed = sorted(set(IDENTITY + features))
    final_test = load_final_test_samples(needed)
    importance_rows: list[dict[str, Any]] = []
    shap_frames: list[pd.DataFrame] = []

    for outcome in config["outcomes"]:
        for horizon in [int(value) for value in config["horizons"]]:
            locked = task_lock(lock, outcome, horizon)
            family = str(locked["reference_tree_family"])
            model_path = FITTED / f"{outcome}_h{horizon}__{family}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model = joblib.load(model_path)
            task = final_test.loc[final_test["horizon"].eq(horizon)].copy()
            seed = stable_task_seed(config["master_seed"], outcome, horizon, "shap")
            explained = balanced_explanation_sample(task, 300, seed)
            x = explained[features].to_numpy(dtype=np.float32)
            explainer = shap.TreeExplainer(model)
            explanation = explainer(x, check_additivity=False)
            values = np.asarray(explanation.values)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.shape != x.shape:
                raise ValueError(
                    f"Unexpected SHAP shape {values.shape} for input {x.shape} ({family})"
                )
            for index, feature in enumerate(features):
                importance_rows.append(
                    {
                        "outcome": outcome,
                        "horizon": horizon,
                        "tree_family": family,
                        "feature": feature,
                        "feature_family": feature_family(feature),
                        "mean_absolute_shap_log1p_target": float(np.mean(np.abs(values[:, index]))),
                        "mean_signed_shap_log1p_target": float(np.mean(values[:, index])),
                        "n_explained": int(len(explained)),
                    }
                )
            shap_frame = explained[IDENTITY].copy()
            shap_frame["outcome"] = outcome
            shap_frame["tree_family"] = family
            shap_frame = pd.concat(
                [
                    shap_frame.reset_index(drop=True),
                    pd.DataFrame(
                        values,
                        columns=[f"shap__{feature}" for feature in features],
                    ),
                ],
                axis=1,
            )
            shap_frames.append(shap_frame)
            print(f"[done] SHAP: {outcome}_h{horizon} {family}", flush=True)

    importance = pd.DataFrame(importance_rows)
    importance["within_task_rank"] = importance.groupby(["outcome", "horizon"])[
        "mean_absolute_shap_log1p_target"
    ].rank(method="min", ascending=False)
    importance = importance.sort_values(
        ["outcome", "horizon", "within_task_rank", "feature"], kind="mergesort"
    ).reset_index(drop=True)
    importance_path = OUT / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False, float_format="%.12g")
    family_importance = (
        importance.groupby(["outcome", "horizon", "tree_family", "feature_family"], as_index=False)[
            "mean_absolute_shap_log1p_target"
        ]
        .sum()
        .rename(columns={"mean_absolute_shap_log1p_target": "sum_mean_absolute_shap"})
    )
    family_importance["proportion_of_total_absolute_shap"] = family_importance[
        "sum_mean_absolute_shap"
    ] / family_importance.groupby(["outcome", "horizon"])["sum_mean_absolute_shap"].transform(
        "sum"
    )
    family_path = OUT / "shap_feature_family_importance.csv"
    family_importance.to_csv(family_path, index=False, float_format="%.12g")
    shap_path = OUT / "shap_values_balanced_test_sample.csv.gz"
    pd.concat(shap_frames, ignore_index=True).to_csv(
        shap_path,
        index=False,
        float_format="%.10g",
        compression=deterministic_gzip_options(),
    )

    audit = {
        "status": "complete",
        "explanation_scale": "model output in log1p target space",
        "explanation_is_descriptive_not_causal": True,
        "sampling": "up to 300 rows per locked final-test partition per task; no target-based sampling",
        "outputs": {
            "feature_importance_sha256": sha256(importance_path),
            "feature_family_importance_sha256": sha256(family_path),
            "shap_values_sha256": sha256(shap_path),
        },
    }
    (OUT / "shap_importance_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
