from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_utils import (
    DATA,
    OUT,
    deterministic_gzip_options,
    get_cv_masks,
    load_config,
    load_development_samples,
    regression_metrics,
    sha256,
    stable_task_seed,
)


IDENTITY = [
    "sample_id",
    "location_id",
    "location_name",
    "origin_year",
    "target_year",
    "horizon",
    "sdi_quintile_2023_assignment_only",
    "partition",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_id(params: dict[str, Any]) -> str:
    return hashlib.sha256(f"gru|{canonical_json(params)}".encode("utf-8")).hexdigest()[:12]


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def candidate_grid(config: dict[str, Any], seed: int, smoke: bool) -> list[dict[str, Any]]:
    spec = config["sequence_model"]
    keys = [
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
    ]
    grid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for values in product(*(spec[key] for key in keys)):
        params = dict(zip(keys, values))
        if int(params["num_layers"]) == 1:
            params["dropout"] = 0.0
        key = canonical_json(params)
        if key not in seen:
            seen.add(key)
            grid.append(params)
    rng = np.random.default_rng(seed)
    limit = 1 if smoke else int(spec["max_random_configs"])
    if len(grid) > limit:
        chosen = np.sort(rng.choice(len(grid), size=limit, replace=False))
        grid = [grid[int(index)] for index in chosen]
    return grid


def build_sequence_tensor(
    samples: pd.DataFrame,
    panel: pd.DataFrame,
    signals: list[str],
    sequence_length: int,
) -> np.ndarray:
    by_location = {
        int(location_id): group.set_index("year").sort_index()
        for location_id, group in panel.groupby("location_id", sort=False)
    }
    output = np.empty((len(samples), sequence_length, len(signals)), dtype=np.float32)
    for row_index, row in enumerate(samples[["location_id", "origin_year"]].itertuples(index=False)):
        start = int(row.origin_year) - sequence_length + 1
        years = list(range(start, int(row.origin_year) + 1))
        values = by_location[int(row.location_id)].reindex(years)[signals].to_numpy(dtype=float)
        if values.shape != (sequence_length, len(signals)) or not np.isfinite(values).all():
            raise ValueError(
                f"Incomplete sequence for location_id={row.location_id}, origin={row.origin_year}"
            )
        # All seven prespecified series are non-negative. A log1p transform
        # reduces cross-country skew before fold-specific channel scaling.
        output[row_index] = np.log1p(values).astype(np.float32)
    return output


def x_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def apply_x_scaler(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - mean[None, None, :]) / scale[None, None, :]).astype(np.float32)


def y_scaler(y: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y))
    scale = float(np.std(y, ddof=0))
    return mean, scale if scale >= 1e-8 else 1.0


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)).reshape(-1, 1),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def validation_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    model.eval()
    loader = make_loader(x, y, batch_size, False, 0)
    losses: list[float] = []
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for xb, yb in loader:
            losses.append(float(criterion(model(xb), yb).item()))
    return float(sum(losses) / len(y))


def fit_with_early_stopping(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
    input_size: int,
    seed: int,
    max_epochs: int,
    patience: int,
) -> int:
    set_deterministic(seed)
    model = GRURegressor(
        input_size,
        int(params["hidden_size"]),
        int(params["num_layers"]),
        float(params["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.MSELoss()
    loader = make_loader(x_train, y_train, int(params["batch_size"]), True, seed)
    best_loss = math.inf
    best_epoch = 1
    wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        current = validation_loss(model, x_valid, y_valid, int(params["batch_size"]))
        if current < best_loss - 1e-6:
            best_loss = current
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return int(best_epoch)


def fit_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    input_size: int,
    seed: int,
    epochs: int,
) -> nn.Module:
    set_deterministic(seed)
    model = GRURegressor(
        input_size,
        int(params["hidden_size"]),
        int(params["num_layers"]),
        float(params["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.MSELoss()
    loader = make_loader(x_train, y_train, int(params["batch_size"]), True, seed)
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


def predict_network(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    dummy = np.zeros(len(x), dtype=np.float32)
    loader = make_loader(x, dummy, batch_size, False, 0)
    values: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            values.append(model(xb).numpy().reshape(-1))
    return np.concatenate(values)


def outer_fold_prediction(
    x: np.ndarray,
    target_delta: np.ndarray,
    base_log: np.ndarray,
    target_year: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    params: dict[str, Any],
    signal_count: int,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, int]:
    outer_max = int(target_year[train_mask].max())
    inner_valid = train_mask & (target_year >= outer_max - 1)
    inner_train = train_mask & (target_year < outer_max - 1)
    if inner_train.sum() == 0 or inner_valid.sum() == 0:
        raise ValueError("GRU inner temporal early-stopping split is empty")

    inner_x_mean, inner_x_scale = x_scaler(x[inner_train])
    inner_y_mean, inner_y_scale = y_scaler(target_delta[inner_train])
    best_epoch = fit_with_early_stopping(
        apply_x_scaler(x[inner_train], inner_x_mean, inner_x_scale),
        ((target_delta[inner_train] - inner_y_mean) / inner_y_scale).astype(np.float32),
        apply_x_scaler(x[inner_valid], inner_x_mean, inner_x_scale),
        ((target_delta[inner_valid] - inner_y_mean) / inner_y_scale).astype(np.float32),
        params,
        signal_count,
        seed,
        max_epochs,
        patience,
    )

    outer_x_mean, outer_x_scale = x_scaler(x[train_mask])
    outer_y_mean, outer_y_scale = y_scaler(target_delta[train_mask])
    model = fit_fixed_epochs(
        apply_x_scaler(x[train_mask], outer_x_mean, outer_x_scale),
        ((target_delta[train_mask] - outer_y_mean) / outer_y_scale).astype(np.float32),
        params,
        signal_count,
        seed,
        best_epoch,
    )
    standardized = predict_network(
        model,
        apply_x_scaler(x[valid_mask], outer_x_mean, outer_x_scale),
        int(params["batch_size"]),
    )
    predicted_delta = standardized * outer_y_scale + outer_y_mean
    prediction = np.maximum(np.expm1(base_log[valid_mask] + predicted_delta), 0.0)
    return prediction, best_epoch


def run_task(
    frame: pd.DataFrame,
    x: np.ndarray,
    outcome: str,
    horizon: int,
    target_column: str,
    signals: list[str],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame[target_column].to_numpy(dtype=float)
    target_channel = signals.index(f"bmi_{outcome}_rate")
    base_log = x[:, -1, target_channel].astype(float)
    target_delta = np.log1p(y) - base_log
    target_year = frame["target_year"].to_numpy(dtype=int)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    model_spec = config["sequence_model"]
    task_seed = stable_task_seed(config["master_seed"], outcome, horizon, "gru")

    for candidate_index, params in enumerate(candidates):
        cid = config_id(params)
        print(
            f"[run]  {outcome}_h{horizon}: gru {candidate_index + 1}/{len(candidates)} {cid}",
            flush=True,
        )
        for fold_index, fold in enumerate(config["cross_validation"]):
            train_mask, valid_mask = get_cv_masks(frame, fold)
            prediction, best_epoch = outer_fold_prediction(
                x,
                target_delta,
                base_log,
                target_year,
                train_mask,
                valid_mask,
                params,
                len(signals),
                task_seed + candidate_index * 10 + fold_index,
                int(model_spec["max_epochs"]),
                int(model_spec["early_stopping_patience"]),
            )
            metrics_rows.append(
                {
                    "outcome": outcome,
                    "horizon": horizon,
                    "family": "gru",
                    "config_id": cid,
                    "candidate_index": candidate_index,
                    "params_json": canonical_json(params),
                    "fold": fold["fold"],
                    "n_train": int(train_mask.sum()),
                    "n_validation": int(valid_mask.sum()),
                    "selected_epochs": best_epoch,
                    **regression_metrics(y[valid_mask], prediction),
                }
            )
            part = frame.loc[valid_mask, IDENTITY].copy()
            part["outcome"] = outcome
            part["family"] = "gru"
            part["fold"] = fold["fold"]
            part["observed"] = y[valid_mask]
            part["prediction"] = prediction
            part["config_id"] = cid
            part["candidate_index"] = candidate_index
            part["selected_epochs"] = best_epoch
            part["log1p_residual"] = np.log1p(part["observed"]) - np.log1p(part["prediction"])
            prediction_rows.append(part)
    return pd.DataFrame(metrics_rows), pd.concat(prediction_rows, ignore_index=True)


def select_config(metrics: pd.DataFrame, tolerance: float) -> pd.Series:
    numeric = [
        "rmsle",
        "mae",
        "rmse",
        "wape",
        "median_absolute_error",
        "r2",
        "spearman",
        "selected_epochs",
    ]
    summary = (
        metrics.groupby(
            ["outcome", "horizon", "family", "config_id", "candidate_index", "params_json"],
            as_index=False,
        )[numeric]
        .mean()
        .rename(columns={name: f"mean_{name}" for name in numeric})
    )
    minimum = float(summary["mean_rmsle"].min())
    eligible = summary.loc[summary["mean_rmsle"].le(minimum * (1.0 + tolerance))].copy()

    def complexity(text: str) -> tuple[Any, ...]:
        params = json.loads(text)
        return (
            int(params["hidden_size"]),
            int(params["num_layers"]),
            float(params["dropout"]),
            -float(params["weight_decay"]),
            -int(params["batch_size"]),
        )

    eligible["_complexity"] = eligible["params_json"].map(complexity)
    eligible = eligible.sort_values(["_complexity", "mean_rmsle", "config_id"], kind="mergesort")
    selected = eligible.iloc[0].copy()
    selected["complexity_key_json"] = canonical_json(list(selected.pop("_complexity")))
    selected["minimum_mean_rmsle"] = minimum
    selected["selection_tolerance"] = tolerance
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested temporal CV for compact GRU")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--outcome", choices=["daly", "death"])
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5])
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    config = load_config()
    model_spec = config["sequence_model"]
    signals = [str(value) for value in model_spec["sequence_signals"]]
    needed = sorted(set(IDENTITY + list(config["outcomes"].values())))
    samples = load_development_samples(needed).reset_index(drop=True)
    if samples["target_year"].max() > 2018:
        raise ValueError("GRU CV received final-test target years")
    panel = pd.read_csv(DATA / "analytic_panel_development_sequence_locked.csv.gz", low_memory=False)
    sequences = build_sequence_tensor(samples, panel, signals, int(model_spec["sequence_length"]))
    if not np.isfinite(sequences).all():
        raise ValueError("Sequence tensor contains non-finite values")

    work = OUT / ("smoke_gru_v2" if args.smoke else "gru_cv_blocks_v2")
    work.mkdir(parents=True, exist_ok=True)
    outcomes = ["daly"] if args.smoke else ([args.outcome] if args.outcome else list(config["outcomes"]))
    horizons = [1] if args.smoke else ([args.horizon] if args.horizon else [int(value) for value in config["horizons"]])

    if not args.aggregate_only:
        for outcome in outcomes:
            for horizon in horizons:
                task = f"{outcome}_h{horizon}"
                metrics_path = work / f"{task}__gru_metrics.csv"
                predictions_path = work / f"{task}__gru_predictions.csv.gz"
                if metrics_path.exists() and predictions_path.exists() and not args.force:
                    print(f"[skip] {task}: completed GRU block", flush=True)
                    continue
                mask = samples["horizon"].eq(horizon).to_numpy()
                frame = samples.loc[mask].reset_index(drop=True)
                task_x = sequences[mask]
                seed = stable_task_seed(config["master_seed"], outcome, horizon, "gru_grid")
                candidates = candidate_grid(config, seed, args.smoke)
                print(f"[task] {task}: {len(frame)} samples, {len(candidates)} configs", flush=True)
                metrics, predictions = run_task(
                    frame,
                    task_x,
                    outcome,
                    horizon,
                    config["outcomes"][outcome],
                    signals,
                    candidates,
                    config,
                )
                metrics.to_csv(metrics_path, index=False, float_format="%.12g")
                predictions.to_csv(
                    predictions_path,
                    index=False,
                    float_format="%.12g",
                    compression=deterministic_gzip_options(),
                )
                print(f"[done] {task}: GRU block", flush=True)

    if args.no_aggregate:
        print("[done] task blocks written; aggregation deferred", flush=True)
        return

    metric_blocks = sorted(work.glob("*__gru_metrics.csv"))
    prediction_blocks = sorted(work.glob("*__gru_predictions.csv.gz"))
    all_metrics = pd.concat([pd.read_csv(path) for path in metric_blocks], ignore_index=True)
    all_predictions = pd.concat([pd.read_csv(path) for path in prediction_blocks], ignore_index=True)
    all_metrics = all_metrics.sort_values(
        ["outcome", "horizon", "candidate_index", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    all_metrics_path = work / "gru_cv_fold_metrics.csv"
    all_metrics.to_csv(all_metrics_path, index=False, float_format="%.12g")

    selected_rows = [
        select_config(group, float(config["selection_tie_relative_tolerance"]))
        for _, group in all_metrics.groupby(["outcome", "horizon"], sort=True)
    ]
    selected = pd.DataFrame(selected_rows).sort_values(["outcome", "horizon"], kind="mergesort")
    selected_path = work / "selected_gru_configurations.csv"
    selected.to_csv(selected_path, index=False, float_format="%.12g")

    key = selected[["outcome", "horizon", "config_id"]].copy()
    chosen_predictions = all_predictions.merge(
        key, on=["outcome", "horizon", "config_id"], how="inner", validate="many_to_one"
    )
    chosen_predictions = chosen_predictions.sort_values(
        ["outcome", "horizon", "fold", "location_id", "target_year"], kind="mergesort"
    ).reset_index(drop=True)
    chosen_path = work / "selected_gru_oof_predictions.csv.gz"
    chosen_predictions.to_csv(
        chosen_path,
        index=False,
        float_format="%.12g",
        compression=deterministic_gzip_options(),
    )

    metadata = {
        "status": "smoke" if args.smoke else "complete",
        "nested_early_stopping": True,
        "sequence_transform": "channelwise log1p followed by fold-train z score",
        "target_parameterization": "log1p future minus log1p current burden",
        "dataset_sha256": sha256(DATA / "supervised_development_locked.csv.gz"),
        "panel_sha256": sha256(DATA / "analytic_panel_development_sequence_locked.csv.gz"),
        "model_config_sha256": sha256(OUT.parent / "02_protocol" / "model_config_locked.json"),
        "implementation_sha256": sha256(Path(__file__)),
        "model_utils_sha256": sha256(Path(__file__).with_name("model_utils.py")),
        "n_metric_rows": int(len(all_metrics)),
        "n_selected_configurations": int(len(selected)),
        "n_selected_oof_predictions": int(len(chosen_predictions)),
        "outputs": {
            "fold_metrics_sha256": sha256(all_metrics_path),
            "selected_configurations_sha256": sha256(selected_path),
            "selected_oof_predictions_sha256": sha256(chosen_path),
        },
    }
    metadata_path = work / "gru_cv_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
