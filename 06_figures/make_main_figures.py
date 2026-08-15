from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from figure_style import (
    FAMILY_COLORS,
    OUTCOME_COLORS,
    PALETTE,
    PARTITION_COLORS,
    clean_axis,
    family_label,
    panel_label,
    save_figure,
    sha256,
    setup_style,
    short_feature,
    task_label,
    write_csv,
)

setup_style()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import patches as mpatches  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REBUILD = ROOT
DATA = REBUILD / "01_data"
PROTOCOL = REBUILD / "02_protocol"
MODELS = REBUILD / "03_models"
EVAL = REBUILD / "04_evaluation"
FIGDIR = REBUILD / "06_figures"
PDFDIR = FIGDIR / "pdf"
FIGDATA = FIGDIR / "figure_data"
ASSETS = FIGDIR / "assets"

PARTITION_LABEL = {
    "test_temporal_seen_country": "Seen countries",
    "test_spatiotemporal_unseen_country": "Unseen countries",
}
OUTCOME_LABEL = {"daly": "DALY", "death": "Death"}
TASK_ORDER = [(outcome, horizon) for outcome in ["daly", "death"] for horizon in [1, 3, 5]]
FAMILY_ORDER = [
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
    "gru",
]


def load_inputs() -> dict[str, Any]:
    paths = {
        "descriptive": DATA / "descriptive_global_sdi_panel.csv.gz",
        "panel": DATA / "analytic_panel_204x34.csv.gz",
        "development": DATA / "supervised_development_locked.csv.gz",
        "sealed": DATA / "supervised_final_tests_sealed.csv.gz",
        "cv": MODELS / "selection_lock" / "cv_model_comparison.csv",
        "cv_fold": MODELS / "selection_lock" / "cv_fold_metrics_selected_models.csv",
        "metrics": EVAL / "final_test_metrics.csv",
        "predictions": EVAL / "final_test_predictions.csv.gz",
        "bootstrap": EVAL / "country_cluster_bootstrap.csv",
        "subgroup": EVAL / "subgroup_performance_and_coverage.csv",
        "residual": EVAL / "residual_calibration_diagnostics.csv",
        "shift": EVAL / "distribution_shift_diagnostics.csv",
        "stability": EVAL / "five_seed_stability_summary.csv",
        "ablation": EVAL / "feature_ablation_metrics.csv",
        "fraction": EVAL / "fraction_definition_sensitivity_metrics.csv",
        "shap_feature": EVAL / "shap_feature_importance.csv",
        "shap_family": EVAL / "shap_feature_family_importance.csv",
        "shap_values": EVAL / "shap_values_balanced_test_sample.csv.gz",
    }
    output = {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}
    output["lock"] = json.loads(
        (MODELS / "selection_lock" / "model_selection_lock.json").read_text(encoding="utf-8")
    )
    output["config"] = json.loads(
        (PROTOCOL / "model_config_locked.json").read_text(encoding="utf-8")
    )
    output["partition_manifest"] = json.loads(
        (DATA / "model_partition_manifest.json").read_text(encoding="utf-8")
    )
    output["world"] = json.loads(
        (ASSETS / "ne_50m_admin_0_countries.geojson").read_text(encoding="utf-8")
    )
    return output


def use_plain_log_tick_labels(ax, axes: str = "both") -> None:
    """Keep logarithmic scales while avoiding math-font tick-label fallback."""
    formatter = FuncFormatter(lambda value, _position: f"{value:g}")
    if axes in {"x", "both"}:
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())
    if axes in {"y", "both"}:
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())


def vector_heatmap(ax, values, **kwargs):
    """Draw cell-based heatmaps as vector polygons rather than embedded images."""
    data = np.asarray(values)
    rows, columns = data.shape
    kwargs.pop("aspect", None)
    mesh = ax.pcolormesh(
        np.arange(columns + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        data,
        shading="flat",
        rasterized=False,
        **kwargs,
    )
    ax.set_xlim(-0.5, columns - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("auto")
    return mesh


def vector_colorbar(fig, mappable, **kwargs):
    """Prevent Matplotlib from rasterising continuous colour-bar solids."""
    colorbar = fig.colorbar(mappable, **kwargs)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    return colorbar


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=4.5)


def add_figure_title(fig, text: str) -> None:
    """Keep figure identification in the legend, not inside the exported art."""
    return None


def draw_box(ax, xy, width, height, text, facecolor, edgecolor=PALETTE["light"], fontsize=6.5):
    box = mpatches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.7,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def arrow(ax, start, end, color=PALETTE["mid"], connectionstyle="arc3,rad=0"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.75,
            color=color,
            shrinkA=1,
            shrinkB=1,
            connectionstyle=connectionstyle,
        ),
    )


def connector(ax, start, end, color=PALETTE["mid"]):
    """Draw a branch or merge segment without stacking arrowheads at a hub."""
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=0.75,
        solid_capstyle="round",
        clip_on=False,
    )


def locked_family_map(lock: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {(row["outcome"], int(row["horizon"])): row["selected_family"] for row in lock["tasks"]}


def reference_family_map(lock: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (row["outcome"], int(row["horizon"])): row["reference_transparent_family"]
        for row in lock["tasks"]
    }


def task_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["task"] = [task_label(o, h) for o, h in zip(out["outcome"], out["horizon"])]
    out["task_order"] = [TASK_ORDER.index((o, int(h))) for o, h in zip(out["outcome"], out["horizon"])]
    return out


def country_rmsle_table(predictions: pd.DataFrame, partition: str, horizon: int) -> pd.DataFrame:
    local = predictions.loc[
        predictions["partition"].eq(partition)
        & predictions["horizon"].eq(horizon)
        & (predictions["selected_by_cv"] | predictions["family"].isin(["persistence", "pooled_ridge"]))
    ].copy()
    local["sq"] = (
        np.log1p(local["prediction"].to_numpy(dtype=float))
        - np.log1p(local["observed"].to_numpy(dtype=float))
    ) ** 2
    result = (
        local.groupby(
            [
                "outcome",
                "location_id",
                "location_name",
                "sdi_quintile_2023_assignment_only",
                "family",
            ],
            as_index=False,
        )["sq"]
        .mean()
        .assign(rmsle=lambda frame: np.sqrt(frame["sq"]))
    )
    wide = result.pivot_table(
        index=["outcome", "location_id", "location_name", "sdi_quintile_2023_assignment_only"],
        columns="family",
        values="rmsle",
    ).reset_index()
    selected = local.loc[local["selected_by_cv"], ["outcome", "family"]].drop_duplicates()
    selected = selected.rename(columns={"family": "selected_family"})
    wide = wide.merge(selected, on="outcome", how="left", validate="many_to_one")
    selected_values = []
    for row in wide.itertuples(index=False):
        selected_values.append(float(getattr(row, str(row.selected_family))))
    wide["selected_rmsle"] = selected_values
    wide["delta_vs_persistence"] = wide["selected_rmsle"] - wide["persistence"]
    wide["delta_vs_pooled"] = wide["selected_rmsle"] - wide["pooled_ridge"]
    return wide


GBD_TO_NE = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Democratic People's Republic of Korea": "North Korea",
    "Iran (Islamic Republic of)": "Iran",
    "Lao People's Democratic Republic": "Laos",
    "Micronesia (Federated States of)": "Federated States of Micronesia",
    "Republic of Moldova": "Moldova",
    "Sao Tome and Principe": "São Tomé and Principe",
    "Syrian Arab Republic": "Syria",
    "Türkiye": "Turkey",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam",
}


def feature_admin_name(feature: dict[str, Any]) -> str:
    properties = feature["properties"]
    return str(properties.get("ADMIN") or properties.get("NAME") or "")


def geometry_polygons(geometry: dict[str, Any]):
    if geometry is None:
        return
    if geometry["type"] == "Polygon":
        for polygon in [geometry["coordinates"]]:
            if polygon:
                yield polygon[0]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            if polygon:
                yield polygon[0]


def plot_world_map(ax, world: dict[str, Any], values: dict[str, float], cmap, norm, title_text: str):
    patches = []
    facevalues = []
    for feature in world["features"]:
        admin = feature_admin_name(feature)
        value = values.get(admin, np.nan)
        for ring in geometry_polygons(feature.get("geometry")):
            coords = np.asarray(ring, dtype=float)
            if len(coords) < 3:
                continue
            if np.ptp(coords[:, 0]) > 300:
                continue
            patches.append(mpatches.Polygon(coords, closed=True))
            facevalues.append(value)
    collection = PatchCollection(
        patches,
        linewidth=0.22,
        edgecolor="#B7B7B7",
        cmap=cmap,
        norm=norm,
    )
    collection.set_array(np.asarray(facevalues, dtype=float))
    collection.set_facecolor(
        [cmap(norm(v)) if np.isfinite(v) else "#EFEFEF" for v in facevalues]
    )
    ax.add_collection(collection)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 88)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    title(ax, title_text)
    return collection


def _figure1_dense_legacy(inputs: dict[str, Any]) -> None:
    metrics = inputs["metrics"]
    selected = metrics.loc[metrics["selected_by_cv"]].copy()
    pooled = metrics.loc[
        metrics["family"].eq("pooled_ridge"),
        ["partition", "outcome", "horizon", "rmsle"],
    ].rename(columns={"rmsle": "pooled_ridge_rmsle"})
    sealed = selected.merge(
        pooled,
        on=["partition", "outcome", "horizon"],
        how="inner",
        validate="one_to_one",
    )
    sealed["skill_vs_pooled_pct"] = 100 * (1 - sealed["rmsle"] / sealed["pooled_ridge_rmsle"])
    if len(sealed) != 12:
        raise RuntimeError(f"Expected 12 sealed locked-model comparisons, found {len(sealed)}")

    locked = sealed[["outcome", "horizon", "family"]].drop_duplicates()
    if len(locked) != 6 or locked.groupby(["outcome", "horizon"])["family"].nunique().max() != 1:
        raise RuntimeError("Locked family is not unique for every outcome-horizon task")
    locked_family = {
        (row.outcome, int(row.horizon)): row.family for row in locked.itertuples(index=False)
    }

    partition_order = ["test_temporal_seen_country", "test_spatiotemporal_unseen_country"]
    skill_rows = [
        (partition, outcome)
        for partition in partition_order
        for outcome in ["daly", "death"]
    ]
    skill_matrix = np.asarray(
        [
            [
                sealed.loc[
                    sealed["partition"].eq(partition)
                    & sealed["outcome"].eq(outcome)
                    & sealed["horizon"].eq(horizon),
                    "skill_vs_pooled_pct",
                ].iloc[0]
                for horizon in [1, 3, 5]
            ]
            for partition, outcome in skill_rows
        ]
    )
    positive_comparisons = int((skill_matrix > 0).sum())
    interval_estimates = int(len(sealed) * 2)
    undercovered = int((sealed["coverage_90"] < 0.90).sum() + (sealed["coverage_95"] < 0.95).sum())

    coverage_summary = []
    for partition in partition_order:
        subset = sealed.loc[sealed["partition"].eq(partition)]
        for level, column, nominal in [(90, "coverage_90", 0.90), (95, "coverage_95", 0.95)]:
            values = subset[column]
            coverage_summary.append(
                {
                    "partition": partition,
                    "level": level,
                    "minimum": float(values.min()),
                    "median": float(values.median()),
                    "maximum": float(values.max()),
                    "nominal": nominal,
                }
            )
    coverage_summary = pd.DataFrame(coverage_summary)

    fig = plt.figure(figsize=(7.2, 6.70))
    add_figure_title(fig, "Figure 1 | Overall model development and geographic validation framework")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")

    panel_x = [0.020, 0.347, 0.674]
    panel_width = 0.306
    panel_bottom = 0.095
    panel_height = 0.855
    contained_text: list[tuple[Any, list[Any], str, float]] = []
    panel_patches: list[Any] = []

    def panel_frame(x: float, number: int, heading: str, accent: str) -> None:
        frame = mpatches.Rectangle(
            (x, panel_bottom),
            panel_width,
            panel_height,
            facecolor="#FCFCFC",
            edgecolor="#D2D2D2",
            linewidth=0.75,
        )
        canvas.add_patch(frame)
        panel_patches.append(frame)
        canvas.add_patch(
            mpatches.Rectangle(
                (x, panel_bottom + panel_height - 0.010),
                panel_width,
                0.010,
                facecolor=accent,
                edgecolor="none",
            )
        )
        canvas.text(
            x + 0.024,
            0.908,
            str(number),
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.23", facecolor=accent, edgecolor="none"),
        )
        canvas.text(
            x + 0.050,
            0.908,
            heading,
            ha="left",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color=PALETTE["dark"],
        )

    def section_label(x: float, y: float, text: str, color: str = PALETTE["mid"]) -> None:
        canvas.text(
            x,
            y,
            text.upper(),
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color=color,
        )

    panel_frame(panel_x[0], 1, "Data & physical isolation", PALETTE["navy"])
    panel_frame(panel_x[1], 2, "Development & immutable lock", PALETTE["teal"])
    panel_frame(panel_x[2], 3, "Sealed evaluation & finding", PALETTE["red"])

    for start, end in [(panel_x[0] + panel_width, panel_x[1]), (panel_x[1] + panel_width, panel_x[2])]:
        canvas.annotate(
            "",
            xy=(end - 0.004, 0.515),
            xytext=(start + 0.004, 0.515),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["mid"], lw=0.8, mutation_scale=8),
        )

    # Column 1: compact source description, target-label timeline and country split.
    x = panel_x[0]
    section_label(x + 0.027, 0.856, "GBD 2023 substrate", PALETTE["navy"])
    canvas.text(
        x + 0.027,
        0.817,
        "BMI-attributable + overall liver-cancer\nDALY/death ASRs • fractions/UI • SDI",
        ha="left",
        va="top",
        fontsize=7.25,
        linespacing=1.35,
        color=PALETTE["dark"],
    )
    canvas.text(
        x + 0.027,
        0.756,
        "204 countries • 34 years\n6,936 country-years",
        ha="left",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        linespacing=1.25,
        color=PALETTE["dark"],
    )

    timeline_ax = fig.add_axes([x + 0.030, 0.604, panel_width - 0.060, 0.090])
    timeline_ax.broken_barh([(1990, 29)], (-0.27, 0.54), facecolors="#DCE8F2", edgecolors="none")
    timeline_ax.broken_barh([(2019, 5)], (-0.27, 0.54), facecolors="#F6D9D4", edgecolors="none")
    timeline_ax.axvline(2018.5, color=PALETTE["dark"], lw=0.65)
    timeline_ax.text(2003.8, 0, "history", ha="center", va="center", fontsize=6.4, color=PALETTE["navy"])
    timeline_ax.text(2021.2, 0, "sealed", ha="center", va="center", fontsize=6.1, color=PALETTE["red"])
    timeline_ax.set_xlim(1990, 2024)
    timeline_ax.set_ylim(-0.34, 0.34)
    timeline_ax.set_yticks([])
    timeline_ax.set_xticks([1990, 2010, 2018, 2023])
    timeline_ax.tick_params(axis="x", labelsize=6.4, length=2.2, pad=1.5)
    timeline_ax.set_title("Target-label timeline", loc="left", fontsize=7.3, pad=2.5)
    for spine in timeline_ax.spines.values():
        spine.set_visible(False)

    sealed_timeline_note = canvas.text(
        x + 0.027,
        0.574,
        "2019–2023 target labels sealed",
        ha="left",
        va="center",
        fontsize=6.4,
        color=PALETTE["red"],
    )
    contained_text.append((panel_patches[0], [sealed_timeline_note], "sealed target-label note", 5.0))

    split_ax = fig.add_axes([x + 0.030, 0.461, panel_width - 0.060, 0.082])
    split_ax.barh([0], [163], color="#8FBAD5", height=0.46)
    split_ax.barh([0], [41], left=[163], color="#E7A098", height=0.46)
    split_ax.text(81.5, 0, "163", ha="center", va="center", fontsize=7.0, fontweight="bold")
    split_ax.text(183.5, 0, "41", ha="center", va="center", fontsize=7.0, fontweight="bold")
    split_ax.set_xlim(0, 204)
    split_ax.set_ylim(-0.40, 0.40)
    split_ax.set_xticks([])
    split_ax.set_yticks([])
    split_ax.set_title("Country-level physical split", loc="left", fontsize=7.3, pad=2.5)
    for spine in split_ax.spines.values():
        spine.set_visible(False)
    canvas.scatter([x + 0.035, x + 0.174], [0.438, 0.438], s=14, color=["#8FBAD5", "#E7A098"], clip_on=False)
    canvas.text(x + 0.048, 0.438, "Development", va="center", fontsize=6.55)
    canvas.text(x + 0.187, 0.438, "Unseen holdout", va="center", fontsize=6.55)

    canvas.add_patch(mpatches.Rectangle((x + 0.027, 0.345), 0.004, 0.052, facecolor=PALETTE["navy"], edgecolor="none"))
    canvas.text(
        x + 0.041,
        0.371,
        "Seen-country temporal test\n815 samples per horizon",
        ha="left",
        va="center",
        fontsize=6.9,
        linespacing=1.25,
    )
    canvas.add_patch(mpatches.Rectangle((x + 0.027, 0.273), 0.004, 0.052, facecolor=PALETTE["red"], edgecolor="none"))
    canvas.text(
        x + 0.041,
        0.299,
        "41 countries never enter fitting\n205 samples per horizon",
        ha="left",
        va="center",
        fontsize=6.9,
        linespacing=1.25,
    )
    section_label(x + 0.027, 0.222, "Primary targets", PALETTE["navy"])
    canvas.text(
        x + 0.027,
        0.183,
        "BMI-attributable liver-cancer\nDALY and death ASRs",
        ha="left",
        va="center",
        fontsize=7.25,
        fontweight="bold",
        linespacing=1.3,
    )

    # Column 2: slim representation lanes and the actual six-task immutable lock.
    x = panel_x[1]
    section_label(x + 0.027, 0.856, "History-only representations", PALETTE["teal"])

    def representation_lane(y: float, label: str, detail: str, color: str) -> None:
        lane = mpatches.Rectangle(
            (x + 0.027, y),
            panel_width - 0.054,
            0.061,
            facecolor=mcolors.to_rgba(color, 0.10),
            edgecolor="#DDDDDD",
            linewidth=0.55,
        )
        canvas.add_patch(lane)
        canvas.add_patch(mpatches.Rectangle((x + 0.027, y), 0.006, 0.061, facecolor=color, edgecolor="none"))
        label_text = canvas.text(
            x + 0.043,
            y + 0.041,
            label,
            ha="left",
            va="center",
            fontsize=6.9,
            fontweight="bold",
        )
        detail_text = canvas.text(
            x + 0.043,
            y + 0.017,
            detail,
            ha="left",
            va="center",
            fontsize=6.2,
            color=PALETTE["mid"],
        )
        contained_text.append((lane, [label_text, detail_text], f"representation lane: {label}", 4.0))

    representation_lane(0.777, "Transparent references", "4 baselines", PALETTE["mid"])
    representation_lane(0.701, "Tabular histories", "179 features • 6 families", PALETTE["teal"])
    representation_lane(0.625, "Ten-year sequences", "7 signals • compact GRU", PALETTE["gold"])
    canvas.annotate(
        "",
        xy=(x + panel_width / 2, 0.581),
        xytext=(x + panel_width / 2, 0.616),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["mid"], lw=0.75, mutation_scale=8),
    )
    fold_text = canvas.text(
        x + panel_width / 2,
        0.555,
        "3 expanding temporal folds • nested stopping",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
    )
    selection_text = canvas.text(
        x + panel_width / 2,
        0.522,
        "RMSLE selection + prespecified 1% simplicity rule",
        ha="center",
        va="center",
        fontsize=6.55,
        color=PALETTE["mid"],
    )
    contained_text.append((panel_patches[1], [fold_text, selection_text], "development selection text", 5.0))

    section_label(x + 0.027, 0.470, "Six task-specific model locks", PALETTE["teal"])
    grid_left = x + 0.076
    cell_width = 0.064
    cell_gap = 0.010
    cell_height = 0.061
    for column, horizon in enumerate([1, 3, 5]):
        centre = grid_left + column * (cell_width + cell_gap) + cell_width / 2
        canvas.text(centre, 0.438, f"{horizon} y", ha="center", va="center", fontsize=6.8, fontweight="bold")
    for row, outcome in enumerate(["daly", "death"]):
        y = 0.360 - row * 0.082
        canvas.text(
            x + 0.027,
            y + cell_height / 2,
            OUTCOME_LABEL[outcome],
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=OUTCOME_COLORS[outcome],
        )
        for column, horizon in enumerate([1, 3, 5]):
            family = locked_family[(outcome, horizon)]
            color = FAMILY_COLORS[family]
            cell_x = grid_left + column * (cell_width + cell_gap)
            cell = mpatches.Rectangle(
                (cell_x, y),
                cell_width,
                cell_height,
                facecolor=mcolors.to_rgba(color, 0.18),
                edgecolor=color,
                linewidth=0.75,
            )
            canvas.add_patch(cell)
            label = "Extra\nTrees" if family == "extra_trees" else family_label(family)
            cell_text = canvas.text(
                cell_x + cell_width / 2,
                y + cell_height / 2,
                label,
                ha="center",
                va="center",
                fontsize=6.35,
                fontweight="bold",
                linespacing=1.05,
            )
            contained_text.append((cell, [cell_text], f"locked model cell: {outcome} {horizon} y", 3.0))
    canvas.add_patch(mpatches.Rectangle((x + 0.027, 0.198), 0.004, 0.052, facecolor=PALETTE["purple"], edgecolor="none"))
    canvas.text(
        x + 0.041,
        0.224,
        "SHA-256 lock frozen before test access",
        ha="left",
        va="center",
        fontsize=6.85,
        fontweight="bold",
    )
    canvas.text(
        x + 0.041,
        0.177,
        "No post-test model reselection",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=PALETTE["red"],
    )
    canvas.text(
        x + panel_width / 2,
        0.143,
        "Bootstrap • calibration • shift\nSeeds • ablation • descriptive SHAP",
        ha="center",
        va="center",
        fontsize=6.15,
        linespacing=1.2,
        color=PALETTE["mid"],
    )

    # Column 3: quantitative sealed-result snapshot, not decorative workflow art.
    x = panel_x[2]
    section_label(x + 0.027, 0.856, "Sealed partitions", PALETTE["red"])
    sample_ax = fig.add_axes([x + 0.060, 0.708, panel_width - 0.092, 0.112])
    sample_counts = [815, 205]
    sample_y = [1, 0]
    sample_colors = [PARTITION_COLORS[name] for name in partition_order]
    sample_ax.barh(sample_y, sample_counts, color=sample_colors, height=0.46)
    for y_value, value in zip(sample_y, sample_counts):
        sample_ax.text(value + 22, y_value, f"{value}", ha="left", va="center", fontsize=6.8, fontweight="bold")
    sample_ax.set_xlim(0, 930)
    sample_ax.set_yticks(sample_y, ["Seen", "Unseen"])
    sample_ax.set_xticks([0, 400, 800])
    sample_ax.tick_params(labelsize=6.3, length=2.2, pad=1.5)
    sample_ax.set_title("Samples per horizon", loc="left", fontsize=7.3, pad=2.5)
    clean_axis(sample_ax, "x")

    section_label(x + 0.027, 0.658, "Locked skill vs pooled ridge (%)", PALETTE["red"])
    skill_ax = fig.add_axes([x + 0.082, 0.482, panel_width - 0.104, 0.145])
    skill_cmap = mcolors.LinearSegmentedColormap.from_list(
        "sealed_skill",
        ["#8E2530", "#D96B5F", "#F7F7F7", "#67A9CF", "#2166AC"],
    )
    skill_norm = mcolors.TwoSlopeNorm(vmin=-170, vcenter=0, vmax=20)
    vector_heatmap(skill_ax, skill_matrix, cmap=skill_cmap, norm=skill_norm)
    for row in range(skill_matrix.shape[0]):
        for column in range(skill_matrix.shape[1]):
            value = skill_matrix[row, column]
            skill_ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.75,
                fontweight="bold" if value <= -45 else "normal",
                color="white" if value <= -45 else PALETTE["dark"],
            )
    skill_ax.set_xticks(range(3), ["1 y", "3 y", "5 y"])
    skill_ax.set_yticks(range(4), ["Seen DALY", "Seen death", "Unseen DALY", "Unseen death"])
    skill_ax.tick_params(axis="both", labelsize=5.9, length=0, pad=1.7)
    for spine in skill_ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#BEBEBE")
    sealed_result_note = canvas.text(
        x + panel_width / 2,
        0.451,
        "Negative values favour pooled ridge.\nSealed results were not used for reselection.",
        ha="center",
        va="center",
        fontsize=6.05,
        linespacing=1.18,
        color=PALETTE["mid"],
    )
    contained_text.append((panel_patches[2], [sealed_result_note], "sealed-result note", 5.0))

    section_label(x + 0.027, 0.407, "Empirical interval coverage", PALETTE["red"])
    coverage_ax = fig.add_axes([x + 0.075, 0.238, panel_width - 0.105, 0.135])
    coverage_rows = [
        ("test_temporal_seen_country", 90, 3, "Seen 90"),
        ("test_spatiotemporal_unseen_country", 90, 2, "Unseen 90"),
        ("test_temporal_seen_country", 95, 1, "Seen 95"),
        ("test_spatiotemporal_unseen_country", 95, 0, "Unseen 95"),
    ]
    for partition, level, y_value, _label in coverage_rows:
        row = coverage_summary.loc[
            coverage_summary["partition"].eq(partition) & coverage_summary["level"].eq(level)
        ].iloc[0]
        color = PARTITION_COLORS[partition]
        coverage_ax.hlines(y_value, row["minimum"], row["maximum"], color=color, lw=1.25)
        coverage_ax.scatter(row["median"], y_value, s=18, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    coverage_ax.vlines(0.90, 1.55, 3.45, color=PALETTE["mid"], ls="--", lw=0.7)
    coverage_ax.vlines(0.95, -0.45, 1.45, color=PALETTE["mid"], ls="--", lw=0.7)
    coverage_ax.set_xlim(0.60, 1.00)
    coverage_ax.set_ylim(-0.55, 3.55)
    coverage_ax.set_xticks([0.60, 0.80, 1.00], ["60", "80", "100"])
    coverage_ax.set_yticks([3, 2, 1, 0], ["Seen 90", "Unseen 90", "Seen 95", "Unseen 95"])
    coverage_ax.set_xlabel("Coverage (%)", labelpad=1.5)
    coverage_ax.tick_params(axis="both", labelsize=5.9, length=2.2, pad=1.5)
    clean_axis(coverage_ax, "x")

    conclusion_box = mpatches.Rectangle(
        (x + 0.027, 0.121),
        panel_width - 0.054,
        0.078,
        facecolor="#F8E8E5",
        edgecolor="#E8C7C2",
        linewidth=0.65,
    )
    canvas.add_patch(conclusion_box)
    canvas.add_patch(mpatches.Rectangle((x + 0.027, 0.121), 0.006, 0.078, facecolor=PALETTE["red"], edgecolor="none"))
    conclusion_primary = canvas.text(
        x + 0.043,
        0.171,
        f"{positive_comparisons}/12 beat pooled ridge",
        ha="left",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#8E2530",
    )
    conclusion_secondary = canvas.text(
        x + 0.043,
        0.142,
        f"{undercovered}/{interval_estimates} intervals undercovered",
        ha="left",
        va="center",
        fontsize=6.35,
        color=PALETTE["dark"],
    )
    contained_text.append(
        (conclusion_box, [conclusion_primary, conclusion_secondary], "sealed-result conclusion", 5.0)
    )

    canvas.text(
        0.5,
        0.043,
        "Outcome scope: aggregate liver cancer attributable to high BMI; not histology-confirmed HCC",
        ha="center",
        va="center",
        fontsize=6.7,
        color=PALETTE["red"],
    )
    # Production guard: fail the build if any protected label crosses its panel or colour block.
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    qa_canvas = FigureCanvasAgg(fig)
    qa_canvas.draw()
    renderer = qa_canvas.get_renderer()
    for container, texts, label, padding in contained_text:
        container_box = container.get_window_extent(renderer)
        for text_artist in texts:
            text_box = text_artist.get_window_extent(renderer)
            inside = (
                text_box.x0 >= container_box.x0 + padding
                and text_box.x1 <= container_box.x1 - padding
                and text_box.y0 >= container_box.y0 + padding
                and text_box.y1 <= container_box.y1 - padding
            )
            if not inside:
                raise RuntimeError(
                    f"Figure 1 text overflow in {label}: {text_artist.get_text()!r}; "
                    f"text={text_box.bounds}, container={container_box.bounds}"
                )

    save_figure(fig, PDFDIR / "Fig1_workflow.pdf", {"Title": "Figure 1 workflow", "Author": "Rebuilt analysis"})
    plt.close(fig)


def figure1(inputs: dict[str, Any]) -> None:
    """Draw a compact, left-to-right overview with details deferred to the legend."""
    metrics = inputs["metrics"]
    selected = metrics.loc[metrics["selected_by_cv"]].copy()
    pooled = metrics.loc[
        metrics["family"].eq("pooled_ridge"),
        ["partition", "outcome", "horizon", "rmsle"],
    ].rename(columns={"rmsle": "pooled_ridge_rmsle"})
    sealed = selected.merge(
        pooled,
        on=["partition", "outcome", "horizon"],
        how="inner",
        validate="one_to_one",
    )
    sealed["skill_vs_pooled_pct"] = 100 * (
        1 - sealed["rmsle"] / sealed["pooled_ridge_rmsle"]
    )
    if len(sealed) != 12:
        raise RuntimeError(f"Expected 12 sealed locked-model comparisons, found {len(sealed)}")

    locked = sealed[["outcome", "horizon", "family"]].drop_duplicates()
    locked_family = {
        (row.outcome, int(row.horizon)): row.family for row in locked.itertuples(index=False)
    }
    if len(locked_family) != 6:
        raise RuntimeError("Expected one locked family for each of six outcome-horizon tasks")
    for horizon in [1, 3, 5]:
        if locked_family[("daly", horizon)] != locked_family[("death", horizon)]:
            raise RuntimeError("Compact Figure 1 lock summary requires matching families across outcomes")

    positive_comparisons = int((sealed["skill_vs_pooled_pct"] > 0).sum())
    interval_estimates = int(len(sealed) * 2)
    undercovered = int(
        (sealed["coverage_90"] < 0.90).sum()
        + (sealed["coverage_95"] < 0.95).sum()
    )

    fig = plt.figure(figsize=(7.2, 3.0))
    add_figure_title(fig, "Figure 1 | Overall model development and geographic validation framework")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set(xlim=(0, 1), ylim=(0, 1))
    canvas.axis("off")

    panel_bottom = 0.105
    panel_height = 0.815
    panel_specs = [
        (0.015, 0.174, "GBD 2023 data", PALETTE["navy"]),
        (0.201, 0.170, "Physical split", PALETTE["blue"]),
        (0.383, 0.200, "History-only models", PALETTE["teal"]),
        (0.595, 0.200, "Immutable lock", PALETTE["gold"]),
        (0.807, 0.178, "Sealed finding", PALETTE["red"]),
    ]
    contained_text: list[tuple[Any, list[Any], str, float]] = []
    frames: list[Any] = []

    def add_panel(x: float, width: float, number: int, heading: str, accent: str):
        frame = mpatches.FancyBboxPatch(
            (x, panel_bottom),
            width,
            panel_height,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=mcolors.to_rgba(accent, 0.055),
            edgecolor=mcolors.to_rgba(accent, 0.32),
            linewidth=0.75,
        )
        canvas.add_patch(frame)
        canvas.add_patch(
            mpatches.Rectangle(
                (x, panel_bottom + panel_height - 0.013),
                width,
                0.013,
                facecolor=accent,
                edgecolor="none",
            )
        )
        canvas.scatter(
            [x + 0.018],
            [0.865],
            s=72,
            color=accent,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
            clip_on=False,
        )
        number_text = canvas.text(
            x + 0.018,
            0.865,
            str(number),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        heading_text = canvas.text(
            x + 0.036,
            0.865,
            heading,
            ha="left",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        contained_text.append((frame, [number_text, heading_text], f"panel {number} header", 3.0))
        frames.append(frame)
        return frame

    def ptext(frame, x, y, text, *, label, padding=3.0, **kwargs):
        artist = canvas.text(x, y, text, **kwargs)
        contained_text.append((frame, [artist], label, padding))
        return artist

    for number, (x, width, heading, accent) in enumerate(panel_specs, start=1):
        add_panel(x, width, number, heading, accent)

    for (x, width, _heading, _accent), (next_x, *_rest) in zip(panel_specs[:-1], panel_specs[1:]):
        canvas.annotate(
            "",
            xy=(next_x - 0.002, 0.515),
            xytext=(x + width + 0.002, 0.515),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#8E989E",
                lw=1.0,
                mutation_scale=9,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=6,
        )

    # 1 | Source data: one compact data glyph and two essential quantities.
    x, width, _heading, accent = panel_specs[0]
    tile_colors = [PALETTE["navy"], PALETTE["cyan"], PALETTE["teal"], PALETTE["purple"], PALETTE["orange"]]
    tile_left = x + 0.040
    for row in range(3):
        for column in range(5):
            color = tile_colors[(row + column) % len(tile_colors)]
            canvas.add_patch(
                mpatches.Rectangle(
                    (tile_left + column * 0.019, 0.718 - row * 0.041),
                    0.012,
                    0.026,
                    facecolor=mcolors.to_rgba(color, 0.82 if (row + column) % 3 else 0.48),
                    edgecolor="none",
                )
            )
    ptext(
        frames[0],
        x + width / 2,
        0.565,
        "204 x 34",
        label="source panel dimensions",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    ptext(
        frames[0],
        x + width / 2,
        0.505,
        "countries x years",
        label="source panel units",
        ha="center",
        va="center",
        fontsize=7.1,
        color=PALETTE["mid"],
    )
    canvas.plot([x + 0.025, x + width - 0.025], [0.450, 0.450], color="#D5D9DC", lw=0.7)
    ptext(
        frames[0],
        x + width / 2,
        0.352,
        "BMI-attributable\nliver-cancer burden",
        label="source outcome",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        linespacing=1.18,
        color=PALETTE["dark"],
    )
    ptext(
        frames[0],
        x + width / 2,
        0.242,
        "DALY + death ASRs",
        label="source outcomes",
        ha="center",
        va="center",
        fontsize=7.0,
        color=PALETTE["navy"],
    )

    # 2 | Physical isolation: time and country axes communicate both locks.
    x, width, _heading, accent = panel_specs[1]
    time_left = x + 0.020
    time_width = width - 0.040
    history_width = time_width * 29 / 34
    canvas.add_patch(
        mpatches.Rectangle((time_left, 0.685), history_width, 0.060, facecolor="#CFE0EC", edgecolor="none")
    )
    canvas.add_patch(
        mpatches.Rectangle(
            (time_left + history_width, 0.685),
            time_width - history_width,
            0.060,
            facecolor="#F3C7C0",
            edgecolor="none",
        )
    )
    canvas.plot(
        [time_left + history_width, time_left + history_width],
        [0.674, 0.755],
        color=PALETTE["dark"],
        lw=0.7,
    )
    for tick_x, tick_text, align in [
        (time_left, "1990", "left"),
        (time_left + history_width, "2018", "center"),
    ]:
        ptext(
            frames[1],
            tick_x,
            0.775,
            tick_text,
            label=f"timeline tick {tick_text}",
            ha=align,
            va="center",
            fontsize=6.6,
            color=PALETTE["mid"],
        )
    ptext(
        frames[1],
        x + width / 2,
        0.635,
        "2019-23 targets sealed",
        label="sealed target note",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["red"],
    )

    split_left = x + 0.020
    split_width = width - 0.040
    development_width = split_width * 163 / 204
    canvas.add_patch(
        mpatches.Rectangle((split_left, 0.465), development_width, 0.105, facecolor="#8FBAD5", edgecolor="none")
    )
    canvas.add_patch(
        mpatches.Rectangle(
            (split_left + development_width, 0.465),
            split_width - development_width,
            0.105,
            facecolor="#E7A098",
            edgecolor="none",
        )
    )
    ptext(
        frames[1],
        split_left + development_width / 2,
        0.518,
        "163",
        label="development country count",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
    )
    ptext(
        frames[1],
        split_left + development_width + (split_width - development_width) / 2,
        0.518,
        "41",
        label="unseen country count",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
    )
    ptext(
        frames[1],
        split_left + development_width / 2,
        0.425,
        "develop",
        label="development split label",
        ha="center",
        va="center",
        fontsize=6.7,
        color=PALETTE["navy"],
    )
    ptext(
        frames[1],
        split_left + development_width + (split_width - development_width) / 2,
        0.425,
        "unseen",
        label="unseen split label",
        ha="center",
        va="center",
        fontsize=6.4,
        color=PALETTE["red"],
    )
    canvas.add_patch(
        mpatches.Rectangle((x + 0.025, 0.245), 0.004, 0.082, facecolor=PALETTE["red"], edgecolor="none")
    )
    ptext(
        frames[1],
        x + 0.041,
        0.286,
        "41 countries\nnever enter fitting",
        label="physical holdout statement",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        linespacing=1.18,
        color=PALETTE["dark"],
    )

    # 3 | Candidate representations: use icons rather than descriptive sentences.
    x, width, _heading, accent = panel_specs[2]
    centres = [x + 0.043, x + 0.100, x + 0.157]
    icon_y = 0.690

    # Trend glyph.
    canvas.plot([centres[0] - 0.023, centres[0] - 0.023], [0.645, 0.738], color="#AAB0B4", lw=0.7)
    canvas.plot([centres[0] - 0.023, centres[0] + 0.024], [0.645, 0.645], color="#AAB0B4", lw=0.7)
    trend_x = np.linspace(centres[0] - 0.018, centres[0] + 0.020, 5)
    trend_y = [0.660, 0.676, 0.671, 0.706, 0.724]
    canvas.plot(trend_x, trend_y, color=PALETTE["navy"], lw=1.4)
    canvas.scatter(trend_x[-1:], trend_y[-1:], s=14, color=PALETTE["navy"], edgecolor="white", linewidth=0.4)

    # Tabular glyph.
    for row in range(3):
        for column in range(3):
            color = [PALETTE["teal"], PALETTE["cyan"], PALETTE["mid"]][(row + column) % 3]
            canvas.add_patch(
                mpatches.Rectangle(
                    (centres[1] - 0.023 + column * 0.017, 0.718 - row * 0.036),
                    0.011,
                    0.024,
                    facecolor=mcolors.to_rgba(color, 0.78),
                    edgecolor="none",
                )
            )

    # Sequence glyph.
    sequence_x = np.linspace(centres[2] - 0.022, centres[2] + 0.022, 4)
    sequence_y = [icon_y - 0.018, icon_y + 0.020, icon_y - 0.006, icon_y + 0.032]
    canvas.plot(sequence_x, sequence_y, color=PALETTE["gold"], lw=1.2)
    canvas.scatter(sequence_x, sequence_y, s=24, color=PALETTE["gold"], edgecolor="white", linewidth=0.5, zorder=3)

    for centre, label in zip(centres, ["Trend", "Table", "Sequence"]):
        ptext(
            frames[2],
            centre,
            0.585,
            label,
            label=f"candidate representation {label}",
            ha="center",
            va="center",
            fontsize=6.9,
            fontweight="bold",
            color=PALETTE["dark"],
        )

    canvas.plot([x + 0.025, x + width - 0.025], [0.520, 0.520], color="#D5D9DC", lw=0.7)
    fold_left = x + 0.038
    for row, fraction in enumerate([0.55, 0.72, 0.90]):
        y = 0.432 - row * 0.055
        full_width = 0.125
        canvas.plot([fold_left, fold_left + full_width], [y, y], color="#D9DEE1", lw=3.2, solid_capstyle="butt")
        canvas.plot([fold_left, fold_left + full_width * fraction], [y, y], color=PALETTE["teal"], lw=3.2, solid_capstyle="butt")
        canvas.scatter([fold_left + full_width * fraction], [y], s=16, color=PALETTE["red"], edgecolor="white", linewidth=0.4, zorder=3)
    ptext(
        frames[2],
        x + width / 2,
        0.245,
        "3 expanding folds",
        label="temporal fold count",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["teal"],
    )
    ptext(
        frames[2],
        x + width / 2,
        0.185,
        "development only",
        label="development-only selection",
        ha="center",
        va="center",
        fontsize=6.8,
        color=PALETTE["mid"],
    )

    # 4 | Immutable lock: the six cells collapse to one shared horizon row.
    x, width, _heading, accent = panel_specs[3]
    lock_x = x + 0.029
    canvas.add_patch(
        mpatches.Arc((lock_x + 0.014, 0.729), 0.030, 0.095, theta1=0, theta2=180, color=PALETTE["purple"], lw=1.2)
    )
    canvas.add_patch(
        mpatches.Rectangle((lock_x, 0.675), 0.028, 0.070, facecolor=mcolors.to_rgba(PALETTE["purple"], 0.18), edgecolor=PALETTE["purple"], linewidth=0.8)
    )
    ptext(
        frames[3],
        x + 0.070,
        0.716,
        "6 task models",
        label="task model count",
        ha="left",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ptext(
        frames[3],
        x + 0.070,
        0.660,
        "SHA-256 frozen",
        label="lock provenance",
        ha="left",
        va="center",
        fontsize=6.7,
        color=PALETTE["purple"],
    )

    cell_left = x + 0.018
    cell_width = 0.052
    cell_gap = 0.006
    cell_y = 0.445
    cell_height = 0.105
    for column, horizon in enumerate([1, 3, 5]):
        family = locked_family[("daly", horizon)]
        color = FAMILY_COLORS[family]
        cell_x = cell_left + column * (cell_width + cell_gap)
        ptext(
            frames[3],
            cell_x + cell_width / 2,
            0.585,
            f"{horizon} y",
            label=f"locked horizon {horizon}",
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=PALETTE["mid"],
        )
        cell = mpatches.Rectangle(
            (cell_x, cell_y),
            cell_width,
            cell_height,
            facecolor=mcolors.to_rgba(color, 0.17),
            edgecolor=color,
            linewidth=0.8,
        )
        canvas.add_patch(cell)
        label = "Trees" if family == "extra_trees" else family_label(family)
        cell_text = canvas.text(
            cell_x + cell_width / 2,
            cell_y + cell_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        contained_text.append((cell, [cell_text], f"locked model {horizon} y", 3.0))

    canvas.scatter([x + 0.055, x + 0.077], [0.365, 0.365], s=23, color=[PALETTE["navy"], PALETTE["red"]], edgecolor="white", linewidth=0.4)
    ptext(
        frames[3],
        x + 0.091,
        0.365,
        "DALY + death",
        label="locked outcomes",
        ha="left",
        va="center",
        fontsize=6.9,
        color=PALETTE["dark"],
    )
    canvas.plot([x + 0.026, x + width - 0.026], [0.305, 0.305], color="#D5D9DC", lw=0.7)
    ptext(
        frames[3],
        x + width / 2,
        0.230,
        "No reselection",
        label="no reselection statement",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["red"],
    )

    # 5 | Sealed outputs: two direct takeaways replace dense micro-plots.
    x, width, _heading, accent = panel_specs[4]
    target_x, target_y = x + 0.042, 0.705
    canvas.scatter([target_x], [target_y], s=380, facecolors="none", edgecolors="#C8D0D4", linewidths=1.0)
    canvas.scatter([target_x], [target_y], s=190, facecolors="none", edgecolors=PALETTE["orange"], linewidths=1.1)
    canvas.scatter([target_x], [target_y], s=35, color=PALETTE["red"], edgecolor="white", linewidth=0.5, zorder=3)
    canvas.annotate(
        "",
        xy=(target_x + 0.031, target_y + 0.068),
        xytext=(target_x + 0.006, target_y + 0.018),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["navy"], lw=1.0, mutation_scale=8),
    )
    ptext(
        frames[4],
        x + 0.119,
        0.718,
        f"{positive_comparisons}/12",
        label="sealed comparison result",
        ha="center",
        va="center",
        fontsize=11.7,
        fontweight="bold",
        color=PALETTE["red"],
    )
    ptext(
        frames[4],
        x + 0.119,
        0.640,
        "beat pooled ridge",
        label="sealed comparator label",
        ha="center",
        va="center",
        fontsize=6.6,
        color=PALETTE["dark"],
    )

    interval_y = 0.468
    canvas.plot([x + 0.021, x + 0.057], [interval_y, interval_y], color=PALETTE["navy"], lw=1.5)
    canvas.scatter([x + 0.040], [interval_y], s=22, color=PALETTE["navy"], edgecolor="white", linewidth=0.4, zorder=3)
    canvas.plot([x + 0.066, x + 0.066], [interval_y - 0.055, interval_y + 0.055], color=PALETTE["mid"], lw=0.8, ls="--")
    ptext(
        frames[4],
        x + 0.121,
        interval_y + 0.015,
        f"{undercovered}/{interval_estimates}",
        label="interval coverage result",
        ha="center",
        va="center",
        fontsize=10.7,
        fontweight="bold",
        color=PALETTE["red"],
    )
    ptext(
        frames[4],
        x + 0.121,
        0.395,
        "below nominal",
        label="interval coverage label",
        ha="center",
        va="center",
        fontsize=6.7,
        color=PALETTE["dark"],
    )
    result_box = mpatches.Rectangle(
        (x + 0.018, 0.205),
        width - 0.036,
        0.100,
        facecolor=mcolors.to_rgba(PALETTE["red"], 0.10),
        edgecolor=mcolors.to_rgba(PALETTE["red"], 0.30),
        linewidth=0.65,
    )
    canvas.add_patch(result_box)
    result_text = canvas.text(
        x + width / 2,
        0.255,
        "No sealed gain",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color="#9E3328",
    )
    contained_text.append((result_box, [result_text], "sealed conclusion", 4.0))

    # Production guard: every protected label must remain inside its intended region.
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    qa_canvas = FigureCanvasAgg(fig)
    qa_canvas.draw()
    renderer = qa_canvas.get_renderer()
    for container, texts, label, padding in contained_text:
        container_box = container.get_window_extent(renderer)
        for text_artist in texts:
            text_box = text_artist.get_window_extent(renderer)
            inside = (
                text_box.x0 >= container_box.x0 + padding
                and text_box.x1 <= container_box.x1 - padding
                and text_box.y0 >= container_box.y0 + padding
                and text_box.y1 <= container_box.y1 - padding
            )
            if not inside:
                raise RuntimeError(
                    f"Figure 1 text overflow in {label}: {text_artist.get_text()!r}; "
                    f"text={text_box.bounds}, container={container_box.bounds}"
                )

    save_figure(
        fig,
        PDFDIR / "Fig1_workflow.pdf",
        {"Title": "Figure 1 workflow", "Author": "Rebuilt analysis"},
    )
    plt.close(fig)


def figure2(inputs: dict[str, Any]) -> None:
    desc = inputs["descriptive"]
    panel = inputs["panel"]
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05], hspace=0.44, wspace=0.30)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]
    add_figure_title(fig, "Figure 2 | Global and country-level BMI-attributable liver-cancer burden")
    for ax, letter in zip(axes, list("abcde")):
        panel_label(ax, letter)

    for ax, outcome, text in [(axes[0], "daly", "Global BMI-attributable DALY ASR"), (axes[1], "death", "Global BMI-attributable death ASR")]:
        d = desc.loc[desc["scope"].eq("global") & desc["outcome"].eq(outcome)].sort_values("year")
        ax.fill_between(d["year"], d["bmi_rate_lower"], d["bmi_rate_upper"], color=OUTCOME_COLORS[outcome], alpha=0.16, lw=0)
        ax.plot(d["year"], d["bmi_rate"], color=OUTCOME_COLORS[outcome])
        ax.scatter([d["year"].iloc[-1]], [d["bmi_rate"].iloc[-1]], color=OUTCOME_COLORS[outcome], s=18, zorder=3)
        ax.annotate(f"{d['bmi_rate'].iloc[-1]:.2f}", (d["year"].iloc[-1], d["bmi_rate"].iloc[-1]), xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=7.2)
        title(ax, text); ax.set_xlabel("Year"); ax.set_ylabel("Age-standardised rate per 100,000"); clean_axis(ax)

    ax = axes[2]
    for outcome in ["daly", "death"]:
        d = desc.loc[desc["scope"].eq("global") & desc["outcome"].eq(outcome)].sort_values("year")
        ax.fill_between(d["year"], d["direct_gbd_fraction_lower"] * 100, d["direct_gbd_fraction_upper"] * 100, color=OUTCOME_COLORS[outcome], alpha=0.12, lw=0)
        ax.plot(d["year"], d["direct_gbd_fraction"] * 100, color=OUTCOME_COLORS[outcome], label=OUTCOME_LABEL[outcome])
    title(ax, "Global direct GBD attributable fraction"); ax.set_xlabel("Year"); ax.set_ylabel("Attributable fraction (%)"); clean_axis(ax); ax.legend(ncol=2, loc="upper left")

    ax = axes[3]
    d = desc.loc[desc["scope"].eq("sdi_group") & desc["year"].eq(2023)].copy()
    d["fraction_pct"] = d["direct_gbd_fraction"] * 100
    order = d[["location_name", "sdi_group_order"]].drop_duplicates().sort_values("sdi_group_order")["location_name"].tolist()
    y = np.arange(len(order)); offsets = {"daly": -0.12, "death": 0.12}
    for outcome in ["daly", "death"]:
        q = d.loc[d["outcome"].eq(outcome)].set_index("location_name").loc[order]
        ax.hlines(y + offsets[outcome], 0, q["fraction_pct"], color=OUTCOME_COLORS[outcome], lw=1.0, alpha=0.8)
        ax.scatter(q["fraction_pct"], y + offsets[outcome], color=OUTCOME_COLORS[outcome], s=20, label=OUTCOME_LABEL[outcome], zorder=3)
    ax.set_yticks(y, [name.replace(" SDI", "") for name in order]); ax.invert_yaxis(); ax.set_xlabel("2023 attributable fraction (%)"); title(ax, "SDI aggregates (High SDI absent)"); clean_axis(ax, "x"); ax.legend(ncol=2, loc="upper right")

    ax = axes[4]
    d = panel.loc[panel["year"].eq(2023)].copy()
    groups = [f"Q{i}" for i in range(1, 6)]
    arrays = [d.loc[d["sdi_quintile_2023"].eq(group), "bmi_daly_rate"].to_numpy() for group in groups]
    vio = ax.violinplot(arrays, positions=np.arange(1, 6), showextrema=False, widths=0.78)
    for body, group in zip(vio["bodies"], groups):
        body.set_facecolor(plt.cm.Blues(0.25 + 0.12 * int(group[1]))); body.set_edgecolor("#6E6E6E"); body.set_alpha(0.8); body.set_linewidth(0.45)
    ax.boxplot(arrays, positions=np.arange(1, 6), widths=0.18, showfliers=False, patch_artist=True, boxprops=dict(facecolor="white", edgecolor=PALETTE["dark"], linewidth=0.6), medianprops=dict(color=PALETTE["red"], linewidth=0.9), whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6))
    ax.set_xticks(np.arange(1, 6), groups); ax.set_xlabel("Country SDI quintile (2023 assignment)"); ax.set_ylabel("BMI-attributable DALY ASR"); title(ax, "Country burden distribution, 2023"); clean_axis(ax)

    write_csv(desc, FIGDATA / "Fig2_global_sdi_series.csv")
    write_csv(panel.loc[panel["year"].eq(2023)], FIGDATA / "Fig2_country_2023.csv")
    fig.subplots_adjust(top=0.985, bottom=0.065, left=0.105, right=0.985)
    save_figure(fig, PDFDIR / "Fig2_burden_landscape.pdf", {"Title": "Figure 2 burden landscape"})
    plt.close(fig)


def figure3(inputs: dict[str, Any]) -> None:
    config = inputs["config"]
    predictions = inputs["predictions"]
    development = inputs["development"]
    shift = inputs["shift"]
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.48, wspace=0.66)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]
    add_figure_title(fig, "Figure 3 | Validation design and quantified distribution shift")
    for ax, letter in zip(axes, list("abcde")):
        panel_label(ax, letter)

    ax = axes[0]
    folds = config["cross_validation"]
    for i, fold in enumerate(folds):
        y = len(folds) - i
        train_start = 1991
        train_end = int(fold["train_target_year_max"])
        valid_start = int(min(fold["validation_target_years"]))
        valid_end = int(max(fold["validation_target_years"]))
        ax.broken_barh([(train_start, train_end - train_start + 1)], (y - 0.28, 0.5), facecolors="#DCE8F2", edgecolors="none")
        ax.broken_barh([(max(train_start, train_end - 1), 2)], (y - 0.28, 0.5), facecolors="#F9EBC8", edgecolors="none")
        ax.broken_barh([(valid_start, valid_end - valid_start + 1)], (y - 0.28, 0.5), facecolors="#F6D9D4", edgecolors="none")
        ax.text(1991.2, y, str(fold["fold"]).upper(), va="center", fontsize=7.2)
    ax.set_xlim(1990, 2019); ax.set_ylim(0.5, 4); ax.set_yticks([]); ax.set_xticks([1991, 2000, 2010, 2014, 2018]); ax.set_xlabel("Target year"); title(ax, "Expanding-window outer CV with inner stop set"); clean_axis(ax, None)
    handles = [mpatches.Patch(color="#DCE8F2", label="outer train"), mpatches.Patch(color="#F9EBC8", label="inner stop"), mpatches.Patch(color="#F6D9D4", label="outer validation")]
    ax.legend(handles=handles, ncol=3, loc="lower left")

    ax = axes[1]
    unique = predictions[["partition", "location_id", "sdi_quintile_2023_assignment_only"]].drop_duplicates()
    counts = unique.groupby(["partition", "sdi_quintile_2023_assignment_only"]).size().unstack(fill_value=0).reindex(columns=[f"Q{i}" for i in range(1, 6)])
    y = np.arange(len(counts)); left = np.zeros(len(counts)); colors = [plt.cm.Blues(0.25 + 0.12 * i) for i in range(1, 6)]
    for index, group in enumerate(counts.columns):
        vals = counts[group].to_numpy(); ax.barh(y, vals, left=left, color=colors[index], height=0.55, label=group)
        for row, (start, value) in enumerate(zip(left, vals)):
            if value >= 5: ax.text(start + value / 2, row, str(int(value)), ha="center", va="center", fontsize=6.8)
        left += vals
    ax.set_yticks(y, [PARTITION_LABEL[name] for name in counts.index]); ax.set_xlabel("Countries"); title(ax, "Country holdout preserves SDI breadth"); clean_axis(ax, "x")
    ax.legend(
        ncol=5,
        loc="center",
        bbox_to_anchor=(0.50, 0.50),
        fontsize=6.7,
        handlelength=0.95,
        handletextpad=0.25,
        columnspacing=0.60,
        borderaxespad=0,
    )

    ax = axes[2]
    dev_counts = development.groupby(["horizon", "target_year"]).size().reset_index(name="n")
    final_unique = predictions[["partition", "horizon", "sample_id"]].drop_duplicates().groupby(["partition", "horizon"]).size().reset_index(name="n")
    matrix = np.zeros((3, 3));
    for i, horizon in enumerate([1, 3, 5]):
        matrix[i, 0] = dev_counts.loc[dev_counts["horizon"].eq(horizon), "n"].sum()
        matrix[i, 1] = final_unique.loc[final_unique["partition"].eq("test_temporal_seen_country") & final_unique["horizon"].eq(horizon), "n"].iloc[0]
        matrix[i, 2] = final_unique.loc[final_unique["partition"].eq("test_spatiotemporal_unseen_country") & final_unique["horizon"].eq(horizon), "n"].iloc[0]
    im = vector_heatmap(ax, matrix, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{int(matrix[i,j]):,}", ha="center", va="center", color="white" if matrix[i,j] > matrix.max() * 0.55 else PALETTE["dark"], fontsize=7.2)
    ax.set_xticks(range(3), ["Development", "Seen test", "Unseen test"], rotation=20, ha="right"); ax.set_yticks(range(3), ["1 y", "3 y", "5 y"]); title(ax, "Samples by horizon and partition")

    ax = axes[3]
    top_features = shift.groupby("feature")["absolute_smd"].max().nlargest(12).index
    q = shift.loc[shift["feature"].isin(top_features)].copy()
    order = q.groupby("feature")["absolute_smd"].max().sort_values().index.tolist()
    y = np.arange(len(order)); offsets = {"test_temporal_seen_country": -0.12, "test_spatiotemporal_unseen_country": 0.12}
    for partition in ["test_temporal_seen_country", "test_spatiotemporal_unseen_country"]:
        z = q.loc[q["partition"].eq(partition)].set_index("feature").reindex(order)
        ax.scatter(z["standardized_mean_difference"], y + offsets[partition], s=18, color=PARTITION_COLORS[partition], label=PARTITION_LABEL[partition], zorder=3)
    ax.axvline(0, color=PALETTE["mid"], lw=0.7); ax.axvline(-0.2, color=PALETTE["light"], ls="--", lw=0.6); ax.axvline(0.2, color=PALETTE["light"], ls="--", lw=0.6)
    ax.set_yticks(y, [short_feature(name, 27) for name in order]); ax.set_xlabel("Standardised mean difference"); title(ax, "Largest feature shifts"); clean_axis(ax, "x"); ax.legend(ncol=1, loc="lower left")

    ax = axes[4]
    for partition in ["test_temporal_seen_country", "test_spatiotemporal_unseen_country"]:
        q = shift.loc[shift["partition"].eq(partition)]
        ax.scatter(q["absolute_smd"], q["ks_statistic"], s=12, alpha=0.58, color=PARTITION_COLORS[partition], label=PARTITION_LABEL[partition], edgecolors="none")
    labels = shift.loc[shift["partition"].eq("test_spatiotemporal_unseen_country")].nlargest(3, "absolute_smd")
    for index, row in enumerate(labels.itertuples(index=False), start=1):
        ax.annotate(str(index), (row.absolute_smd, row.ks_statistic), xytext=(3, (index - 2) * 7), textcoords="offset points", fontsize=6.8, fontweight="bold")
    label_key = "\n".join(f"{index}  {short_feature(row.feature, 22)}" for index, row in enumerate(labels.itertuples(index=False), start=1))
    ax.text(0.02, 0.98, label_key, transform=ax.transAxes, va="top", fontsize=6.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.7))
    ax.set_xlabel("Absolute standardised mean difference"); ax.set_ylabel("Kolmogorov-Smirnov statistic"); title(ax, "Shift diagnostics agree across metrics"); clean_axis(ax); ax.legend(loc="lower right")

    write_csv(shift, FIGDATA / "Fig3_distribution_shift.csv")
    write_csv(unique, FIGDATA / "Fig3_country_split.csv")
    fig.subplots_adjust(top=0.985, bottom=0.07, left=0.14, right=0.985)
    save_figure(fig, PDFDIR / "Fig3_validation_and_shift.pdf", {"Title": "Figure 3 validation and shift"})
    plt.close(fig)


def figure4(inputs: dict[str, Any]) -> None:
    cv = task_frame(inputs["cv"])
    cv_fold = task_frame(inputs["cv_fold"])
    selected_map = locked_family_map(inputs["lock"])
    fig = plt.figure(figsize=(7.2, 8.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.92], hspace=0.53, wspace=0.42)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),
    ]
    add_figure_title(fig, "Figure 4 | Development-only benchmark across 11 model families")
    for ax, letter in zip(axes, list("abcde")): panel_label(ax, letter)

    ax = axes[0]
    pivot = cv.pivot(index="task", columns="family", values="skill_vs_persistence").reindex(
        index=[task_label(*task) for task in TASK_ORDER], columns=FAMILY_ORDER
    ) * 100
    shown = np.clip(pivot.to_numpy(), -100, 30)
    norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=30)
    im = vector_heatmap(ax, shown, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(FAMILY_ORDER)), [family_label(name) for name in FAMILY_ORDER], rotation=55, ha="right")
    ax.set_yticks(range(len(TASK_ORDER)), [task_label(*task) for task in TASK_ORDER])
    for row, task in enumerate(TASK_ORDER):
        column = FAMILY_ORDER.index(selected_map[task])
        ax.add_patch(mpatches.Rectangle((column - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor=PALETTE["gold"], linewidth=1.15))
    title(ax, "Skill versus persistence (gold = locked)")
    cb = vector_colorbar(fig, im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Skill (%)"); cb.ax.tick_params(labelsize=6.9)

    ax = axes[1]
    ranks = cv.groupby("family")["rank_rmsle"].agg(["mean", "min", "max"]).sort_values("mean")
    y = np.arange(len(ranks))
    for i, (family, row) in enumerate(ranks.iterrows()):
        ax.hlines(i, row["min"], row["max"], color=FAMILY_COLORS[family], alpha=0.65, lw=1.1)
        ax.scatter(row["mean"], i, s=22, color=FAMILY_COLORS[family], edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_yticks(y, [family_label(name) for name in ranks.index]); ax.invert_yaxis(); ax.set_xlabel("Rank across six tasks (lower is better)"); title(ax, "Mean and range of CV rank"); clean_axis(ax, "x")

    show_families = ["persistence", "pooled_ridge", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "gru"]
    for ax, outcome, panel_title in [(axes[2], "daly", "DALY error grows with horizon"), (axes[3], "death", "Death error grows with horizon")]:
        for family in show_families:
            q = cv.loc[cv["outcome"].eq(outcome) & cv["family"].eq(family)].sort_values("horizon")
            ax.plot(q["horizon"], q["mean_fold_rmsle"], marker="o", color=FAMILY_COLORS[family], label=family_label(family), alpha=0.9)
        ax.set_xticks([1, 3, 5]); ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Mean-fold RMSLE"); ax.set_yscale("log"); use_plain_log_tick_labels(ax, "y"); title(ax, panel_title); clean_axis(ax)
    axes[3].legend(ncol=2, loc="upper left", fontsize=6.7)

    ax = axes[4]
    selected_rows = []
    for task in TASK_ORDER:
        outcome, horizon = task; family = selected_map[task]
        q_selected = cv_fold.loc[cv_fold["outcome"].eq(outcome) & cv_fold["horizon"].eq(horizon) & cv_fold["family"].eq(family), ["fold", "rmsle"]].rename(columns={"rmsle": "selected_rmsle"})
        q_ref = cv_fold.loc[cv_fold["outcome"].eq(outcome) & cv_fold["horizon"].eq(horizon) & cv_fold["family"].eq("pooled_ridge"), ["fold", "rmsle"]].rename(columns={"rmsle": "pooled_rmsle"})
        q = q_selected.merge(q_ref, on="fold", validate="one_to_one")
        q["outcome"] = outcome; q["horizon"] = horizon; q["delta"] = q["selected_rmsle"] - q["pooled_rmsle"]
        selected_rows.append(q)
    paired = pd.concat(selected_rows, ignore_index=True)
    for outcome, marker in [("daly", "o"), ("death", "s")]:
        q = paired.loc[paired["outcome"].eq(outcome)]
        for fold, alpha in [("cv1", 0.45), ("cv2", 0.7), ("cv3", 1.0)]:
            z = q.loc[q["fold"].eq(fold)].sort_values("horizon")
            ax.plot(z["horizon"], z["delta"], marker=marker, color=OUTCOME_COLORS[outcome], alpha=alpha, label=f"{OUTCOME_LABEL[outcome]} {fold.upper()}")
    ax.axhline(0, color=PALETTE["dark"], lw=0.7); ax.set_xticks([1, 3, 5]); ax.set_xlabel("Forecast horizon (years)"); ax.set_ylabel("Locked minus pooled-ridge RMSLE"); title(ax, "Foldwise gain is not uniform"); clean_axis(ax); ax.legend(ncol=3, fontsize=6.8, loc="best")

    write_csv(cv, FIGDATA / "Fig4_cv_model_comparison.csv")
    write_csv(paired, FIGDATA / "Fig4_foldwise_selected_vs_pooled.csv")
    fig.subplots_adjust(top=0.985, bottom=0.085, left=0.16, right=0.985)
    save_figure(fig, PDFDIR / "Fig4_cv_benchmark.pdf", {"Title": "Figure 4 cross-validation benchmark"})
    plt.close(fig)


def comparison_metrics(metrics: pd.DataFrame, partition: str) -> pd.DataFrame:
    local = metrics.loc[metrics["partition"].eq(partition)].copy()
    selected = local.loc[local["selected_by_cv"]].copy()
    persistence = local.loc[local["family"].eq("persistence"), ["outcome", "horizon", "rmsle"]].rename(columns={"rmsle": "persistence_rmsle"})
    pooled = local.loc[local["family"].eq("pooled_ridge"), ["outcome", "horizon", "rmsle"]].rename(columns={"rmsle": "pooled_rmsle"})
    return selected.merge(persistence, on=["outcome", "horizon"], validate="one_to_one").merge(pooled, on=["outcome", "horizon"], validate="one_to_one")


def forest_panel(ax, bootstrap: pd.DataFrame, partition: str, panel_title: str) -> pd.DataFrame:
    q = bootstrap.loc[
        bootstrap["partition"].eq(partition)
        & bootstrap["estimand"].eq("selected_minus_reference_rmsle")
    ].copy()
    q = task_frame(q).sort_values("task_order", ascending=False)
    y = np.arange(len(q))
    ax.hlines(y, q["ci_lower"], q["ci_upper"], color=[OUTCOME_COLORS[o] for o in q["outcome"]], lw=1.2)
    ax.scatter(q["estimate"], y, color=[OUTCOME_COLORS[o] for o in q["outcome"]], s=22, zorder=3)
    ax.axvline(0, color=PALETTE["dark"], lw=0.7)
    ax.set_yticks(y, q["task"]); ax.set_xlabel("Selected minus pooled-ridge RMSLE (95% CI)"); title(ax, panel_title); clean_axis(ax, "x")
    return q


def figure5(inputs: dict[str, Any]) -> None:
    metrics = inputs["metrics"]
    predictions = inputs["predictions"]
    bootstrap = inputs["bootstrap"]
    residual = inputs["residual"]
    partition = "test_temporal_seen_country"
    comp = task_frame(comparison_metrics(metrics, partition)).sort_values("task_order")
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.65))
    add_figure_title(fig, "Figure 5 | Sealed 2019-2023 evaluation in previously seen countries")
    for ax, letter in zip(axes.flat, list("abcdef")): panel_label(ax, letter)

    ax = axes[0, 0]
    y = np.arange(len(comp))[::-1]
    for index, row in enumerate(comp.itertuples(index=False)):
        yi = y[index]
        ax.plot([row.persistence_rmsle, row.pooled_rmsle], [yi, yi], color=PALETTE["light"], lw=1.0)
        ax.scatter(row.persistence_rmsle, yi, marker="x", color=FAMILY_COLORS["persistence"], s=24)
        ax.scatter(row.pooled_rmsle, yi, marker="o", facecolor="white", edgecolor=FAMILY_COLORS["pooled_ridge"], s=24)
        ax.scatter(row.rmsle, yi, marker="D", color=FAMILY_COLORS[row.family], s=24, zorder=4)
    ax.set_yticks(y, comp["task"]); ax.set_xlabel("RMSLE"); title(ax, "Locked model versus transparent baselines"); clean_axis(ax, "x")
    ax.legend(handles=[Line2D([], [], marker="x", color=FAMILY_COLORS["persistence"], ls="", label="Persistence"), Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=FAMILY_COLORS["pooled_ridge"], ls="", label="Pooled ridge"), Line2D([], [], marker="D", color=PALETTE["gold"], ls="", label="CV-locked model")], ncol=1, loc="lower right", fontsize=6.7)

    ax = axes[0, 1]
    comp["skill_persistence"] = (1 - comp["rmsle"] / comp["persistence_rmsle"]) * 100
    comp["skill_pooled"] = (1 - comp["rmsle"] / comp["pooled_rmsle"]) * 100
    x = np.arange(len(comp)); width = 0.34
    ax.bar(x - width / 2, comp["skill_persistence"], width, color=PALETTE["teal"], label="vs persistence")
    ax.bar(x + width / 2, comp["skill_pooled"], width, color=PALETTE["red"], label="vs pooled ridge")
    ax.axhline(0, color=PALETTE["dark"], lw=0.7); ax.set_xticks(x, comp["task"], rotation=35, ha="right"); ax.set_ylabel("Skill (%)"); title(ax, "Apparent gain depends on comparator"); clean_axis(ax); ax.legend(ncol=2, loc="lower left")

    forest = forest_panel(axes[1, 0], bootstrap, partition, "Country-clustered difference from pooled ridge")

    ax = axes[1, 1]
    q = predictions.loc[predictions["partition"].eq(partition) & predictions["outcome"].eq("daly") & predictions["horizon"].eq(5) & predictions["selected_by_cv"]].copy()
    ax.scatter(q["observed"], q["prediction"], s=10, alpha=0.45, color=PALETTE["teal"], edgecolors="none")
    limits = [max(0.01, min(q["observed"].min(), q["prediction"].min()) * 0.8), max(q["observed"].max(), q["prediction"].max()) * 1.2]
    ax.plot(limits, limits, ls="--", color=PALETTE["mid"], lw=0.8); ax.set(xlim=limits, ylim=limits); ax.set_xscale("log"); ax.set_yscale("log"); use_plain_log_tick_labels(ax)
    q["abs_log_error"] = (np.log1p(q["prediction"]) - np.log1p(q["observed"])).abs()
    labelled = q.nlargest(4, "abs_log_error")
    for index, row in enumerate(labelled.itertuples(index=False), start=1):
        ax.annotate(str(index), (row.observed, row.prediction), xytext=(3, 2), textcoords="offset points", fontsize=6.6, fontweight="bold")
    label_key = "\n".join(f"{index}  {row.location_name}" for index, row in enumerate(labelled.itertuples(index=False), start=1))
    ax.text(0.03, 0.97, label_key, transform=ax.transAxes, va="top", fontsize=6.4, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.7))
    ax.set_xlabel("Observed DALY ASR"); ax.set_ylabel("Predicted DALY ASR"); title(ax, "Five-year locked predictions"); clean_axis(ax)

    ax = axes[2, 0]
    r = task_frame(residual.loc[residual["partition"].eq(partition) & residual["selected_by_cv"]]).sort_values("task_order")
    for outcome, marker in [("daly", "o"), ("death", "s")]:
        z = r.loc[r["outcome"].eq(outcome)].sort_values("horizon")
        ax.plot(z["horizon"], z["calibration_slope_observed_on_predicted"], marker=marker, color=OUTCOME_COLORS[outcome], label=OUTCOME_LABEL[outcome])
    ax.axhline(1, color=PALETTE["dark"], ls="--", lw=0.7); ax.set_xticks([1, 3, 5]); ax.set_xlabel("Horizon (years)"); ax.set_ylabel("Calibration slope"); title(ax, "Calibration slope deviates from unity"); clean_axis(ax); ax.legend()

    ax = axes[2, 1]
    x = np.arange(len(comp));
    ax.scatter(x - 0.10, comp["coverage_90"], color=PALETTE["navy"], marker="o", label="90% interval")
    ax.scatter(x + 0.10, comp["coverage_95"], color=PALETTE["red"], marker="s", label="95% interval")
    ax.axhline(0.90, color=PALETTE["navy"], ls="--", lw=0.65); ax.axhline(0.95, color=PALETTE["red"], ls="--", lw=0.65)
    ax.set_ylim(0.55, 1.01); ax.set_xticks(x, comp["task"], rotation=35, ha="right"); ax.set_ylabel("Empirical coverage"); title(ax, "Cross-conformal intervals undercover"); clean_axis(ax); ax.legend(ncol=2, loc="lower left")

    write_csv(comp, FIGDATA / "Fig5_seen_performance.csv")
    write_csv(forest, FIGDATA / "Fig5_seen_bootstrap.csv")
    write_csv(q, FIGDATA / "Fig5_seen_daly_h5_predictions.csv")
    fig.subplots_adjust(top=0.985, bottom=0.09, left=0.13, right=0.985, hspace=0.52, wspace=0.36)
    save_figure(fig, PDFDIR / "Fig5_seen_country_test.pdf", {"Title": "Figure 5 seen country test"})
    plt.close(fig)


def figure6(inputs: dict[str, Any]) -> None:
    metrics = inputs["metrics"]
    predictions = inputs["predictions"]
    bootstrap = inputs["bootstrap"]
    panel = inputs["panel"]
    world = inputs["world"]
    partition = "test_spatiotemporal_unseen_country"
    country = country_rmsle_table(predictions, partition, 5)
    comp = task_frame(comparison_metrics(metrics, partition)).sort_values("task_order")
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.7))
    add_figure_title(fig, "Figure 6 | Geographic transportability to 41 countries excluded from fitting")
    for ax, letter in zip(axes.flat, list("abcdef")): panel_label(ax, letter)

    limit = max(0.02, float(country["delta_vs_persistence"].abs().quantile(0.95)))
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    for ax, outcome, panel_title in [(axes[0, 0], "daly", "DALY 5-y country delta vs persistence"), (axes[0, 1], "death", "Death 5-y country delta vs persistence")]:
        q = country.loc[country["outcome"].eq(outcome)]
        values = {GBD_TO_NE.get(name, name): value for name, value in zip(q["location_name"], q["delta_vs_persistence"])}
        collection = plot_world_map(ax, world, values, cmap, norm, panel_title)
        cb = vector_colorbar(fig, collection, ax=ax, fraction=0.032, pad=0.015, orientation="horizontal"); cb.set_label("Selected minus persistence RMSLE"); cb.ax.tick_params(labelsize=6.6)

    ax = axes[1, 0]
    y = np.arange(len(comp))[::-1]
    for index, row in enumerate(comp.itertuples(index=False)):
        yi = y[index]
        ax.scatter(row.persistence_rmsle, yi, marker="x", color=PALETTE["mid"], s=24)
        ax.scatter(row.pooled_rmsle, yi, marker="o", facecolor="white", edgecolor=PALETTE["navy"], s=24)
        ax.scatter(row.rmsle, yi, marker="D", color=FAMILY_COLORS[row.family], s=24)
    ax.set_yticks(y, comp["task"]); ax.set_xlabel("RMSLE"); title(ax, "Locked models lose at long horizons"); clean_axis(ax, "x")
    ax.legend(handles=[Line2D([], [], marker="x", color=PALETTE["mid"], ls="", label="Persistence"), Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=PALETTE["navy"], ls="", label="Pooled ridge"), Line2D([], [], marker="D", color=PALETTE["gold"], ls="", label="CV-locked")], ncol=1, loc="lower right", fontsize=6.6)

    forest = forest_panel(axes[1, 1], bootstrap, partition, "Country-clustered difference from pooled ridge")

    ax = axes[2, 0]
    positions = []; arrays = []; colors = []; labels = []
    position = 1
    for outcome in ["daly", "death"]:
        for group in [f"Q{i}" for i in range(1, 6)]:
            values = country.loc[country["outcome"].eq(outcome) & country["sdi_quintile_2023_assignment_only"].eq(group), "selected_rmsle"].to_numpy()
            if len(values):
                positions.append(position); arrays.append(values); colors.append(OUTCOME_COLORS[outcome]); labels.append(f"{OUTCOME_LABEL[outcome]} {group}")
            position += 1
        position += 1
    box = ax.boxplot(arrays, positions=positions, widths=0.58, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=0.8), whiskerprops=dict(lw=0.55), capprops=dict(lw=0.55))
    for patch, color in zip(box["boxes"], colors): patch.set_facecolor(color); patch.set_alpha(0.78); patch.set_edgecolor(PALETTE["dark"])
    ax.set_xticks(positions, labels, rotation=55, ha="right"); ax.set_ylabel("Country-level RMSLE"); title(ax, "Error heterogeneity by SDI quintile"); clean_axis(ax)

    ax = axes[2, 1]
    sdi = panel.loc[panel["year"].eq(2023), ["location_id", "sdi"]].drop_duplicates()
    q = country.merge(sdi, on="location_id", validate="many_to_one")
    for outcome, marker in [("daly", "o"), ("death", "s")]:
        z = q.loc[q["outcome"].eq(outcome)]
        ax.scatter(z["sdi"], z["delta_vs_pooled"], marker=marker, s=20, alpha=0.7, color=OUTCOME_COLORS[outcome], label=OUTCOME_LABEL[outcome])
    ax.axhline(0, color=PALETTE["dark"], lw=0.7)
    labelled = q.nlargest(6, "delta_vs_pooled")
    for index, row in enumerate(labelled.itertuples(index=False), start=1):
        ax.annotate(str(index), (row.sdi, row.delta_vs_pooled), xytext=(3, 2), textcoords="offset points", fontsize=6.5, fontweight="bold")
    label_key = "\n".join(f"{index}  {OUTCOME_LABEL[row.outcome]} | {row.location_name}" for index, row in enumerate(labelled.itertuples(index=False), start=1))
    ax.text(0.02, 0.97, label_key, transform=ax.transAxes, va="top", fontsize=6.2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.7))
    ax.set_xlabel("SDI, 2023"); ax.set_ylabel("Selected minus pooled-ridge RMSLE"); title(ax, "Failures concentrate in specific countries"); clean_axis(ax); ax.legend()

    write_csv(country, FIGDATA / "Fig6_country_h5_errors.csv")
    write_csv(comp, FIGDATA / "Fig6_unseen_performance.csv")
    write_csv(forest, FIGDATA / "Fig6_unseen_bootstrap.csv")
    fig.subplots_adjust(top=0.985, bottom=0.12, left=0.13, right=0.985, hspace=0.56, wspace=0.34)
    save_figure(fig, PDFDIR / "Fig6_unseen_country_test.pdf", {"Title": "Figure 6 unseen country test"})
    plt.close(fig)


def figure7(inputs: dict[str, Any]) -> None:
    shap_feature = inputs["shap_feature"]
    shap_family = inputs["shap_family"]
    shap_values = inputs["shap_values"]
    sealed = inputs["sealed"]
    ablation = inputs["ablation"]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.0))
    add_figure_title(fig, "Figure 7 | Tree-model explanation and prespecified feature-family ablation")
    for ax, letter in zip(axes.flat, list("abcdef")): panel_label(ax, letter)

    ax = axes[0, 0]
    family_order = [
        "BMI-attributable burden history",
        "GBD attributable fraction history",
        "overall liver-cancer burden history",
        "sociodemographic index history",
        "derived definition/gap features",
    ]
    family_short = ["BMI burden", "GBD fraction", "Overall burden", "SDI", "Derived/gaps"]
    q = task_frame(shap_family)
    pivot = q.pivot(index="task", columns="feature_family", values="proportion_of_total_absolute_shap").reindex(
        index=[task_label(*task) for task in TASK_ORDER], columns=family_order
    ) * 100
    im = vector_heatmap(ax, pivot.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value) and value >= 2: ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.7, color="white" if value > 55 else PALETTE["dark"])
    ax.set_xticks(range(len(family_short)), family_short, rotation=35, ha="right"); ax.set_yticks(range(len(TASK_ORDER)), [task_label(*task) for task in TASK_ORDER]); title(ax, "Share of absolute SHAP by feature family")
    cb = vector_colorbar(fig, im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("SHAP share (%)"); cb.ax.tick_params(labelsize=6.7)

    for ax, outcome, panel_title in [(axes[0, 1], "daly", "Top features: DALY 5-y tree"), (axes[1, 0], "death", "Top features: death 5-y tree")]:
        q = shap_feature.loc[shap_feature["outcome"].eq(outcome) & shap_feature["horizon"].eq(5)].nsmallest(12, "within_task_rank").sort_values("mean_absolute_shap_log1p_target")
        colors = [OUTCOME_COLORS[outcome] if rank <= 3 else "#AEBAC5" for rank in q["within_task_rank"]]
        ax.barh(np.arange(len(q)), q["mean_absolute_shap_log1p_target"], color=colors)
        ax.set_yticks(np.arange(len(q)), [short_feature(name, 25) for name in q["feature"]]); ax.tick_params(axis="y", labelsize=6.6); ax.set_xlabel("Mean |SHAP| in log1p target"); title(ax, panel_title); clean_axis(ax, "x")

    ax = axes[1, 1]
    top = shap_feature.loc[shap_feature["outcome"].eq("daly") & shap_feature["horizon"].eq(5)].nsmallest(8, "within_task_rank")["feature"].tolist()
    q = shap_values.loc[shap_values["outcome"].eq("daly") & shap_values["horizon"].eq(5)].merge(
        sealed[["sample_id", *top]], on="sample_id", how="left", validate="many_to_one"
    )
    rng = np.random.default_rng(20260811)
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-2, vmax=2), cmap="coolwarm")
    for index, feature in enumerate(top[::-1]):
        values = q[feature].to_numpy(dtype=float)
        scale = np.std(values)
        z = np.zeros_like(values) if scale == 0 else (values - np.mean(values)) / scale
        shap_column = f"shap__{feature}"
        ax.scatter(q[shap_column], index + rng.normal(0, 0.085, len(q)), c=np.clip(z, -2, 2), cmap="coolwarm", vmin=-2, vmax=2, s=7, alpha=0.58, edgecolors="none")
    ax.axvline(0, color=PALETTE["mid"], lw=0.65); ax.set_yticks(range(len(top)), [short_feature(name, 27) for name in top[::-1]]); ax.set_xlabel("SHAP value in log1p target"); title(ax, "DALY 5-y SHAP distribution")
    cb = vector_colorbar(fig, sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_ticks([-2, 2])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("Feature value")

    ablation_labels = {
        "remove_attributable_fraction": "Fraction history",
        "remove_attributable_history": "BMI burden history",
        "remove_overall_history": "Overall burden",
        "remove_sdi": "SDI",
        "remove_uncertainty_width": "Uncertainty widths",
    }
    ax = axes[2, 0]
    q = ablation.loc[
        ablation["partition"].eq("test_temporal_seen_country")
        & ablation["horizon"].eq(5)
        & ~ablation["ablation"].eq("none_full_model")
    ].copy()
    order = list(ablation_labels)
    x = np.arange(len(order)); width = 0.35
    for offset, outcome in [(-width / 2, "daly"), (width / 2, "death")]:
        z = q.loc[q["outcome"].eq(outcome)].set_index("ablation").reindex(order)
        ax.bar(x + offset, z["delta_rmsle_vs_full"], width, color=OUTCOME_COLORS[outcome], label=OUTCOME_LABEL[outcome])
    ax.axhline(0, color=PALETTE["dark"], lw=0.7); ax.set_xticks(x, [ablation_labels[name] for name in order], rotation=45, ha="right"); ax.set_ylabel("Delta RMSLE vs full tree"); title(ax, "Seen-country five-year ablation"); clean_axis(ax); ax.legend()

    ax = axes[2, 1]
    q = task_frame(ablation.loc[ablation["partition"].eq("test_spatiotemporal_unseen_country") & ~ablation["ablation"].eq("none_full_model")])
    pivot = q.pivot(index="task", columns="ablation", values="delta_rmsle_vs_full").reindex(
        index=[task_label(*task) for task in TASK_ORDER], columns=order
    )
    limit = max(0.005, float(np.nanquantile(np.abs(pivot.to_numpy()), 0.95)))
    im = vector_heatmap(ax, pivot.to_numpy(), cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(range(len(order)), [ablation_labels[name] for name in order], rotation=40, ha="right"); ax.set_yticks(range(len(TASK_ORDER)), [task_label(*task) for task in TASK_ORDER]); title(ax, "Unseen-country ablation sensitivity")
    cb = vector_colorbar(fig, im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Delta RMSLE"); cb.ax.tick_params(labelsize=6.7)

    write_csv(shap_family, FIGDATA / "Fig7_shap_family_importance.csv")
    write_csv(shap_feature, FIGDATA / "Fig7_shap_feature_importance.csv")
    write_csv(ablation, FIGDATA / "Fig7_feature_ablation.csv")
    fig.subplots_adjust(top=0.985, bottom=0.13, left=0.19, right=0.985, hspace=0.60, wspace=0.64)
    save_figure(fig, PDFDIR / "Fig7_explanation_ablation.pdf", {"Title": "Figure 7 explanation and ablation"})
    plt.close(fig)


def figure8(inputs: dict[str, Any]) -> None:
    stability = task_frame(inputs["stability"])
    fraction = task_frame(inputs["fraction"])
    subgroup = task_frame(inputs["subgroup"])
    metrics = task_frame(inputs["metrics"])
    selected_map = locked_family_map(inputs["lock"])
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.8))
    add_figure_title(fig, "Figure 8 | Seed, fraction-definition, subgroup and interval robustness")
    for ax, letter in zip(axes.flat, list("abcdef")): panel_label(ax, letter)

    for ax, partition, panel_title in [
        (axes[0, 0], "test_temporal_seen_country", "Five-seed range: seen countries"),
        (axes[0, 1], "test_spatiotemporal_unseen_country", "Five-seed range: unseen countries"),
    ]:
        q = stability.loc[stability["partition"].eq(partition)].copy()
        for index, task in enumerate(TASK_ORDER):
            outcome, horizon = task
            z = q.loc[q["outcome"].eq(outcome) & q["horizon"].eq(horizon)]
            for offset, row in zip([-0.12, 0.12], z.sort_values("family").itertuples(index=False)):
                color = FAMILY_COLORS.get(row.family, PALETTE["mid"])
                ax.hlines(index + offset, row.min_rmsle, row.max_rmsle, color=color, lw=1.1)
                ax.scatter(row.mean_rmsle, index + offset, s=17, color=color, zorder=3)
        ax.set_yticks(range(len(TASK_ORDER)), [task_label(*task) for task in TASK_ORDER]); ax.invert_yaxis(); ax.set_xlabel("RMSLE mean and min-max across 5 seeds"); title(ax, panel_title); clean_axis(ax, "x")
    axes[0, 1].legend(handles=[Line2D([], [], color=PALETTE["teal"], marker="o", label="Tree model"), Line2D([], [], color=PALETTE["gold"], marker="o", label="GRU")], loc="lower right")

    ax = axes[1, 0]
    rows = []
    for row in stability.itertuples(index=False):
        if row.family == selected_map[(row.outcome, int(row.horizon))]: rows.append(row)
    q = pd.DataFrame(rows).sort_values(["partition", "task_order"])
    for partition, marker in [("test_temporal_seen_country", "o"), ("test_spatiotemporal_unseen_country", "s")]:
        z = q.loc[q["partition"].eq(partition)].sort_values("task_order")
        ax.plot(z["task_order"], z["p95_prediction_seed_cv"] * 100, marker=marker, color=PARTITION_COLORS[partition], label=PARTITION_LABEL[partition])
    ax.set_xticks(range(len(TASK_ORDER)), [task_label(*task) for task in TASK_ORDER], rotation=35, ha="right"); ax.set_ylabel("95th percentile prediction CV (%)"); title(ax, "Locked-model prediction variability by seed"); clean_axis(ax); ax.legend()

    ax = axes[1, 1]
    q = fraction.loc[fraction["fraction_definition"].eq("ASR_ratio_sensitivity")].copy()
    q = q.loc[
        q.apply(
            lambda row: row["family"]
            == selected_map[(row["outcome"], int(row["horizon"]))],
            axis=1,
        )
    ].sort_values(["partition", "task_order"])
    expected_fraction_rows = len(TASK_ORDER) * len(PARTITION_LABEL)
    if len(q) != expected_fraction_rows:
        raise ValueError(
            f"Expected {expected_fraction_rows} locked-model fraction-sensitivity rows; found {len(q)}"
        )
    y = np.arange(len(TASK_ORDER)); offsets = {"test_temporal_seen_country": -0.11, "test_spatiotemporal_unseen_country": 0.11}
    for partition in offsets:
        z = q.loc[q["partition"].eq(partition)].sort_values("task_order")
        ax.scatter(z["delta_rmsle_vs_direct_percent"] * 1000, y + offsets[partition], s=20, marker="o" if partition.endswith("seen_country") and "unseen" not in partition else "s", color=PARTITION_COLORS[partition], label=PARTITION_LABEL[partition])
    ax.axvline(0, color=PALETTE["dark"], lw=0.7); ax.set_yticks(y, [task_label(*task) for task in TASK_ORDER]); ax.invert_yaxis(); ax.set_xlabel("Delta RMSLE x 1,000"); title(ax, "ASR-ratio versus direct-Percent definition"); clean_axis(ax, "x"); ax.legend()

    ax = axes[2, 0]
    selected_sub = subgroup.loc[
        subgroup["partition"].eq("test_spatiotemporal_unseen_country")
        & subgroup["selected_by_cv"]
        & subgroup["subgroup_dimension"].eq("sdi_quintile")
    ].copy()
    overall = subgroup.loc[
        subgroup["partition"].eq("test_spatiotemporal_unseen_country")
        & subgroup["selected_by_cv"]
        & subgroup["subgroup_dimension"].eq("overall")
    ][["outcome", "horizon", "rmsle"]].rename(columns={"rmsle": "overall_rmsle"})
    selected_sub = selected_sub.merge(overall, on=["outcome", "horizon"], validate="many_to_one")
    selected_sub["relative_rmsle"] = selected_sub["rmsle"] / selected_sub["overall_rmsle"]
    pivot = task_frame(selected_sub).pivot(index="task", columns="subgroup_value", values="relative_rmsle").reindex(index=[task_label(*task) for task in TASK_ORDER], columns=[f"Q{i}" for i in range(1, 6)])
    im = vector_heatmap(ax, pivot.to_numpy(), cmap="PuOr_r", norm=mcolors.TwoSlopeNorm(vmin=0.4, vcenter=1, vmax=max(1.8, float(np.nanmax(pivot.to_numpy())))), aspect="auto")
    ax.set_xticks(range(5), [f"SDI Q{i}" for i in range(1, 6)]); ax.set_yticks(range(6), [task_label(*task) for task in TASK_ORDER]); title(ax, "Unseen-country subgroup error / overall")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if value < 0.62 or value > 1.45 else PALETTE["dark"])
    ax.text(0.99, -0.16, "1.00 = overall task error", transform=ax.transAxes, ha="right", fontsize=6.4, color=PALETTE["mid"])

    ax = axes[2, 1]
    q = metrics.loc[metrics["selected_by_cv"]].copy().sort_values(["partition", "task_order"])
    rows = []
    for row in q.itertuples(index=False):
        partition_short = "Seen" if row.partition == "test_temporal_seen_country" else "Unseen"
        rows.append({"row": f"{partition_short} | {row.task}", "c90": row.coverage_90 - 0.90, "c95": row.coverage_95 - 0.95})
    coverage = pd.DataFrame(rows)
    matrix = coverage[["c90", "c95"]].to_numpy()
    im = vector_heatmap(ax, matrix, cmap="RdBu", norm=mcolors.TwoSlopeNorm(vmin=-0.35, vcenter=0, vmax=0.10), aspect="auto")
    ax.set_xticks([0, 1], ["90% interval", "95% interval"]); ax.set_yticks(range(len(coverage)), coverage["row"]); ax.tick_params(axis="y", labelsize=6.3); title(ax, "Coverage minus nominal level")
    cb = vector_colorbar(fig, im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Coverage difference")

    write_csv(stability, FIGDATA / "Fig8_seed_stability.csv")
    write_csv(fraction, FIGDATA / "Fig8_fraction_definition.csv")
    write_csv(selected_sub, FIGDATA / "Fig8_unseen_sdi_subgroups.csv")
    fig.subplots_adjust(top=0.985, bottom=0.11, left=0.20, right=0.985, hspace=0.52, wspace=0.64)
    save_figure(fig, PDFDIR / "Fig8_robustness.pdf", {"Title": "Figure 8 robustness"})
    plt.close(fig)


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.maximum(np.asarray(values, dtype=float), 0.0))
    if values.sum() == 0:
        return np.linspace(0, 1, len(values) + 1), np.linspace(0, 1, len(values) + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(values) / values.sum()])
    population = np.linspace(0, 1, len(values) + 1)
    return population, cumulative


def figure9(inputs: dict[str, Any]) -> None:
    metrics = task_frame(inputs["metrics"])
    predictions = inputs["predictions"]
    bootstrap = inputs["bootstrap"]
    comp_rows = []
    for partition in PARTITION_LABEL:
        q = task_frame(comparison_metrics(inputs["metrics"], partition))
        q["skill_persistence"] = (1 - q["rmsle"] / q["persistence_rmsle"]) * 100
        q["skill_pooled"] = (1 - q["rmsle"] / q["pooled_rmsle"]) * 100
        comp_rows.append(q)
    comp = pd.concat(comp_rows, ignore_index=True)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.75))
    add_figure_title(fig, "Figure 9 | Synthesis: apparent temporal skill does not transport geographically")
    for ax, letter in zip(axes.flat, list("abcdef")): panel_label(ax, letter)

    for ax, value, panel_title in [(axes[0, 0], "skill_persistence", "Skill versus persistence"), (axes[0, 1], "skill_pooled", "Skill versus pooled ridge")]:
        pivot = comp.pivot(index="partition", columns="task", values=value).reindex(index=["test_temporal_seen_country", "test_spatiotemporal_unseen_country"], columns=[task_label(*task) for task in TASK_ORDER])
        limit = max(25, float(np.nanquantile(np.abs(pivot.to_numpy()), 0.95)))
        im = vector_heatmap(ax, pivot.to_numpy(), cmap="RdBu", norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value_here = float(pivot.iloc[i, j])
                shown_text = "0" if abs(value_here) < 0.5 else f"{value_here:+.0f}"
                ax.text(j, i, shown_text, ha="center", va="center", fontsize=6.8, color="white" if abs(value_here) > limit * 0.55 else PALETTE["dark"])
        ax.set_xticks(range(6), [task_label(*task) for task in TASK_ORDER], rotation=35, ha="right"); ax.set_yticks(range(2), ["Seen", "Unseen"]); title(ax, panel_title)
        cb = vector_colorbar(fig, im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Skill (%)")

    ax = axes[1, 0]
    for task in TASK_ORDER:
        q = comp.loc[(comp["outcome"].eq(task[0])) & (comp["horizon"].eq(task[1]))].set_index("partition")
        ys = [q.loc["test_temporal_seen_country", "skill_pooled"], q.loc["test_spatiotemporal_unseen_country", "skill_pooled"]]
        ax.plot([0, 1], ys, marker="o" if task[0] == "daly" else "s", color=OUTCOME_COLORS[task[0]], alpha=0.35 + 0.12 * task[1], label=task_label(*task))
        endpoint_offset = -7 if task == ("daly", 1) else (5 if task == ("death", 1) else 0)
        ax.annotate(task_label(*task), (1, ys[1]), xytext=(4, endpoint_offset), textcoords="offset points", fontsize=6.4, va="center", color=OUTCOME_COLORS[task[0]])
    ax.axhline(0, color=PALETTE["dark"], lw=0.7); ax.set_xticks([0, 1], ["Seen countries", "Unseen countries"]); ax.set_xlim(-0.1, 1.42); ax.set_ylabel("Skill versus pooled ridge (%)"); title(ax, "Transportability collapse by task"); clean_axis(ax)

    ax = axes[1, 1]
    local = predictions.loc[predictions["partition"].eq("test_spatiotemporal_unseen_country") & predictions["horizon"].eq(5) & (predictions["selected_by_cv"] | predictions["family"].eq("pooled_ridge"))].copy()
    local["absolute_log_error"] = (np.log1p(local["prediction"]) - np.log1p(local["observed"])).abs()
    for outcome in ["daly", "death"]:
        for family, linestyle in [("pooled_ridge", "--"), (None, "-")]:
            q = local.loc[local["outcome"].eq(outcome)]
            if family is None: q = q.loc[q["selected_by_cv"]]; label = f"{OUTCOME_LABEL[outcome]} locked"
            else: q = q.loc[q["family"].eq(family)]; label = f"{OUTCOME_LABEL[outcome]} pooled"
            country_error = q.groupby("location_id")["absolute_log_error"].sum().to_numpy()
            x, y = lorenz_curve(country_error)
            ax.plot(x * 100, y * 100, color=OUTCOME_COLORS[outcome], ls=linestyle, label=label)
    ax.plot([0, 100], [0, 100], color=PALETTE["light"], ls=":", lw=0.7); ax.set_xlabel("Countries ordered by error contribution (%)"); ax.set_ylabel("Cumulative absolute log error (%)"); title(ax, "Country-level concentration of 5-y error"); clean_axis(ax); ax.legend(ncol=2, fontsize=6.6)

    ax = axes[2, 0]
    country = country_rmsle_table(predictions, "test_spatiotemporal_unseen_country", 5)
    rank = pd.concat([country.nsmallest(5, "delta_vs_pooled"), country.nlargest(7, "delta_vs_pooled")]).sort_values("delta_vs_pooled")
    y = np.arange(len(rank)); colors = [OUTCOME_COLORS[o] for o in rank["outcome"]]
    ax.hlines(y, 0, rank["delta_vs_pooled"], color=colors, lw=1.0); ax.scatter(rank["delta_vs_pooled"], y, color=colors, s=20)
    ax.axvline(0, color=PALETTE["dark"], lw=0.7); ax.set_yticks(y, [f"{OUTCOME_LABEL[o]} | {n}" for o, n in zip(rank["outcome"], rank["location_name"])]); ax.set_xlabel("Selected minus pooled-ridge RMSLE"); title(ax, "Countries driving improvement or failure"); clean_axis(ax, "x")

    ax = axes[2, 1]
    worst = country.loc[country["outcome"].eq("daly")].nlargest(1, "delta_vs_pooled").iloc[0]
    q = predictions.loc[
        predictions["partition"].eq("test_spatiotemporal_unseen_country")
        & predictions["outcome"].eq("daly")
        & predictions["horizon"].eq(5)
        & predictions["location_id"].eq(worst["location_id"])
        & (predictions["selected_by_cv"] | predictions["family"].eq("pooled_ridge"))
    ].copy()
    observed = q[["target_year", "observed"]].drop_duplicates().sort_values("target_year")
    ax.plot(observed["target_year"], observed["observed"], marker="o", color=PALETTE["dark"], label="Observed")
    selected = q.loc[q["selected_by_cv"]].sort_values("target_year")
    pooled = q.loc[q["family"].eq("pooled_ridge")].sort_values("target_year")
    ax.plot(selected["target_year"], selected["prediction"], marker="D", color=PALETTE["teal"], label="Locked Extra Trees")
    ax.plot(pooled["target_year"], pooled["prediction"], marker="o", markerfacecolor="white", color=PALETTE["navy"], label="Pooled ridge")
    ax.set_xlabel("Target year"); ax.set_ylabel("BMI-attributable DALY ASR"); title(ax, f"Failure case: {worst['location_name']}"); clean_axis(ax); ax.legend()

    write_csv(comp, FIGDATA / "Fig9_task_skill_synthesis.csv")
    write_csv(country, FIGDATA / "Fig9_country_h5_comparison.csv")
    write_csv(rank, FIGDATA / "Fig9_ranked_country_examples.csv")
    fig.subplots_adjust(top=0.985, bottom=0.11, left=0.24, right=0.985, hspace=0.56, wspace=0.42)
    save_figure(fig, PDFDIR / "Fig9_synthesis.pdf", {"Title": "Figure 9 synthesis"})
    plt.close(fig)


def write_manifest() -> None:
    rows = [
        (1, "Fig1_workflow.pdf", 1, "Overall model and validation framework"),
        (2, "Fig2_burden_landscape.pdf", 5, "Descriptive burden landscape"),
        (3, "Fig3_validation_and_shift.pdf", 5, "Validation design and distribution shift"),
        (4, "Fig4_cv_benchmark.pdf", 5, "Development-only model benchmark"),
        (5, "Fig5_seen_country_test.pdf", 6, "Sealed seen-country test"),
        (6, "Fig6_unseen_country_test.pdf", 6, "Sealed unseen-country test"),
        (7, "Fig7_explanation_ablation.pdf", 6, "Explanation and ablation"),
        (8, "Fig8_robustness.pdf", 6, "Robustness analyses"),
        (9, "Fig9_synthesis.pdf", 6, "Transportability synthesis"),
    ]
    manifest = []
    for number, filename, panels, purpose in rows:
        path = PDFDIR / filename
        svg_path = FIGDIR / "svg" / Path(filename).with_suffix(".svg").name
        manifest.append(
            {
                "figure": f"Fig{number}",
                "panels": panels,
                "purpose": purpose,
                "pdf_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "pdf_bytes": path.stat().st_size,
                "pdf_sha256": sha256(path),
                "svg_path": str(svg_path.relative_to(ROOT)).replace("\\", "/"),
                "svg_bytes": svg_path.stat().st_size,
                "svg_sha256": sha256(svg_path),
                "font_requirement": "Helvetica (PDF Core 14)",
                "vector_requirement": "PDF plus raster-free SVG with Helvetica text styling",
                "raster_requirement": "TIF and PNG 600 dpi generated from this PDF",
            }
        )
    pd.DataFrame(manifest).to_csv(FIGDIR / "figure_manifest.csv", index=False)


def main() -> None:
    PDFDIR.mkdir(parents=True, exist_ok=True)
    FIGDATA.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    for number, function in enumerate(
        [figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8, figure9],
        start=1,
    ):
        function(inputs)
        print(f"[done] Figure {number}", flush=True)
    write_manifest()
    print(json.dumps({"status": "complete", "figures": 9, "manifest": str(FIGDIR / "figure_manifest.csv")}, indent=2))


if __name__ == "__main__":
    main()
