from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib as mpl


PALETTE = {
    "navy": "#3C5488",
    "cyan": "#4DBBD5",
    "teal": "#00A087",
    "red": "#E64B35",
    "orange": "#F39B7F",
    "purple": "#7E62A3",
    "gold": "#EFC000",
    "blue": "#2F6B9A",
    "green": "#3C8D6E",
    "dark": "#2B2B2B",
    "mid": "#737373",
    "light": "#D9D9D9",
    "pale": "#F2F2F2",
    "white": "#FFFFFF",
}

FAMILY_COLORS = {
    "persistence": "#8A8A8A",
    "logtrend5": "#B6B6B6",
    "logtrend10": "#6F6F6F",
    "pooled_ridge": PALETTE["navy"],
    "ridge": "#8DA0CB",
    "elastic_net": PALETTE["purple"],
    "random_forest": PALETTE["green"],
    "extra_trees": PALETTE["teal"],
    "hist_gradient_boosting": PALETTE["orange"],
    "xgboost": PALETTE["red"],
    "gru": PALETTE["gold"],
}

OUTCOME_COLORS = {"daly": PALETTE["navy"], "death": PALETTE["red"]}
PARTITION_COLORS = {
    "test_temporal_seen_country": PALETTE["navy"],
    "test_spatiotemporal_unseen_country": PALETTE["red"],
}


def setup_style() -> None:
    mpl.use("pdf")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial"],
            "pdf.use14corefonts": True,
            "ps.useafm": True,
            # These sizes are deliberately larger than Matplotlib defaults for
            # dense multi-panel biomedical figures. At the 6.2-inch manuscript
            # insertion width, the smallest standard tick/legend text remains
            # approximately 6.3-6.6 pt rather than becoming unreadable.
            "font.size": 8.3,
            "font.weight": "normal",
            "axes.unicode_minus": False,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": PALETTE["dark"],
            "axes.labelcolor": PALETTE["dark"],
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": 7.1,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.1,
            "patch.linewidth": 0.65,
            "figure.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
            "savefig.edgecolor": PALETTE["white"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "figure.dpi": 150,
            # Keep SVG text explicitly tagged as Helvetica. Arial is the
            # metric-compatible fallback on systems without Helvetica.
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E6E6E6", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color=PALETTE["dark"],
        clip_on=False,
    )


def task_label(outcome: str, horizon: int) -> str:
    return f"{'DALY' if outcome == 'daly' else 'Death'} {int(horizon)} y"


def family_label(family: str) -> str:
    return {
        "persistence": "Persistence",
        "logtrend5": "5-y trend",
        "logtrend10": "10-y trend",
        "pooled_ridge": "Pooled ridge",
        "ridge": "Ridge",
        "elastic_net": "Elastic net",
        "random_forest": "Random forest",
        "extra_trees": "Extra Trees",
        "hist_gradient_boosting": "HistGB",
        "xgboost": "XGBoost",
        "gru": "GRU",
    }.get(family, family.replace("_", " ").title())


def short_feature(name: str, max_length: int = 33) -> str:
    replacements = {
        "bmi_daly_rate": "BMI-DALY",
        "bmi_death_rate": "BMI-death",
        "gbd_daly_fraction": "DALY fraction",
        "gbd_death_fraction": "Death fraction",
        "overall_daly_rate": "Overall DALY",
        "overall_death_rate": "Overall death",
        "derived": "Derived",
        "sdi": "SDI",
        "rel_ui_width": "UI width",
        "log_growth": "log growth",
        "log_slope": "log slope",
        "definition_gap_pp": "definition gap",
        "current_daly_rate_gap": "current DALY gap",
        "current_death_rate_gap": "current death gap",
    }
    text = name
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("__", " ").replace("_", " ")
    text = " ".join(text.split())
    if len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig, path: Path, metadata: dict[str, str] | None = None) -> None:
    """Save the editable PDF and a Word-compatible SVG vector counterpart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {"Creator": "Reproducible Matplotlib workflow"}
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.035,
        metadata=metadata,
    )
    if path.parent.name == "pdf":
        svg_dir = path.parent.parent / "svg"
    elif path.parent.name == "main_pdf":
        svg_dir = path.parent.parent / "main_svg"
    elif path.parent.name == "supplement_pdf":
        svg_dir = path.parent.parent / "supplement_svg"
    else:
        svg_dir = path.parent / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_metadata = {
        "Title": metadata.get("Title", path.stem),
        "Creator": metadata.get("Creator", metadata.get("Author", "Reproducible Matplotlib workflow")),
    }
    fig.savefig(
        svg_dir / path.with_suffix(".svg").name,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.035,
        metadata=svg_metadata,
    )


def write_csv(frame, path: Path, columns: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, list(columns)] if columns is not None else frame
    output.to_csv(path, index=False, float_format="%.10g")
