from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data"
PROTOCOL = ROOT / "02_protocol"
SOURCE = DATA / "supervised_samples_h1_h3_h5.csv.gz"
DEVELOPMENT = DATA / "supervised_development_locked.csv.gz"
FINAL_TEST = DATA / "supervised_final_tests_sealed.csv.gz"
RESERVED_HISTORY = DATA / "reserved_unseen_country_history_sealed.csv.gz"
MANIFEST = DATA / "model_partition_manifest.json"
PANEL_SOURCE = DATA / "analytic_panel_204x34.csv.gz"
DEVELOPMENT_PANEL = DATA / "analytic_panel_development_sequence_locked.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_options() -> dict[str, object]:
    return {"method": "gzip", "compresslevel": 6, "mtime": 0}


def main() -> None:
    config = json.loads((PROTOCOL / "model_config_locked.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(SOURCE, low_memory=False)
    development = frame.loc[frame["partition"].eq(config["development_partition"])].copy()
    final_test = frame.loc[frame["partition"].isin(config["final_test_partitions"])].copy()
    reserved_history = frame.loc[frame["partition"].eq("reserved_unseen_country_history")].copy()

    if len(development) + len(final_test) + len(reserved_history) != len(frame):
        allowed = {
            config["development_partition"],
            *config["final_test_partitions"],
            "reserved_unseen_country_history",
        }
        unexpected = sorted(set(frame["partition"]) - allowed)
        raise ValueError(f"Unexpected or unassigned partitions: {unexpected}")
    if set(development["sample_id"]).intersection(final_test["sample_id"]):
        raise ValueError("Development/final-test sample overlap")
    if development["target_year"].max() > 2018:
        raise ValueError("Development file contains target years after 2018")
    if sorted(final_test["target_year"].unique().tolist()) != config["final_test_target_years"]:
        raise ValueError("Sealed test target years do not match the lock")
    if list(development.columns) != list(final_test.columns) or list(development.columns) != list(reserved_history.columns):
        raise ValueError("Partition schemas differ")

    sort_columns = ["horizon", "target_year", "location_id", "sample_id"]
    development = development.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    final_test = final_test.sort_values(
        ["partition", *sort_columns], kind="mergesort"
    ).reset_index(drop=True)
    development.to_csv(
        DEVELOPMENT,
        index=False,
        float_format="%.12g",
        compression=gzip_options(),
    )
    final_test.to_csv(
        FINAL_TEST,
        index=False,
        float_format="%.12g",
        compression=gzip_options(),
    )
    reserved_history = reserved_history.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    reserved_history.to_csv(
        RESERVED_HISTORY,
        index=False,
        float_format="%.12g",
        compression=gzip_options(),
    )
    panel = pd.read_csv(PANEL_SOURCE, low_memory=False)
    development_locations = set(development["location_id"].astype(int).unique())
    max_development_origin = int(development["origin_year"].max())
    development_panel = panel.loc[
        panel["location_id"].astype(int).isin(development_locations)
        & panel["year"].le(max_development_origin)
    ].copy()
    development_panel = development_panel.sort_values(
        ["location_id", "year"], kind="mergesort"
    ).reset_index(drop=True)
    if development_panel["year"].max() != max_development_origin:
        raise ValueError("Development sequence panel cutoff mismatch")
    if development_panel["location_id"].nunique() != len(development_locations):
        raise ValueError("Development sequence panel country mismatch")
    development_panel.to_csv(
        DEVELOPMENT_PANEL,
        index=False,
        float_format="%.12g",
        compression=gzip_options(),
    )

    manifest = {
        "partitioning_role": "one-time data-steward materialization before formal model search",
        "source": {"path": SOURCE.name, "rows": int(len(frame)), "sha256": sha256(SOURCE)},
        "development": {
            "path": DEVELOPMENT.name,
            "rows": int(len(development)),
            "target_year_min": int(development["target_year"].min()),
            "target_year_max": int(development["target_year"].max()),
            "partitions": sorted(development["partition"].unique().tolist()),
            "sha256": sha256(DEVELOPMENT),
        },
        "sealed_final_test": {
            "path": FINAL_TEST.name,
            "rows": int(len(final_test)),
            "target_years": config["final_test_target_years"],
            "partitions": config["final_test_partitions"],
            "sha256": sha256(FINAL_TEST),
            "access_policy": "Do not load before model_selection_lock.json is finalized and hashed.",
        },
        "sealed_unseen_country_history": {
            "path": RESERVED_HISTORY.name,
            "rows": int(len(reserved_history)),
            "partitions": ["reserved_unseen_country_history"],
            "sha256": sha256(RESERVED_HISTORY),
            "access_policy": "Never use for fitting or model selection; retained only as an audit trace of excluded rows.",
        },
        "development_sequence_panel": {
            "path": DEVELOPMENT_PANEL.name,
            "rows": int(len(development_panel)),
            "locations": int(development_panel["location_id"].nunique()),
            "year_min": int(development_panel["year"].min()),
            "year_max": int(development_panel["year"].max()),
            "sha256": sha256(DEVELOPMENT_PANEL),
            "source_sha256": sha256(PANEL_SOURCE),
        },
        "schema": {"n_columns": int(len(frame.columns)), "columns": frame.columns.tolist()},
        "checks": {
            "exhaustive_partition": True,
            "sample_id_overlap": 0,
            "schema_identical": True,
            "development_target_year_le_2018": True,
            "sealed_target_years_match_lock": True,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
