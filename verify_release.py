"""Verify the public release directory before upload."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROHIBITED_SUFFIXES = {".gz", ".joblib", ".pt", ".pth", ".pkl", ".docx", ".tif", ".tiff"}
PROHIBITED_NAMES = {"data_raw", "__pycache__", ".venv"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

files = [p for p in ROOT.rglob("*") if p.is_file()]
prohibited = [p.relative_to(ROOT).as_posix() for p in files if p.suffix.lower() in PROHIBITED_SUFFIXES or any(part in PROHIBITED_NAMES for part in p.parts)]
secrets = []
pattern = re.compile(r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*\S+")
for path in files:
    if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".cff", ".yml", ".yaml"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text):
            secrets.append(path.relative_to(ROOT).as_posix())
lock = ROOT / "03_models" / "selection_lock" / "model_selection_lock.json"
expected_lock = "49bdbb1929a31d257aedc7bd747c05691e222db2ba28e2a607f52cad9c74d147"
manifest = ROOT / "MANIFEST_SHA256.csv"
manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8"))) if manifest.exists() else []
manifest_errors = []
for row in manifest_rows:
    path = ROOT / row["path"]
    if not path.exists() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
        manifest_errors.append(row["path"])
checks = {
    "prohibited_public_files": prohibited,
    "possible_secrets": secrets,
    "selection_lock_matches": lock.exists() and sha256(lock) == expected_lock,
    "figure_data_files": len(list((ROOT / "06_figures" / "figure_data").glob("*.csv"))),
    "figure_pdfs": len(list((ROOT / "06_figures" / "pdf").glob("*.pdf"))),
    "figure_svgs": len(list((ROOT / "06_figures" / "svg").glob("*.svg"))),
    "manifest_entries": len(manifest_rows),
    "manifest_errors": manifest_errors,
}
status = not prohibited and not secrets and checks["selection_lock_matches"] and checks["figure_data_files"] == 21 and checks["figure_pdfs"] == 9 and checks["figure_svgs"] == 9 and not manifest_errors
print(json.dumps({"status": "PASS" if status else "FAIL", **checks}, indent=2))
raise SystemExit(0 if status else 1)
