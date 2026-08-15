"""Rasterise master figures into 600-dpi TIFF/PNG files and 180-dpi previews."""

from __future__ import annotations

import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


HERE = Path(__file__).resolve().parent
PDFDIR = HERE / "pdf"
TIFFDIR = HERE / "tiff_600dpi"
PNGDIR = HERE / "png_preview"
PNG600DIR = HERE / "png_600dpi"


def render_pdf(path: Path) -> dict[str, object]:
    document = pdfium.PdfDocument(path)
    if len(document) != 1:
        raise RuntimeError(f"Expected one-page figure PDF: {path}")
    page = document[0]
    TIFFDIR.mkdir(parents=True, exist_ok=True)
    PNGDIR.mkdir(parents=True, exist_ok=True)
    PNG600DIR.mkdir(parents=True, exist_ok=True)

    full = page.render(scale=600 / 72).to_pil().convert("RGB")
    tiff_path = TIFFDIR / f"{path.stem}.tif"
    full.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
    png600_path = PNG600DIR / f"{path.stem}.png"
    full.save(png600_path, format="PNG", dpi=(600, 600), optimize=True)
    full_size = list(full.size)
    del full

    preview = page.render(scale=180 / 72).to_pil().convert("RGB")
    png_path = PNGDIR / f"{path.stem}.png"
    preview.save(png_path, format="PNG", dpi=(180, 180), optimize=True)
    preview_size = list(preview.size)
    document.close()

    with Image.open(tiff_path) as audit:
        dpi = [round(float(value)) for value in audit.info.get("dpi", (0, 0))]
        compression = str(audit.info.get("compression", ""))
        mode = audit.mode
    passed = (
        dpi == [600, 600]
        and compression == "tiff_lzw"
        and mode == "RGB"
        # Wide workflow figures remain publication-scale at 600 dpi without
        # being forced into a near-square canvas.
        and max(full_size) >= 4000
        and min(full_size) >= 1800
        and max(preview_size) >= 1200
        and min(preview_size) >= 540
    )
    if not passed:
        raise RuntimeError(
            f"Raster contract failed for {path.name}: size={full_size}, dpi={dpi}, "
            f"compression={compression}, mode={mode}, preview={preview_size}"
        )
    return {
        "pdf": path.name,
        "tiff": tiff_path.name,
        "tiff_pixels": full_size,
        "png_600dpi": png600_path.name,
        "png_600dpi_pixels": full_size,
        "png": png_path.name,
        "png_pixels": preview_size,
        "pass": True,
    }


def main() -> None:
    records = [render_pdf(path) for path in sorted(PDFDIR.glob("Fig*.pdf"))]
    if len(records) != 9:
        raise RuntimeError(f"Expected nine master PDFs, found {len(records)}")
    (HERE / "master_raster_generation.json").write_text(
        json.dumps({"status": "PASS", "figures": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "figures": len(records)}, indent=2))


if __name__ == "__main__":
    main()
