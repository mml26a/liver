# Stage 4 figure-production gate

**Status: PASS — 98.5/100 (required: >=90).**

| Dimension | Points | Maximum |
|---|---:|---:|
| Complete PDF/SVG/TIFF/PNG asset set | 20.0 | 20.0 |
| Panel-count specification | 10.0 | 10.0 |
| Helvetica-only PDF typography | 15.0 | 15.0 |
| Single-page PDF/SVG vector and 600-dpi raster preflight | 15.0 | 15.0 |
| Reproducible code, hashes and figure-data inventory | 15.0 | 15.0 |
| Non-empty and finite figure-data integrity | 10.0 | 10.0 |
| Manual visual QA | 8.5 | 10.0 |
| Captions, map provenance and pinned geometry | 5.0 | 5.0 |

- Final assets: 9 vector PDFs, 9 raster-free SVGs, 9 600-dpi LZW-TIFFs, 9 600-dpi PNG fallbacks and 9 QA previews.
- Composition: Figure 1 is one integrated framework diagram; Figures 2-9 contain four to nine panels each (46 total graphical units).
- Typography: all PDF font resources are Helvetica or Helvetica-Bold; SVG text is explicitly styled Helvetica with Arial only as a system fallback; no DejaVu or Times resource remains.
- Word readiness: each SVG is fully vector (no embedded image payload), and exported artwork contains no redundant global figure-title banner.
- Reproducibility: 21 non-empty figure-data tables are hashed; generation and audit sources compile.
- Integrity: every fixed-size benchmark table has its expected row count and no numeric infinity was found.
- Visual QA: every final preview passed inspection after correction of signed-minus glyphs, annotation collisions, cross-panel colour-bar intrusion and legend obstruction.
- Scope: the scientific result is intentionally negative where supported. The figures do not claim that the locked complex models outperform pooled ridge, and they retain the aggregate liver-cancer rather than histology-confirmed HCC label.

A 1.5-point reserve is retained for a journal-specific compositor proof and publisher preflight, which cannot be completed locally. This reserve does not block progression because every scientific and locally testable production criterion passed.
