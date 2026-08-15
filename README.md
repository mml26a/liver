# GBD-LiverBench

**GBD-LiverBench: A sealed temporal and geographic benchmark of machine learning for BMI-attributable liver-cancer burden forecasting**

GBD-LiverBench is a leakage-controlled retrospective benchmark for evaluating whether machine-learning gains in country-level burden forecasting survive temporal and geographic shift. It uses one finalized GBD 2023 vintage for 204 countries (1990-2023) and predicts BMI-attributable liver-cancer DALY and death age-standardised rates at 1-, 3-, and 5-year horizons.

This is an evaluation framework, not a newly proposed prediction algorithm. The benchmark compares transparent trends, regularised regression, tree ensembles, gradient boosting, XGBoost, and a compact GRU under an immutable model-selection lock.

## Study design

- 163 development countries and 41 whole-country geographic holdouts.
- Development targets end in 2018; 2019-2023 targets are sealed.
- Three expanding temporal validation folds.
- Eleven candidate model families and six outcome-horizon tasks.
- Locked selection SHA-256: `49bdbb1929a31d257aedc7bd747c05691e222db2ba28e2a607f52cad9c74d147`.
- Selected families: GRU at 1 year and Extra Trees at 3 and 5 years for both outcomes.

The final result is deliberately negative: development-only gains did not transport reliably. All six locked models were worse than pooled ridge in both sealed partitions, and empirical predictive-interval coverage was below nominal.

## Repository contents

| Path | Contents |
|---|---|
| `00_audit/` | Raw-input validation code |
| `01_data/` | Panel construction, split and partition scripts; public split only |
| `02_protocol/` | Locked protocol, configuration, feature lineage and environment |
| `03_models/` | CV/model-lock code and aggregate locked outputs |
| `04_evaluation/` | Sealed evaluation, robustness, calibration and explainability code/results |
| `06_figures/` | Figure code, source tables, PDF/SVG figures and map asset |
| `DATA_ACCESS.md` | Required inputs and redistribution boundary |
| `REPRODUCE.md` | Reproduction commands and expected checks |
| `EXCLUDED_FILES.csv` | Hashes of controlled artifacts not uploaded |
| `MANIFEST_SHA256.csv` | Hash and size of every release file |

## Data boundary

Raw IHME/GBD exports, reconstructed analytic panels, complete sample-level prediction outputs, and fitted models are **not included**. The release candidate retains only the aggregate verification tables and figure-level subsets needed to support the reported graphics; public release of those subsets still requires author confirmation under the agreement accepted by the actual data downloader. Obtain the six source exports directly from IHME, place them in `data_raw/`, and follow `REPRODUCE.md`.

## Citation and status

Use `CITATION.cff`. Replace the repository placeholders there after the GitHub URL and archival DOI are created. This directory is a **release candidate** until the corresponding authors approve the code license, public data boundary, author order, and repository metadata.

Source: Institute for Health Metrics and Evaluation (IHME). Used with permission. All rights reserved.
