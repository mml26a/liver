# Stage 2 data, leakage, and bias gate

**Gate date:** 2026-08-10  
**Decision:** **PASS — 96/100; model development is authorized**  
**Scope:** rebuilt analytic panel, feature lineage, target construction, country/temporal splits, dependence and bias audit, deterministic reproducibility, and prespecified model protocol

The score applies only to Stage 2. No fitted ML/DL model or performance claim exists yet.

## 1. Score

| Domain | Weight | Score | Evidence |
|---|---:|---:|---|
| ID-keyed panel integrity | 20 | 20 | 6,936 rows = 204 locations x 34 years; zero missing cells, duplicate keys, interval-order errors, or attributable-rate-above-overall errors |
| Estimand and target validity | 15 | 14 | Liver-cancer ecological outcomes and 1/3/5-year targets are explicit; one point withheld because no histology-specific or patient-level data exist |
| Feature lineage and leakage prevention | 20 | 19 | 179 features use only origin-year or earlier values; targets, identity, splits, future UIs, and 2023 assignment strata are excluded; one point withheld for the unavoidable single-vintage retrospective GBD limitation |
| Geographic and temporal validation design | 20 | 20 | 41 whole-country holdouts; 163-country development set; 2019–2023 locked temporal and spatiotemporal tests; three expanding CV blocks; separate models by horizon |
| Bias, shift, and dependence audit | 15 | 14 | All SDI strata represented; top-20 burden split is proportional; max absolute SMD 0.281 and KS 0.223; equal country counts; autocorrelation explicitly handled; one point withheld for moderate retained test shift |
| Reproducibility and protocol lock | 10 | 9 | Deterministic seed/config, SHA-256 outputs, two identical rebuilds, feature manifest and machine-readable search space; one point withheld until the ML/DL software environment is installed and frozen |
| **Total** | **100** | **96** | **PASS (threshold: >=90)** |

## 2. Rebuilt data products

| Product | Rows | Key facts | SHA-256 |
|---|---:|---|---|
| `analytic_panel_204x34.csv.gz` | 6,936 | 204 countries, 1990–2023, no missing/duplicate keys | `31493664da2bdfc93fff2d3fcdffd713da1830e0be8c4bfd7c9d9f8c8a1359bc` |
| `country_split_locked.csv` | 204 | 163 development, 41 geographic holdout | `05629470fa024429af80e63b62f4b80564b2c491bcfef0ea7acfad82053da0a9` |
| `supervised_samples_h1_h3_h5.csv.gz` | 12,852 | 1/3/5-year samples, 179 features, fixed partitions | `d8478c293b339e58486783ad10b71254947c2667d1b6052eff8b801b58386d78` |
| `feature_manifest.csv` | 179 | Availability and maximum source-year rule for every feature | `5ca626587728ea419b40e5066df03dea71a1a9f2e4fc84eff59e12cfea656dd8` |
| `model_config_locked.json` | — | Outcomes, horizons, folds, metrics, search spaces, seeds, bootstrap, conformal rules | `cbe6f035fdca5d863edd7fcdf7529f6ed01f8a849eade1f991b8f0e41f24c6c8` |

The panel and supervised sample files were rebuilt twice after fixing gzip metadata to `mtime=0`; all four hashes were identical across runs.

## 3. Partition accounting

| Partition | Rows | Use |
|---|---:|---|
| Development | 7,824 | Training and expanding-window CV only |
| Temporal test, seen countries | 2,445 | 163 countries x 5 target years x 3 horizons |
| Spatiotemporal test, unseen countries | 615 | 41 countries x 5 target years x 3 horizons |
| Reserved history from held-out countries | 1,968 | May construct their own lagged inputs; never used to fit parameters or calibration |

For every final test and horizon, each country contributes exactly five target years. Development-country counts are also equal within horizon, enabling country-macro evaluation without implicit population-size weighting.

## 4. Leakage controls that passed

- Maximum feature source year minus forecast-origin year: **0**.
- Minimum history required per sample: **11 annual observations**.
- Country ID/name included as model feature: **no**.
- 2023 SDI quintile used as model feature: **no**; it is split-assignment metadata only.
- Geographic-holdout rows in development: **0**.
- Targets after 2018 in development: **0**.
- Pre-split scaling/imputation/PCA/selection: **none**.
- Separate model per horizon: **required**, preventing cross-horizon origin-time leakage.
- Test partitions prohibited from preprocessing, tuning, early stopping, family selection, conformal calibration, and thresholds: **locked in config**.

## 5. Bias and dependence findings

- Maximum absolute development-versus-geographic-holdout SMD: **0.281**.
- Maximum empirical KS distance: **0.223**.
- Four of the 20 highest 2023 BMI-attributable DALY-ASR countries are held out, exactly matching the 20% allocation expectation.
- The holdout includes Mongolia and Nauru; the resulting distribution shift is deliberately retained as a rigorous transportability stress test.
- Median country-specific lag-1 autocorrelation is 0.982 for BMI-attributable DALY ASR and 0.983 for death ASR; direct attributable fractions and SDI exceed 0.999. Country-year rows therefore cannot be treated as independent.
- The outcome is strongly right-skewed (skewness 6.23 for DALY ASR and 6.98 for death ASR); primary tuning uses RMSLE and all error analyses retain original-scale metrics.

## 6. Metric-definition correction

The direct GBD `Percent` metric and `BMI-attributable ASR / overall ASR` are separate quantities after age standardization. Their median difference is approximately 0.012 percentage points; the maximum absolute difference is 0.985 percentage points for DALYs and 0.894 for deaths. The rebuilt panel preserves both and prohibits relabeling one as the other.

## 7. Known limitations carried forward, not hidden

1. GBD 2023 is one retrospectively harmonized vintage; historical estimates can reflect information unavailable in real time. Validation is therefore retrospective pseudo-out-of-sample evaluation.
2. The geographic test assumes that a held-out country has its own 10-year pre-origin history. It tests model-parameter transportability, not no-history cold start.
3. GBD posterior draws and temporal covariance are unavailable; source UIs and predictive intervals must remain separate.
4. Results are ecological and cannot support patient-level, causal, clinical-prognostic, or histology-specific HCC claims.

## 8. Gate decision

There are no unresolved Stage-2 P0 findings. Formal model development may begin using only the locked configuration. Stage 3 remains blocked until the software environment is frozen, every baseline/ML/DL family completes cross-validation, and final-test evaluation is executed exactly once after model selection.
