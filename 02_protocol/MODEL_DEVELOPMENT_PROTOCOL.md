# Prespecified model-development and validation protocol

**Protocol version:** 1.0  
**Locked before test-set access:** 2026-08-10  
**Study type:** retrospective ecological prediction study using one finalized GBD 2023 data vintage

## 1. Scientific question and estimand

Can a country-pooled, history-aware machine-learning or compact deep sequence model improve 1-, 3-, and 5-year prediction of age-standardized liver-cancer burden attributable to high body-mass index over strong transparent trend baselines, while retaining performance in countries excluded from parameter estimation?

The estimand is country-level future burden, not patient-level risk. The outcome is **liver cancer**, not histology-confirmed hepatocellular carcinoma (HCC).

## 2. Outcomes and horizons

- Primary outcome: BMI-attributable liver-cancer DALY age-standardized rate per 100,000.
- Key secondary outcome: BMI-attributable liver-cancer death age-standardized rate per 100,000.
- Forecast horizons: 1, 3, and 5 years.
- Primary policy horizon: 5 years. Results for 1 and 3 years remain mandatory and cannot be omitted based on performance.
- No projection beyond the observed 2023 endpoint, including 2050 projection, will be reported until this shorter-horizon validation is completed and the model gate passes.

Models are trained separately for each horizon. This prevents a short-horizon sample with a later origin year from leaking future information into a long-horizon model evaluated from an earlier origin.

## 3. Analytic panel

- 204 countries/territories.
- Annual observations from 1990 through 2023.
- 6,936 unique `location_id`-year rows.
- Primary key: numeric `location_id` plus year.
- Display names are never used for joins or features.
- Direct GBD `Percent` estimates are retained as attributable fractions. Ratios of age-standardized rates are stored separately as sensitivity quantities because the two definitions are not identical.

Seven longitudinal signals available at each forecast origin are used:

1. BMI-attributable DALY ASR.
2. BMI-attributable death ASR.
3. Direct GBD BMI-attributable DALY fraction.
4. Direct GBD BMI-attributable death fraction.
5. Overall liver-cancer DALY ASR.
6. Overall liver-cancer death ASR.
7. SDI.

The lower/upper GBD uncertainty width at the origin is available as a model feature. Future lower/upper GBD estimates are outcomes for reporting only and cannot enter features.

## 4. Feature engineering

For each signal and forecast origin, the tabular pipeline computes only information dated at or before the origin:

- current value;
- lags 1, 3, 5, and 10 years;
- absolute changes and annualized log changes over the same lags;
- rolling means, standard deviations, and coefficients of variation over 3, 5, and 10 years;
- log-linear slopes over 5 and 10 years;
- current relative GBD uncertainty-interval width;
- current attributable-versus-overall burden gaps and the difference between direct GBD Percent and ASR-ratio definitions.

There are 179 prespecified tabular features. Country ID, country name, SDI quintile based on 2023, split labels, target year, target values, and future uncertainty intervals are excluded. No imputation, scaling, feature selection, or dimensionality reduction is performed before fold assignment.

The deep sequence input uses the trailing 10 annual observations ending at the origin. Static input is limited to current SDI and uncertainty-width summaries. It does not use a learned country embedding, so unseen-country evaluation is possible.

## 5. Locked country split

A deterministic seed (`20260810`) assigned 41/204 countries (20.1%) to a geographic holdout, stratified only by 2023 SDI quintile. Outcomes were not used to choose or reroll the split.

| 2023 SDI quintile | Development | Geographic holdout |
|---|---:|---:|
| Q1 | 32 | 9 |
| Q2 | 33 | 8 |
| Q3 | 32 | 8 |
| Q4 | 33 | 8 |
| Q5 | 33 | 8 |
| **Total** | **163** | **41** |

The holdout includes exactly 4 of the 20 highest-burden countries, matching the 20% expectation. Maximum absolute standardized difference between development and holdout variables is 0.281 and maximum empirical KS distance is 0.223. This moderate shift is retained as a stress test, not optimized away.

Interpretation: the geographic test measures parameter transportability to countries excluded from model fitting **with their own pre-origin history available**. It is not a no-history cold-start experiment.

## 6. Temporal development and final tests

For every outcome-horizon combination:

- Development countries and target years through 2018 form the development set.
- Development-country targets in 2019–2023 form the locked temporal test.
- Geographic-holdout-country targets in 2019–2023 form the locked spatiotemporal transportability test.
- Pre-2019 rows from geographic-holdout countries are reserved only to construct their historical input features and are never used to estimate parameters, preprocessing, hyperparameters, or conformal residual quantiles.

The final tests contain five target years per country and horizon:

- Temporal/seen-country test: 163 x 5 = 815 rows per horizon.
- Spatiotemporal/unseen-country test: 41 x 5 = 205 rows per horizon.

## 7. Expanding-window cross-validation

Hyperparameters are selected only within development countries using three expanding validation blocks:

| Fold | Training target years | Validation target years |
|---|---|---|
| CV1 | <=2012 | 2013–2014 |
| CV2 | <=2014 | 2015–2016 |
| CV3 | <=2016 | 2017–2018 |

All preprocessing is fitted anew on each fold's training rows. Validation rows are never used to select transformations. The 2019–2023 test labels remain unopened until candidate families, search spaces, and selection rules are frozen.

## 8. Prespecified model families

### Transparent baselines

1. Persistence (last observed value).
2. Five-year local log trend.
3. Ten-year local log trend.
4. Fold-fitted pooled regularized log-linear trend.
5. ARIMA/ETS sensitivity where convergence and runtime permit; these cannot be declared primary without winning validation.

### Machine learning

1. Ridge/Elastic Net on log-transformed outcome.
2. Random Forest / Extra Trees.
3. Histogram gradient boosting.
4. XGBoost with conservative depth, subsampling, and early stopping.

### Deep learning

A compact gated recurrent unit (GRU) or temporal convolutional network consumes the trailing 10-year multivariate sequence. Capacity is constrained, dropout and weight decay are tuned, early stopping uses only the current CV validation block, and five training seeds are retained for the locked configuration. A deep model is reported even if it does not win; superiority will never be assumed from architecture alone.

## 9. Hyperparameter selection

- Separate search for each outcome and horizon.
- Primary tuning loss: mean validation RMSLE across CV1–CV3.
- Ties within 1% of the best mean RMSLE are resolved in favor of the simpler model, lower fold-to-fold variance, and fewer effective parameters, in that order.
- Search spaces and random seeds are written to machine-readable configuration before final-test evaluation.
- Test performance cannot change the selected family or hyperparameters.

## 10. Evaluation metrics

Metrics are reported on the original rate scale unless marked otherwise:

- RMSLE (primary selection metric).
- MAE.
- RMSE.
- WAPE.
- Median absolute error.
- R-squared, with negative values retained.
- Spearman rank correlation.
- Skill relative to persistence and relative to the best transparent trend baseline: `1 - error_model / error_baseline`.

MAPE is not primary because death rates can be close to zero. Metrics are computed both across all country-year rows and as country-macro averages. Inferential uncertainty uses country-clustered bootstrap resampling; country-year rows are not treated as independent.

## 11. Prediction intervals and calibration

- Out-of-fold log1p residuals from CV1–CV3 form a cross-conformal calibration set.
- Prespecified 90% and 95% predictive intervals use absolute log-residual quantiles and are transformed back to non-negative rate space.
- Report empirical coverage, mean/median interval width, and coverage by horizon, outcome, test partition, SDI quintile, and burden quartile.
- GBD lower/upper source intervals are reported separately from predictive intervals. They answer different uncertainty questions and are never merged or relabeled.

## 12. Robustness and bias analyses

Mandatory analyses are:

1. Feature-family ablation: attributable-history only; + overall liver-cancer history; + direct attributable fractions; + SDI; + source-uncertainty width.
2. Five-seed stability for the selected tree and deep configurations.
3. Performance by 2023 SDI quintile, forecast horizon, target year, and observed-burden quartile.
4. Error analysis for the top 20 burden countries and the four held-out top-20 locations.
5. Country-clustered bootstrap confidence intervals for error and paired model-minus-baseline differences.
6. Distribution-shift diagnostics between development and each final test.
7. Permutation or SHAP feature importance with features aggregated into interpretable families; explanations are descriptive, not causal.
8. Sensitivity excluding the largest 1% of target rates.
9. Sensitivity using the ASR ratio instead of direct GBD Percent as a historical fraction feature.

## 13. Model-stage pass criteria

The model/evaluation stage cannot score >=90 unless all of the following are satisfied:

1. Every outcome-horizon combination has complete CV and both locked test predictions.
2. Strong transparent baselines are included and no baseline is selectively omitted.
3. The selected model improves primary RMSLE over persistence; any superiority over the best trend baseline is reported with a country-bootstrap interval.
4. No test data affect preprocessing, tuning, early stopping, conformal calibration, or model-family selection.
5. 90% and 95% interval coverage and width are reported, including subgroup failures.
6. Deep-model results are reproducible across five seeds and are honestly labeled if inferior.
7. All predictions, fitted objects, configurations, environment versions, and random seeds are saved and hashed.

If advanced models do not outperform simple trends, the scientific conclusion will be a rigorously validated negative benchmark rather than a fabricated performance claim.

## 14. Structural limitations fixed in advance

- Retrospective GBD estimates are generated in one finalized vintage and may incorporate information that was unavailable in the historical year being predicted. Results are therefore retrospective pseudo-out-of-sample validation, not a real-time deployment study.
- GBD source estimates are modeled and strongly autocorrelated; they are not independent surveillance observations.
- Posterior draws and temporal covariance of source uncertainty are unavailable.
- Country-level findings cannot be interpreted as individual risk, causal effect, clinical prognosis, or histology-specific HCC prediction.
