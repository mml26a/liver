# Model-partition isolation addendum

## Reason for the addendum

The first supervised artifact stored development, final-test, and reserved-history rows in one compressed table. The initial smoke test filtered to `partition == development` before any fit, so no final-test target affected a transformation, parameter, stopping decision, or score. Nevertheless, parsing one monolithic file is weaker than a physically auditable blind-test boundary.

Before the formal search, the one-time data-steward script `01_data/materialize_model_partitions.py` therefore created three mutually exclusive, deterministic, separately hashed artifacts. Formal model outputs are written to `cv_search_blocks_v2` and `gru_cv_blocks_v2`; earlier smoke/search blocks are quarantined engineering traces and are not eligible evidence.

## Locked artifacts

| Role | Rows | Permitted access before selection lock | SHA-256 |
|---|---:|---|---|
| `supervised_development_locked.csv.gz` | 7,824 | CV fitting, preprocessing, tuning, inner early stopping, conformal calibration | `1053b8f86cb6d6f5aed1b4ce3f3fa00a4dd13155d56b90ee1056d8eac262375f` |
| `supervised_final_tests_sealed.csv.gz` | 3,060 | None | `99dea846e8ebca71d10df81f49b51f154a3e33f3511f309c06af6ea10568cd48` |
| `reserved_unseen_country_history_sealed.csv.gz` | 1,968 | None; never eligible for model fitting or selection | `464ca70bbc983efe91e5fe3417857f8d17f98b3bf14a787dd5512e70ae2ffda7` |
| `analytic_panel_development_sequence_locked.csv.gz` | 4,564 | GRU development sequences only (163 development countries, 1990–2017) | `699f4821ceb3ba249a93e739534b6346bf67fd6e34a5b8a84fcea4d5269ea1ea` |

The source supervised table is exhausted exactly by the three row partitions, sample identifiers do not overlap, schemas are identical, development targets end in 2018, and sealed-test targets are exactly 2019–2023. Full machine-readable checks and paths are recorded in `01_data/model_partition_manifest.json`.

## Test-opening rule

The sealed final-test artifact may be opened only after:

1. all tabular and GRU CV blocks are complete;
2. one configuration per family and one final family per outcome–horizon task are selected from development-only OOF predictions;
3. `03_models/selection_lock/model_selection_lock.json` is written and its SHA-256 is frozen; and
4. the final evaluation command is called with the explicit `--confirm-open-sealed-test` flag.

After opening, model family, hyperparameters, preprocessing, early-stopping rounds, conformal quantiles, and transparent reference models cannot change.
