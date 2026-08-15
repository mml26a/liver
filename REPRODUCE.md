# Reproducing GBD-LiverBench

## 1. Environment

Python 3.12 was used for the locked analysis. Create a clean environment and install:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Compare installed versions with `02_protocol/environment_versions.json`. Small platform-dependent numerical differences are possible; do not change the prespecified design or reselect models after viewing sealed tests.

## 2. Inputs

Obtain all six files listed in `DATA_ACCESS.md` and place them in `data_raw/`. The build scripts validate cause, risk, age, sex, metric, location and year fields before analysis.

## 3. Locked run order

From the repository root:

```bash
python 00_audit/audit_raw_data.py
python 01_data/build_analytic_panel.py
python 01_data/audit_split_bias.py
python 01_data/materialize_model_partitions.py
python 01_data/build_descriptive_aggregate_panel.py
python 03_models/run_tabular_cv.py
python 03_models/run_gru_cv.py
python 03_models/audit_cv_completion.py
python 03_models/lock_model_selection.py
```

Before opening the test data, verify that `03_models/selection_lock/model_selection_lock.json` has SHA-256:

```text
49bdbb1929a31d257aedc7bd747c05691e222db2ba28e2a607f52cad9c74d147
```

Then run the one-time sealed evaluation and downstream analyses:

```bash
python 04_evaluation/fit_final_and_evaluate.py --confirm-open-sealed-test
python 04_evaluation/analyze_final_predictions.py
python 04_evaluation/run_seed_stability.py
python 04_evaluation/run_feature_ablations.py
python 04_evaluation/run_fraction_definition_sensitivity.py
python 04_evaluation/compute_shap_importance.py
python 04_evaluation/score_model_stage.py
python 06_figures/make_main_figures.py
python 06_figures/rasterize_master_figures.py
python 06_figures/audit_figures.py
```

## 4. Expected high-level checks

- 204 countries: 163 development and 41 geographic holdout.
- Development targets end in 2018; sealed targets span 2019-2023.
- Six locked tasks: GRU at 1 year; Extra Trees at 3 and 5 years.
- Twenty-one figure-level CSV files and nine PDF/SVG figure pairs.
- Aggregate verification values are supplied under `03_models/`, `04_evaluation/`, and `06_figures/figure_data/`.

This is a retrospective one-vintage pseudo-out-of-sample benchmark, not a real-time forecast archive, patient-level prediction model, or causal analysis.
