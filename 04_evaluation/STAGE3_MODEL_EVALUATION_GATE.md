# Stage 3 model and evaluation gate

**Score: 91.7/100 — PASS (required ≥90)**

This score is generated from artifact counts, hashes, locked selections, and recorded metrics. It is a workflow-quality gate, not a probability of journal acceptance.

| Criterion | Earned | Maximum | Evidence |
|---|---:|---:|---|
| Physical development/test isolation | 6.0 | 6.0 | development=7824; sealed test=3060; checks={'development_target_year_le_2018': True, 'exhaustive_partition': True, 'sample_id_overlap': 0, 'schema_identical': True, 'sealed_target_years_match_lock': True} |
| Pre-test family/configuration lock | 6.0 | 6.0 | lock_sha256=49bdbb1929a31d257aedc7bd747c05691e222db2ba28e2a607f52cad9c74d147; sealed_final_test_not_loaded=True |
| Complete CV comparison with strong baselines | 8.0 | 8.0 | rows=66; tasks=6; families_per_task=[np.int64(11)] |
| Nested temporal early stopping and three outer folds | 7.0 | 7.0 | xgboost_rows=540; xgboost_effective_rounds_complete=True; gru_nested=True |
| Both locked final tests predicted for every task/family | 7.0 | 7.0 | prediction_rows=67320; metric_rows=132 |
| No post-test reselection or target-informed preprocessing | 5.0 | 5.0 | {'model_family_selection_changed_after_test': False, 'preprocessing_or_early_stopping_used_test_targets': False, 'selection_lock_sha256': '49bdbb1929a31d257aedc7bd747c05691e222db2ba28e2a607f52cad9c74d147'} |
| Selected-model improvement over persistence across final tasks | 6.7 | 10.0 | improved=8/12 task-partitions |
| Primary DALY five-year improvement in both test partitions | 0.0 | 5.0 | [{'partition': 'test_spatiotemporal_unseen_country', 'family': 'extra_trees', 'rmsle': 0.210053692843, 'persistence_rmsle': 0.161190484079, 'improves_persistence': False}, {'partition': 'test_temporal_seen_country', 'family': 'extra_trees', 'rmsle': 0.124420886538, 'persistence_rmsle': 0.160827792128, 'improves_persistence': True}] |
| Country-clustered bootstrap for paired model/reference differences | 10.0 | 10.0 | rows=60; replicates=[np.int64(2000)] |
| 90% and 95% predictive interval coverage and width | 5.0 | 5.0 | coverage90_range=(0.652, 0.902); coverage95_range=(0.741, 0.966) |
| Prespecified subgroup coverage and extreme-burden sensitivity | 5.0 | 5.0 | dimensions=['observed_burden_quartile', 'overall', 'sdi_quintile', 'target_year', 'top_1pct_sensitivity']; rows=2244 |
| Five-seed tree and GRU stability with saved objects | 8.0 | 8.0 | metric_rows=120; fitted_objects=60 |
| Ablation, definition, shift, and model-explanation analyses | 8.0 | 8.0 | {'feature ablation': True, 'fraction-definition sensitivity': True, 'distribution shift': True, 'SHAP': True} |
| Fitted-object preservation and hash verification | 10.0 | 10.0 | final=48/48; stability=60/60; ablation=30/30; fraction=12/12 |

## Locked final-test performance versus persistence

| Partition | Outcome | Horizon | Selected family | Selected RMSLE | Persistence RMSLE | Improved |
|---|---|---:|---|---:|---:|---|
| test_spatiotemporal_unseen_country | daly | 1 | gru | 0.05855 | 0.06439 | True |
| test_spatiotemporal_unseen_country | daly | 3 | extra_trees | 0.13290 | 0.11920 | False |
| test_spatiotemporal_unseen_country | daly | 5 | extra_trees | 0.21005 | 0.16119 | False |
| test_spatiotemporal_unseen_country | death | 1 | gru | 0.02608 | 0.02812 | True |
| test_spatiotemporal_unseen_country | death | 3 | extra_trees | 0.11205 | 0.05071 | False |
| test_spatiotemporal_unseen_country | death | 5 | extra_trees | 0.11811 | 0.06825 | False |
| test_temporal_seen_country | daly | 1 | gru | 0.06258 | 0.06717 | True |
| test_temporal_seen_country | daly | 3 | extra_trees | 0.10271 | 0.12173 | True |
| test_temporal_seen_country | daly | 5 | extra_trees | 0.12442 | 0.16083 | True |
| test_temporal_seen_country | death | 1 | gru | 0.02845 | 0.03020 | True |
| test_temporal_seen_country | death | 3 | extra_trees | 0.04778 | 0.05338 | True |
| test_temporal_seen_country | death | 5 | extra_trees | 0.05334 | 0.06610 | True |

## Gate decision

The figure stage is authorized because the objective model/evaluation score is at least 90.
