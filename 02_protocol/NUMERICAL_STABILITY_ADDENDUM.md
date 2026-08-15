# Numerical-stability addendum

The initial development-only Ridge search emitted ill-conditioned-matrix warnings at very small regularisation values. Before any model-family lock or access to the sealed tests, the Ridge implementation was changed to the deterministic LSQR solver (`tol=1e-10`, `max_iter=10000`). The warning-version task blocks were preserved under `numerical_warning_archive_pre_lsqr`; only the LSQR outputs are eligible for selection.

**CV completion gate: PASS.** Tabular rows=2,412; tabular OOF=58,680; GRU rows=216; GRU OOF=5,868.

| Outcome | Horizon | Family | Pre-LSQR config | LSQR config | Unchanged | Pre-LSQR RMSLE | LSQR RMSLE | Change |
|---|---:|---|---|---|---|---:|---:|---:|
| daly | 1 | pooled_ridge | 391f696a66fa | 391f696a66fa | True | 0.035540 | 0.035540 | -1.43e-09 |
| daly | 1 | ridge | b06ea01f7f1d | b06ea01f7f1d | True | 0.238556 | 0.238556 | +1.89e-08 |
| daly | 3 | pooled_ridge | 391f696a66fa | 391f696a66fa | True | 0.076975 | 0.076975 | +1.83e-08 |
| daly | 3 | ridge | 41da9d593261 | 41da9d593261 | True | 0.255470 | 0.255470 | +2.96e-08 |
| daly | 5 | pooled_ridge | 391f696a66fa | 391f696a66fa | True | 0.113641 | 0.113641 | +1.56e-08 |
| daly | 5 | ridge | 41da9d593261 | 41da9d593261 | True | 0.273311 | 0.273311 | +6.44e-08 |
| death | 1 | pooled_ridge | 391f696a66fa | 391f696a66fa | True | 0.014118 | 0.014118 | -5.52e-09 |
| death | 1 | ridge | 41da9d593261 | 41da9d593261 | True | 0.063100 | 0.063100 | -1.62e-08 |
| death | 3 | pooled_ridge | 391f696a66fa | 391f696a66fa | True | 0.030790 | 0.030790 | -1.70e-09 |
| death | 3 | ridge | 41da9d593261 | 41da9d593261 | True | 0.069364 | 0.069364 | +3.64e-08 |
| death | 5 | pooled_ridge | 391f696a66fa | 391f696a66fa | True | 0.045639 | 0.045639 | -8.46e-10 |
| death | 5 | ridge | 0a0d7970fc7f | 0a0d7970fc7f | True | 0.077656 | 0.077656 | +4.66e-09 |
