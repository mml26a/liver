# GBD BMI-Liver Cancer Pipeline

Run in order:

1. `scripts/01_data_audit.R`
2. `scripts/02_build_master_dataset.R`
3. `scripts/03_decomposition.R`
4. `scripts/04_joinpoint.R`
5. `scripts/05_projection.R`

Or run all:

- `scripts/00_run_all.R`
- `Rscript -e "renv::restore(prompt = FALSE)"` (non-interactive dependency restore)

## Required inputs (project root)

- `gbd2023_BMI_HCC_global_alllevels.csv`
- `gbd2023_BMI_HCC_SDI_1990_2023.csv`
- `gbd2023_BMI_HCC_country_1990_2023.csv`
- `gbd2023_allHCC_global_1990_2023.csv`
- `gbd2023_allHCC_country_1990_2023.csv`
- `gbd2023_SDI_values_1950_2023.csv`
- population CSV (`*population*.csv`; for decomposition/projection with counts)

## Key outputs

- `outputs/audit/*` (filter rules, before/after location lists, ambiguous-name resolution)
- `data_clean/analytic_master_long.csv` and `.rds`
- `outputs/decomposition/*`
- `outputs/joinpoint/*`
- `outputs/projection/*`
