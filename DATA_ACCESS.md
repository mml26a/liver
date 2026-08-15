# Data access and redistribution boundary

## Obtain the source files

Download the required GBD 2023 result exports from the [IHME GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) or the [Global Health Data Exchange](https://ghdx.healthdata.org/gbd-results-tool), using the terms accepted by the actual downloader. Do not commit these exports to this repository.

Place the following filenames in `data_raw/` at the repository root:

| Local filename | Required content |
|---|---|
| `gbd2023_BMI_HCC_country_1990_2023.csv` | Countries; 1990-2023; liver cancer; high BMI; DALYs and deaths; age-standardised; both sexes; Rate and Percent |
| `gbd2023_allHCC_country_1990_2023.csv` | Countries; 1990-2023; liver cancer; DALYs and deaths; age-standardised; both sexes; Rate |
| `gbd2023_SDI_values_1950_2023.csv` | SDI series containing all 204 study locations through 2023 |
| `gbd2023_BMI_HCC_global_alllevels.csv` | Global; 1990-2023; liver cancer; high BMI; DALYs and deaths; Rate and Percent |
| `gbd2023_BMI_HCC_SDI_1990_2023.csv` | SDI-level series used for descriptive panels; same outcomes/metrics |
| `gbd2023_allHCC_global_1990_2023.csv` | Global overall liver-cancer DALY/death rates, 1990-2023 |

`HCC` is retained only in the legacy local filenames. The verified GBD cause field and study outcome are aggregate **liver cancer**, not histology-confirmed hepatocellular carcinoma.

## What is excluded

The repository does not redistribute complete source exports, reconstructed country-year panels, supervised sample tables, complete sample-level prediction outputs, SHAP values at sample level, or fitted model objects. Their local paths, sizes, and SHA-256 hashes are recorded in `EXCLUDED_FILES.csv` for identity checking.

The release candidate does contain aggregate verification tables and minimal figure-level subsets. In particular, `Fig2_country_2023.csv` supports the country-level descriptive panels and `Fig5_seen_daly_h5_predictions.csv` supports a published observed-versus-predicted panel. These are not substitutes for the complete analytic or prediction tables. The corresponding authors must decide whether each included subset is necessary for publication under the agreement accepted by the data downloader; otherwise remove the table and retain the corresponding vector figure only.

IHME's non-commercial agreement permits analyses and publication of necessary portions of results but imposes conditions on the underlying data. The corresponding authors must confirm that the agreement accepted by the downloader permits every proposed public result table. This repository uses a conservative boundary and is not legal advice.

Required acknowledgment: **Source: Institute for Health Metrics and Evaluation (IHME). Used with permission. All rights reserved.**
