#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(stringr)
  library(tibble)
  library(fs)
})

source("scripts/utils_gbd_pipeline.R")

ensure_dir("outputs")
ensure_dir("outputs/master")
ensure_dir("data_clean")

country_roster <- load_country_roster("config/country_roster_204.csv")
overrides <- load_ambiguous_overrides("config/ambiguous_location_overrides.csv")

disease_sources <- tribble(
  ~raw_file, ~risk_group, ~scope_hint,
  "gbd2023_BMI_HCC_global_alllevels.csv", "BMI_attributable", "global_sdi",
  "gbd2023_BMI_HCC_SDI_1990_2023.csv", "BMI_attributable", "sdi",
  "gbd2023_BMI_HCC_country_1990_2023.csv", "BMI_attributable", "country",
  "gbd2023_allHCC_global_1990_2023.csv", "Overall", "global_sdi",
  "gbd2023_allHCC_country_1990_2023.csv", "Overall", "country"
)

read_one_disease <- function(raw_file, risk_group, scope_hint) {
  clean_file <- fs::path(
    "data_clean",
    paste0(fs::path_ext_remove(fs::path_file(raw_file)), "_clean.csv")
  )

  raw_candidates <- c(raw_file, fs::path("data_raw", raw_file))
  existing_raw <- raw_candidates[file_exists(raw_candidates)]
  chosen_file <- if (file_exists(clean_file)) {
    clean_file
  } else if (length(existing_raw) > 0) {
    existing_raw[[1]]
  } else {
    raw_file
  }
  if (!file_exists(chosen_file)) {
    return(NULL)
  }

  d <- safe_read_csv(chosen_file)
  d <- standardize_location_col(d)
  if (!"location_name" %in% names(d)) {
    return(NULL)
  }

  if (scope_hint == "country") {
    clean_country <- clean_location_frame(
      df = d,
      country_roster = country_roster,
      overrides = overrides,
      dataset_name = raw_file,
      allowed_locations = country_roster
    )
    d <- clean_country$clean_data
  } else if (scope_hint == "sdi") {
    d <- d %>% filter(location_name %in% SCOPE_SDI_GROUPS)
  } else if (scope_hint == "global_sdi") {
    d <- d %>% filter(location_name %in% c("Global", SCOPE_SDI_GROUPS))
  }

  d <- standardize_gbd_schema(d, source_file = raw_file, risk_group = risk_group)

  if ("cause_name" %in% names(d)) {
    d <- d %>%
      filter(is.na(cause_name) | str_detect(cause_name, regex("^Liver cancer$", ignore_case = TRUE)))
  }

  if (risk_group == "BMI_attributable" && "rei_name" %in% names(d)) {
    d <- d %>%
      filter(
        is.na(rei_name) |
          str_detect(rei_name, regex("body[- ]?mass index|high bmi", ignore_case = TRUE))
      )
  }

  d %>%
    mutate(
      scope = case_when(
        location_name == "Global" ~ "global",
        location_name %in% SCOPE_SDI_GROUPS ~ "sdi",
        TRUE ~ "country"
      )
    )
}

disease_long <- pmap_dfr(disease_sources, read_one_disease)

if (nrow(disease_long) == 0) {
  stop("No disease datasets found. Place the gbd2023_*.csv files in the project root.", call. = FALSE)
}

sdi_tbl <- prepare_sdi_table("gbd2023_SDI_values_1950_2023.csv")

pop_file <- find_population_file()
pop_tbl <- prepare_population_table(pop_file)

master_long <- disease_long %>%
  left_join(sdi_tbl, by = c("location_name", "year")) %>%
  left_join(pop_tbl, by = c("location_name", "year", "age_name", "sex_name")) %>%
  mutate(
    measure_name = normalize_measure_name(measure_name),
    metric_name = normalize_metric_name(metric_name),
    year = as.integer(year)
  ) %>%
  filter(!is.na(year), year >= 1990, year <= 2023) %>%
  arrange(scope, location_name, risk_group, measure_name, metric_name, age_name, year)

write_csv(master_long, "data_clean/analytic_master_long.csv")
saveRDS(master_long, "data_clean/analytic_master_long.rds")

# --- Quality checks ---
dup_key_strict <- master_long %>%
  count(
    scope, risk_group, location_name, year, age_name, sex_name,
    measure_name, metric_name,
    name = "n"
  ) %>%
  filter(n > 1) %>%
  arrange(desc(n))

dup_key_requested <- master_long %>%
  count(location_name, year, age_name, measure_name, metric_name, name = "n") %>%
  filter(n > 1) %>%
  arrange(desc(n))

coverage_tbl <- master_long %>%
  count(scope, risk_group, measure_name, metric_name, name = "n_rows") %>%
  arrange(scope, risk_group, measure_name, metric_name)

year_range_tbl <- master_long %>%
  summarise(
    min_year = min(year, na.rm = TRUE),
    max_year = max(year, na.rm = TRUE),
    n_rows = n()
  )

location_counts_tbl <- master_long %>%
  distinct(scope, location_name) %>%
  count(scope, name = "n_locations")

qc_summary <- tibble(
  check_item = c(
    "strict_uniqueness(scope+risk+location+year+age+sex+measure+metric)",
    "requested_uniqueness(location+year+age+measure+metric)",
    "year_range_1990_2023",
    "population_joined_rows",
    "sdi_joined_rows"
  ),
  status = c(
    ifelse(nrow(dup_key_strict) == 0, "PASS", "FAIL"),
    ifelse(nrow(dup_key_requested) == 0, "PASS", "FAIL"),
    ifelse(year_range_tbl$min_year >= 1990 && year_range_tbl$max_year <= 2023, "PASS", "FAIL"),
    ifelse(sum(!is.na(master_long$population)) > 0, "PASS", "WARN"),
    ifelse(sum(!is.na(master_long$sdi)) > 0, "PASS", "WARN")
  ),
  detail = c(
    paste("duplicate rows =", nrow(dup_key_strict)),
    paste("duplicate rows =", nrow(dup_key_requested)),
    paste("range =", year_range_tbl$min_year, "-", year_range_tbl$max_year),
    paste("non-missing population rows =", sum(!is.na(master_long$population))),
    paste("non-missing SDI rows =", sum(!is.na(master_long$sdi)))
  )
)

write_csv(dup_key_strict, "outputs/master/check_duplicates_strict.csv")
write_csv(dup_key_requested, "outputs/master/check_duplicates_requested_key.csv")
write_csv(coverage_tbl, "outputs/master/check_coverage_summary.csv")
write_csv(year_range_tbl, "outputs/master/check_year_range.csv")
write_csv(location_counts_tbl, "outputs/master/check_location_counts.csv")
write_csv(qc_summary, "outputs/master/master_dataset_qc_summary.csv")

message("Master dataset complete: data_clean/analytic_master_long.csv")
if (nrow(dup_key_strict) > 0) {
  message("WARNING: strict uniqueness check failed. See outputs/master/check_duplicates_strict.csv")
}
