#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(tidyr)
  library(stringr)
  library(tibble)
  library(fs)
})

source("scripts/utils_gbd_pipeline.R")

ensure_dir("outputs")
ensure_dir("outputs/audit")
ensure_dir("data_clean")
ensure_dir("config")

country_roster <- load_country_roster("config/country_roster_204.csv")
overrides <- load_ambiguous_overrides("config/ambiguous_location_overrides.csv")

read_calls <- parse_read_csv_calls("Code.R")

if (nrow(read_calls) == 0) {
  stop("No read.csv calls found in Code.R", call. = FALSE)
}

read_objects <- read_calls %>%
  count(file, name = "n_calls") %>%
  arrange(desc(n_calls), file) %>%
  mutate(file_exists = file_exists(file))

write_csv(read_objects, "outputs/audit/gbd_read_objects_from_code.csv")
write_csv(tibble(location_name = sort(country_roster)), "outputs/audit/country_roster_204_used.csv")

audit_summaries <- list()
union_diffs <- list()
union_resolutions <- list()

process_one_dataset <- function(dataset_file) {
  df <- safe_read_csv(dataset_file)
  loc_col <- detect_column(df, c("location_name", "location", "Country"))

  if (is.na(loc_col)) {
    return(
      list(
        summary = tibble(
          dataset = dataset_file,
          has_location_column = FALSE,
          n_rows = nrow(df),
          before_unique_locations = NA_integer_,
          after_unique_locations = NA_integer_,
          dropped_unique_locations = NA_integer_,
          kept_scope = NA_character_
        ),
        diff = tibble(),
        resolution = tibble(),
        cleaned = df
      )
    )
  }

  targets <- location_target_from_filename(dataset_file, country_roster)
  clean_res <- clean_location_frame(
    df = df,
    country_roster = country_roster,
    overrides = overrides,
    dataset_name = dataset_file,
    allowed_locations = targets
  )

  dataset_stem <- fs::path_ext_remove(fs::path_file(dataset_file))
  clean_path <- fs::path("data_clean", paste0(dataset_stem, "_clean.csv"))
  write_csv(clean_res$clean_data, clean_path)

  write_csv(
    clean_res$location_audit,
    fs::path("outputs/audit", paste0(dataset_stem, "_location_classification.csv"))
  )
  write_csv(
    clean_res$location_diff,
    fs::path("outputs/audit", paste0(dataset_stem, "_before_after_diff.csv"))
  )
  write_csv(
    clean_res$resolution_log,
    fs::path("outputs/audit", paste0(dataset_stem, "_ambiguous_resolution.csv"))
  )

  before_n <- clean_res$location_diff %>% filter(before) %>% nrow()
  after_n <- clean_res$location_diff %>% filter(after) %>% nrow()

  summary_row <- tibble(
    dataset = dataset_file,
    has_location_column = TRUE,
    n_rows = nrow(df),
    before_unique_locations = before_n,
    after_unique_locations = after_n,
    dropped_unique_locations = before_n - after_n,
    kept_scope = case_when(
      all(targets %in% country_roster) & length(targets) == 204 ~ "country_204",
      identical(sort(targets), sort(SCOPE_SDI_GROUPS)) ~ "sdi_5_groups",
      identical(sort(targets), sort(c("Global", SCOPE_SDI_GROUPS))) ~ "global_plus_sdi",
      TRUE ~ "custom"
    )
  )

  list(
    summary = summary_row,
    diff = clean_res$location_diff %>% mutate(dataset = dataset_file),
    resolution = clean_res$resolution_log %>% mutate(dataset = dataset_file),
    cleaned = clean_res$clean_data
  )
}

for (dataset_file in read_objects$file[read_objects$file_exists]) {
  res <- process_one_dataset(dataset_file)
  audit_summaries[[dataset_file]] <- res$summary
  union_diffs[[dataset_file]] <- res$diff
  union_resolutions[[dataset_file]] <- res$resolution
}

# Optional auxiliary audit on current STable1 output (if present).
if (file_exists("archived_tables/STable1_Country_full_metrics.csv")) {
  st <- safe_read_csv("archived_tables/STable1_Country_full_metrics.csv")
  st <- st %>% rename(location_name = Country)
  st_res <- clean_location_frame(
    df = st,
    country_roster = country_roster,
    overrides = overrides,
    dataset_name = "archived_tables/STable1_Country_full_metrics.csv",
    allowed_locations = country_roster
  )

  write_csv(
    st_res$location_diff,
    "outputs/audit/STable1_before_after_diff.csv"
  )
  write_csv(
    st_res$resolution_log,
    "outputs/audit/STable1_ambiguous_resolution.csv"
  )
  write_csv(
    st_res$location_audit,
    "outputs/audit/STable1_location_classification.csv"
  )

  audit_summaries[["archived_STable1"]] <- tibble(
    dataset = "archived_tables/STable1_Country_full_metrics.csv",
    has_location_column = TRUE,
    n_rows = nrow(st),
    before_unique_locations = n_distinct(st$location_name),
    after_unique_locations = st_res$location_diff %>% filter(after) %>% nrow(),
    dropped_unique_locations = n_distinct(st$location_name) - (st_res$location_diff %>% filter(after) %>% nrow()),
    kept_scope = "country_204"
  )

  union_diffs[["archived_STable1"]] <- st_res$location_diff %>%
    mutate(dataset = "archived_tables/STable1_Country_full_metrics.csv")
  union_resolutions[["archived_STable1"]] <- st_res$resolution_log %>%
    mutate(dataset = "archived_tables/STable1_Country_full_metrics.csv")
}

audit_summary_tbl <- bind_rows(audit_summaries) %>% arrange(desc(before_unique_locations), dataset)
write_csv(audit_summary_tbl, "outputs/audit/location_audit_summary.csv")

diff_tbl <- bind_rows(union_diffs)
if (nrow(diff_tbl) > 0) {
  write_csv(diff_tbl, "outputs/audit/location_before_after_by_dataset.csv")

  diff_union <- diff_tbl %>%
    group_by(location_name) %>%
    summarise(
      in_any_before = any(before),
      in_any_after = any(after),
      drop_reason = first(na.omit(drop_reason)),
      .groups = "drop"
    ) %>%
    mutate(status = ifelse(in_any_after, "kept", "dropped")) %>%
    arrange(status, location_name)

  write_csv(diff_union, "outputs/audit/location_list_difference_union.csv")

  primary_dataset <- audit_summary_tbl %>%
    filter(has_location_column) %>%
    arrange(desc(before_unique_locations)) %>%
    slice(1) %>%
    pull(dataset)

  write_lines(primary_dataset, "outputs/audit/primary_dataset_for_before_after.txt")

  primary_diff <- diff_tbl %>%
    filter(dataset == primary_dataset) %>%
    arrange(desc(before), desc(after), location_name)
  write_csv(primary_diff, "outputs/audit/location_list_difference_primary.csv")
}

resolution_tbl <- bind_rows(union_resolutions)
write_csv(resolution_tbl, "outputs/audit/ambiguous_name_resolution_union.csv")

filter_rules <- c(
  "# Location Filtering Rules",
  "",
  "1. Country scope: retain only names in `config/country_roster_204.csv` (204 countries/territories).",
  "2. SDI scope: retain only `High SDI`, `High-middle SDI`, `Middle SDI`, `Low-middle SDI`, `Low SDI`.",
  "3. Global scope: retain only `Global` plus the five SDI groups.",
  "4. Drop any name outside the target scope, and classify the drop reason as:",
  "   - `aggregate_or_group` if matching SDI/World Bank/WHO/super-region style labels.",
  "   - `subnational` if matching state/province/urban-rural/admin-unit patterns.",
  "   - `other_non_country_or_not_targeted` for all remaining non-target locations.",
  "5. Resolve duplicate country names (e.g., `Georgia`, `Niger`) using `config/ambiguous_location_overrides.csv`:",
  "   - Prefer `preferred_ihme_loc_id` if available.",
  "   - Else prefer `preferred_location_id` if available.",
  "   - Else prefer ISO3-like `ihme_loc_id` (`^[A-Z]{3}$`).",
  "   - Else keep the first record and log it for manual review."
)
write_lines(filter_rules, "outputs/audit/filter_rules.md")

message("Data audit complete.")
message("Summary written to outputs/audit/location_audit_summary.csv")
