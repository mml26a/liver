suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(readr)
  library(purrr)
  library(tibble)
  library(fs)
})

SCOPE_SDI_GROUPS <- c(
  "High SDI",
  "High-middle SDI",
  "Middle SDI",
  "Low-middle SDI",
  "Low SDI"
)

DEFAULT_AGGREGATE_PATTERN <- paste(
  c(
    "^Global$",
    "\\bSDI\\b",
    "World Bank",
    "^WHO\\b",
    "\\bWHO\\s+region\\b",
    "European Union",
    "African Union",
    "\\bIncome\\b",
    "\\bsuper[- ]?region\\b",
    "\\bregion\\b",
    "G20",
    "OECD"
  ),
  collapse = "|"
)

DEFAULT_SUBNATIONAL_PATTERN <- paste(
  c(
    ",\\s*(Urban|Rural)$",
    "\\bstate\\b",
    "\\bprovince\\b",
    "\\boblast\\b",
    "\\bcounty\\b",
    "\\bdistrict\\b",
    "\\bmunicipality\\b",
    "\\bgovernorate\\b",
    "\\bprefecture\\b",
    "\\bautonomous\\b",
    "\\bmetro\\b",
    "\\bcity\\b"
  ),
  collapse = "|"
)

ensure_dir <- function(path) {
  if (!fs::dir_exists(path)) {
    fs::dir_create(path, recurse = TRUE)
  }
  invisible(path)
}

detect_column <- function(df, candidates) {
  hit <- intersect(candidates, names(df))
  if (length(hit) == 0) {
    return(NA_character_)
  }
  hit[[1]]
}

trim_location <- function(x) {
  stringr::str_squish(as.character(x))
}

normalize_name_key <- function(x) {
  x <- trim_location(x)
  x <- iconv(x, from = "", to = "ASCII//TRANSLIT")
  x <- ifelse(is.na(x), "", x)
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", " ", x)
  key <- stringr::str_squish(x)
  dplyr::case_when(
    key %in% c("turkiye", "turkey") ~ "turkiye",
    key %in% c("cote d ivoire", "c te d ivoire", "ivory coast") ~ "cote d ivoire",
    TRUE ~ key
  )
}

normalize_measure_name <- function(x) {
  x <- as.character(x)
  dplyr::case_when(
    stringr::str_detect(x, regex("DALY", ignore_case = TRUE)) ~ "DALYs",
    stringr::str_detect(x, regex("^Deaths?$", ignore_case = TRUE)) ~ "Deaths",
    TRUE ~ x
  )
}

normalize_metric_name <- function(x) {
  x <- as.character(x)
  dplyr::case_when(
    stringr::str_detect(x, regex("^Rate$", ignore_case = TRUE)) ~ "Rate",
    stringr::str_detect(x, regex("^Number$", ignore_case = TRUE)) ~ "Number",
    TRUE ~ x
  )
}

parse_read_csv_calls <- function(code_file = "Code.R") {
  if (!file.exists(code_file)) {
    return(tibble(line = integer(), code = character(), file = character()))
  }

  lines <- readLines(code_file, warn = FALSE, encoding = "UTF-8")

  tibble(
    line = seq_along(lines),
    code = lines
  ) %>%
    mutate(file = stringr::str_match(code, 'read\\.csv\\("([^"]+)"')[, 2]) %>%
    filter(!is.na(file))
}

safe_read_csv <- function(path) {
  tryCatch(
    readr::read_csv(path, show_col_types = FALSE, progress = FALSE),
    error = function(e1) {
      tryCatch(
        as_tibble(utils::read.csv(path, check.names = FALSE)),
        error = function(e2) {
          stop(
            sprintf(
              "Failed to read '%s'. readr error: %s | base error: %s",
              path, e1$message, e2$message
            ),
            call. = FALSE
          )
        }
      )
    }
  )
}

load_country_roster <- function(roster_path = "config/country_roster_204.csv") {
  if (file.exists(roster_path)) {
    roster <- safe_read_csv(roster_path) %>%
      pull(location_name) %>%
      trim_location() %>%
      unique() %>%
      sort()
    return(roster)
  }

  if (file.exists("archived_tables/STable1_Country_full_metrics.csv")) {
    st <- safe_read_csv("archived_tables/STable1_Country_full_metrics.csv")
    loc_col <- detect_column(st, c("Country", "location_name", "location"))
    score_col <- detect_column(st, c("Priority_score", "priority_score"))

    if (!is.na(loc_col) && !is.na(score_col)) {
      roster <- st %>%
        filter(.data[[score_col]] != "NA") %>%
        pull(.data[[loc_col]]) %>%
        trim_location() %>%
        unique() %>%
        sort()

      if (length(roster) == 204) {
        ensure_dir(dirname(roster_path))
        write_csv(tibble(location_name = roster), roster_path)
        return(roster)
      }
    }
  }

  stop(
    paste0(
      "Unable to derive a 204-country roster. Provide ",
      "`config/country_roster_204.csv` with one column `location_name`."
    ),
    call. = FALSE
  )
}

load_ambiguous_overrides <- function(path = "config/ambiguous_location_overrides.csv") {
  if (!file.exists(path)) {
    ensure_dir(dirname(path))
    defaults <- tribble(
      ~location_name, ~preferred_ihme_loc_id, ~preferred_location_id, ~preferred_region, ~notes,
      "Georgia", "GEO", NA_character_, "Central Asia", "Country Georgia; remove US state Georgia.",
      "Niger", "NER", NA_character_, "Western Sub-Saharan Africa", "Country Niger; remove Niger state in Nigeria."
    )
    write_csv(defaults, path)
  }

  safe_read_csv(path) %>%
    mutate(
      location_name = trim_location(location_name),
      preferred_ihme_loc_id = as.character(preferred_ihme_loc_id),
      preferred_location_id = as.character(preferred_location_id),
      preferred_region = as.character(preferred_region)
    )
}

location_target_from_filename <- function(dataset_name, country_roster) {
  d <- tolower(dataset_name)
  if (str_detect(d, "country")) {
    return(country_roster)
  }
  if (str_detect(d, "sdi")) {
    return(SCOPE_SDI_GROUPS)
  }
  if (str_detect(d, "global")) {
    return(c("Global", SCOPE_SDI_GROUPS))
  }
  country_roster
}

standardize_location_col <- function(df) {
  loc_col <- detect_column(df, c("location_name", "location", "Country"))
  if (is.na(loc_col)) {
    return(df)
  }
  if (loc_col != "location_name") {
    df <- rename(df, location_name = all_of(loc_col))
  }
  df %>% mutate(location_name = trim_location(location_name))
}

resolve_duplicate_rows <- function(df, overrides, dataset_name, key_cols) {
  if (!"location_name" %in% names(df)) {
    return(list(clean = df, log = tibble()))
  }

  if (length(key_cols) == 0) {
    key_cols <- "location_name"
  }

  work <- df %>%
    left_join(overrides, by = "location_name") %>%
    mutate(
      ihme_loc_id_chr = if ("ihme_loc_id" %in% names(.)) as.character(ihme_loc_id) else NA_character_,
      location_id_chr = if ("location_id" %in% names(.)) as.character(location_id) else NA_character_,
      region_name_chr = if ("region_name" %in% names(.)) as.character(region_name) else NA_character_,
      score_pref_ihme = ifelse(
        !is.na(preferred_ihme_loc_id) &
          !is.na(ihme_loc_id_chr) &
          ihme_loc_id_chr == preferred_ihme_loc_id,
        100L, 0L
      ),
      score_pref_locid = ifelse(
        !is.na(preferred_location_id) &
          !is.na(location_id_chr) &
          location_id_chr == preferred_location_id,
        90L, 0L
      ),
      score_iso3 = ifelse(
        !is.na(ihme_loc_id_chr) & stringr::str_detect(ihme_loc_id_chr, "^[A-Z]{3}$"),
        40L, 0L
      ),
      score_no_underscore = ifelse(
        !is.na(ihme_loc_id_chr) & !stringr::str_detect(ihme_loc_id_chr, "_"),
        10L, 0L
      ),
      score_region = ifelse(
        !is.na(preferred_region) &
          !is.na(region_name_chr) &
          trim_location(region_name_chr) == trim_location(preferred_region),
        20L, 0L
      ),
      score_total = score_pref_ihme + score_pref_locid + score_iso3 + score_no_underscore + score_region
    ) %>%
    group_by(across(all_of(key_cols))) %>%
    mutate(group_n = n()) %>%
    arrange(desc(score_total), ihme_loc_id_chr, location_id_chr, .by_group = TRUE) %>%
    mutate(rank_in_group = row_number()) %>%
    ungroup()

  duplicate_log <- work %>%
    filter(group_n > 1) %>%
    mutate(
      kept = rank_in_group == 1,
      resolution_rule = case_when(
        score_pref_ihme > 0 ~ "preferred_ihme_loc_id",
        score_pref_locid > 0 ~ "preferred_location_id",
        score_iso3 > 0 ~ "iso3_like_ihme_loc_id",
        TRUE ~ "fallback_first_record"
      ),
      dataset = dataset_name
    ) %>%
    select(
      dataset, all_of(key_cols), location_name, ihme_loc_id_chr, location_id_chr,
      score_total, kept, resolution_rule, notes
    )

  clean <- work %>%
    filter(rank_in_group == 1) %>%
    select(-starts_with("preferred_"), -notes,
           -ihme_loc_id_chr, -location_id_chr, -region_name_chr,
           -starts_with("score_"), -group_n, -rank_in_group)

  list(clean = clean, log = duplicate_log)
}

clean_location_frame <- function(
  df,
  country_roster,
  overrides,
  dataset_name,
  aggregate_pattern = DEFAULT_AGGREGATE_PATTERN,
  subnational_pattern = DEFAULT_SUBNATIONAL_PATTERN,
  allowed_locations = NULL
) {
  df <- standardize_location_col(df)

  if (!"location_name" %in% names(df)) {
    return(
      list(
        clean_data = df,
        location_audit = tibble(),
        location_diff = tibble(),
        resolution_log = tibble()
      )
    )
  }

  if (is.null(allowed_locations)) {
    allowed_locations <- location_target_from_filename(dataset_name, country_roster)
  }

  roster_keys <- normalize_name_key(country_roster)
  allowed_keys <- normalize_name_key(allowed_locations)

  df1 <- df %>%
    mutate(
      location_name = trim_location(location_name),
      location_key = normalize_name_key(location_name),
      in_country_roster = location_key %in% roster_keys,
      in_allowed_scope = location_key %in% allowed_keys,
      aggregate_flag = stringr::str_detect(location_name, regex(aggregate_pattern, ignore_case = TRUE)),
      subnational_flag = stringr::str_detect(location_name, regex(subnational_pattern, ignore_case = TRUE)),
      drop_reason = case_when(
        in_allowed_scope ~ NA_character_,
        aggregate_flag ~ "aggregate_or_group",
        subnational_flag ~ "subnational",
        in_country_roster & !in_allowed_scope ~ "scope_mismatch",
        TRUE ~ "other_non_country_or_not_targeted"
      )
    )

  location_audit <- df1 %>%
    distinct(location_name, in_country_roster, in_allowed_scope, aggregate_flag, subnational_flag, drop_reason) %>%
    arrange(location_name)

  kept <- df1 %>%
    filter(in_allowed_scope) %>%
    select(-in_country_roster, -in_allowed_scope, -location_key, -aggregate_flag, -subnational_flag, -drop_reason)

  key_cols <- intersect(
    c(
      "location_name", "year", "year_id", "age_name", "sex_name",
      "measure_name", "metric_name", "rei_name", "cause_name"
    ),
    names(kept)
  )

  dedup <- resolve_duplicate_rows(kept, overrides, dataset_name, key_cols)
  clean_data <- dedup$clean

  before_locations <- sort(unique(df1$location_name))
  after_locations <- sort(unique(clean_data$location_name))

  location_diff <- tibble(location_name = sort(unique(c(before_locations, after_locations)))) %>%
    mutate(
      before = location_name %in% before_locations,
      after = location_name %in% after_locations
    ) %>%
    left_join(
      location_audit %>%
        select(location_name, drop_reason) %>%
        distinct(),
      by = "location_name"
    ) %>%
    mutate(drop_reason = ifelse(after, NA_character_, drop_reason))

  list(
    clean_data = clean_data,
    location_audit = location_audit,
    location_diff = location_diff,
    resolution_log = dedup$log
  )
}

standardize_gbd_schema <- function(df, source_file, risk_group) {
  df <- standardize_location_col(df)

  rename_pairs <- list(
    year = c("year", "year_id"),
    age_name = c("age_name", "age"),
    sex_name = c("sex_name", "sex"),
    measure_name = c("measure_name", "measure"),
    metric_name = c("metric_name", "metric"),
    rei_name = c("rei_name", "risk"),
    cause_name = c("cause_name", "cause"),
    val = c("val", "value")
  )

  for (nm in names(rename_pairs)) {
    col <- detect_column(df, rename_pairs[[nm]])
    if (!is.na(col) && col != nm) {
      df <- rename(df, !!nm := all_of(col))
    }
  }

  if (!"year" %in% names(df)) {
    df$year <- NA_integer_
  }
  if (!"age_name" %in% names(df)) {
    df$age_name <- NA_character_
  }
  if (!"sex_name" %in% names(df)) {
    df$sex_name <- NA_character_
  }
  if (!"measure_name" %in% names(df)) {
    df$measure_name <- NA_character_
  }
  if (!"metric_name" %in% names(df)) {
    df$metric_name <- NA_character_
  }
  if (!"val" %in% names(df)) {
    df$val <- NA_real_
  }
  if (!"upper" %in% names(df)) {
    df$upper <- NA_real_
  }
  if (!"lower" %in% names(df)) {
    df$lower <- NA_real_
  }
  if (!"rei_name" %in% names(df)) {
    df$rei_name <- NA_character_
  }
  if (!"cause_name" %in% names(df)) {
    df$cause_name <- NA_character_
  }
  if (!"location_id" %in% names(df)) {
    df$location_id <- NA_integer_
  }
  if (!"ihme_loc_id" %in% names(df)) {
    df$ihme_loc_id <- NA_character_
  }

  df %>%
    mutate(
      source_file = source_file,
      risk_group = risk_group,
      location_name = trim_location(location_name),
      year = suppressWarnings(as.integer(year)),
      age_name = trim_location(age_name),
      sex_name = trim_location(sex_name),
      measure_name = normalize_measure_name(measure_name),
      metric_name = normalize_metric_name(metric_name),
      rei_name = trim_location(rei_name),
      cause_name = trim_location(cause_name),
      val = suppressWarnings(as.numeric(val)),
      upper = suppressWarnings(as.numeric(upper)),
      lower = suppressWarnings(as.numeric(lower))
    ) %>%
    filter(is.na(year) | (year >= 1990 & year <= 2023))
}

find_population_file <- function() {
  preferred <- c(
    "gbd2023_population_1990_2050.csv",
    "gbd2023_population_1990_2023.csv",
    "gbd_population_age_specific.csv",
    fs::path("data_raw", "gbd2023_population_1990_2050.csv"),
    fs::path("data_raw", "gbd2023_population_1990_2023.csv"),
    fs::path("data_raw", "gbd_population_age_specific.csv")
  )

  for (p in preferred) {
    if (file.exists(p)) {
      return(p)
    }
  }

  candidates <- fs::dir_ls(
    path = ".",
    recurse = TRUE,
    type = "file",
    regexp = "(?i)population.*\\.csv$"
  )

  if (length(candidates) == 0) {
    return(NA_character_)
  }

  candidates[[1]]
}

prepare_sdi_table <- function(path = "gbd2023_SDI_values_1950_2023.csv") {
  if (!file.exists(path) && file.exists(fs::path("data_raw", path))) {
    path <- fs::path("data_raw", path)
  }

  if (!file.exists(path)) {
    return(tibble(location_name = character(), year = integer(), sdi = numeric()))
  }

  d <- safe_read_csv(path)
  d <- standardize_location_col(d)
  year_col <- detect_column(d, c("year_id", "year"))
  value_col <- detect_column(d, c("mean_value", "val", "value", "sdi"))
  if (is.na(year_col) || is.na(value_col) || !"location_name" %in% names(d)) {
    return(tibble(location_name = character(), year = integer(), sdi = numeric()))
  }

  d %>%
    transmute(
      location_name = trim_location(location_name),
      year = suppressWarnings(as.integer(.data[[year_col]])),
      sdi = suppressWarnings(as.numeric(.data[[value_col]]))
    ) %>%
    filter(!is.na(year), year >= 1990, year <= 2023) %>%
    distinct(location_name, year, .keep_all = TRUE)
}

prepare_population_table <- function(path) {
  if (is.na(path) || !file.exists(path)) {
    return(tibble(
      location_name = character(),
      year = integer(),
      age_name = character(),
      sex_name = character(),
      population = numeric()
    ))
  }

  d <- safe_read_csv(path)
  d <- standardize_location_col(d)

  year_col <- detect_column(d, c("year_id", "year"))
  age_col <- detect_column(d, c("age_name", "age"))
  sex_col <- detect_column(d, c("sex_name", "sex"))
  val_col <- detect_column(d, c("val", "value", "population", "pop"))

  if (is.na(year_col) || is.na(age_col) || is.na(val_col) || !"location_name" %in% names(d)) {
    return(tibble(
      location_name = character(),
      year = integer(),
      age_name = character(),
      sex_name = character(),
      population = numeric()
    ))
  }

  if (is.na(sex_col)) {
    d$sex_name <- "Both"
    sex_col <- "sex_name"
  }

  d %>%
    transmute(
      location_name = trim_location(location_name),
      year = suppressWarnings(as.integer(.data[[year_col]])),
      age_name = trim_location(.data[[age_col]]),
      sex_name = trim_location(.data[[sex_col]]),
      population = suppressWarnings(as.numeric(.data[[val_col]]))
    ) %>%
    filter(!is.na(year), year >= 1990, year <= 2050) %>%
    distinct(location_name, year, age_name, sex_name, .keep_all = TRUE)
}
