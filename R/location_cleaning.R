source("R/utils_helpers.R")
source("R/io_paths.R")

location_level_guess <- function(location_name) {
  nm <- as.character(location_name)
  if (grepl("^Global$", nm, ignore.case = TRUE)) return("global")
  if (grepl("\\bSDI\\b|World Bank|^WHO\\b|region|income|European Union|African Union", nm, ignore.case = TRUE)) return("aggregate")
  if (grepl(",\\s*(Urban|Rural)$|\\bstate\\b|\\bprovince\\b|\\boblast\\b|\\bcounty\\b|\\bdistrict\\b|\\bmunicipality\\b", nm, ignore.case = TRUE)) return("subnational")
  "country_or_territory_candidate"
}

build_location_master <- function(locations, country_roster = NULL) {
  if (is.null(country_roster)) {
    roster_path <- path_in_project("config", "country_roster_204.csv")
    if (file.exists(roster_path)) {
      country_roster <- read.csv(roster_path, stringsAsFactors = FALSE)$location_name
    } else {
      country_roster <- character(0)
    }
  }

  roster_keys <- unique(normalize_name_key(country_roster))
  roster_key_all <- normalize_name_key(country_roster)
  roster_map <- stats::setNames(country_roster, roster_key_all)
  loc_u <- sort(unique(trimws(as.character(locations))))

  out <- data.frame(
    original_location_name = loc_u,
    cleaned_location_name = trimws(loc_u),
    stringsAsFactors = FALSE
  )
  out$location_level_guess <- vapply(out$original_location_name, location_level_guess, FUN.VALUE = character(1))
  out$name_key <- normalize_name_key(out$cleaned_location_name)
  out$cleaned_location_name <- ifelse(
    out$name_key %in% names(roster_map),
    unname(roster_map[out$name_key]),
    out$cleaned_location_name
  )
  out$keep_for_country_analysis <- out$name_key %in% roster_keys
  out$exclusion_reason <- ifelse(
    out$keep_for_country_analysis,
    "",
    ifelse(
      out$location_level_guess == "aggregate",
      "aggregate",
      ifelse(out$location_level_guess == "subnational", "subnational", "not_in_country_roster")
    )
  )
  out$note_on_ambiguity <- ifelse(
    out$cleaned_location_name %in% c("Georgia", "Niger"),
    "Ambiguous label; resolve by ihme_loc_id override (Georgia=GEO, Niger=NER).",
    ""
  )
  out$name_key <- NULL
  out
}

assert_country_only <- function(df, location_col = "location_name", location_master = NULL) {
  if (!location_col %in% names(df)) stop("location column not found: ", location_col, call. = FALSE)
  if (is.null(location_master)) {
    location_master <- build_location_master(df[[location_col]])
  }
  bad <- location_master$original_location_name[
    is.na(location_master$keep_for_country_analysis) | !location_master$keep_for_country_analysis
  ]
  if (length(bad) > 0) {
    stop(
      paste0(
        "Country-level assertion failed. Non-country entries detected: ",
        paste(head(sort(unique(bad)), 20), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}
