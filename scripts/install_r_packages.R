suppressPackageStartupMessages({
  options(repos = c(CRAN = "http://cran.r-project.org"))
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

script_file_arg <- commandArgs(trailingOnly = FALSE)
script_file_arg <- script_file_arg[grepl("^--file=", script_file_arg)]
script_file <- if (length(script_file_arg) > 0) sub("^--file=", "", script_file_arg[[1]]) else NA_character_

project_root <- if (file.exists("renv.lock")) {
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
} else if (!is.na(script_file)) {
  normalizePath(file.path(dirname(script_file), ".."), winslash = "/", mustWork = FALSE)
} else {
  normalizePath(".", winslash = "/", mustWork = TRUE)
}

log_dir <- file.path(project_root, "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

r_version_parts <- strsplit(as.character(getRversion()), ".", fixed = TRUE)[[1]]
project_lib <- file.path(project_root, ".r_libs", paste0("R-", r_version_parts[1], ".", r_version_parts[2]))
dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(project_lib, .libPaths())))
user_lib <- project_lib

required_pkgs <- c(
  "dplyr",
  "tidyr",
  "ggplot2",
  "patchwork",
  "sf",
  "rnaturalearth",
  "rnaturalearthdata",
  "RColorBrewer",
  "scales",
  "forecast",
  "segmented",
  "cluster",
  "data.table",
  "readr",
  "purrr",
  "stringr",
  "broom",
  "janitor",
  "openxlsx",
  "writexl",
  "renv"
)

optional_pkgs <- c("nordpred", "BAPC", "INLA")

message("[INFO] Installing required R packages...")
installed <- rownames(installed.packages())
missing_required <- setdiff(required_pkgs, installed)

if (length(missing_required) > 0) {
  install.packages(missing_required, lib = user_lib, dependencies = TRUE, Ncpus = max(1L, parallel::detectCores() - 1L))
}

installed_after <- rownames(installed.packages())
still_missing_required <- setdiff(required_pkgs, installed_after)

if (length(still_missing_required) > 0) {
  message("[WARN] Retrying failed required packages one-by-one: ", paste(still_missing_required, collapse = ", "))
  for (pkg in still_missing_required) {
    try(install.packages(pkg, lib = user_lib, dependencies = TRUE), silent = FALSE)
  }
}

installed_after_retry <- rownames(installed.packages())
missing_required_final <- setdiff(required_pkgs, installed_after_retry)
missing_optional <- setdiff(optional_pkgs, installed_after_retry)

status <- data.frame(
  package = c(required_pkgs, optional_pkgs),
  required = c(rep(TRUE, length(required_pkgs)), rep(FALSE, length(optional_pkgs))),
  installed = c(
    required_pkgs %in% installed_after_retry,
    optional_pkgs %in% installed_after_retry
  ),
  stringsAsFactors = FALSE
)

write.csv(status, file.path(log_dir, "package_install_status.csv"), row.names = FALSE)

if (length(missing_required_final) > 0) {
  message("[ERROR] Missing required packages: ", paste(missing_required_final, collapse = ", "))
  quit(status = 1)
}

if (length(missing_optional) > 0) {
  message("[INFO] Optional packages not installed (non-blocking): ", paste(missing_optional, collapse = ", "))
}

message("[INFO] Required package installation complete.")
