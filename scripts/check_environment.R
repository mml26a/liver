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

rscript_path <- Sys.which("Rscript")
r_version <- R.version.string
r_version_parts <- strsplit(as.character(getRversion()), ".", fixed = TRUE)[[1]]
project_lib <- file.path(project_root, ".r_libs", paste0("R-", r_version_parts[1], ".", r_version_parts[2]))
dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(project_lib, .libPaths())))
udunits_xml <- file.path(project_lib, "units", "share", "udunits", "udunits2.xml")
if (file.exists(udunits_xml)) {
  udunits_ascii_dir <- file.path(tempdir(), "codex_udunits")
  dir.create(udunits_ascii_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(
    list.files(dirname(udunits_xml), full.names = TRUE),
    udunits_ascii_dir,
    overwrite = TRUE
  )
  Sys.setenv(UDUNITS2_XML_PATH = file.path(udunits_ascii_dir, "udunits2.xml"))
}

renv_available <- requireNamespace("renv", quietly = TRUE)
renv_library <- NA_character_
if (renv_available && file.exists(file.path(project_root, "renv.lock"))) {
  tryCatch(
    {
      renv::load(project = project_root)
      renv_library <<- renv::paths$library(project = project_root)
    },
    error = function(e) {
      cat("[WARN] renv project library could not be loaded; using active R libraries.\n")
    }
  )
}

installed_tbl <- installed.packages()
installed <- rownames(installed_tbl)
missing_required <- setdiff(required_pkgs, installed)
missing_optional <- setdiff(optional_pkgs, installed)

namespace_failures <- character(0)
for (pkg in required_pkgs) {
  pkg_loadable <- tryCatch(
    isTRUE(requireNamespace(pkg, quietly = TRUE)),
    error = function(e) FALSE
  )
  if (!pkg_loadable) {
    namespace_failures <- c(namespace_failures, pkg)
  }
}

lockfile_path <- file.path(project_root, "renv.lock")
lockfile_exists <- file.exists(lockfile_path)
lock_missing_required <- character(0)
lock_version_mismatch <- character(0)
lock_check_pass <- FALSE

if (lockfile_exists && renv_available && requireNamespace("jsonlite", quietly = TRUE) && !is.na(renv_library)) {
  lock <- jsonlite::fromJSON(lockfile_path, simplifyVector = FALSE)
  lock_packages <- names(lock$Packages %||% list())
  lock_missing_required <- setdiff(required_pkgs, lock_packages)
  for (pkg in intersect(required_pkgs, lock_packages)) {
    lock_ver <- as.character(lock$Packages[[pkg]]$Version %||% NA_character_)
    lib_ver <- if (pkg %in% rownames(installed_tbl)) as.character(installed_tbl[pkg, "Version"]) else NA_character_
    if (is.na(lib_ver) || is.na(lock_ver) || lock_ver != lib_ver) {
      lock_version_mismatch <- c(lock_version_mismatch, sprintf("%s(lock=%s,lib=%s)", pkg, lock_ver, lib_ver))
    }
  }

  lock_check_pass <- length(lock_missing_required) == 0 && length(lock_version_mismatch) == 0
}

cat("[INFO] Rscript path: ", rscript_path, "\n", sep = "")
cat("[INFO] R version: ", r_version, "\n", sep = "")
cat("[INFO] project root: ", project_root, "\n", sep = "")
cat("[INFO] active libraries: ", paste(.libPaths(), collapse = " | "), "\n", sep = "")
cat("[INFO] renv available: ", renv_available, "\n", sep = "")
cat("[INFO] renv project library: ", renv_library, "\n", sep = "")
cat("[INFO] Non-interactive restore command: Rscript -e \"renv::restore(prompt = FALSE)\"\n")
cat("[INFO] Required packages installed: ", length(required_pkgs) - length(missing_required), "/", length(required_pkgs), "\n", sep = "")
if (length(missing_required) > 0) {
  cat("[ERROR] Missing required packages: ", paste(missing_required, collapse = ", "), "\n", sep = "")
}
if (length(namespace_failures) > 0) {
  cat("[ERROR] Required packages not loadable from renv project library: ", paste(namespace_failures, collapse = ", "), "\n", sep = "")
}
if (length(missing_optional) > 0) {
  cat("[INFO] Missing optional packages (non-blocking): ", paste(missing_optional, collapse = ", "), "\n", sep = "")
}
cat("[INFO] renv.lock exists: ", lockfile_exists, "\n", sep = "")
if (lockfile_exists) {
  if (length(lock_missing_required) > 0) {
    cat("[ERROR] Required packages missing from renv.lock: ", paste(lock_missing_required, collapse = ", "), "\n", sep = "")
  }
  if (length(lock_version_mismatch) > 0) {
    cat("[ERROR] renv lock/library version mismatch: ", paste(lock_version_mismatch, collapse = "; "), "\n", sep = "")
  }
  cat("[INFO] renv lockfile consistency check pass: ", lock_check_pass, "\n", sep = "")
}

pkg_status <- data.frame(
  package = c(required_pkgs, optional_pkgs),
  required = c(rep(TRUE, length(required_pkgs)), rep(FALSE, length(optional_pkgs))),
  installed = c(
    required_pkgs %in% installed,
    optional_pkgs %in% installed
  ),
  loadable_from_project_library = c(
    required_pkgs %in% setdiff(required_pkgs, namespace_failures),
    rep(NA, length(optional_pkgs))
  ),
  stringsAsFactors = FALSE
)
write.csv(pkg_status, file.path(log_dir, "check_environment_package_status.csv"), row.names = FALSE)

if (!nzchar(rscript_path)) {
  quit(status = 1)
}
if (length(missing_required) > 0) {
  quit(status = 1)
}
if (length(namespace_failures) > 0) {
  quit(status = 1)
}
if (!lockfile_exists) {
  cat("[WARN] renv.lock is not present.\n")
} else if (!lock_check_pass) {
  cat("[WARN] renv lockfile consistency check did not pass; continuing because required packages are loadable.\n")
}

quit(status = 0)
