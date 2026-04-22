.project_root_path <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

.local_r_lib_path <- file.path(
  .project_root_path,
  ".r_libs",
  paste0("R-", R.version$major, ".", strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1])
)

if (dir.exists(.local_r_lib_path)) {
  .libPaths(unique(c(normalizePath(.local_r_lib_path, winslash = "/", mustWork = TRUE), .libPaths())))
}

.udunits_xml_path <- file.path(.local_r_lib_path, "units", "share", "udunits", "udunits2.xml")
if (file.exists(.udunits_xml_path) && !nzchar(Sys.getenv("UDUNITS2_XML_PATH"))) {
  .udunits_ascii_dir <- file.path(tempdir(), "codex_udunits")
  dir.create(.udunits_ascii_dir, recursive = TRUE, showWarnings = FALSE)
  file.copy(
    list.files(dirname(.udunits_xml_path), full.names = TRUE),
    .udunits_ascii_dir,
    overwrite = TRUE
  )
  Sys.setenv(UDUNITS2_XML_PATH = file.path(.udunits_ascii_dir, "udunits2.xml"))
}

project_root <- function() {
  .project_root_path
}

path_in_project <- function(...) {
  normalizePath(file.path(project_root(), ...), winslash = "/", mustWork = FALSE)
}

project_dirs <- function() {
  list(
    data_raw = path_in_project("data_raw"),
    data_intermediate = path_in_project("data_intermediate"),
    outputs = path_in_project("outputs"),
    outputs_tables = path_in_project("outputs", "tables"),
    outputs_figures = path_in_project("outputs", "figures"),
    logs = path_in_project("logs")
  )
}

ensure_project_dirs <- function() {
  dirs <- unlist(project_dirs(), use.names = FALSE)
  for (d in dirs) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(dirs)
}

required_raw_inputs <- function() {
  c(
    "gbd2023_BMI_HCC_global_alllevels.csv",
    "gbd2023_BMI_HCC_SDI_1990_2023.csv",
    "gbd2023_BMI_HCC_country_1990_2023.csv",
    "gbd2023_allHCC_global_1990_2023.csv",
    "gbd2023_allHCC_country_1990_2023.csv",
    "gbd2023_SDI_values_1950_2023.csv"
  )
}

resolve_raw_input_path <- function(filename) {
  candidates <- c(
    path_in_project(filename),
    path_in_project("data_raw", filename)
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) NA_character_ else existing[[1]]
}

raw_input_inventory <- function() {
  req <- required_raw_inputs()
  resolved <- vapply(req, resolve_raw_input_path, FUN.VALUE = character(1))
  data.frame(
    input_file = req,
    expected_path = path_in_project("data_raw", req),
    resolved_path = resolved,
    exists = !is.na(resolved),
    stringsAsFactors = FALSE
  )
}
