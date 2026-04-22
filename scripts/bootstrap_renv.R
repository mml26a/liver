options(repos = c(CRAN = "https://cloud.r-project.org"))

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

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

renv::consent(provided = TRUE)

if (!file.exists("renv.lock")) {
  renv::init(bare = TRUE, restart = FALSE)
}

renv::activate()
renv::settings$snapshot.type("explicit")
renv::install(required_pkgs, prompt = FALSE)
renv::snapshot(packages = required_pkgs, prompt = FALSE, force = TRUE)
renv::restore(prompt = FALSE)

cat("[INFO] renv bootstrap completed.\n")
