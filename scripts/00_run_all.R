#!/usr/bin/env Rscript

steps <- c(
  "scripts/01_data_audit.R",
  "scripts/02_build_master_dataset.R",
  "scripts/03_decomposition.R",
  "scripts/04_joinpoint.R",
  "scripts/05_projection.R"
)

for (s in steps) {
  message("Running: ", s)
  source(s, echo = FALSE)
}

message("Pipeline finished.")
