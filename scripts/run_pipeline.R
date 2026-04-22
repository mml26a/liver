source("R/io_paths.R")

ensure_project_dirs()

phases <- data.frame(
  phase = c(
    "environment_bootstrap",
    "renv",
    "repo_map",
    "location_audit",
    "master_dataset",
    "refresh_existing_analyses",
    "decomposition",
    "joinpoint",
    "projection",
    "output_inventory",
    "manuscript_claim_validation",
    "manuscript_rewriting"
  ),
  script = c(
    NA,
    NA,
    "analysis/00_repo_map.R",
    "analysis/01_data_audit.R",
    "analysis/02_master_dataset.R",
    "analysis/03_refresh_existing_analyses.R",
    "analysis/04_decomposition.R",
    "analysis/05_joinpoint.R",
    "analysis/06_projection.R",
    "analysis/07_tables_figures.R",
    "analysis/08_extract_manuscript_numbers.R",
    NA
  ),
  stringsAsFactors = FALSE
)

results <- data.frame(
  phase = phases$phase,
  script = phases$script,
  status = NA_character_,
  message = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(phases))) {
  ph <- phases$phase[i]
  sc <- phases$script[i]

  if (is.na(sc)) {
    results$status[i] <- "MANUAL_OR_EXTERNAL"
    results$message[i] <- "Handled outside this runner."
    next
  }

  ok <- TRUE
  msg <- "OK"
  tryCatch(
    {
      source(sc, echo = FALSE, chdir = FALSE, local = new.env(parent = globalenv()))
    },
    error = function(e) {
      ok <<- FALSE
      msg <<- conditionMessage(e)
    }
  )
  results$status[i] <- if (ok) {
    "PASS"
  } else if (
    ph == "decomposition" &&
      grepl("No age-specific Number \\+ population rows available", msg, fixed = FALSE)
  ) {
    "BLOCKED_INPUT"
  } else {
    "FAIL"
  }
  results$message[i] <- msg
}

write.csv(results, path_in_project("logs", "pipeline_status.csv"), row.names = FALSE)

first_fail_idx <- which(results$status == "FAIL")[1]
first_blocked_idx <- which(results$status == "BLOCKED_INPUT")[1]
next_cmd <- if (!is.na(first_fail_idx)) {
  paste("Rscript", results$script[first_fail_idx])
} else if (!is.na(first_blocked_idx)) {
  "add age-specific Number rows plus matched population, then Rscript analysis/04_decomposition.R"
} else {
  "none"
}

md <- c(
  "# Pipeline Status",
  "",
  "Phase | Status | Script | Message",
  "---|---|---|---"
)
for (i in seq_len(nrow(results))) {
  md <- c(md, paste(results$phase[i], results$status[i], results$script[i], results$message[i], sep = " | "))
}
md <- c(
  md,
  "",
  paste0("First unresolved phase next action: `", next_cmd, "`")
)

writeLines(md, path_in_project("logs", "pipeline_status.md"))
