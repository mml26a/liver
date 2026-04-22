#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(ggplot2)
  library(tibble)
  library(fs)
})

source("scripts/utils_gbd_pipeline.R")

if (!requireNamespace("segmented", quietly = TRUE)) {
  stop("Package `segmented` is required. Install with install.packages('segmented').", call. = FALSE)
}

ensure_dir("outputs")
ensure_dir("outputs/joinpoint")

master_path_rds <- "data_clean/analytic_master_long.rds"
master_path_csv <- "data_clean/analytic_master_long.csv"

if (file_exists(master_path_rds)) {
  master <- readRDS(master_path_rds)
} else if (file_exists(master_path_csv)) {
  master <- safe_read_csv(master_path_csv)
} else {
  stop("Master dataset not found. Run scripts/02_build_master_dataset.R first.", call. = FALSE)
}

target_locations <- c("Global", SCOPE_SDI_GROUPS)
target_measures <- c("DALYs", "Deaths")

joinpoint_input <- master %>%
  filter(
    risk_group == "BMI_attributable",
    location_name %in% target_locations,
    measure_name %in% target_measures,
    metric_name == "Rate",
    age_name %in% c("Age-standardized", "Age standardized"),
    !is.na(val),
    val > 0
  ) %>%
  mutate(
    year = as.integer(year),
    asr = as.numeric(val),
    indicator = ifelse(measure_name == "DALYs", "DALY_ASR", "ASDR")
  ) %>%
  arrange(location_name, indicator, year)

if (nrow(joinpoint_input) == 0) {
  stop("No annual age-standardized rate series found for joinpoint analysis.", call. = FALSE)
}

fit_joinpoint_series <- function(dat, max_joinpoints = 3) {
  dat <- dat %>%
    arrange(year) %>%
    filter(!is.na(asr), asr > 0)

  if (nrow(dat) < 8) {
    return(NULL)
  }

  fit_lm <- lm(log(asr) ~ year, data = dat)
  best_model <- fit_lm
  best_bic <- BIC(fit_lm)
  best_k <- 0L

  for (k in seq_len(max_joinpoints)) {
    try({
      seg_fit <- segmented::segmented(
        fit_lm,
        seg.Z = ~year,
        npsi = k,
        control = segmented::seg.control(
          n.boot = 50,
          it.max = 100,
          tol = 1e-6
        )
      )
      this_bic <- BIC(seg_fit)
      if (is.finite(this_bic) && this_bic < best_bic) {
        best_model <- seg_fit
        best_bic <- this_bic
        best_k <- k
      }
    }, silent = TRUE)
  }

  if (best_k == 0L) {
    slope_est <- coef(best_model)[["year"]]
    slope_ci <- confint(best_model)["year", ]
    slope_se <- summary(best_model)$coefficients["year", "Std. Error"]
    segments <- tibble(
      segment = 1L,
      start_year = min(dat$year),
      end_year = max(dat$year),
      slope_est = slope_est,
      slope_se = slope_se,
      apc = 100 * (exp(slope_est) - 1),
      apc_lower = 100 * (exp(slope_ci[1]) - 1),
      apc_upper = 100 * (exp(slope_ci[2]) - 1)
    )
    joinpoints <- numeric()
  } else {
    jp_est <- as.numeric(best_model$psi[, "Est."])
    joinpoints <- sort(unique(round(jp_est)))

    breakpoints <- sort(unique(c(min(dat$year), joinpoints, max(dat$year))))
    if (length(breakpoints) < 2) {
      breakpoints <- c(min(dat$year), max(dat$year))
    }

    slopes <- segmented::slope(best_model)$year
    n_seg <- nrow(slopes)
    seg_start <- breakpoints[seq_len(n_seg)]
    seg_end <- breakpoints[seq_len(n_seg) + 1]

    segments <- tibble(
      segment = seq_len(n_seg),
      start_year = seg_start,
      end_year = seg_end,
      slope_est = slopes[, "Est."],
      slope_se = slopes[, "St.Err."]
    ) %>%
      mutate(
        apc = 100 * (exp(slope_est) - 1),
        apc_lower = 100 * (exp(slope_est - 1.96 * slope_se) - 1),
        apc_upper = 100 * (exp(slope_est + 1.96 * slope_se) - 1)
      )
  }

  segments <- segments %>%
    mutate(
      seg_length = pmax(1, end_year - start_year),
      weight = seg_length / sum(seg_length)
    )

  weighted_beta <- sum(segments$weight * segments$slope_est)
  weighted_se <- sqrt(sum((segments$weight * segments$slope_se)^2))
  aapc <- 100 * (exp(weighted_beta) - 1)
  aapc_lower <- 100 * (exp(weighted_beta - 1.96 * weighted_se) - 1)
  aapc_upper <- 100 * (exp(weighted_beta + 1.96 * weighted_se) - 1)

  fitted <- tibble(
    year = dat$year,
    observed = dat$asr,
    fitted = as.numeric(exp(predict(best_model)))
  )

  list(
    model = best_model,
    n_joinpoints = best_k,
    bic = best_bic,
    joinpoints = joinpoints,
    segments = segments,
    aapc = aapc,
    aapc_lower = aapc_lower,
    aapc_upper = aapc_upper,
    fitted = fitted
  )
}

series_index <- joinpoint_input %>%
  distinct(location_name, indicator) %>%
  arrange(location_name, indicator)

joinpoint_models <- pmap(
  series_index,
  function(location_name, indicator) {
    dat <- joinpoint_input %>%
      filter(location_name == !!location_name, indicator == !!indicator)
    fit_joinpoint_series(dat, max_joinpoints = 3)
  }
)
names(joinpoint_models) <- paste(series_index$location_name, series_index$indicator, sep = " | ")

summary_tbl <- map2_dfr(
  joinpoint_models,
  names(joinpoint_models),
  function(res, nm) {
    if (is.null(res)) {
      parts <- str_split(nm, "\\s\\|\\s", simplify = TRUE)
      return(tibble(
        location_name = parts[1],
        indicator = parts[2],
        n_joinpoints = NA_integer_,
        bic = NA_real_,
        joinpoints = NA_character_,
        aapc = NA_real_,
        aapc_lower = NA_real_,
        aapc_upper = NA_real_
      ))
    }
    parts <- str_split(nm, "\\s\\|\\s", simplify = TRUE)
    tibble(
      location_name = parts[1],
      indicator = parts[2],
      n_joinpoints = res$n_joinpoints,
      bic = res$bic,
      joinpoints = ifelse(length(res$joinpoints) == 0, "None", paste(res$joinpoints, collapse = ", ")),
      aapc = res$aapc,
      aapc_lower = res$aapc_lower,
      aapc_upper = res$aapc_upper
    )
  }
)

segments_tbl <- map2_dfr(
  joinpoint_models,
  names(joinpoint_models),
  function(res, nm) {
    parts <- str_split(nm, "\\s\\|\\s", simplify = TRUE)
    if (is.null(res)) {
      return(tibble())
    }
    res$segments %>%
      mutate(
        location_name = parts[1],
        indicator = parts[2]
      ) %>%
      select(location_name, indicator, everything())
  }
)

table1_ready <- summary_tbl %>%
  mutate(
    aapc_ci = ifelse(
      is.na(aapc),
      NA_character_,
      sprintf("%.2f%% (%.2f to %.2f)", aapc, aapc_lower, aapc_upper)
    )
  ) %>%
  select(location_name, indicator, n_joinpoints, joinpoints, aapc_ci, bic)

write_csv(summary_tbl, "outputs/joinpoint/joinpoint_summary.csv")
write_csv(segments_tbl, "outputs/joinpoint/joinpoint_segments.csv")
write_csv(table1_ready, "outputs/joinpoint/table1_joinpoint_ready.csv")

# Figure-ready fitted object
fitted_tbl <- map2_dfr(
  joinpoint_models,
  names(joinpoint_models),
  function(res, nm) {
    if (is.null(res)) {
      return(tibble())
    }
    parts <- str_split(nm, "\\s\\|\\s", simplify = TRUE)
    res$fitted %>%
      mutate(
        location_name = parts[1],
        indicator = parts[2]
      )
  }
)

joinpoint_lines <- summary_tbl %>%
  filter(!is.na(joinpoints), joinpoints != "None") %>%
  separate_rows(joinpoints, sep = ",\\s*") %>%
  mutate(joinpoints = as.numeric(joinpoints))

p <- ggplot(fitted_tbl, aes(x = year)) +
  geom_point(aes(y = observed), size = 1.1, alpha = 0.55, color = "grey35") +
  geom_line(aes(y = fitted), linewidth = 0.8, color = "#1f78b4") +
  geom_vline(
    data = joinpoint_lines,
    aes(xintercept = joinpoints),
    linetype = "dashed",
    linewidth = 0.35,
    color = "#e31a1c"
  ) +
  facet_grid(indicator ~ location_name, scales = "free_y") +
  labs(
    title = "Joinpoint regression (BIC-selected, 0-3 joinpoints)",
    subtitle = "BMI-attributable liver cancer age-standardized rates, 1990-2023",
    x = "Year",
    y = "Rate per 100,000"
  ) +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95"),
    strip.text = element_text(face = "bold")
  )

ggsave("outputs/joinpoint/figure1_joinpoint_replacement.pdf", p, width = 14, height = 7)
ggsave("outputs/joinpoint/figure1_joinpoint_replacement.png", p, width = 14, height = 7, dpi = 320)

saveRDS(
  list(
    input = joinpoint_input,
    models = joinpoint_models,
    summary = summary_tbl,
    segments = segments_tbl,
    table1_ready = table1_ready,
    fitted = fitted_tbl
  ),
  "outputs/joinpoint/joinpoint_results_object.rds"
)

message("Joinpoint analysis complete: outputs/joinpoint/")
