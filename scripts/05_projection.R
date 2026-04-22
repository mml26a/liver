#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(ggplot2)
  library(tibble)
  library(fs)
  library(stringr)
})

source("scripts/utils_gbd_pipeline.R")

if (!requireNamespace("forecast", quietly = TRUE)) {
  stop("Package `forecast` is required. Install with install.packages('forecast').", call. = FALSE)
}

ensure_dir("outputs")
ensure_dir("outputs/projection")

master_path_rds <- "data_clean/analytic_master_long.rds"
master_path_csv <- "data_clean/analytic_master_long.csv"

if (file_exists(master_path_rds)) {
  master <- readRDS(master_path_rds)
} else if (file_exists(master_path_csv)) {
  master <- safe_read_csv(master_path_csv)
} else {
  stop("Master dataset not found. Run scripts/02_build_master_dataset.R first.", call. = FALSE)
}

choose_top10_countries <- function(master_df) {
  if (file_exists("archived_tables/Table4_Top20_priority_countries.csv")) {
    t4 <- safe_read_csv("archived_tables/Table4_Top20_priority_countries.csv")
    c_col <- detect_column(t4, c("Country", "location_name"))
    if (!is.na(c_col)) {
      top <- t4 %>%
        mutate(country = trim_location(.data[[c_col]])) %>%
        pull(country) %>%
        unique() %>%
        .[seq_len(min(10, length(.)))]
      if (length(top) > 0) {
        return(top)
      }
    }
  }

  master_df %>%
    filter(
      scope == "country",
      risk_group == "BMI_attributable",
      measure_name == "DALYs",
      metric_name == "Rate",
      age_name %in% c("Age-standardized", "Age standardized"),
      year == 2023
    ) %>%
    arrange(desc(val)) %>%
    distinct(location_name, .keep_all = TRUE) %>%
    slice_head(n = 10) %>%
    pull(location_name)
}

top10_countries <- choose_top10_countries(master)
main_locations <- c("Global", SCOPE_SDI_GROUPS)
target_locations <- unique(c(main_locations, top10_countries))
target_measures <- c("DALYs", "Deaths")

projection_input <- master %>%
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
    rate = as.numeric(val),
    indicator = ifelse(measure_name == "DALYs", "DALY_ASR", "ASDR")
  ) %>%
  arrange(location_name, indicator, year)

if (nrow(projection_input) == 0) {
  stop("No age-standardized rate series found for projection.", call. = FALSE)
}

forecast_one_series <- function(dat, method = c("arima", "ets", "loglinear"), end_year = 2050) {
  method <- match.arg(method)
  dat <- dat %>% arrange(year)
  max_year <- max(dat$year, na.rm = TRUE)
  horizon <- end_year - max_year
  if (horizon <= 0) {
    return(tibble())
  }

  ts_obj <- stats::ts(dat$rate, start = min(dat$year), frequency = 1)
  future_years <- seq(max_year + 1, end_year)

  if (method == "arima") {
    fit <- forecast::auto.arima(
      ts_obj,
      seasonal = FALSE,
      stepwise = FALSE,
      approximation = FALSE
    )
    fc <- forecast::forecast(fit, h = horizon, level = 95)
    out <- tibble(
      year = future_years,
      predicted = as.numeric(fc$mean),
      lower_95 = as.numeric(fc$lower[, 1]),
      upper_95 = as.numeric(fc$upper[, 1])
    )
  } else if (method == "ets") {
    fit <- forecast::ets(ts_obj)
    fc <- forecast::forecast(fit, h = horizon, level = 95)
    out <- tibble(
      year = future_years,
      predicted = as.numeric(fc$mean),
      lower_95 = as.numeric(fc$lower[, 1]),
      upper_95 = as.numeric(fc$upper[, 1])
    )
  } else {
    fit <- stats::lm(log(rate) ~ year, data = dat)
    pred <- stats::predict(
      fit,
      newdata = tibble(year = future_years),
      interval = "prediction",
      level = 0.95
    )
    out <- tibble(
      year = future_years,
      predicted = as.numeric(exp(pred[, "fit"])),
      lower_95 = as.numeric(exp(pred[, "lwr"])),
      upper_95 = as.numeric(exp(pred[, "upr"]))
    )
  }

  out
}

methods <- c("arima", "ets", "loglinear")

projection_results <- projection_input %>%
  distinct(location_name, indicator) %>%
  arrange(location_name, indicator) %>%
  pmap_dfr(function(location_name, indicator) {
    loc_dat <- projection_input %>%
      filter(location_name == !!location_name, indicator == !!indicator)

    map_dfr(methods, function(m) {
      fc <- forecast_one_series(loc_dat, method = m, end_year = 2050)
      if (nrow(fc) == 0) {
        return(tibble())
      }
      fc %>%
        mutate(
          location_name = location_name,
          indicator = indicator,
          method = m
        )
    })
  }) %>%
  select(location_name, indicator, method, year, predicted, lower_95, upper_95)

observed_results <- projection_input %>%
  transmute(
    location_name,
    indicator,
    method = "observed",
    year,
    predicted = rate,
    lower_95 = NA_real_,
    upper_95 = NA_real_
  )

projection_all <- bind_rows(observed_results, projection_results) %>%
  arrange(location_name, indicator, method, year)

write_csv(projection_all, "outputs/projection/projection_all_methods.csv")

milestones <- projection_results %>%
  filter(year %in% c(2030, 2040, 2050)) %>%
  select(location_name, indicator, method, year, predicted, lower_95, upper_95) %>%
  pivot_wider(
    names_from = year,
    values_from = c(predicted, lower_95, upper_95),
    names_glue = "{.value}_{year}"
  )

base_2023 <- observed_results %>%
  filter(year == 2023) %>%
  select(location_name, indicator, baseline_2023 = predicted)

milestones <- milestones %>%
  left_join(base_2023, by = c("location_name", "indicator")) %>%
  mutate(
    change_2023_2050_pct = 100 * (predicted_2050 - baseline_2023) / baseline_2023
  ) %>%
  arrange(method, indicator, desc(change_2023_2050_pct))

write_csv(milestones, "outputs/projection/projection_milestones_2030_2040_2050.csv")

# Publication-ready primary table (ARIMA)
table_primary <- milestones %>%
  filter(method == "arima") %>%
  mutate(
    `2030 (95% PI)` = sprintf("%.2f (%.2f to %.2f)", predicted_2030, lower_95_2030, upper_95_2030),
    `2040 (95% PI)` = sprintf("%.2f (%.2f to %.2f)", predicted_2040, lower_95_2040, upper_95_2040),
    `2050 (95% PI)` = sprintf("%.2f (%.2f to %.2f)", predicted_2050, lower_95_2050, upper_95_2050)
  ) %>%
  select(
    location_name, indicator, baseline_2023,
    `2030 (95% PI)`, `2040 (95% PI)`, `2050 (95% PI)`,
    change_2023_2050_pct
  )
write_csv(table_primary, "outputs/projection/table_projection_primary_arima.csv")

# Figure A: Global + SDI (ARIMA primary)
plot_main <- projection_all %>%
  filter(location_name %in% main_locations, method %in% c("observed", "arima"))

p_main <- ggplot(plot_main, aes(x = year, y = predicted, color = location_name)) +
  geom_line(data = subset(plot_main, method == "observed"), linewidth = 0.8) +
  geom_line(data = subset(plot_main, method == "arima"), linewidth = 0.85, linetype = "dashed") +
  geom_ribbon(
    data = subset(plot_main, method == "arima"),
    aes(ymin = lower_95, ymax = upper_95, fill = location_name),
    alpha = 0.15,
    color = NA
  ) +
  geom_vline(xintercept = 2023, linetype = "dotted", color = "grey40") +
  facet_wrap(~indicator, scales = "free_y", ncol = 1) +
  scale_x_continuous(breaks = seq(1990, 2050, 10)) +
  labs(
    title = "Projected BMI-attributable liver cancer rates to 2050",
    subtitle = "Main analysis: ARIMA; solid = observed, dashed = projected",
    x = "Year",
    y = "Rate per 100,000",
    color = "Location",
    fill = "Location"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave("outputs/projection/fig_projection_global_sdi_arima.pdf", p_main, width = 12, height = 8)
ggsave("outputs/projection/fig_projection_global_sdi_arima.png", p_main, width = 12, height = 8, dpi = 320)

# Figure B: Sensitivity methods for Global
plot_sens <- projection_all %>%
  filter(location_name == "Global", method %in% c("arima", "ets", "loglinear", "observed"))

p_sens <- ggplot(plot_sens, aes(x = year, y = predicted, color = method)) +
  geom_line(linewidth = 0.85) +
  geom_ribbon(
    data = subset(plot_sens, method %in% c("arima", "ets", "loglinear")),
    aes(ymin = lower_95, ymax = upper_95, fill = method),
    alpha = 0.10,
    color = NA
  ) +
  geom_vline(xintercept = 2023, linetype = "dotted", color = "grey40") +
  facet_wrap(~indicator, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c(
    observed = "#333333",
    arima = "#1f78b4",
    ets = "#33a02c",
    loglinear = "#e31a1c"
  )) +
  scale_fill_manual(values = c(arima = "#1f78b4", ets = "#33a02c", loglinear = "#e31a1c")) +
  labs(
    title = "Sensitivity analysis for Global projections",
    subtitle = "ARIMA vs ETS vs log-linear extrapolation",
    x = "Year",
    y = "Rate per 100,000",
    color = "Method",
    fill = "Method"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave("outputs/projection/fig_projection_global_sensitivity.pdf", p_sens, width = 11, height = 8)
ggsave("outputs/projection/fig_projection_global_sensitivity.png", p_sens, width = 11, height = 8, dpi = 320)

# Figure C: Top 10 priority countries (ARIMA)
plot_top10 <- projection_all %>%
  filter(
    location_name %in% top10_countries,
    method %in% c("observed", "arima"),
    indicator == "DALY_ASR"
  )

p_top10 <- ggplot(plot_top10, aes(x = year, y = predicted)) +
  geom_line(data = subset(plot_top10, method == "observed"), color = "#1f78b4", linewidth = 0.7) +
  geom_line(data = subset(plot_top10, method == "arima"), color = "#e31a1c", linetype = "dashed", linewidth = 0.75) +
  geom_ribbon(
    data = subset(plot_top10, method == "arima"),
    aes(ymin = lower_95, ymax = upper_95),
    fill = "#e31a1c",
    alpha = 0.14,
    color = NA
  ) +
  geom_vline(xintercept = 2023, linetype = "dotted", color = "grey40", linewidth = 0.3) +
  facet_wrap(~location_name, scales = "free_y", ncol = 5) +
  scale_x_continuous(breaks = c(1990, 2010, 2030, 2050)) +
  labs(
    title = "Top 10 priority countries: projected DALY ASR to 2050",
    subtitle = "Solid = observed; dashed = ARIMA projection with 95% PI",
    x = "Year",
    y = "DALY ASR per 100,000"
  ) +
  theme_bw(base_size = 9) +
  theme(
    strip.text = element_text(face = "bold", size = 8),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.minor = element_blank()
  )

ggsave("outputs/projection/fig_projection_top10_countries_arima.pdf", p_top10, width = 14, height = 8)
ggsave("outputs/projection/fig_projection_top10_countries_arima.png", p_top10, width = 14, height = 8, dpi = 320)

# Optional APC/Nordpred-style age-specific projection (runs only when future pop is available)
run_apc_optional <- function(master_df, target_locs) {
  age_counts <- master_df %>%
    filter(
      risk_group == "BMI_attributable",
      metric_name == "Number",
      location_name %in% target_locs,
      measure_name %in% target_measures,
      !is.na(population),
      !is.na(val),
      !grepl("^Age-standardized$|^All ages$", age_name, ignore.case = TRUE)
    ) %>%
    mutate(
      year = as.integer(year),
      cases = as.numeric(val),
      pop = as.numeric(population),
      indicator = ifelse(measure_name == "DALYs", "DALY_ASR", "ASDR")
    )

  if (nrow(age_counts) == 0) {
    return(list(results = tibble(), note = "No age-specific Number+population rows found."))
  }

  if (!any(age_counts$year > 2023)) {
    return(list(results = tibble(), note = "No future population years (>2023) found; APC projection skipped."))
  }

  hist <- age_counts %>% filter(year <= 2023, pop > 0)
  fut <- age_counts %>% filter(year >= 2024, year <= 2050, pop > 0)
  if (nrow(fut) == 0) {
    return(list(results = tibble(), note = "Future population table has no usable rows for 2024-2050."))
  }

  age_weights <- hist %>%
    filter(location_name == "Global", year == 2023) %>%
    group_by(age_name) %>%
    summarise(w = sum(pop, na.rm = TRUE), .groups = "drop") %>%
    mutate(w = w / sum(w, na.rm = TRUE))

  apc_preds <- hist %>%
    distinct(location_name, indicator, age_name) %>%
    pmap_dfr(function(location_name, indicator, age_name) {
      h <- hist %>%
        filter(location_name == !!location_name, indicator == !!indicator, age_name == !!age_name)
      f <- fut %>%
        filter(location_name == !!location_name, indicator == !!indicator, age_name == !!age_name)

      if (nrow(h) < 8 || nrow(f) == 0) {
        return(tibble())
      }

      fit <- try(
        glm(cases ~ year + offset(log(pop)), family = poisson(link = "log"), data = h),
        silent = TRUE
      )
      if (inherits(fit, "try-error")) {
        return(tibble())
      }

      pred_link <- predict(fit, newdata = f, type = "link", se.fit = TRUE)

      tibble(
        location_name = location_name,
        indicator = indicator,
        age_name = age_name,
        year = f$year,
        pop = f$pop,
        rate = exp(pred_link$fit),
        lower_rate = exp(pred_link$fit - 1.96 * pred_link$se.fit),
        upper_rate = exp(pred_link$fit + 1.96 * pred_link$se.fit)
      )
    })

  if (nrow(apc_preds) == 0) {
    return(list(results = tibble(), note = "APC models could not be fitted for available age series."))
  }

  apc_asr <- apc_preds %>%
    left_join(age_weights, by = "age_name") %>%
    mutate(w = ifelse(is.na(w), 0, w)) %>%
    group_by(location_name, indicator, year) %>%
    summarise(
      predicted = sum(rate * w, na.rm = TRUE),
      lower_95 = sum(lower_rate * w, na.rm = TRUE),
      upper_95 = sum(upper_rate * w, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(method = "apc_poisson_age_specific")

  list(results = apc_asr, note = "APC age-specific projection computed.")
}

apc_out <- run_apc_optional(master, main_locations)
write_lines(apc_out$note, "outputs/projection/apc_projection_note.txt")

if (nrow(apc_out$results) > 0) {
  write_csv(apc_out$results, "outputs/projection/projection_apc_optional.csv")
}

message("Projection analysis complete: outputs/projection/")
