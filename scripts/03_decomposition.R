#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(ggplot2)
  library(scales)
  library(tibble)
  library(fs)
})

source("scripts/utils_gbd_pipeline.R")

ensure_dir("outputs")
ensure_dir("outputs/decomposition")

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

decomp_input <- master %>%
  filter(
    risk_group == "BMI_attributable",
    metric_name == "Number",
    location_name %in% target_locations,
    measure_name %in% target_measures,
    !is.na(population),
    !is.na(val),
    !is.na(age_name)
  ) %>%
  filter(
    !grepl("^Age-standardized$|^All ages$", age_name, ignore.case = TRUE)
  ) %>%
  mutate(
    year = as.integer(year),
    population = as.numeric(population),
    val = as.numeric(val)
  )

if (nrow(decomp_input) == 0) {
  stop(
    paste0(
      "No age-specific Number + population rows available for decomposition. ",
      "Ensure master dataset includes age-specific counts and population."
    ),
    call. = FALSE
  )
}

das_gupta_three_factor <- function(data, year1, year2, location, measure) {
  d1 <- data %>%
    filter(location_name == location, measure_name == measure, year == year1) %>%
    select(age_name, cases1 = val, pop1 = population)
  d2 <- data %>%
    filter(location_name == location, measure_name == measure, year == year2) %>%
    select(age_name, cases2 = val, pop2 = population)

  dd <- inner_join(d1, d2, by = "age_name") %>%
    filter(pop1 > 0, pop2 > 0) %>%
    mutate(
      rate1 = cases1 / pop1,
      rate2 = cases2 / pop2,
      prop1 = pop1 / sum(pop1),
      prop2 = pop2 / sum(pop2)
    )

  if (nrow(dd) < 2) {
    return(tibble(
      location_name = location,
      measure_name = measure,
      year1 = year1,
      year2 = year2,
      observed_change = NA_real_,
      population_effect = NA_real_,
      aging_effect = NA_real_,
      rate_effect = NA_real_,
      decomp_sum = NA_real_,
      residual = NA_real_,
      pct_population = NA_real_,
      pct_aging = NA_real_,
      pct_rate = NA_real_
    ))
  }

  p1 <- sum(dd$pop1)
  p2 <- sum(dd$pop2)
  n1 <- sum(dd$cases1)
  n2 <- sum(dd$cases2)

  rate_effect <- mean(c(
    p1 * sum(dd$prop1 * (dd$rate2 - dd$rate1)),
    p1 * sum(dd$prop2 * (dd$rate2 - dd$rate1)),
    p2 * sum(dd$prop1 * (dd$rate2 - dd$rate1)),
    p2 * sum(dd$prop2 * (dd$rate2 - dd$rate1))
  ))

  aging_effect <- mean(c(
    p1 * sum((dd$prop2 - dd$prop1) * dd$rate1),
    p1 * sum((dd$prop2 - dd$prop1) * dd$rate2),
    p2 * sum((dd$prop2 - dd$prop1) * dd$rate1),
    p2 * sum((dd$prop2 - dd$prop1) * dd$rate2)
  ))

  population_effect <- mean(c(
    (p2 - p1) * sum(dd$prop1 * dd$rate1),
    (p2 - p1) * sum(dd$prop1 * dd$rate2),
    (p2 - p1) * sum(dd$prop2 * dd$rate1),
    (p2 - p1) * sum(dd$prop2 * dd$rate2)
  ))

  observed_change <- n2 - n1
  decomp_sum <- population_effect + aging_effect + rate_effect
  residual <- observed_change - decomp_sum

  tibble(
    location_name = location,
    measure_name = measure,
    year1 = year1,
    year2 = year2,
    observed_change = observed_change,
    population_effect = population_effect,
    aging_effect = aging_effect,
    rate_effect = rate_effect,
    decomp_sum = decomp_sum,
    residual = residual,
    pct_population = ifelse(observed_change == 0, NA_real_, 100 * population_effect / observed_change),
    pct_aging = ifelse(observed_change == 0, NA_real_, 100 * aging_effect / observed_change),
    pct_rate = ifelse(observed_change == 0, NA_real_, 100 * rate_effect / observed_change)
  )
}

periods <- tribble(
  ~period, ~year1, ~year2,
  "1990-2023", 1990L, 2023L,
  "1990-2006", 1990L, 2006L,
  "2006-2023", 2006L, 2023L
)

decomp_results <- crossing(
  location_name = target_locations,
  measure_name = target_measures,
  periods
) %>%
  pmap_dfr(function(location_name, measure_name, period, year1, year2) {
    das_gupta_three_factor(
      data = decomp_input,
      year1 = year1,
      year2 = year2,
      location = location_name,
      measure = measure_name
    ) %>%
      mutate(period = period)
  }) %>%
  mutate(
    residual_abs = abs(residual),
    residual_rel = ifelse(abs(observed_change) < 1e-9, NA_real_, residual / observed_change),
    residual_pass = ifelse(is.na(residual), FALSE, abs(residual) <= pmax(1e-6, abs(observed_change) * 1e-8))
  )

publication_table <- decomp_results %>%
  transmute(
    period,
    location_name,
    measure_name,
    observed_change = round(observed_change, 2),
    population_effect = round(population_effect, 2),
    aging_effect = round(aging_effect, 2),
    rate_effect = round(rate_effect, 2),
    pct_population = round(pct_population, 1),
    pct_aging = round(pct_aging, 1),
    pct_rate = round(pct_rate, 1),
    residual = signif(residual, 4),
    residual_pass
  )

write_csv(decomp_results, "outputs/decomposition/decomposition_results_full.csv")
write_csv(publication_table, "outputs/decomposition/table_decomposition_publication.csv")

residual_check <- decomp_results %>%
  select(period, location_name, measure_name, observed_change, decomp_sum, residual, residual_abs, residual_rel, residual_pass)
write_csv(residual_check, "outputs/decomposition/decomposition_residual_check.csv")

plot_data <- decomp_results %>%
  select(period, location_name, measure_name, population_effect, aging_effect, rate_effect) %>%
  pivot_longer(
    cols = c(population_effect, aging_effect, rate_effect),
    names_to = "component",
    values_to = "contribution"
  ) %>%
  mutate(
    component = factor(
      component,
      levels = c("population_effect", "aging_effect", "rate_effect"),
      labels = c("Population growth", "Age structure", "Age-specific rate")
    ),
    location_name = factor(location_name, levels = rev(target_locations))
  )

p_stacked <- ggplot(plot_data, aes(x = location_name, y = contribution, fill = component)) +
  geom_col(width = 0.72) +
  facet_grid(measure_name ~ period, scales = "free_x") +
  coord_flip() +
  scale_y_continuous(labels = scales::comma) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B", "#E15759")) +
  labs(
    x = NULL,
    y = "Contribution to change in counts",
    fill = NULL,
    title = "Das Gupta three-factor decomposition of BMI-attributable liver cancer burden",
    subtitle = "Global and SDI groups; residual-checked decomposition"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey95"),
    strip.text = element_text(face = "bold")
  )

ggsave("outputs/decomposition/fig_decomposition_stacked.pdf", p_stacked, width = 12, height = 7)
ggsave("outputs/decomposition/fig_decomposition_stacked.png", p_stacked, width = 12, height = 7, dpi = 320)

message("Decomposition complete: outputs/decomposition/")
