source("R/io_paths.R")

library(xml2)
library(zip)

ensure_project_dirs()

original_docx <- path_in_project("Liver cancer_manuscript.docx")
number_bank_path <- path_in_project("outputs", "tables", "manuscript_number_bank.csv")
out_docx <- path_in_project("manuscript", "Liver_cancer_manuscript_revised_submission_checked_20260416.docx")
validation_log <- path_in_project("logs", "revised_submission_checked_docx_validation.md")

if (!file.exists(original_docx)) {
  stop("Original manuscript docx not found.", call. = FALSE)
}
if (!file.exists(number_bank_path)) {
  stop("Manuscript number bank not found.", call. = FALSE)
}

claims <- read.csv(number_bank_path, stringsAsFactors = FALSE, check.names = FALSE)
claim <- function(id) {
  value <- claims$value[claims$claim_id == id]
  if (length(value) == 0 || is.na(value[1]) || !nzchar(value[1])) {
    stop(sprintf("Required claim_id missing from number bank: %s", id), call. = FALSE)
  }
  value[1]
}

split_arrow <- function(value) {
  parts <- strsplit(value, " -> ", fixed = TRUE)[[1]]
  if (length(parts) != 2) stop(sprintf("Expected arrow-separated claim value: %s", value), call. = FALSE)
  parts
}

daly_asr <- split_arrow(claim("global_daly_asr_1990_2023"))
asdr <- split_arrow(claim("global_asdr_1990_2023"))
daly_share <- split_arrow(claim("global_daly_share_1990_2023"))
death_share <- split_arrow(claim("global_death_share_1990_2023"))

text_escape <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  x
}

make_p <- function(text, style = NULL, bold = FALSE) {
  style_xml <- if (!is.null(style)) {
    sprintf("<w:pPr><w:pStyle w:val=\"%s\"/></w:pPr>", text_escape(style))
  } else {
    ""
  }
  bold_xml <- if (bold) "<w:rPr><w:b/></w:rPr>" else ""
  read_xml(sprintf(
    paste0(
      "<w:p xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" ",
      "xmlns:xml=\"http://www.w3.org/XML/1998/namespace\">",
      "%s<w:r>%s<w:t xml:space=\"preserve\">%s</w:t></w:r></w:p>"
    ),
    style_xml,
    bold_xml,
    text_escape(text)
  ))
}

temp_dir <- tempfile("revised_docx_")
dir.create(temp_dir, recursive = TRUE)
unzip(original_docx, exdir = temp_dir)

document_xml <- file.path(temp_dir, "word", "document.xml")
doc <- read_xml(document_xml)
ns <- xml_ns(doc)
body <- xml_find_first(doc, "//w:body", ns)

get_text <- function(node) {
  paste(xml_text(xml_find_all(node, ".//w:t", ns)), collapse = "")
}

children <- function() xml_children(body)
child_texts <- function() vapply(children(), get_text, FUN.VALUE = character(1))

find_child <- function(text) {
  hits <- which(child_texts() == text)
  if (length(hits) == 0) stop(sprintf("Could not find manuscript marker: %s", text), call. = FALSE)
  hits[1]
}

insert_after <- function(anchor_text, new_nodes) {
  kids <- children()
  anchor <- kids[[find_child(anchor_text)]]
  for (node in rev(new_nodes)) {
    xml_add_sibling(anchor, node, .where = "after")
  }
}

insert_before <- function(anchor_text, new_nodes) {
  kids <- children()
  anchor <- kids[[find_child(anchor_text)]]
  for (node in new_nodes) {
    xml_add_sibling(anchor, node, .where = "before")
  }
}

replace_between <- function(start_text, end_text, new_nodes) {
  kids <- children()
  texts <- child_texts()
  start_idx <- which(texts == start_text)[1]
  end_idx <- which(texts == end_text)[1]
  if (is.na(start_idx) || is.na(end_idx) || end_idx <= start_idx) {
    stop(sprintf("Invalid replacement range: %s -> %s", start_text, end_text), call. = FALSE)
  }
  if (end_idx > start_idx + 1) {
    xml_remove(kids[(start_idx + 1):(end_idx - 1)])
  }
  insert_after(start_text, new_nodes)
}

replace_paragraph_text <- function(marker_text, replacement) {
  node <- children()[[find_child(marker_text)]]
  text_nodes <- xml_find_all(node, ".//w:t", ns)
  if (length(text_nodes) == 0) stop(sprintf("No text node found for: %s", marker_text), call. = FALSE)
  xml_text(text_nodes[[1]]) <- replacement
  if (length(text_nodes) > 1) {
    for (extra in text_nodes[-1]) xml_text(extra) <- ""
  }
}

normal <- function(...) lapply(c(...), make_p)
heading2 <- function(text) make_p(text, style = "4", bold = TRUE)
heading3 <- function(text) make_p(text, style = "5", bold = TRUE)

make_cell <- function(text, header = FALSE) {
  bold_xml <- if (header) "<w:rPr><w:b/></w:rPr>" else ""
  sprintf(
    "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr><w:p><w:r>%s<w:t xml:space=\"preserve\">%s</w:t></w:r></w:p></w:tc>",
    bold_xml,
    text_escape(text)
  )
}

make_tbl <- function(df) {
  header <- paste(vapply(names(df), make_cell, character(1), header = TRUE), collapse = "")
  rows <- c(sprintf("<w:tr>%s</w:tr>", header))
  for (i in seq_len(nrow(df))) {
    cells <- paste(vapply(as.character(df[i, ]), make_cell, character(1)), collapse = "")
    rows <- c(rows, sprintf("<w:tr>%s</w:tr>", cells))
  }
  read_xml(sprintf(
    paste0(
      "<w:tbl xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" ",
      "xmlns:xml=\"http://www.w3.org/XML/1998/namespace\">",
      "<w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/><w:tblBorders>",
      "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>",
      "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>",
      "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>",
      "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>",
      "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>",
      "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"auto\"/>",
      "</w:tblBorders></w:tblPr>%s</w:tbl>"
    ),
    paste(rows, collapse = "")
  ))
}

replace_paragraph_text(
  "Global trends, trajectories, and country typologies of liver cancer attributable to high body-mass index",
  "Global trends, joinpoints, projections, trajectories, and country typologies of liver cancer attributable to high body-mass index"
)

abstract_nodes <- normal(
  "Background: The global burden of liver cancer is increasingly influenced by metabolic risk, but cross-country heterogeneity and time-varying trends in liver cancer attributable to high body-mass index (BMI) remain incompletely characterized.",
  paste0(
    "Methods: Using GBD 2023 estimates for ", claim("country_cohort_n"),
    " countries and territories from 1990 to 2023, we quantified age-standardized death rates (ASDRs), age-standardized DALY rates (DALY ASRs), and BMI-attributable shares for liver cancer attributable to high BMI. Long-term trends were summarized using EAPC and joinpoint regression with BIC-based model selection. We projected DALY ASR and ASDR to 2050 using ARIMA as the primary model, with ETS and log-linear models as sensitivity analyses."
  ),
  paste0(
    "Results: Globally, BMI-attributable liver-cancer DALY ASR increased from ", daly_asr[1],
    " to ", daly_asr[2], " per 100,000 from 1990 to 2023, and ASDR increased from ",
    asdr[1], " to ", asdr[2], " per 100,000. The corresponding BMI-attributable DALY share rose from ",
    daly_share[1], " to ", daly_share[2], ", and the death share rose from ",
    death_share[1], " to ", death_share[2], ". Joinpoint analysis identified global DALY ASR and ASDR inflection years in ",
    "1998, 2017, and 2020; the corresponding AAPCs were 1.88% (95% CI 1.71 to 2.04) for DALY ASR and 1.94% (95% CI 1.77 to 2.11) for ASDR. Under the ARIMA model, the global DALY ASR was projected to reach ",
    claim("global_daly_asr_projection_2050"), " in 2050, and ASDR was projected to reach ",
    claim("global_asdr_projection_2050"), "."
  ),
  "Conclusions: Liver cancer attributable to high BMI increased substantially from 1990 to 2023 and is projected to remain a growing burden under historical trends. These findings support integrating obesity prevention, metabolic-risk management, and risk-based liver-cancer surveillance within national cancer-control strategies.",
  "Keywords: liver cancer; high body-mass index; obesity; Global Burden of Disease; joinpoint; projection; Socio-demographic Index; MASLD"
)
replace_between("Abstract", "1. Introduction", abstract_nodes)

intro_nodes <- normal(
  "Liver cancer remains a major source of global premature mortality, and hepatocellular carcinoma accounts for most primary liver cancers. Although viral hepatitis, alcohol, and liver-disease progression remain central to the disease landscape, advances in HCC biology and clinical management have not eliminated the need for population-level prevention [1-5].",
  "Metabolic dysfunction is now an increasingly important contributor to chronic liver disease and liver carcinogenesis, particularly in the context of MASLD/MASH terminology and changing NASH-related liver-cancer epidemiology [6,7]. In parallel, global overweight and obesity prevalence has risen sharply across adult and pediatric populations, creating sustained upstream pressure on metabolic liver disease and downstream cancer risk [8,9].",
  "Several GBD-based analyses have described high-BMI-attributable disease burden, cancer burden, metabolic-risk-attributable cancer, and liver cancer attributable to high BMI or related metabolic risks, but most used GBD 2021 or earlier datasets or focused on selected regions and projections rather than a unified GBD 2023 country-level framework [10-18]. Evidence also remains limited on how country-level heterogeneity can be linked with reproducible location cleaning and manuscript-level claim validation.",
  "Using GBD 2023 estimates, we therefore combined descriptive epidemiology with two executable analytic layers in the current dataset: joinpoint regression for trend segmentation and ARIMA-based projection to 2050. Das Gupta decomposition was prespecified in the revision plan but is not reported here because the current project lacks age-specific Number data and matched population."
)
replace_between("1. Introduction", "2. Materials and methods", intro_nodes)

trend_nodes <- normal(
  "Long-term trends were summarized using estimated annual percentage change (EAPC). For each series, we fitted log-linear models of the form ln(rate_t) = alpha + beta x year_t and calculated EAPC = 100 x (exp(beta) - 1). Positive EAPC indicates increasing rates and negative EAPC indicates decreasing rates.",
  "To detect changes in trend slope over time, we fitted joinpoint regression using segmented log-linear models for annual age-standardized DALY ASR and ASDR from 1990 to 2023. For each available location-measure series, candidate models with 0 to 3 joinpoints were compared using Bayesian Information Criterion (BIC), and the model with the minimum BIC was selected. We reported joinpoint years, segment-specific annual percentage change, average annual percentage change (AAPC), and 95% confidence intervals."
)
replace_between("2.3 Trend analysis", "2.4 Spatial analysis, trajectory clustering, and country typologies", trend_nodes)

spatial_cluster_nodes <- normal(
  "Spatial analyses summarized country-level BMI-attributable liver-cancer DALY ASRs and BMI-attributable shares in 2023, with regional hotspot insets for the Middle East and North Africa, East Asia, Southern Europe, and Latin America and the Caribbean.",
  "Trajectory clustering used annual BMI-attributable DALY ASR profiles from 1990 to 2023. Country trajectories were standardized within country using row-wise z-scores to emphasize trajectory shape rather than absolute level, and grouped by k-means clustering. The number of clusters was evaluated using silhouette width, Calinski-Harabasz index, and Davies-Bouldin index, and k-means estimation used the Hartigan-Wong algorithm [23-26].",
  "A separate four-dimensional typology analysis characterized each country by 2023 BMI-attributable DALY ASR, 2023 BMI-attributable share of liver-cancer DALYs, EAPC_BMI, and 2023 SDI. These variables were standardized and clustered by k-means to identify policy-relevant phenotypes."
)
replace_between("2.4 Spatial analysis, trajectory clustering, and country typologies", "2.5 Composite priority index and inequality analysis", spatial_cluster_nodes)

priority_inequality_nodes <- normal(
  "To prioritize countries in which BMI-attributable liver cancer is both substantial and evolving unfavorably, we constructed a composite priority score from standardized 2023 BMI-attributable DALY ASR, standardized BMI-attributable share of liver-cancer DALYs, and standardized EAPC_BMI, with an inverse SDI component to emphasize settings with lower development resources.",
  "To assess weight sensitivity, the ranking was repeated under alternative SDI coefficients (0, -0.25, -0.75, and -1.0), and rank stability was summarized using Spearman correlations and top-20 overlap with the baseline model.",
  "Cross-country inequality in BMI-attributable liver-cancer burden was evaluated using Lorenz curves and the Gini coefficient [27,28]."
)
replace_between("2.5 Composite priority index and inequality analysis", "2.6 Statistical software", priority_inequality_nodes)

software_projection_nodes <- c(
  normal("All analyses were conducted in R 4.5.3 (R Foundation for Statistical Computing, Vienna, Austria). Data management used dplyr, tidyr, readr, and data.table; visualization used ggplot2; spatial outputs used sf and rnaturalearth; joinpoint modeling used segmented; and time-series forecasting used forecast. Scripted outputs included machine-readable tables, diagnostics, and figure-ready objects to reduce manual table drift."),
  list(heading2("2.7 Projection analysis")),
  normal("We projected BMI-attributable liver-cancer DALY ASR and ASDR from 2024 to 2050. ARIMA served as the primary method, and ETS plus log-linear extrapolation were prespecified as sensitivity analyses. Projections were generated for Global, available SDI strata, and selected high-priority countries. We reported annual point estimates with 95% prediction intervals and extracted milestones for 2030, 2040, and 2050.")
)
replace_between("2.6 Statistical software", "3. Results", software_projection_nodes)

results31_nodes <- normal(
  paste0(
    "Globally, BMI-attributable liver-cancer burden increased from 1990 to 2023. DALY ASR rose from ",
    daly_asr[1], " to ", daly_asr[2], " per 100,000, and ASDR rose from ",
    asdr[1], " to ", asdr[2], " per 100,000. The BMI-attributable DALY share increased from ",
    daly_share[1], " to ", daly_share[2], ", and the death share increased from ",
    death_share[1], " to ", death_share[2], " (Figure 1; Table 1)."
  ),
  paste0(
    "EAPC estimates confirmed sustained increases in the global rate series, with DALY ASR EAPC of ",
    claim("global_daly_asr_eapc_1990_2023"), " and ASDR EAPC of ",
    claim("global_asdr_eapc_1990_2023"), "."
  ),
  "Joinpoint analysis supported non-uniform temporal change. For DALY ASR, the BIC-selected model identified joinpoints in 1998, 2017, and 2020, with an AAPC of 1.88% (95% CI 1.71 to 2.04). For ASDR, the selected model identified joinpoints in 1998, 2017, and 2020, with an AAPC of 1.94% (95% CI 1.77 to 2.11).",
  paste0(
    "Available SDI-stratified source data included High-middle, Middle, Low-middle, and Low SDI groups but did not include High SDI rows; therefore complete five-stratum SDI interpretation was not made in this revision."
  )
)
replace_between("3.1 Global and SDI-level trends", "3.2 Spatial heterogeneity and country-level BMI-attributable share", results31_nodes)

projection_results_nodes <- c(
  list(heading2("3.7 Projection analysis")),
  normal(
    paste0(
      "Under the ARIMA primary model, BMI-attributable liver-cancer DALY ASR and ASDR were projected to increase through 2050 at the global level. The projected global DALY ASR reached ",
      claim("global_daly_asr_projection_2050"), " in 2050, representing a ",
      claim("global_daly_asr_projection_2023_2050_pct_change"), " increase from 2023. The projected global ASDR reached ",
      claim("global_asdr_projection_2050"), " in 2050, representing a ",
      claim("global_asdr_projection_2023_2050_pct_change"), " increase from 2023 (Figure 8; Table 5)."
    ),
    paste0(
      "Among available SDI strata, ", claim("available_sdi_highest_projected_daly_asr_2050"),
      " showed the highest projected 2050 DALY ASR under ARIMA, and ",
      claim("available_sdi_fastest_projected_daly_asr_growth"),
      " showed the steepest relative increase from 2023. Complete five-stratum SDI interpretation is not possible from the current source file because High SDI rows are absent."
    )
  )
)
insert_before("4. Discussion", projection_results_nodes)

discussion_nodes <- normal(
  "This study provides a comprehensive, policy-oriented analysis of liver cancer attributable to high BMI using GBD 2023 estimates, extending beyond standard burden tables through trajectory clustering, multidimensional typologies, joinpoint regression, projection analysis, and a composite priority framework.",
  "The joinpoint findings indicate that the global increase in BMI-attributable liver-cancer burden was not a constant linear process. The inflection points around 1998, 2017, and 2020 suggest periods of changing slope, with the most recent segment showing renewed acceleration in both DALY ASR and ASDR. These findings support reporting both long-term average change and segmented trend structure.",
  "The projection analyses suggest that, under historical trends, BMI-attributable liver-cancer burden is likely to remain substantial through 2050. Projection uncertainty widens over time, so these estimates should be interpreted as conditional forecasts rather than deterministic predictions. They can inform prioritization of obesity control, metabolic-risk management, MASLD-related care pathways, and surveillance resource allocation.",
  "The SDI pattern observed here likely reflects both epidemiologic transition and competing etiologies. Diet quality and obesity policy are directly relevant to metabolic-risk prevention, while population ageing and persistent viral-hepatitis burden can modify the observed country-level liver-cancer profile [29-32].",
  "The country typology and trajectory analyses help disentangle these patterns. Hotspot and rapidly rising phenotypes identify settings that may benefit from immediate integration of obesity prevention, MASLD case-finding, and risk-based liver-cancer surveillance into liver-cancer control programs. These prevention-oriented priorities complement, rather than replace, advances in HCC precision treatment and clinical management [33,34].",
  "From a policy perspective, the results argue against a liver-cancer control model that focuses only on viral hepatitis and treatment expansion. BMI reduction, diabetes control, MASLD/MASH management, and socioeconomic context need to be integrated explicitly into national cancer-control and chronic-liver-disease strategies [35].",
  "This study has several strengths, including the use of GBD 2023 estimates, reproducible location cleaning, country-level typology, formal trend segmentation, and projection to 2050. Limitations include the ecological nature of GBD estimates, the use of modeled inputs, the absence of complete High SDI rows in the current BMI-attributable SDI source file, and the inability to report Das Gupta decomposition without age-specific Number data and matched population.",
  "Taken together, the present findings suggest that obesity-related liver carcinogenesis is no longer a peripheral issue in global liver-cancer control. The BMI-attributable component is rising, increasingly concentrated, and projected to remain a growing burden under historical trends."
)
replace_between("4. Discussion", "5. Conclusions", discussion_nodes)

conclusion_nodes <- normal(
  "From 1990 to 2023, liver cancer attributable to high BMI increased globally, with rising age-standardized DALY and death rates and increasing BMI-attributable shares. Joinpoint and projection analyses indicate a non-uniform historical trajectory and continued burden increase through 2050 under historical trends. These findings support integrated liver-cancer control strategies that combine obesity prevention, metabolic-risk management, and context-specific surveillance prioritization."
)
replace_between("5. Conclusions", "Declarations", conclusion_nodes)

figure8_nodes <- normal(
  "Figure 8. ARIMA projections of liver cancer attributable to high BMI through 2050. The figure shows projected DALY ASR and ASDR trajectories for Global and available SDI strata, with 95% prediction intervals and sensitivity-model comparisons where available."
)
insert_before("Supplementary figure and table", figure8_nodes)

projection_table_path <- path_in_project("outputs", "tables", "Table5_projection_milestones.csv")
if (file.exists(projection_table_path)) {
  projection_table <- read.csv(projection_table_path, stringsAsFactors = FALSE, check.names = FALSE)
  keep_locations <- c("Global", "High-middle SDI", "Middle SDI", "Low-middle SDI", "Low SDI")
  projection_table <- projection_table[
    projection_table$method == "arima" &
      projection_table$location_name %in% keep_locations &
      projection_table$indicator %in% c("DALY_ASR", "ASDR"),
    ,
    drop = FALSE
  ]
  projection_table <- projection_table[order(match(projection_table$location_name, keep_locations), projection_table$indicator), ]
  projection_table_display <- data.frame(
    Location = projection_table$location_name,
    Indicator = projection_table$indicator,
    `2030` = sprintf("%.2f", projection_table$predicted_2030),
    `2040` = sprintf("%.2f", projection_table$predicted_2040),
    `2050` = sprintf("%.2f", projection_table$predicted_2050),
    `2050 95% PI` = paste0(sprintf("%.2f", projection_table$lower_95_2050), " to ", sprintf("%.2f", projection_table$upper_95_2050)),
    `Change 2023-2050` = paste0(sprintf("%.2f", projection_table$change_2023_2050_pct), "%"),
    check.names = FALSE
  )
  table5_nodes <- c(
    list(heading3("Table 5. ARIMA projection milestones for liver cancer attributable to high BMI, 2030-2050")),
    list(make_tbl(projection_table_display)),
    normal("Note: Values are age-standardized rates per 100,000. High SDI is not shown because High SDI rows were absent from the current BMI-attributable SDI source file.")
  )
  sect_pr <- xml_find_first(body, "./w:sectPr", ns)
  if (length(sect_pr) == 0) {
    for (node in table5_nodes) xml_add_child(body, node)
  } else {
    for (node in table5_nodes) xml_add_sibling(sect_pr, node, .where = "before")
  }
}

write_xml(doc, document_xml)

if (file.exists(out_docx)) {
  invisible(file.remove(out_docx))
}

old_wd <- getwd()
setwd(temp_dir)
on.exit(setwd(old_wd), add = TRUE)
zip::zipr(
  zipfile = out_docx,
  files = list.files(".", all.files = TRUE, no.. = TRUE, recursive = FALSE),
  recurse = TRUE,
  include_directories = FALSE,
  root = ".",
  mode = "mirror"
)
setwd(old_wd)

check_doc <- officer::read_docx(out_docx)
check_summary <- officer::docx_summary(check_doc)
check_text <- paste(check_summary$text, collapse = "\n")

required_phrases <- c(
  "joinpoints in 1998, 2017, and 2020",
  "projected global DALY ASR reached 21.53",
  "Das Gupta decomposition without age-specific Number data and matched population",
  "High SDI rows are absent"
)
missing_required <- required_phrases[
  !vapply(required_phrases, function(pattern) grepl(pattern, check_text, fixed = TRUE), logical(1))
]
if (length(missing_required) > 0) {
  stop(paste("Revised docx validation failed. Missing:", paste(missing_required, collapse = "; ")), call. = FALSE)
}

writeLines(c(
  "# Revised Manuscript DOCX Validation",
  "",
  "- Output: `manuscript/Liver_cancer_manuscript_revised_submission_checked_20260416.docx`",
  "- Source manuscript: `Liver cancer_manuscript.docx`",
  "- Number bank: `outputs/tables/manuscript_number_bank.csv`",
  "- Validation: PASS",
  "",
  "## Skipped blockers",
  "- Decomposition was not fabricated; the manuscript states that decomposition cannot be reported without age-specific Number data and matched population.",
  "- Complete five-stratum SDI interpretation was not claimed because High SDI rows are absent from the current source input."
), validation_log)

message("Revised manuscript written to: ", out_docx)
