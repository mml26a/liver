calc_eapc <- function(df, year_col = "year", value_col = "val") {
  y <- as.numeric(df[[value_col]])
  x <- as.numeric(df[[year_col]])
  keep <- !is.na(x) & !is.na(y) & y > 0
  if (sum(keep) < 2) {
    return(data.frame(EAPC = NA_real_, CI_low = NA_real_, CI_high = NA_real_))
  }
  fit <- stats::lm(log(y[keep]) ~ x[keep])
  beta <- stats::coef(fit)[2]
  se <- summary(fit)$coefficients[2, "Std. Error"]
  data.frame(
    EAPC = 100 * (exp(beta) - 1),
    CI_low = 100 * (exp(beta - 1.96 * se) - 1),
    CI_high = 100 * (exp(beta + 1.96 * se) - 1)
  )
}

clean_location_name <- function(x) {
  x <- tolower(as.character(x))
  x <- gsub("&", "and", x, fixed = TRUE)
  x <- gsub("[[:punct:]]", " ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

normalize_name_key <- function(x) {
  x <- clean_location_name(iconv(as.character(x), from = "", to = "ASCII//TRANSLIT"))
  x <- ifelse(x %in% c("turkey", "turkiye"), "turkiye", x)
  x <- ifelse(x %in% c("c te d ivoire", "cote d ivoire", "ivory coast"), "cote d ivoire", x)
  x
}

zscore <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }
  s <- stats::sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) {
    return(rep(0, length(x)))
  }
  (x - mean(x, na.rm = TRUE)) / s
}

lorenz_points <- function(x) {
  x <- sort(as.numeric(x[!is.na(x)]))
  if (length(x) == 0 || sum(x) == 0) {
    return(data.frame(p = c(0, 1), L = c(0, 1)))
  }
  n <- length(x)
  p <- c(0, seq_len(n) / n)
  L <- c(0, cumsum(x) / sum(x))
  data.frame(p = p, L = L)
}
