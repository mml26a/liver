param(
  [switch]$SkipDecomposition
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:R_LIBS_USER = Join-Path $projectRoot ".r_libs\R-4.5"
New-Item -ItemType Directory -Force -Path $env:R_LIBS_USER | Out-Null
$udunitsXml = Join-Path $env:R_LIBS_USER "units\share\udunits\udunits2.xml"
if (Test-Path -LiteralPath $udunitsXml) {
  $udunitsAsciiDir = Join-Path $env:TEMP "codex_udunits"
  New-Item -ItemType Directory -Force -Path $udunitsAsciiDir | Out-Null
  Copy-Item -Path (Join-Path (Split-Path -Parent $udunitsXml) "*") -Destination $udunitsAsciiDir -Force
  $env:UDUNITS2_XML_PATH = Join-Path $udunitsAsciiDir "udunits2.xml"
}

$requiredInputs = @(
  "gbd2023_BMI_HCC_global_alllevels.csv",
  "gbd2023_BMI_HCC_SDI_1990_2023.csv",
  "gbd2023_BMI_HCC_country_1990_2023.csv",
  "gbd2023_allHCC_global_1990_2023.csv",
  "gbd2023_allHCC_country_1990_2023.csv",
  "gbd2023_SDI_values_1950_2023.csv"
)

$dataRaw = Join-Path $projectRoot "data_raw"
New-Item -ItemType Directory -Force -Path $dataRaw | Out-Null

$missing = @()
foreach ($inputFile in $requiredInputs) {
  if (-not (Test-Path -LiteralPath (Join-Path $dataRaw $inputFile))) {
    $missing += $inputFile
  }
}

if ($missing.Count -gt 0) {
  $sourceDir = Join-Path (Split-Path -Parent $projectRoot) "epidemic"
  foreach ($inputFile in $missing) {
    $sourcePath = Join-Path $sourceDir $inputFile
    if (-not (Test-Path -LiteralPath $sourcePath)) {
      throw "Missing required input: $inputFile"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $dataRaw $inputFile) -Force
  }
}

$rscript = Get-Command Rscript -ErrorAction SilentlyContinue
if (-not $rscript) {
  $rscriptCandidates = @(
    "C:\Program Files\R\R-4.5.3\bin\Rscript.exe",
    "C:\Program Files\R\R-4.5.2\bin\Rscript.exe",
    "C:\Program Files\R\R-4.5.1\bin\Rscript.exe",
    "C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
    "C:\Program Files\R\R-4.4.3\bin\Rscript.exe",
    "C:\Program Files\R\R-4.4.2\bin\Rscript.exe",
    "C:\Program Files\R\R-4.4.1\bin\Rscript.exe",
    "C:\Program Files\R\R-4.4.0\bin\Rscript.exe"
  )
  foreach ($candidate in $rscriptCandidates) {
    if (Test-Path -LiteralPath $candidate) {
      $rscript = Get-Item -LiteralPath $candidate
      break
    }
  }
}
if (-not $rscript) {
  throw "Rscript is not available in this shell. Run the project R setup first, then rerun this script."
}
$rscriptExe = if ($rscript.Source) { $rscript.Source } else { $rscript.FullName }

$stages = @(
  "scripts/check_environment.R",
  "analysis/01_data_audit.R",
  "analysis/02_master_dataset.R",
  "analysis/03_refresh_existing_analyses.R",
  "analysis/05_joinpoint.R",
  "analysis/06_projection.R"
)

if (-not $SkipDecomposition) {
  $populationCandidates = @(
    "gbd2023_population_1990_2050.csv",
    "gbd2023_population_1990_2023.csv",
    "gbd_population_age_specific.csv"
  )
  $hasPopulation = $false
  foreach ($populationFile in $populationCandidates) {
    if (Test-Path -LiteralPath (Join-Path $dataRaw $populationFile)) {
      $hasPopulation = $true
      break
    }
  }
  if ($hasPopulation) {
    $stages += "analysis/04_decomposition.R"
  } else {
    Write-Warning "Population CSV not found. Skipping decomposition; rerun without -SkipDecomposition after adding population data."
  }
}

$stages += @(
  "analysis/07_tables_figures.R",
  "analysis/08_extract_manuscript_numbers.R",
  "scripts/run_pipeline.R"
)

foreach ($stage in $stages) {
  Write-Host "Running $stage"
  & $rscriptExe --vanilla $stage
  if ($LASTEXITCODE -ne 0) {
    throw "Stage failed: $stage"
  }
}

Write-Host "Full revision pipeline completed."
