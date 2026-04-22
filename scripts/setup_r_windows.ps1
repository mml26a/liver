Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "setup_r_windows.log"

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "[INFO] setup_r_windows.ps1 started at $(Get-Date -Format o)"
    Write-Host "[INFO] project root: $ProjectRoot"

    $rscript = Get-Command Rscript -ErrorAction SilentlyContinue
    if (-not $rscript) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-Host "[INFO] Installing R via winget..."
            winget install --id RProject.R --source winget --silent --accept-package-agreements --accept-source-agreements
        }
        else {
            throw "winget not found. Please install R manually and rerun this script."
        }
    }

    $candidatePaths = @(
        "C:\Program Files\R\R-4.4.1\bin",
        "C:\Program Files\R\R-4.4.0\bin",
        "C:\Program Files\R\R-4.3.3\bin",
        "C:\Program Files\R\R-4.3.2\bin",
        "C:\Program Files\R\R-4.3.1\bin",
        "C:\Program Files\R\R-4.3.0\bin",
        "C:\Program Files\R\R-4.2.3\bin",
        "C:\Program Files\R\R-4.2.2\bin"
    )
    foreach ($p in $candidatePaths) {
        if ((Test-Path $p) -and ($env:Path -notlike "*$p*")) {
            $env:Path = "$p;$env:Path"
        }
    }

    $rscript = Get-Command Rscript -ErrorAction SilentlyContinue
    if (-not $rscript) {
        throw "Rscript still not found after attempting installation."
    }

    & Rscript (Join-Path $ProjectRoot "scripts\install_r_packages.R")
    & Rscript (Join-Path $ProjectRoot "scripts\check_environment.R")

    Write-Host "[INFO] setup_r_windows.ps1 completed at $(Get-Date -Format o)"
}
finally {
    Stop-Transcript | Out-Null
}
