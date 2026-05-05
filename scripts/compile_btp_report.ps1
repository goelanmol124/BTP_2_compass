# Build all COMPASS report PDFs: BTP report, JCTC manuscript, and Supporting Information.
# Prefers Tectonic in tools/tectonic if pdflatex is not on PATH.
# Run from anywhere:  pwsh -File scripts/compile_btp_report.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$paper = Join-Path $root "paper"
Set-Location $root

function Find-Tectonic {
    $local = Join-Path $root "tools\tectonic\tectonic.exe"
    if (Test-Path $local) { return $local }
    $cmd = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Find-PdfLatex {
    $cmd = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "$env:ProgramFiles\MiKTeX\miktex\bin\x64\pdflatex.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

& python (Join-Path $root "scripts\gen_si_sweep_table.py")

Set-Location $paper
$tectonic = Find-Tectonic
$pdflatex = Find-PdfLatex

if ($tectonic) {
    Write-Host "Using Tectonic: $tectonic" -ForegroundColor Cyan
    Write-Host "Building BTP report..." -ForegroundColor Cyan
    & $tectonic --reruns 8 "COMPASS_BTP_REPORT.tex"
    if ($LASTEXITCODE -ne 0) { throw "tectonic failed for BTP report" }
    Write-Host "Building JCTC manuscript..." -ForegroundColor Cyan
    & $tectonic --reruns 6 "COMPASS_JCTC_MANUSCRIPT.tex"
    if ($LASTEXITCODE -ne 0) { throw "tectonic failed for main manuscript" }
    & $tectonic --reruns 8 "COMPASS_JCTC_SUPPORTING_INFORMATION.tex"
    if ($LASTEXITCODE -ne 0) { throw "tectonic failed for SI" }
}
elseif ($pdflatex) {
    Write-Host "Using pdflatex: $pdflatex" -ForegroundColor Cyan
    function Build-TeX($name) {
        & $pdflatex -interaction=nonstopmode -halt-on-error "$name.tex"
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed for $name" }
    }
    Write-Host "Building BTP report..." -ForegroundColor Cyan
    Build-TeX "COMPASS_BTP_REPORT"
    & bibtex COMPASS_BTP_REPORT
    Build-TeX "COMPASS_BTP_REPORT"
    Build-TeX "COMPASS_BTP_REPORT"
    Write-Host "Building JCTC manuscript..." -ForegroundColor Cyan
    Build-TeX "COMPASS_JCTC_MANUSCRIPT"
    & bibtex COMPASS_JCTC_MANUSCRIPT
    Build-TeX "COMPASS_JCTC_MANUSCRIPT"
    Build-TeX "COMPASS_JCTC_MANUSCRIPT"
    Build-TeX "COMPASS_JCTC_SUPPORTING_INFORMATION"
    & bibtex COMPASS_JCTC_SUPPORTING_INFORMATION
    Build-TeX "COMPASS_JCTC_SUPPORTING_INFORMATION"
    Build-TeX "COMPASS_JCTC_SUPPORTING_INFORMATION"
}
else {
    Write-Error "No TeX engine found. Either install MiKTeX/TeX Live (pdflatex + bibtex on PATH) or extract Tectonic to tools/tectonic/tectonic.exe (see script header)."
}

Write-Host "Done. Outputs:" -ForegroundColor Green
Get-Item (Join-Path $paper "COMPASS_BTP_REPORT.pdf"), (Join-Path $paper "COMPASS_JCTC_MANUSCRIPT.pdf"), (Join-Path $paper "COMPASS_JCTC_SUPPORTING_INFORMATION.pdf") -ErrorAction SilentlyContinue | Format-Table Name, Length, LastWriteTime
