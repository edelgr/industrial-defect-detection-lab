<#
.SYNOPSIS
  Launch IDD Lab.  Fallback:  .\.venv\Scripts\python.exe -m app.main
.PARAMETER Workshop
  Start with the Workshop offline demo profile for this session only: Methods Laboratory and
  Compare Methods prefer Workshop Verified checkpoints and ask before any training.
#>
param([switch]$Workshop)
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "Virtual environment not found. Run .\setup_student.ps1 first (instructor installation: .\setup.ps1)."; exit 1 }
$env:PYTHONIOENCODING = "utf-8"
if ($Workshop) { $env:IDD_WORKSHOP = "1" } else { Remove-Item Env:IDD_WORKSHOP -ErrorAction SilentlyContinue }
& $py -m app.main @args
