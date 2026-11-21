<#
.SYNOPSIS
  Dataset preparation (selective; never downloads > 1 GB without -ApproveLarge).
.EXAMPLE
  .\prepare_datasets.ps1                                   # status
  .\prepare_datasets.ps1 -Synthetic                        # SYNTHETIC demo data
  .\prepare_datasets.ps1 -MulSen capsule,nut,flat_pad      # show plan (dry run)
  .\prepare_datasets.ps1 -MulSen capsule,nut,flat_pad -Yes # fetch ≈ 0.65 GB via HTTP range
  .\prepare_datasets.ps1 -Import D05 -Archive C:\Downloads\sheet_metal.tar.gz -Categories sheet_metal
#>
param(
    [switch]$Synthetic,
    [string[]]$MulSen,
    [string]$Import,
    [string]$Archive,
    [string[]]$Categories,
    [switch]$Yes,
    [switch]$ApproveLarge
)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if ($Synthetic) { & $py -m scripts.prepare_datasets synthetic }
if ($MulSen) {
    $a = @("mulsen", "--categories") + $MulSen
    if ($Yes) { $a += "--yes" }
    if ($ApproveLarge) { $a += "--approve-large" }
    & $py -m scripts.prepare_datasets @a
}
if ($Import) {
    $a = @("import", "--dataset", $Import, "--archive", $Archive)
    if ($Categories) { $a += @("--categories") + $Categories }
    & $py -m scripts.prepare_datasets @a
}
& $py -m scripts.prepare_datasets status
