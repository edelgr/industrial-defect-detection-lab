<#
.SYNOPSIS
  Download and cache pretrained backbones into checkpoints\ (run once while online).
  Each resource is < 1 GB: ResNet-18 (45 MB), Wide-ResNet-50-2 (263 MB), DINOv2 ViT-S/14 (84 MB),
  OpenCLIP ViT-B-16-plus-240 (795 MB, only with -WithWinCLIP).
#>
param([switch]$WithWinCLIP, [switch]$WithEfficientAD)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$a = @()
if ($WithWinCLIP) { $a += "--with-winclip" }
if ($WithEfficientAD) { $a += "--with-efficientad" }   # teacher weights 40 MB + Imagenette2-320 342 MB
& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") -m scripts.prepare_models @a
