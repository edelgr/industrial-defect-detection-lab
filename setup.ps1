<#
.SYNOPSIS
  Create the IDD Lab virtual environment and install dependencies (no admin rights needed).
.PARAMETER Python
  Python interpreter to use (3.11-3.13). Default: the 'py -3.12' launcher, then 'python'.
.PARAMETER Cuda
  PyTorch wheel index: cu130 (default, NVIDIA driver >= 580), cu128, or cpu.
.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -Cuda cpu
#>
param(
    [string]$Python = "",
    [ValidateSet("cu130", "cu128", "cpu")][string]$Cuda = "cu130",
    [switch]$No3D
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Python) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $Python = (& py -3.12 -c "import sys;print(sys.executable)") }
    else { $Python = (Get-Command python).Source }
}
Write-Host "Using Python: $Python"
& $Python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), sys.version; print(sys.version)"

if (-not (Test-Path .venv)) { & $Python -m venv .venv }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "Installing PyTorch ($Cuda)…"
& $py -m pip install "torch==2.14.1" "torchvision==0.29.1" --index-url "https://download.pytorch.org/whl/$Cuda"

$extras = if ($No3D) { "deep,dev" } else { "deep,3d,dev" }
& $py -m pip install -e ".[$extras]"

& $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
& $py -m scripts.prepare_datasets synthetic
Write-Host "`nSetup complete. Next steps:" -ForegroundColor Green
Write-Host "  .\prepare_models.ps1     # cache pretrained backbones (online, ~1.2 GB in total)"
Write-Host "  .\prepare_datasets.ps1   # dataset status / import / MulSen-AD subset"
Write-Host "  .\start.ps1              # launch IDD Lab"
