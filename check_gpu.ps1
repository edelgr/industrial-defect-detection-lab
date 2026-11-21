<#
.SYNOPSIS
  Report GPU / CUDA / VRAM / RAM / disk and library versions as seen by IDD Lab.
#>
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
    $drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1)
    $major = [int]($drv.Trim().Split(".")[0])
    if ($major -ge 580) { Write-Host "Recommended PyTorch build: cu130 (driver $drv >= 580)" }
    else { Write-Host "Recommended PyTorch build: cpu (driver $drv < 580; update the NVIDIA driver to use the CUDA 13.0 build)" }
} else { Write-Host "No NVIDIA GPU / nvidia-smi: use the CPU build (setup_student.ps1 -Profile full -Torch cpu) or the core profile." }
& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") -c @"
import json
from core import settings
from core.system.diagnostics import full_report
print(json.dumps(full_report(settings.load().data_root_path), indent=2))
"@
