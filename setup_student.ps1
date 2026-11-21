<#
.SYNOPSIS
  IDD Lab — Student Edition setup (Windows 10/11, no administrator rights needed).

.DESCRIPTION
  1. Detects Python (3.11-3.13)        6. Creates a private virtual environment (.venv)
  2. Verifies the version               7. Installs pinned dependencies (requirements/*.txt)
  3. Checks disk space and path length  8. Validates the essential packages
  4. Detects an NVIDIA GPU and driver   9. Prepares local folders + Synthetic Educational Data
  5. Explains CPU vs GPU and chooses   10. Runs the offline launch-readiness check
     the PyTorch build (full profile)

  Profiles
    core  (default, ~0.8 GB) : Atlas, Methods Library, Start Learning, Dataset Explorer, classical
                               methods, Educational Experiments (stored results), Guided Workshop.
                               No PyTorch. Works on any recent Windows laptop.
    full  (~3-6 GB)          : + PaDiM, PatchCore, EfficientAD, AnomalyDINO, WinCLIP (PyTorch, Anomalib)
                               and 3D (Open3D, PyVista). CPU build by default; CUDA build only if a
                               compatible NVIDIA driver is detected (or requested with -Torch).

  Nothing in an existing installation is deleted. Pretrained weights and datasets are NOT downloaded
  (use the Model Manager / prepare_models.ps1 and DATASETS.md later, if you want them).

.EXAMPLE
  .\setup_student.ps1                      # core profile
  .\setup_student.ps1 -Profile full        # deep learning + 3D, CPU or CUDA chosen automatically
  .\setup_student.ps1 -Profile full -Torch cpu
#>
param(
    [ValidateSet("core", "full")][string]$Profile = "core",
    [ValidateSet("auto", "cpu", "cu130", "cu128")][string]$Torch = "auto",
    [string]$Python = "",
    [string]$VenvDir = ".venv",
    [switch]$SkipReadiness
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Step($n, $msg) { Write-Host ""; Write-Host "[$n/10] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---- 1-2. Python ---------------------------------------------------------------------------------
Step 1 "Detecting Python"
$candidates = @()
if ($Python) { $candidates += ,@($Python) }
if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += ,@("py", "-3.12"); $candidates += ,@("py", "-3.11"); $candidates += ,@("py", "-3.13") }
if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += ,@((Get-Command python).Source) }
$pyExe = $null
foreach ($c in $candidates) {
    try {
        $exe = if ($c.Count -gt 1) { & $c[0] $c[1] -c "import sys;print(sys.executable)" 2>$null } else { & $c[0] -c "import sys;print(sys.executable)" 2>$null }
        if ($LASTEXITCODE -eq 0 -and $exe) { $pyExe = $exe.Trim(); break }
    } catch { }
}
if (-not $pyExe) { Fail "Python 3.11-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH') and run this script again." }
Write-Host "  Python: $pyExe"

Step 2 "Verifying the Python version"
& $pyExe -c "import sys; v=sys.version_info[:2]; print('  version', sys.version.split()[0]); sys.exit(0 if (3,11) <= v <= (3,13) else 1)"
if ($LASTEXITCODE -ne 0) { Fail "IDD Lab needs Python 3.11, 3.12 or 3.13 (3.12 recommended)." }

# ---- 3. Disk space and path length ----------------------------------------------------------------
Step 3 "Checking disk space and path length"
$needGB = if ($Profile -eq "core") { 2 } else { 8 }
$drive = (Get-Item $PSScriptRoot).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "  Free on drive $($drive.Name): $freeGB GB (this profile needs about $needGB GB, plus a 5 GB safety margin)"
if ($freeGB -lt ($needGB + 5)) { Fail "Not enough free disk space on drive $($drive.Name): ($freeGB GB free)." }
$site = Join-Path $PSScriptRoot "$VenvDir\Lib\site-packages\"
$longPaths = $false
try { $longPaths = ((Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled -eq 1) } catch { }
# PySide6 contains files ~140 characters below site-packages; Windows limits paths to 260 characters
# unless Long Path support is enabled.
if (-not $longPaths -and ($site.Length + 140) -ge 260) {
    Fail ("The installation folder path is too long for Windows without Long Path support:`n  $PSScriptRoot`n" +
          "Move the IDD_Lab folder to a short path such as C:\IDD_Lab (no administrator rights needed) and run this " +
          "script again, or ask an administrator to enable Windows Long Path support.")
}
Write-Host "  Path length OK ($($site.Length + 140) of 260 characters; Long Path support: $longPaths)"

# ---- 4-5. GPU and PyTorch build -------------------------------------------------------------------
Step 4 "Detecting GPU and CUDA"
$gpu = $null; $driverMajor = 0
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        $line = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($line) { $gpu = $line.Trim(); $driverMajor = [int](($line -split ",")[1].Trim().Split(".")[0]) }
    } catch { }
}
if ($gpu) { Write-Host "  NVIDIA GPU: $gpu" } else { Write-Host "  No NVIDIA GPU detected (or nvidia-smi not available) - CPU mode." }

Step 5 "Choosing CPU or GPU configuration"
$index = $null
if ($Profile -eq "core") {
    Write-Host "  Core profile: no PyTorch. Classical methods run on the CPU; deep-learning results are shown as stored results."
} else {
    if ($Torch -eq "auto") {
        # The CUDA 13.0 PyTorch wheels need a recent NVIDIA driver (580 or newer). Never install a CUDA build
        # just because an NVIDIA GPU is present.
        if ($gpu -and $driverMajor -ge 580) { $Torch = "cu130" } else { $Torch = "cpu" }
    }
    $index = "https://download.pytorch.org/whl/$Torch"
    if ($Torch -eq "cpu") {
        Write-Host "  PyTorch CPU build. PaDiM/PatchCore/AnomalyDINO fitting and inference work but are slower;"
        Write-Host "  EfficientAD's 10 000-step training is impractical on a CPU (use stored results instead)."
        if ($gpu) { Write-Host "  (Your NVIDIA driver $driverMajor is older than 580; update it and re-run with -Torch cu130 for GPU use.)" }
    } else {
        Write-Host "  PyTorch CUDA build ($Torch) for: $gpu"
    }
}

# ---- 6. Virtual environment -----------------------------------------------------------------------
Step 6 "Creating the virtual environment ($VenvDir)"
$py = Join-Path $PSScriptRoot "$VenvDir\Scripts\python.exe"
if (Test-Path $py) { Write-Host "  Reusing existing environment $VenvDir (nothing is deleted)." }
else { & $pyExe -m venv $VenvDir; if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." } }

# ---- 7. Dependencies ------------------------------------------------------------------------------
Step 7 "Installing pinned dependencies (this can take several minutes)"
if ($Profile -eq "core") {
    & $py -m pip install --disable-pip-version-check -r requirements\student-core-lock.txt
} else {
    & $py -m pip install --disable-pip-version-check "torch==2.14.1" "torchvision==0.29.1" --index-url $index
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed (index $index)." }
    & $py -m pip install --disable-pip-version-check -r requirements\student-full-lock.txt
}
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed - see the messages above and TROUBLESHOOTING.md." }

# ---- 8. Validation --------------------------------------------------------------------------------
Step 8 "Validating essential packages"
$mods = "PySide6.QtWidgets, PySide6.QtSvgWidgets, numpy, cv2, skimage, sklearn, matplotlib, yaml, jsonschema"
if ($Profile -eq "full") { $mods += ", torch, torchvision, anomalib, timm, open3d, pyvistaqt" }
& $py -c "import $mods; print('  essential packages import correctly')"
if ($LASTEXITCODE -ne 0) { Fail "An essential package cannot be imported." }
if ($Profile -eq "full") { & $py -c "import torch; print('  torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" }

# ---- 9. Local folders and synthetic data ----------------------------------------------------------
Step 9 "Preparing local folders and Synthetic Educational Data"
& $py -c "from core import paths, settings; paths.ensure_runtime_dirs(); s = settings.load(); import sys; (None if paths.USER_SETTINGS.exists() else (setattr(s, 'profile', 'student'), settings.save(s))); print('  folders ready; profile:', s.profile)"
& $py -m scripts.prepare_datasets synthetic
if ($LASTEXITCODE -ne 0) { Fail "Could not generate the synthetic data." }

# ---- 10. Readiness --------------------------------------------------------------------------------
Step 10 "Launch-readiness check (offline)"
if (-not $SkipReadiness) {
    & $py -m scripts.readiness --profile student
    if ($LASTEXITCODE -ne 0) { Fail "Readiness check failed - see the messages above." }
}
Write-Host ""
Write-Host "IDD Lab Student Edition is ready." -ForegroundColor Green
Write-Host "  Start:            .\start.ps1"
Write-Host "  First steps:      STUDENT_QUICK_START.md"
Write-Host "  Optional data:    DATASETS.md (official sources; import with .\prepare_datasets.ps1)"
Write-Host "  Optional models:  Model Manager page, or .\prepare_models.ps1 (full profile, online)"
