<#
.SYNOPSIS
  Run the IDD Lab test suite (pytest + pytest-qt). Extra arguments are passed to pytest.
.EXAMPLE
  .\test.ps1            # all tests
  .\test.ps1 -k catalog # subset
#>
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") -m pytest -q @args
exit $LASTEXITCODE
