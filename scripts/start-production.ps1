param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv .venv
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

if ($Install) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r backend\requirements.txt
    & $venvPython -m pip install -r backend\requirements-ml.txt
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & $venvPython -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128
    }
    Push-Location frontend
    pnpm.cmd install
    Pop-Location
}

$env:PYTHONPATH = "backend"
& $venvPython backend\scripts\check_models.py
if ($LASTEXITCODE -ne 0) {
    throw "Production preflight failed. Fix the missing dependency above and run again."
}

New-Item -ItemType Directory -Force storage\logs | Out-Null

Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -in @("python.exe", "pythonw.exe", "node.exe")) -and
        (
            $_.CommandLine -like "*uvicorn*app.main:app*" -or
            $_.CommandLine -like "*-m app.worker*" -or
            $_.CommandLine -like "*vite*--host 127.0.0.1*"
        )
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$apiCommand = "`$env:PYTHONPATH='backend'; & '$venvPython' -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 *>> storage\logs\api.log"
$workerCommand = "`$env:PYTHONPATH='backend'; & '$venvPython' -m app.worker *>> storage\logs\worker.log"
$frontendCommand = "pnpm.cmd dev --host 127.0.0.1 --port 5173 *>> ..\storage\logs\frontend.log"

Start-Process -FilePath powershell.exe -WorkingDirectory $root -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $apiCommand)
Start-Process -FilePath powershell.exe -WorkingDirectory $root -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $workerCommand)
Start-Process -FilePath powershell.exe -WorkingDirectory (Join-Path $root "frontend") -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand)

Start-Sleep -Seconds 3
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 10
Write-Host "DubLok production stack is live at http://127.0.0.1:5173/"
