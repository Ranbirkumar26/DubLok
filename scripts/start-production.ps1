param(
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-Pnpm {
    foreach ($name in @("pnpm.cmd", "pnpm")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            return @{
                Path = $command.Source
                PrefixArgs = @()
            }
        }
    }

    foreach ($name in @("corepack.cmd", "corepack")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) {
            return @{
                Path = $command.Source
                PrefixArgs = @("pnpm")
            }
        }
    }

    throw "pnpm was not found. Install pnpm globally (`npm install -g pnpm`) or install Node.js 20+ with Corepack enabled."
}

function Clear-LogFile($path) {
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            Set-Content -Path $path -Value ""
            return
        } catch {
            if ($attempt -eq 10) {
                throw
            }
            Start-Sleep -Milliseconds 300
        }
    }
}

$pnpm = Resolve-Pnpm

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv .venv
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

Get-Content ".env" |
    Where-Object { $_ -match "^\s*[^#][^=]*=" } |
    ForEach-Object {
        $key, $value = $_ -split "=", 2
        [Environment]::SetEnvironmentVariable(
            $key.Trim(),
            $value.Trim().Trim('"').Trim("'"),
            "Process"
        )
    }

if ($Install) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r backend\requirements.txt
    & $venvPython -m pip install -r backend\requirements-ml.txt
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & $venvPython -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128
    }
    Push-Location frontend
    & $pnpm.Path @($pnpm.PrefixArgs + @("install"))
    Pop-Location
}

$env:PYTHONPATH = "backend"
& $venvPython backend\scripts\check_models.py
if ($LASTEXITCODE -ne 0) {
    throw "Production preflight failed. Fix the missing dependency above and run again."
}

New-Item -ItemType Directory -Force storage\logs | Out-Null
if ($env:HF_HOME) {
    New-Item -ItemType Directory -Force $env:HF_HOME | Out-Null
}

Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -in @("python.exe", "pythonw.exe", "node.exe", "powershell.exe", "pwsh.exe", "cmd.exe")) -and
        (
            $_.CommandLine -like "*uvicorn*app.main:app*" -or
            $_.CommandLine -like "*-m app.worker*" -or
            $_.CommandLine -like "*vite*--host 127.0.0.1*" -or
            $_.CommandLine -like "*storage\logs\api.log*" -or
            $_.CommandLine -like "*storage\logs\worker.log*" -or
            $_.CommandLine -like "*storage\logs\frontend.log*"
        )
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Milliseconds 500
foreach ($logName in @("api.log", "worker.log", "frontend.log")) {
    Clear-LogFile (Join-Path $root "storage\logs\$logName")
}

$apiCommand = "set PYTHONPATH=backend&& `"$venvPython`" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 >> storage\logs\api.log 2>&1"
$workerCommand = "set PYTHONPATH=backend&& `"$venvPython`" -m app.worker >> storage\logs\worker.log 2>&1"
$frontendArgs = @($pnpm.PrefixArgs + @("dev", "--host", "127.0.0.1", "--port", "5173")) -join " "
$frontendCommand = "`"$($pnpm.Path)`" $frontendArgs >> ..\storage\logs\frontend.log 2>&1"

Start-Process -FilePath cmd.exe -WorkingDirectory $root -WindowStyle Hidden -ArgumentList @("/d", "/s", "/c", $apiCommand)
Start-Process -FilePath cmd.exe -WorkingDirectory $root -WindowStyle Hidden -ArgumentList @("/d", "/s", "/c", $workerCommand)
Start-Process -FilePath cmd.exe -WorkingDirectory (Join-Path $root "frontend") -WindowStyle Hidden -ArgumentList @("/d", "/s", "/c", $frontendCommand)

Start-Sleep -Seconds 3
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 10
Write-Host "DubLok production stack is live at http://127.0.0.1:5173/"
