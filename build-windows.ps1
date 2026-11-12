$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run python -m venv .venv first.' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $python -m pip install 'pyinstaller>=6.10,<7'
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller installation failed.' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; build stopped.' }
& $python -m PyInstaller --noconfirm --clean --windowed --onedir --name CrashWatcher --icon 'assets\crash_watcher.ico' --collect-all rapidocr --collect-all onnxruntime --paths . desktop_app.py
if ($LASTEXITCODE -ne 0) { throw 'Windows build failed.' }
Copy-Item -LiteralPath (Join-Path $root 'settings.json') -Destination (Join-Path $root 'dist\CrashWatcher\settings.json') -Force
Write-Host "Build: $root\dist\CrashWatcher\CrashWatcher.exe"
