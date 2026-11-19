@echo off
setlocal
cd /d "%~dp0"

if exist "dist\CrashWatcher\CrashWatcher.exe" (
    start "" "dist\CrashWatcher\CrashWatcher.exe"
    exit /b 0
)

echo First run: building CrashWatcher.exe ... this takes a few minutes.
echo Requires Python 3.11+ installed with "Add python.exe to PATH".
if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found in PATH. Install it from python.org, then run START.bat again.
        pause
        exit /b 1
    )
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create .venv
        pause
        exit /b 1
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-windows.ps1"
if errorlevel 1 (
    echo.
    echo Build failed. See the messages above.
    pause
    exit /b 1
)

if not exist "dist\CrashWatcher\CrashWatcher.exe" (
    echo ERROR: the build finished but CrashWatcher.exe was not found.
    pause
    exit /b 1
)
start "" "dist\CrashWatcher\CrashWatcher.exe"
exit /b 0
