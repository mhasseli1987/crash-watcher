@echo off
setlocal
cd /d "%~dp0"
if not exist "dist\CrashWatcherNew\CrashWatcherNew.exe" (
    echo ERROR: Build not found. Run build-windows.ps1 first.
    pause
    exit /b 1
)
echo Starting Crash Watcher...
start "" "dist\CrashWatcherNew\CrashWatcherNew.exe"
exit /b 0