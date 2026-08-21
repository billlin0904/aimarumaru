@echo off
setlocal

cd /d "%~dp0"

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo [PO Token Provider] ERROR: powershell.exe was not found.
    exit /b 1
)

echo [PO Token Provider] Starting local token server...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0start_pot_provider.ps1" %*

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo [PO Token Provider] Startup failed with exit code %EXIT_CODE%.
    echo [PO Token Provider] Make sure Docker Desktop is running.
    exit /b %EXIT_CODE%
)

echo [PO Token Provider] Startup check completed.
exit /b 0
