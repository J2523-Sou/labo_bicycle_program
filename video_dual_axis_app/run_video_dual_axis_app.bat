@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_PATH=%SCRIPT_DIR%video_dual_axis_app.py"
set "BUNDLED_PYTHON=C:\Users\sou90\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%BUNDLED_PYTHON%" (
    "%BUNDLED_PYTHON%" "%SCRIPT_PATH%"
    goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%SCRIPT_PATH%"
    goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
    py "%SCRIPT_PATH%"
    goto :eof
)

echo Python runtime was not found.
echo Please run video_dual_axis_app.py with a valid Python environment.
pause
