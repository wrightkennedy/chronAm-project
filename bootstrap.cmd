@echo off
setlocal enabledelayedexpansion

REM Bootstrap script for ChronAm on Windows. Creates a virtual environment next to the repo,
REM installs dependencies, and launches the GUI.

set SCRIPT_DIR=%~dp0
set ENV_DIR=%SCRIPT_DIR%chronam-env

if not exist "%ENV_DIR%" (
    echo Creating ChronAm virtual environment at %ENV_DIR%
    py -3 -m venv "%ENV_DIR%"
)

call "%ENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip
python -m pip install -r "%SCRIPT_DIR%requirements.txt"

python "%SCRIPT_DIR%app.py"

pause
