@echo off
REM One-time setup: creates a private Python environment and installs everything.
cd /d "%~dp0"
echo Setting up Puppet Animator...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
) else (
    python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo.
    echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation, then run setup.bat again.
    pause
    exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installing packages failed - please send the messages above to Claude.
    pause
    exit /b 1
)
echo Downloading the AI pose-detection model...
.venv\Scripts\python -c "from animator.core.skeleton import ensure_model; print('Model:', ensure_model())"
echo.
echo Setup finished. Double-click run.bat to start.
pause
