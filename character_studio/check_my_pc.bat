@echo off
REM Checks whether this PC can run the local image model. Installs nothing.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 check_pc.py
) else (
    python check_pc.py
)
if errorlevel 1 echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
pause
