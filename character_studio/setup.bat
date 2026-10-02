@echo off
REM One-time setup for Character Studio (separate from the animation programme).
cd /d "%~dp0"
echo Setting up Character Studio - this downloads about 20 GB and can take a while.
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
echo.
echo Installing PyTorch for NVIDIA graphics cards (about 3 GB)...
.venv\Scripts\python install_torch.py
if errorlevel 1 goto failed
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
.venv\Scripts\python -c "import torch; ok = torch.cuda.is_available(); print('NVIDIA graphics card usable:', ok, '-', torch.cuda.get_device_name(0) if ok else 'NOT FOUND - update the NVIDIA driver')"
echo.
echo Downloading the image model FLUX.2 [klein] 4B (about 16 GB) into the models folder...
.venv\Scripts\python -c "import studio.engine as e; from huggingface_hub import snapshot_download; print(snapshot_download(e.MODEL_ID))"
if errorlevel 1 goto failed
echo.
echo Setup finished. Double-click run_poc.bat to try it.
pause
exit /b 0

:failed
echo.
echo Setup failed - please send the messages above to Claude.
pause
exit /b 1
