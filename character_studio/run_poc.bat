@echo off
REM Drag a character picture onto this file to use it as the reference,
REM or double-click it and type a description of a new character.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
if not "%~1"=="" goto reference
echo Describe your new character, e.g.
echo   a cheerful pig girl with long bright blue hair, mint green t-shirt, blue jeans, white trainers
echo.
set /p DESC=Description: 
.venv\Scripts\python poc.py --describe "%DESC%"
goto done

:reference
.venv\Scripts\python poc.py --reference "%~1"

:done
echo.
pause
