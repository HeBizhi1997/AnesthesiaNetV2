@echo off
cd /d "%~dp0"
echo Starting EEG Processing Service on http://localhost:8765 ...
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

:: Install dependencies if needed
python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

python main.py --host 0.0.0.0 --port 8765
pause
