@echo off
REM =============================================================================
REM LUPE Analysis App - Windows Launcher
REM =============================================================================
REM Double-click this file to launch LUPE!
REM =============================================================================

title LUPE Analysis App

echo ============================================
echo        LUPE Analysis App Launcher
echo ============================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Check if conda is available
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda not found in PATH!
    echo.
    echo Please ensure Anaconda or Miniconda is installed and added to PATH.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo During installation, check "Add to PATH" option.
    echo.
    pause
    exit /b 1
)

echo Activating LUPE2APP environment...
call conda activate LUPE2APP

if errorlevel 1 (
    echo.
    echo ERROR: Could not activate LUPE2APP environment!
    echo.
    echo Please create it first by running:
    echo   conda env create -f LUPE2_App.yaml
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   The app will open in your browser at:
echo   http://localhost:8501
echo ============================================
echo.
echo Keep this window open while using LUPE.
echo Press Ctrl+C or close this window to stop.
echo.

streamlit run lupe_analysis.py --server.headless=false

echo.
echo LUPE has been closed.
pause
