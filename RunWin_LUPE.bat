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

REM Try to find conda in common locations
set CONDA_PATH=

REM Check if conda is in PATH first
where conda >nul 2>nul
if not errorlevel 1 (
    echo Found conda in PATH
    goto :activate
)

REM Check common Anaconda/Miniconda locations
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set CONDA_PATH=%USERPROFILE%\anaconda3
    goto :found_conda
)
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set CONDA_PATH=%USERPROFILE%\miniconda3
    goto :found_conda
)
if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" (
    set CONDA_PATH=%USERPROFILE%\Anaconda3
    goto :found_conda
)
if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" (
    set CONDA_PATH=%USERPROFILE%\Miniconda3
    goto :found_conda
)
if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" (
    set CONDA_PATH=%LOCALAPPDATA%\anaconda3
    goto :found_conda
)
if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    set CONDA_PATH=%LOCALAPPDATA%\miniconda3
    goto :found_conda
)
if exist "%LOCALAPPDATA%\Continuum\anaconda3\Scripts\activate.bat" (
    set CONDA_PATH=%LOCALAPPDATA%\Continuum\anaconda3
    goto :found_conda
)
if exist "%LOCALAPPDATA%\Continuum\miniconda3\Scripts\activate.bat" (
    set CONDA_PATH=%LOCALAPPDATA%\Continuum\miniconda3
    goto :found_conda
)
if exist "C:\anaconda3\Scripts\activate.bat" (
    set CONDA_PATH=C:\anaconda3
    goto :found_conda
)
if exist "C:\miniconda3\Scripts\activate.bat" (
    set CONDA_PATH=C:\miniconda3
    goto :found_conda
)
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set CONDA_PATH=C:\ProgramData\anaconda3
    goto :found_conda
)
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set CONDA_PATH=C:\ProgramData\miniconda3
    goto :found_conda
)

REM Conda not found anywhere
echo ERROR: Could not find Anaconda or Miniconda installation!
echo.
echo Please install Anaconda or Miniconda from:
echo   https://docs.conda.io/en/latest/miniconda.html
echo.
echo Or if already installed, the script could not find it.
echo Common install locations checked:
echo   - %USERPROFILE%\anaconda3
echo   - %USERPROFILE%\miniconda3
echo   - %LOCALAPPDATA%\anaconda3
echo   - C:\anaconda3
echo.
pause
exit /b 1

:found_conda
echo Found conda at: %CONDA_PATH%
call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"

:activate
echo.
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
