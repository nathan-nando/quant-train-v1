@echo off
setlocal

:: Navigate to the engine directory since the python script lives there
cd /d "%~dp0\..\..\quant-engine-v1"

echo ==============================================
echo     QUANT-V1 HISTORICAL DATA INGESTION        
echo ==============================================
echo Please select the ingestion method:
echo 1. By total row (e.g. 50000)
echo 2. By date range (e.g. 2020-01-01 -^> 2023-12-31)
echo ==============================================

set /p choice="Enter choice [1 or 2]: "

if "%choice%"=="1" goto opt1
if "%choice%"=="2" goto opt2

echo Invalid choice. Exiting.
goto end

:opt1
set /p count="Enter total rows to pull (e.g. 10000): "
set /p tf="Enter timeframe (e.g. H1, M15) [Press Enter for H1]: "
if "%tf%"=="" set tf=H1
echo Running ingestion...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)
python scripts\ingest_historical.py --count %count% --timeframe %tf%
goto end

:opt2
set /p start_date="Enter start date (YYYY-MM-DD): "
set /p end_date="Enter end date (YYYY-MM-DD): "
set /p tf="Enter timeframe (e.g. H1, M15) [Press Enter for H1]: "
if "%tf%"=="" set tf=H1
echo Running ingestion...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)
python scripts\ingest_historical.py --start %start_date% --end %end_date% --timeframe %tf%
goto end

:end
endlocal
