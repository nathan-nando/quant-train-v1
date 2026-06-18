@echo off
setlocal

cd /d "%~dp0\..\..\quant-engine-v1"

echo ==============================================
echo     QUANT-V1 XGBOOST BULL TREND TRAINING        
echo ==============================================
echo Mengaktifkan Virtual Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual Environment tidak ditemukan!
    goto end
)

echo Memeriksa dan memasang pustaka Machine Learning...
python -m pip install xgboost scikit-learn pandas onnx onnxmltools onnxconverter-common matplotlib --quiet

echo.
echo.
echo Menarik data harga terbaru (Live) dari MT5...
python scripts\ingest_historical.py --count 10000 --timeframe H1

echo.
echo Memulai proses pelatihan AI...
echo.
python "..\quant-train-v1\notebooks\xgboost\bull_trend\train_xgboost_bull_trend.py"

:end
echo.
pause
endlocal
