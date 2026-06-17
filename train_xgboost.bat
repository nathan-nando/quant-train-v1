@echo off
setlocal

:: Berpindah ke folder quant-engine-v1 karena virtual environment ada di sana
cd /d "%~dp0\..\quant-engine-v1"

echo ==============================================
echo     QUANT-V1 XGBOOST LOCAL TRAINING        
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
echo Memulai proses pelatihan AI...
echo.
python "..\quant-train-v1\notebooks\xgboost\train_xgboost.py"

:end
echo.
pause
endlocal
