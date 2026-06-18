@echo off
setlocal

cd /d "%~dp0"

echo ==============================================
echo     QUANT-V1 MLFLOW TRACKING SERVER        
echo ==============================================
echo Mengaktifkan Virtual Environment...
if exist "..\..\quant-engine-v1\.venv\Scripts\activate.bat" (
    call ..\..\quant-engine-v1\.venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual Environment tidak ditemukan!
    goto end
)

echo.
echo Memulai MLflow UI pada http://127.0.0.1:5000 ...
echo Silakan buka URL tersebut di browser Anda.
echo.
set MLFLOW_ALLOW_FILE_STORE=true
python -m mlflow ui --backend-store-uri file:///c:/code/quant-v1/quant-train-v1/mlflow --host 0.0.0.0 --port 5000

:end
echo.
pause
endlocal
