# Quant Train V1 (Pusat Pelatihan & Ingestion)

`quant-train-v1` adalah lingkungan laboratorium khusus untuk *data ingestion*, riset, dan *hyperparameter tuning* Model *Machine Learning* untuk keluarga algoritma Quant-V1.

Sistem pelatihan ini didesain independen secara penyimpanan agar tidak membebani eksekusi transaksi di pasar nyata, namun sepenuhnya terintegrasi sehingga prosesnya dapat dikendalikan dari jarak jauh melalui **Quant Dashboard V1**.

## 🚀 Pipa Pelatihan Otomatis (Auto-Tuning Pipeline)

Sistem telah dielevasi untuk menggunakan format *XGBoost Ensemble*, yang memecah pembelajaran menjadi 3 langkah berurutan:
1. **Pemilahan Data:** Memfilter *dataset* CSV berdasarkan kriteria *Market Regime* yang dituju (misal: `TREND_BULL`).
2. **Tuning Classifier (Arah):** Menjalankan studi **Optuna** secara otomatis guna menemukan pohon keputusan optimal penentu arah tren (Buy/Sell/Hold).
3. **Tuning SL & TP Regressors (Batas Harga):** Melatih dua *regressor* tambahan untuk memprediksi probabilitas batas terburuk (*Stop Loss*/MAE) dan kemungkinan lonjakan terbaik (*Take Profit*/MFE).
4. **ONNX Export:** Ketiga model di atas langsung diekspor dari memori ke dalam format `.onnx` yang ringan, cepat dieksekusi, dan terstandarisasi.
5. **Registrasi MLflow & Database:** Secara mandiri mencatat *metric accuracy/loss*, argumen model ke *local* **MLflow** backend, menghasilkan file PDF laporan, dan mendaftarkannya via HTTP POST ke Registry `quant-engine-v1`.

## 📁 Struktur Direktori

```text
quant-train-v1/
├── dataset/                    # Rumah bagi CSV hasil injeksi fitur (MACD, ATR, EMA, dll)
├── mlflow/                     # Penyimpanan backend MLflow untuk catatan trials
├── notebooks/
│   └── xgboost/                # Titik kumpul skrip engine tuning (seperti: train_xgboost_bull_trend.py)
└── ingest.bat                  # Utilitas Windows untuk menarik data baris per baris dari MT5
```

## 🛠️ Cara Penggunaan

### Opsi 1: Mode Dashboard (Otomatis & Real-time)
Pelatihan kini terintegrasi secara modular. Buka antarmuka web (Quant Dashboard V1) pada menu **Models -> Train**.
Proses inisialisasi, *loop trials* Optuna, serta konfirmasi penyelesaian akan muncul secara langsung (*streaming*) ke browser.

### Opsi 2: Mode CLI (Manual Tuning)
Jika Anda butuh iterasi *hyperparameter* tertentu tanpa campur tangan API, pastikan Anda memiliki *dataset* yang matang (`ingest.bat`), lalu jalankan:
```cmd
python notebooks/xgboost/bull_trend/train_xgboost_bull_trend.py --optuna_trials 15 --model_name xgboost_my_custom_v1
```
Model ONNX akan dicetak dan dimasukkan ke dalam arsip pendaftaran tanpa perlu dipindahkan secara manual.
