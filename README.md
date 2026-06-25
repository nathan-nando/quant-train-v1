# Quant Train V1 (Pusat Pelatihan AI)

`quant-train-v1` adalah lingkungan laboratorium khusus untuk *data ingestion*, riset, eksperimen, dan *hyperparameter tuning* Model XGBoost untuk keluarga algoritma Quant-V1.

Repositori ini berperan melatih, mengukur, dan merakit logika otak utama sebelum diserahkan ke `quant-engine-v1`.

## Alur Pipeline Pelatihan (Auto-Tuning)

1. **Pemilahan Rezim (Market Regime):** Data diproses dan dipisahkan menjadi *Bear Trend*, *Bull Trend*, dan *Mean Reverting*.
2. **Optuna Hyperparameter Tuning:** Mencari parameter pohon keputusan XGBoost paling mutakhir secara otomatis tanpa campur tangan manusia.
3. **Feature Drift Tracking:** Saat masa pelatihan, sistem melacak pergeseran kepentingan fitur (*Feature Importance Drift*) menggunakan `drift_tracker.py` dan menyimpannya secara mutlak ke folder `feature_drift/`. Fitur ini menjaga dan mendeteksi anomali pada tingkat kontribusi indikator.
4. **Ekspor ONNX:** Model *Classifier* (Arah) dan *Regressor* (Batas TP/SL) di-*compile* ulang ke dalam bentuk `.onnx` demi kecepatan eksekusi (*inference latency* < 1ms) di sisi Engine.
5. **Registrasi Otomatis:** Model hasil lulus uji didorong langsung ke folder `ml_models` pada Engine beserta konfigurasi metadata-nya.

## Struktur Direktori

```text
quant-train-v1/
├── dataset/                    # Rumah bagi CSV hasil injeksi fitur historis MT5
├── feature_drift/              # Penyimpanan riwayat log Feature Importance (menghindari error path di engine)
├── notebooks/
│   └── xgboost/                # Titik kumpul skrip engine tuning (bear, bull, mean_reverting)
├── utils/
│   └── drift_tracker.py        # Modul tracker otomatis untuk pergeseran indikator
└── ingest.bat                  # Utilitas Windows untuk menarik historis tick-by-tick
```

## Cara Penggunaan

Masuk ke *virtual environment* Python yang relevan dan navigasikan ke skrip tujuan.

```cmd
# Contoh melatih model Bear Trend secara mandiri
python notebooks/xgboost/bear_trend/train_xgboost_bear_trend.py --optuna_trials 20
```

Seluruh hasil ONNX akan didorong secara otomatis tanpa memicu gangguan pada mesin Engine yang sedang berjalan.
