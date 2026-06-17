# Quant-Train-V1

Quant-Train-V1 adalah lingkungan pelatihan (Training Environment) khusus untuk proyek Algorithmic Trading Quant-V1. Repositori ini difokuskan sepenuhnya untuk pengumpulan data historis (ingestion), penelitian (research), dan pelatihan Alpha Model (Machine Learning).

Proyek ini sengaja dipisahkan dari `quant-engine-v1` (Backend/Sistem Saraf) dan `quant-dashboard-v1` (Frontend/Portal) agar proses *data science* tidak mengganggu sistem eksekusi *live trading*.

## 🚀 Fitur Utama

- **Data Ingestion Otomatis:** Skrip penarikan data OHLCV langsung dari MetaTrader 5 yang telah dilengkapi dengan injeksi fitur teknikal (*RSI, MACD, ATR, OBV* dll).
- **Fleksibilitas Pengambilan:** Mendukung penarikan data berdasarkan jumlah baris (misal: 50.000 baris ke belakang) atau rentang waktu historis (misal: `2020-01-01` sampai `2023-12-31`).
- **Sistem Kapsul Versi (Versioning):** Setiap penarikan data akan membuat satu direktori kapsul unik yang berisi kumpulan `dataset` (CSV) dan folder `model` yang siap digunakan, memastikan hasil pelatih model ML tidak saling tindih.

## 📁 Struktur Direktori Otomatis

Setiap kali Anda menjalankan skrip *ingestion*, sistem akan menghasilkan struktur versi seperti ini:

```text
quant-train-v1/
├── ingest.bat                          # Pelatuk utama penarik data (Windows)
├── README.md                           # Penjelasan repositori
└── {TIMEFRAME}_{OPTION}_{TIMESTAMP}/   # Contoh: H1_50000rows_20260617_161500
    ├── dataset/
    │   └── XAUUSD_H1_features.csv      # File dataset hasil ekstraksi siap pakai
    └── model/                          # (Kosong) Tempat Anda menyimpan model (.pkl / .joblib)
```

## 🛠️ Cara Penggunaan (Data Ingestion)

Pastikan Anda sudah menjalankan MetaTrader 5 di mesin Anda, kemudian:

1. Buka *Command Prompt* atau *PowerShell*
2. Masuk ke direktori `quant-train-v1`
3. Jalankan berkas *batch* interaktif:
   ```cmd
   ingest.bat
   ```
4. Ikuti instruksi interaktif yang muncul di layar:
   - Pilih mode `Total Row` (Jumlah Baris) atau `Date Range` (Rentang Waktu).
   - Masukkan *Timeframe* yang Anda inginkan (tekan *Enter* untuk *default* `H1`).
5. Selesai! Kapsul versi baru Anda siap digunakan untuk pelatihan model.

## 🧠 Langkah Selanjutnya (Data Science)

1. Buat berkas *Jupyter Notebook* (`.ipynb`) atau *Python script* Anda di dalam folder versi spesifik yang baru terbentuk.
2. Muat (*load*) file `.csv` dari sub-folder `dataset/`.
3. Latih model *Machine Learning* Anda (XGBoost, Random Forest, LSTM, dll).
4. Simpan model final (*dump*) ke dalam sub-folder `model/`.
5. Pindahkan model yang sudah matang dari folder `model/` ke dalam `quant-engine-v1/app/services/alpha_engine.py` untuk dihidupkan di pasar nyata!
