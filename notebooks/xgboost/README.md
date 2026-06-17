# XGBoost Baseline untuk Quant-Engine-V1

## Deskripsi Data & Fitur
Dataset utama yang digunakan: `XAUUSD_H1_features.csv` (Periode: 2020 - 2023).
Data ini merupakan rekaman pergerakan harga emas (XAUUSD) dalam rentang waktu 1-Jam (H1). Selain informasi pergerakan harga dasar OHLCV (*Open, High, Low, Close, Volume*), data ini sudah diperkaya (*enriched*) dengan lebih dari 30 fitur teknikal hasil ekstraksi.
- **Trend Indicators:** EMA (9, 21, 50, 200), MACD, ADX.
- **Momentum Indicators:** RSI (7, 14), Stochastic, CCI.
- **Volatility & Volume Indicators:** ATR, Bollinger Bands, Keltner Channel, OBV, VWAP.
- **Custom Features:** Jarak persentase harga terhadap EMA 50 & 200, histori *return* (return_1, return_5).

## Rencana Data Terhadap Model (Data Plan to Model)
Model yang dilatih adalah **XGBoost Classifier**. 
Karakteristik XGBoost yang berbasis *decision tree* (*Gradient Boosting*) sangat ideal untuk memproses data finansial berdimensi tinggi yang memiliki variasi skala angka ekstrem tanpa perlu normalisasi ketat.
Langkah-langkah yang akan dijalankan oleh *notebook*:
1. **Pembersihan Data (Cleaning):** Menghapus *missing values* (NaN) yang muncul secara alami akibat kalkulasi fitur masa lalu (*lagging indicators*).
2. **Feature Engineering Temporal:** Menambahkan fitur waktu sekunder (contoh: *return lag-1*) agar XGBoost dapat menangkap momentum atau lintasan pergerakan jangka pendek.
3. **Validasi Berkala (Walk-Forward Validation):** Tidak akan menggunakan pembagian data secara acak (*random split*). Model akan divalidasi dengan membelah (*split*) data secara kronologis seiring berjalannya waktu (TimeSeriesSplit) untuk menjamin akurasi yang bebas dari kebocoran data masa depan (*data leakage*).

## Penentuan Fitur (X) dan Target (y)

### X (Fitur Input)
Seluruh indikator teknikal, OHLCV (kecuali baris waktu aktual `time`), dan kolom hasil *feature engineering* akan dilemparkan ke model. XGBoost akan menyeleksi secara otomatis fitur mana yang paling berpengaruh (*Feature Importance*).

### y (Target Label)
Menggunakan desain **Threshold Return (3-Class)**.
Model tidak meramal harga eksak, melainkan arah probabilitas pergerakan yang aman dari biaya *spread*:
- Horizon Waktu: `N_BARS = 3` (Target pergerakan diukur berdasarkan penutupan harga 3 jam ke depan).
- **Label 1 (BUY):** Terpicu jika prediksi persentase kenaikan harga lebih besar dari ambang batas aman (+0.15%).
- **Label -1 (SELL):** Terpicu jika prediksi persentase penurunan harga lebih tajam dari minus ambang batas (-0.15%).
- **Label 0 (NEUTRAL):** Terpicu jika harga bergerak stagnan (berada di antara -0.15% hingga +0.15%). Sinyal ini membuat *engine* Anda diam (*HOLD*) demi mengamankan ekuitas dari *noise* dan biaya *spread*.

## Standarisasi Output (ONNX)
Sebagai bagian dari standarisasi sistem *production* kita, *notebook* ini menggunakan pustaka `onnxmltools` untuk mengonversi dan menyimpan (*export*) model XGBoost yang telah dilatih secara utuh ke dalam format seragam **`.onnx`** (bukan `.joblib`). 
Format ONNX memberikan jaminan kecepatan eksekusi (*Inference Speed*) di dalam mesin `quant-engine-v1` serta keluwesan (*interoperability*) jika ke depannya kita menggunakan bahasa pemrograman lain (misalnya C++ atau Rust).
