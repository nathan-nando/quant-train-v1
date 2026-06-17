import pandas as pd
import numpy as np
import os
import glob
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

print("=========================================")
print("  Memulai Pelatihan XGBoost Lokal...")
print("=========================================")

# 1. Mencari dataset terbaru secara dinamis dari folder kapsul versi (ingestion)
# base_dir saat ini adalah quant-train-v1/notebooks/xgboost
base_dir = os.path.dirname(os.path.abspath(__file__))
# Mundur 2 folder ke atas untuk mencapai root quant-train-v1
train_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))

dataset_pattern = os.path.join(train_dir, "*", "dataset", "XAUUSD_H1_features.csv")
list_of_files = glob.glob(dataset_pattern)

if not list_of_files:
    print("Error: Dataset XAUUSD_H1_features.csv tidak ditemukan!")
    print("Silakan jalankan ingest.bat terlebih dahulu.")
    exit(1)

# Mengambil dataset yang paling baru dibuat
csv_path = max(list_of_files, key=os.path.getctime)
print(f"Dataset Ditemukan:\n{csv_path}\n")

# 2. Load Data
df = pd.read_csv(csv_path)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
print(f"Data termuat: {len(df)} baris.")

# 3. Labeling (Threshold Return 3-Class)
N_BARS = 3
THRESHOLD = 0.0015
df['future_return'] = (df['close'].shift(-N_BARS) - df['close']) / df['close']

conditions = [
    (df['future_return'] > THRESHOLD),
    (df['future_return'] < -THRESHOLD)
]
choices = [1, -1]
df['target'] = np.select(conditions, choices, default=0)

# 4. Feature Engineering
df['close_lag_1'] = df['close'].shift(1)
df['return_lag_1'] = (df['close'] - df['close_lag_1']) / df['close_lag_1']
df.dropna(inplace=True)

fitur_kategori = ['time', 'future_return', 'target']
X = df.drop(columns=fitur_kategori)
X = X.astype(np.float32)
y = df['target']
y_mapped = y.map({-1: 0, 0: 1, 1: 2})

print(f"Matriks Fitur (X) siap: {X.shape[1]} indikator.")

# 5. Walk-Forward Validation
tscv = TimeSeriesSplit(n_splits=5)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1  # Menggunakan seluruh CPU core agar super cepat
)

print("\nMenjalankan Walk-Forward Validation (5 Folds)...")
fold = 1
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_mapped.iloc[train_index], y_mapped.iloc[test_index]
    
    # Gunakan .values agar onnxmltools tidak error membaca string nama kolom
    model.fit(X_train.values, y_train)
    y_pred = model.predict(X_test.values)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold} | Akurasi: {acc:.4f} | Test Range: {df['time'].iloc[test_index[0]].date()} s/d {df['time'].iloc[test_index[-1]].date()}")
    fold += 1

print("\n--- Classification Report (Siklus Pengujian Terakhir) ---")
target_names = ['SELL (0)', 'NEUTRAL (1)', 'BUY (2)']
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Export ke ONNX
print("\nMengekspor model ke format ONNX...")
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# Sinkronisasi ke Folder Engine
engine_dir = os.path.abspath(os.path.join(train_dir, "..", "quant-engine-v1", "ml_models"))
os.makedirs(engine_dir, exist_ok=True)
engine_filename = os.path.join(engine_dir, 'xgboost_baseline_v1.onnx')
onnx.save(onnx_model, engine_filename)
print(f"ONNX tersimpan di Engine: {engine_filename}")

# Sinkronisasi ke Folder Kapsul Train
kapsul_model_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "..", "model"))
os.makedirs(kapsul_model_dir, exist_ok=True)
kapsul_filename = os.path.join(kapsul_model_dir, 'xgboost_baseline_v1.onnx')
onnx.save(onnx_model, kapsul_filename)
print(f"ONNX tersimpan di Kapsul: {kapsul_filename}")

# 6b. Export Metadata (Fitur urutan untuk Engine)
import json
metadata = {
    "model_name": "xgboost_baseline_v1",
    "features": list(X.columns),
    "classes": {0: "SELL", 1: "NEUTRAL", 2: "BUY"}
}
engine_meta = os.path.join(engine_dir, 'xgboost_baseline_v1_metadata.json')
kapsul_meta = os.path.join(kapsul_model_dir, 'xgboost_baseline_v1_metadata.json')
with open(engine_meta, 'w') as f:
    json.dump(metadata, f, indent=4)
with open(kapsul_meta, 'w') as f:
    json.dump(metadata, f, indent=4)
print("Metadata model (urutan fitur) tersimpan.")

# 7. Membangun PDF Report
pdf_path = os.path.join(kapsul_model_dir, 'xgboost_baseline_v1_report.pdf')
print(f"\nMembangun PDF Report di: {pdf_path}")

with PdfPages(pdf_path) as pdf:
    # Halaman 1: Summary Teks
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    report_text = (
        "XGBoost Baseline Model Report\n"
        "======================================\n\n"
        f"Dataset: {os.path.basename(csv_path)}\n"
        f"Total Data (Rows): {len(df)}\n"
        f"Total Features (Cols): {X.shape[1]}\n"
        f"Validation Accuracy (Last Fold): {acc:.4f}\n\n"
        "Classification Report (Last Fold):\n"
        f"{classification_report(y_test, y_pred, target_names=target_names)}\n\n"
        "Model Configuration:\n"
        " - n_estimators = 100\n"
        " - max_depth = 4\n"
        " - learning_rate = 0.05\n"
        " - Walk-Forward Folds = 5\n"
    )
    plt.text(0.05, 0.95, report_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    pdf.savefig()
    plt.close()

    # Halaman 2: Feature Importance Plot
    plt.figure(figsize=(10, 8))
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(15).sort_values()
    feat_imp.plot(kind='barh', color='steelblue')
    plt.title("Top 15 Feature Importance (Penggerak Prediksi)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("=========================================")
print("SUKSES! Semua proses pelaporan & sinkronisasi selesai.")
print("=========================================")
