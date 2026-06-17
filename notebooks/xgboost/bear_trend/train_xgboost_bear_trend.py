import pandas as pd
import numpy as np
import os
import glob
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import warnings
warnings.filterwarnings('ignore')

print("=========================================")
print("  Memulai Pelatihan Bear Trend Ensemble...")
print("=========================================")

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
dataset_pattern = os.path.join(train_dir, "*", "dataset", "XAUUSD_H1_features.csv")
list_of_files = glob.glob(dataset_pattern)

if not list_of_files:
    print("Error: Dataset XAUUSD_H1_features.csv tidak ditemukan!")
    exit(1)

csv_path = max(list_of_files, key=os.path.getctime)
print(f"Dataset Ditemukan:\n{csv_path}\n")

df = pd.read_csv(csv_path)
df['time'] = pd.to_datetime(df['time'])

# FILTER REGIME
if 'regime' in df.columns:
    df = df[df['regime'] == 'TREND_BEAR'].copy()
    print(f"Data difilter untuk Bear Trend: {len(df)} baris.")
else:
    print("WARNING: Kolom 'regime' tidak ditemukan, menggunakan semua data.")

df = df.sort_values('time').reset_index(drop=True)

N_BARS = 3
THRESHOLD = 0.0015
df['future_return'] = (df['close'].shift(-N_BARS) - df['close']) / df['close']

conditions = [
    (df['future_return'] > THRESHOLD),
    (df['future_return'] < -THRESHOLD)
]
choices = [1, -1]
df['target'] = np.select(conditions, choices, default=0)

# MAE / MFE untuk SL & TP
df['highest_high_future'] = df['high'].rolling(window=N_BARS).max().shift(-N_BARS)
df['lowest_low_future'] = df['low'].rolling(window=N_BARS).min().shift(-N_BARS)

# SL Calculation (MAE)
df['mae_buy'] = df['close'] - df['lowest_low_future']
df['mae_sell'] = df['highest_high_future'] - df['close']
df['sl_target'] = np.where(df['target'] == 1, df['mae_buy'], np.where(df['target'] == -1, df['mae_sell'], 0.0))
df['sl_pips'] = df['sl_target'] * 10
df['sl_pips'] = np.clip(df['sl_pips'], 15.0, None)

# TP Calculation (MFE)
df['mfe_buy'] = df['highest_high_future'] - df['close']
df['mfe_sell'] = df['close'] - df['lowest_low_future']
df['tp_target'] = np.where(df['target'] == 1, df['mfe_buy'], np.where(df['target'] == -1, df['mfe_sell'], 0.0))
df['tp_pips'] = df['tp_target'] * 10
df['tp_pips'] = np.clip(df['tp_pips'], 20.0, None)

df['close_lag_1'] = df['close'].shift(1)
df.dropna(inplace=True)

fitur_kategori = ['time', 'future_return', 'target', 'regime', 'highest_high_future', 'lowest_low_future', 'mae_buy', 'mae_sell', 'sl_target', 'sl_pips', 'mfe_buy', 'mfe_sell', 'tp_target', 'tp_pips']
fitur_kategori = [c for c in fitur_kategori if c in df.columns]

X_all = df.drop(columns=fitur_kategori)
X_all = X_all.astype(np.float32)

y_cls = df['target'].map({-1: 0, 0: 1, 1: 2})

print("\n[1/3] Melatih CLASSIFIER...")
tscv = TimeSeriesSplit(n_splits=5)
cls_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss', n_jobs=-1)

acc_last = 0
for train_index, test_index in tscv.split(X_all):
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_cls.iloc[train_index], y_cls.iloc[test_index]
    cls_model.fit(X_train.values, y_train)
    acc_last = accuracy_score(y_test, cls_model.predict(X_test.values))
print(f"Classifier Akurasi Akhir: {acc_last:.4f}")

df_trend = df[df['target'] != 0].copy()
X_trend = df_trend.drop(columns=fitur_kategori).astype(np.float32)
y_sl = df_trend['sl_pips']
y_tp = df_trend['tp_pips']

print("\n[2/3] Melatih SL REGRESSOR...")
sl_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
if len(X_trend) > 20:
    for train_index, test_index in tscv.split(X_trend):
        X_train, X_test = X_trend.iloc[train_index], X_trend.iloc[test_index]
        y_train, y_test = y_sl.iloc[train_index], y_sl.iloc[test_index]
        sl_model.fit(X_train.values, y_train)
        mae_sl = mean_absolute_error(y_test, sl_model.predict(X_test.values))
    print(f"SL MAE Akhir: {mae_sl:.4f} pips")
else:
    sl_model.fit(X_trend.values, y_sl)
    mae_sl = 0

print("\n[3/3] Melatih TP REGRESSOR...")
tp_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
if len(X_trend) > 20:
    for train_index, test_index in tscv.split(X_trend):
        X_train, X_test = X_trend.iloc[train_index], X_trend.iloc[test_index]
        y_train, y_test = y_tp.iloc[train_index], y_tp.iloc[test_index]
        tp_model.fit(X_train.values, y_train)
        mae_tp = mean_absolute_error(y_test, tp_model.predict(X_test.values))
    print(f"TP MAE Akhir: {mae_tp:.4f} pips")
else:
    tp_model.fit(X_trend.values, y_tp)
    mae_tp = 0

print("\nMengekspor 3 model ke format ONNX...")
initial_type = [('float_input', FloatTensorType([None, X_all.shape[1]]))]

onnx_cls = onnxmltools.convert_xgboost(cls_model, initial_types=initial_type)
onnx_sl = onnxmltools.convert_xgboost(sl_model, initial_types=initial_type)
onnx_tp = onnxmltools.convert_xgboost(tp_model, initial_types=initial_type)

engine_dir = os.path.abspath(os.path.join(train_dir, "..", "quant-engine-v1", "ml_models", "bear_trend"))
kapsul_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "..", "model", "bear_trend"))
os.makedirs(engine_dir, exist_ok=True)
os.makedirs(kapsul_dir, exist_ok=True)

names = [
    ('xgboost_bear_trend.onnx', onnx_cls),
    ('xgboost_bear_trend_sl.onnx', onnx_sl),
    ('xgboost_bear_trend_tp.onnx', onnx_tp)
]

for filename, model_onnx in names:
    onnx.save(model_onnx, os.path.join(engine_dir, filename))
    onnx.save(model_onnx, os.path.join(kapsul_dir, filename))

stats = {}
for col in X_all.columns:
    stats[col] = {
        "mean": float(X_all[col].mean()),
        "std": float(X_all[col].std()),
        "min": float(X_all[col].min()),
        "max": float(X_all[col].max())
    }

metadata = {
    "model_name": "xgboost_bear_trend",
    "features": list(X_all.columns),
    "classes": {0: "SELL", 1: "NEUTRAL", 2: "BUY"},
    "baseline_stats": stats
}
with open(os.path.join(engine_dir, 'xgboost_bear_trend_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)
with open(os.path.join(kapsul_dir, 'xgboost_bear_trend_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

pdf_path = os.path.join(kapsul_dir, 'xgboost_bear_trend_report.pdf')
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    report_text = f"ENSEMBLE REPORT: BEAR TREND\n"
    report_text += "="*40 + "\n"
    report_text += f"Classifier Acc : {acc_last:.4f}\n"
    if len(X_trend) > 20:
        report_text += f"SL Regressor MAE: {mae_sl:.4f} pips\n"
        report_text += f"TP Regressor MAE: {mae_tp:.4f} pips\n"
    plt.text(0.05, 0.95, report_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace')
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize=(10, 8))
    feat_imp = pd.Series(cls_model.feature_importances_, index=X_all.columns).nlargest(15).sort_values()
    feat_imp.plot(kind='barh', color='steelblue')
    plt.title(f"Top 15 Features - Bear Trend (Classifier)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"\nSUKSES! 3 Model ONNX untuk Bear Trend telah dibuat dan disimpan di direktori bear_trend.")
