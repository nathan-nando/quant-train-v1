import uuid
import requests
run_id = str(uuid.uuid4())[:6]


import pandas as pd
import numpy as np
import os
import glob
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, fbeta_score, mean_absolute_error, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import optuna
import warnings
import argparse
import time
from datetime import datetime

train_start_time = datetime.utcnow().isoformat()
start_time_ts = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--optuna_trials', type=int, default=50)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--dataset_file', type=str, default='XAUUSDm_H1_features.csv')
parser.add_argument('--use_meta', type=str, default='1')
args = parser.parse_args()
model_prefix = args.model_name if args.model_name else f'xgboost_mean_reverting_{run_id}'

warnings.filterwarnings('ignore')

print("=========================================")
print("  Memulai Pelatihan Mean Reverting Ensemble...")
print("=========================================")
print(f"  [Config] Model Prefix  : {model_prefix}")
print(f"  [Config] Dataset File  : {args.dataset_file}")
print(f"  [Config] Optuna Trials : {args.optuna_trials}")
print(f"  [Config] Meta Labeling : {'ON' if str(args.use_meta) == '1' else 'OFF'}")
print("=========================================")

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
csv_path = os.path.join(train_dir, "dataset", args.dataset_file)


if not os.path.exists(csv_path):
    print("Error: Dataset XAUUSDm_H1_features.csv tidak ditemukan!")
    exit(1)


print(f"Dataset Ditemukan:\n{csv_path}\n")

df = pd.read_csv(csv_path)
df['time'] = pd.to_datetime(df['time'])

# FILTER REGIME
if 'regime' in df.columns:
    df = df[df['regime'] == 'MEAN_REVERTING'].copy()
    print(f"Data difilter untuk Mean Reverting: {len(df)} baris.")
else:
    print("WARNING: Kolom 'regime' tidak ditemukan, menggunakan semua data.")

df = df.sort_values('time').reset_index(drop=True)

N_BARS = 6
df['future_return'] = (df['close'].shift(-N_BARS) - df['close']) / df['close']
df['atr_normalized'] = df['atr_pct'] / 100.0

conditions = [
    ((df['bb_position'] < 0.3) & (df['future_return'] > df['atr_normalized'] * 1.0)),
    ((df['bb_position'] > 0.7) & (df['future_return'] < -df['atr_normalized'] * 1.0))
]
choices = [1, 1]
df['target'] = np.select(conditions, choices, default=0)

# Swing Structure untuk SL & TP AI Regressor Target
LOOKBACK = 12
df['swing_low_12'] = df['low'].rolling(window=LOOKBACK).min()
df['swing_high_12'] = df['high'].rolling(window=LOOKBACK).max()
df['atr_dollars'] = df['atr_normalized'] * df['close']

# SL Calculation (Market Structure)
df['sl_buy_dist'] = df['close'] - (df['swing_low_12'] - 0.5 * df['atr_dollars'])
df['sl_buy_dist'] = np.maximum(df['sl_buy_dist'], 1.0 * df['atr_dollars'])
df['sl_sell_dist'] = (df['swing_high_12'] + 0.5 * df['atr_dollars']) - df['close']
df['sl_sell_dist'] = np.maximum(df['sl_sell_dist'], 1.0 * df['atr_dollars'])

df['sl_target'] = np.where((df['target'] == 1) & (df['future_return'] > 0), df['sl_buy_dist'], 
                  np.where((df['target'] == 1) & (df['future_return'] < 0), df['sl_sell_dist'], 0.0))
df['sl_pips'] = df['sl_target'] * 10
df['sl_pips'] = np.clip(df['sl_pips'], 15.0, None)

# TP Calculation (Dynamic Risk-Reward)
df['tp_buy_dist'] = df['sl_buy_dist'] * 2.0
df['tp_sell_dist'] = df['sl_sell_dist'] * 2.0
df['tp_target'] = np.where((df['target'] == 1) & (df['future_return'] > 0), df['tp_buy_dist'], 
                  np.where((df['target'] == 1) & (df['future_return'] < 0), df['tp_sell_dist'], 0.0))
df['tp_pips'] = df['tp_target'] * 10
df['tp_pips'] = np.clip(df['tp_pips'], 20.0, None)

df.dropna(inplace=True)

fitur_kategori = ['time', 'future_return', 'target', 'regime', 'swing_low_12', 'swing_high_12', 'atr_dollars', 'sl_buy_dist', 'sl_sell_dist', 'sl_target', 'sl_pips', 'tp_buy_dist', 'tp_sell_dist', 'tp_target', 'tp_pips', 'open', 'high', 'low', 'close', 'atr_normalized', 'obv']
fitur_kategori = [c for c in fitur_kategori if c in df.columns]

X_all = df.drop(columns=fitur_kategori)
X_all = X_all.astype(np.float32)

# Convert to Binary Classification (1 = MEAN_REVERT, 0 = NEUTRAL)
y_cls = df['target'].copy()
y_cls = pd.Series(y_cls, index=df.index)

# Hitung class imbalance weight
pos_count = sum(y_cls == 1)
neg_count = sum(y_cls == 0)
scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
print(f"Class Distribution: {neg_count} NEUTRAL vs {pos_count} MEAN_REVERT (Weight: {scale_pos_weight:.2f})")

print("\n[0/3] Feature Selection (Top 20)...")
fs_model = XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
train_size = int(len(X_all) * 0.8)
fs_model.fit(X_all.iloc[:train_size].values, y_cls.iloc[:train_size])
feat_imp = pd.Series(fs_model.feature_importances_, index=X_all.columns).nlargest(20)
top_features = feat_imp.index.tolist()
X_all = X_all[top_features]
print(f"Selected Top 20 Features: {top_features}")

print("\nPROGRESS: 40% - Tuning Classifier...\n[1/3] Melatih CLASSIFIER...")
days_from_end = (df['time'].max() - df['time']).dt.days
sample_weights = pd.Series(np.exp(-0.001 * days_from_end), index=df.index)
tscv = TimeSeriesSplit(n_splits=5, gap=N_BARS * 2)

def cls_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic'
    }
    model = XGBClassifier(**params, random_state=42, eval_metric='logloss', n_jobs=-1)
    scores = []
    for train_index, test_index in tscv.split(X_all):
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_cls.iloc[train_index], y_cls.iloc[test_index]
        w_train = sample_weights.iloc[train_index]
        model.fit(X_train.values, y_train, sample_weight=w_train.values)
        scores.append(fbeta_score(y_test, model.predict(X_test.values), beta=0.5, average='macro'))
    return np.mean(scores)

print("Starting Optuna Study for CLASSIFIER...")
cls_study = optuna.create_study(direction='maximize')
cls_study.optimize(cls_objective, n_trials=args.optuna_trials)
best_cls_params = cls_study.best_params
print(f"Best Classifier Params: {best_cls_params}")

cls_model = XGBClassifier(**best_cls_params, objective='binary:logistic', random_state=42, eval_metric='logloss', n_jobs=-1)

f1_scores = []
report_dict = {}
for train_index, test_index in tscv.split(X_all):
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_cls.iloc[train_index], y_cls.iloc[test_index]
    w_train = sample_weights.iloc[train_index]
    cls_model.fit(X_train.values, y_train, sample_weight=w_train.values)
    y_pred = cls_model.predict(X_test.values)
    f1_scores.append(fbeta_score(y_test, y_pred, beta=0.5, average='macro'))
    report_dict = classification_report([str(x) for x in y_test], [str(x) for x in y_pred], output_dict=True)

f1_last = f1_scores[-1] if f1_scores else 0
print(f"Folds Macro F0.5: {[round(a, 4) for a in f1_scores]}")
print(f"Mean F0.5: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}, Min: {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}")
print(f"Classifier Macro F0.5 Akhir: {f1_last:.4f}")
print("Retraining CLASSIFIER on full dataset...")
cls_model.fit(X_all.values, y_cls, sample_weight=sample_weights.values)

df_trend = df[df['target'] != 0].copy()
sample_weights_trend = sample_weights[df['target'] != 0]
X_trend = df_trend.drop(columns=fitur_kategori).astype(np.float32)
X_trend = X_trend[top_features]
y_sl = df_trend['sl_pips']
y_tp = df_trend['tp_pips']

print("\nPROGRESS: 60% - Tuning SL Regressor...\n[2/3] Melatih SL REGRESSOR...")
def sl_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }
    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    scores = []
    if len(X_trend) > 20:
        for train_index, test_index in tscv.split(X_trend):
            X_train, X_test = X_trend.iloc[train_index], X_trend.iloc[test_index]
            y_train, y_test = y_sl.iloc[train_index], y_sl.iloc[test_index]
            w_train = sample_weights_trend.iloc[train_index]
            model.fit(X_train.values, y_train, sample_weight=w_train.values)
            scores.append(mean_absolute_error(y_test, model.predict(X_test.values)))
        return np.mean(scores)
    return 999.0

if len(X_trend) > 20:
    print("Starting Optuna Study for SL REGRESSOR...")
    sl_study = optuna.create_study(direction='minimize')
    sl_study.optimize(sl_objective, n_trials=args.optuna_trials)
    best_sl_params = sl_study.best_params
    print(f"Best SL Regressor Params: {best_sl_params}")
else:
    best_sl_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}

sl_model = XGBRegressor(**best_sl_params, random_state=42, n_jobs=-1)
if len(X_trend) > 20:
    sl_scores = []
    for train_index, test_index in tscv.split(X_trend):
        X_train, X_test = X_trend.iloc[train_index], X_trend.iloc[test_index]
        y_train, y_test = y_sl.iloc[train_index], y_sl.iloc[test_index]
        w_train = sample_weights_trend.iloc[train_index]
        sl_model.fit(X_train.values, y_train, sample_weight=w_train.values)
        sl_scores.append(mean_absolute_error(y_test, sl_model.predict(X_test.values)))
    mae_sl = sl_scores[-1] if sl_scores else 0
    print(f"Folds MAE: {[round(a, 4) for a in sl_scores]}")
    print(f"SL MAE Akhir: {mae_sl:.4f} pips")
    print("Retraining SL REGRESSOR on full dataset...")
    sl_model.fit(X_trend.values, y_sl, sample_weight=sample_weights_trend.values)
else:
    sl_model.fit(X_trend.values, y_sl, sample_weight=sample_weights_trend.values)
    mae_sl = 0

print("\nPROGRESS: 80% - Tuning TP Regressor...\n[3/3] Melatih TP REGRESSOR...")
def tp_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
    }
    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    scores = []
    if len(X_trend) > 20:
        for train_index, test_index in tscv.split(X_trend):
            X_train, X_test = X_trend.iloc[train_index], X_trend.iloc[test_index]
            y_train, y_test = y_tp.iloc[train_index], y_tp.iloc[test_index]
            w_train = sample_weights_trend.iloc[train_index]
            model.fit(X_train.values, y_train, sample_weight=w_train.values)
            scores.append(mean_absolute_error(y_test, model.predict(X_test.values)))
        return np.mean(scores)
    return 999.0

if len(X_trend) > 20:
    print("Starting Optuna Study for TP REGRESSOR...")
    tp_study = optuna.create_study(direction='minimize')
    tp_study.optimize(tp_objective, n_trials=args.optuna_trials)
    best_tp_params = tp_study.best_params
    print(f"Best TP Regressor Params: {best_tp_params}")
else:
    best_tp_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}

tp_model = XGBRegressor(**best_tp_params, random_state=42, n_jobs=-1)
if len(X_trend) > 20:
    tp_scores = []
    for train_index, test_index in tscv.split(X_trend):
        X_train, X_test = X_trend.iloc[train_index], X_trend.iloc[test_index]
        y_train, y_test = y_tp.iloc[train_index], y_tp.iloc[test_index]
        w_train = sample_weights_trend.iloc[train_index]
        tp_model.fit(X_train.values, y_train, sample_weight=w_train.values)
        tp_scores.append(mean_absolute_error(y_test, tp_model.predict(X_test.values)))
    mae_tp = tp_scores[-1] if tp_scores else 0
    print(f"Folds MAE: {[round(a, 4) for a in tp_scores]}")
    print(f"TP MAE Akhir: {mae_tp:.4f} pips")
    print("Retraining TP REGRESSOR on full dataset...")
    tp_model.fit(X_trend.values, y_tp, sample_weight=sample_weights_trend.values)
else:
    tp_model.fit(X_trend.values, y_tp, sample_weight=sample_weights_trend.values)
    mae_tp = 0

if args.use_meta == "1":
    print("\n[Meta] Melatih Meta-Model untuk Confidence Calibration...")
    from sklearn.model_selection import cross_val_predict

# Generate OOF predictions and probabilities
    y_pred_oof = cross_val_predict(cls_model, X_all.values, y_cls, cv=5, method='predict', n_jobs=-1)
    y_proba_oof = cross_val_predict(cls_model, X_all.values, y_cls, cv=5, method='predict_proba', n_jobs=-1)
    confidence_oof = np.max(y_proba_oof, axis=1)

# Create Meta-Labels: 1 if Primary Model is correct, 0 otherwise
    y_meta = (y_pred_oof == y_cls).astype(int)

# Create Meta-Features: Original features + primary confidence
    X_meta = X_all.copy()
    X_meta['primary_confidence'] = confidence_oof

# Train Meta-Model
    meta_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')
    meta_model.fit(X_meta.values, y_meta)
    meta_acc = accuracy_score(y_meta, meta_model.predict(X_meta.values))
    print(f"Meta-Model Akurasi: {meta_acc:.4f}")

print("\nMengekspor model ke format ONNX...")
initial_type = [('float_input', FloatTensorType([None, X_all.shape[1]]))]

onnx_cls = onnxmltools.convert_xgboost(cls_model, initial_types=initial_type)
onnx_sl = onnxmltools.convert_xgboost(sl_model, initial_types=initial_type)
onnx_tp = onnxmltools.convert_xgboost(tp_model, initial_types=initial_type)

engine_dir = os.path.abspath(os.path.join(train_dir, "..", "quant-engine-v1", "ml_models", "mean_reverting"))
kapsul_dir = os.path.abspath(os.path.join(train_dir, "mlflow", "models", "mean_reverting"))
os.makedirs(engine_dir, exist_ok=True)
os.makedirs(kapsul_dir, exist_ok=True)

names = [
    (f'{model_prefix}.onnx', onnx_cls),
    (f'{model_prefix}_sl.onnx', onnx_sl),
    (f'{model_prefix}_tp.onnx', onnx_tp)
]

if args.use_meta == "1":
    meta_initial_type = [('float_input', FloatTensorType([None, X_meta.shape[1]]))]
    onnx_meta = onnxmltools.convert_xgboost(meta_model, initial_types=meta_initial_type)
    names.append((f'{model_prefix}_meta.onnx', onnx_meta))

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
    "model_name": model_prefix,
    "features": list(X_all.columns),
    "classes": {0: "NEUTRAL", 1: "MEAN_REVERT"},
    "baseline_stats": stats,
    "uses_meta": (args.use_meta == "1")
}
with open(os.path.join(engine_dir, f'{model_prefix}_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)
with open(os.path.join(kapsul_dir, f'{model_prefix}_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

pdf_path = os.path.join(kapsul_dir, f'{model_prefix}_report.pdf')
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    report_text = f"ENSEMBLE REPORT: MEAN REVERTING\n"
    report_text += "="*40 + "\n"
    report_text += f"Classifier F0.5 : {f1_last:.4f}\n"
    report_text += f"Prec (BUY): {report_dict.get('1', {}).get('precision', 0.0):.4f} | Prec (SELL): {report_dict.get('2', {}).get('precision', 0.0):.4f}\n"
    report_text += f"Best Params (Cls): {best_cls_params}\n"
    if len(X_trend) > 20:
        report_text += f"SL Regressor MAE: {mae_sl:.4f} pips\n"
        report_text += f"TP Regressor MAE: {mae_tp:.4f} pips\n"
    plt.text(0.05, 0.95, report_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace')
    pdf.savefig()
    plt.close()
    
    plt.figure(figsize=(10, 8))
    feat_imp = pd.Series(cls_model.feature_importances_, index=X_all.columns).nlargest(15).sort_values()
    feat_imp.plot(kind='barh', color='steelblue')
    plt.title(f"Top 15 Features - Mean Reverting (Classifier)")
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# ==========================================
# MLFLOW TRACKING
# ==========================================
import os
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
import mlflow
from mlflow.exceptions import MlflowException

mlflow.set_tracking_uri("file:///c:/code/quant-v1/quant-train-v1/mlflow")
try:
    mlflow.set_experiment("XAUUSD_H1_mean_reverting")
except MlflowException:
    experiment = mlflow.get_experiment_by_name("XAUUSD_H1_mean_reverting")
    if experiment and experiment.lifecycle_stage == 'deleted':
        mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
        mlflow.set_experiment("XAUUSD_H1_mean_reverting")

with mlflow.start_run(run_name="Optuna_Auto_Tuning"):
    # Log parameters
    mlflow.log_params({"cls_" + k: v for k, v in best_cls_params.items()})
    mlflow.log_params({"sl_" + k: v for k, v in best_sl_params.items()})
    mlflow.log_params({"tp_" + k: v for k, v in best_tp_params.items()})
    
    # Log metrics
    mlflow.log_metric("cls_f05", f1_last)
    if 'mae_sl' in locals():
        mlflow.log_metric("sl_mae", mae_sl)
        mlflow.log_metric("tp_mae", mae_tp)
        
    # Log artifacts
    mlflow.log_artifact(os.path.join(kapsul_dir, f'{model_prefix}.onnx'))
    mlflow.log_artifact(os.path.join(kapsul_dir, f'{model_prefix}_sl.onnx'))
    mlflow.log_artifact(os.path.join(kapsul_dir, f'{model_prefix}_tp.onnx'))
    mlflow.log_artifact(os.path.join(kapsul_dir, f'{model_prefix}_metadata.json'))
    mlflow.log_artifact(pdf_path)
    
print("\n[MLFlow] Run recorded successfully!")
print(f"\nSUKSES! 3 Model ONNX untuk Mean Reverting telah dibuat dan disimpan di direktori mean_reverting.")

# Auto-register to Engine Database
try:
    print("\n[Registry] Mendaftarkan model ke Engine...")
    train_duration_sec = str(int(time.time() - start_time_ts))
    
    try:
        report_str = json.dumps(report_dict)
    except:
        report_str = "{}"
        
    try:
        hyper_str = json.dumps({"cls": best_cls_params, "sl": best_sl_params, "tp": best_tp_params})
    except:
        hyper_str = "{}"

    print("\n=========================================")
    print("  FINAL CONFIGURATION & HYPERPARAMETERS  ")
    print("=========================================")
    try:
        hyper_dict = json.loads(hyper_str)
        print("--- HYPERPARAMETERS ---")
        for m_type, params in hyper_dict.items():
            print(f"[{m_type.upper()}]")
            for k, v in params.items():
                print(f"  > {k:<14} : {v}")
    except: pass

    print("\n--- METADATA ---")
    for k, v in metadata.items():
        if isinstance(v, list):
            print(f"  > {k:<14} : [{len(v)} items]")
        else:
            print(f"  > {k:<14} : {v}")
    print("=========================================\n")

    payload = {
        "name": model_prefix,
        "algorithm_type": "XGBoost Ensemble",
        "accuracy": f"{f1_last*100:.2f}%",
        "status": "Inactive",
        "train_start_time": train_start_time,
        "train_duration_sec": train_duration_sec,
        "metrics_report": report_str,
        "hyperparameters": hyper_str,
        "metadata": json.dumps(metadata),
        "regime": "MEAN_REVERTING"
    }
    res = requests.post("http://127.0.0.1:8000/api/models/", json=payload)
    if res.status_code == 200:
        print(f"[Registry] Berhasil mendaftarkan {model_prefix} ke Dashboard!")
    else:
        print(f"[Registry] Gagal mendaftar: {res.text}")
except Exception as e:
    print(f"[Registry] Gagal terhubung ke API Engine: {e}")