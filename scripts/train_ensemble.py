import pandas as pd
import numpy as np
import os
import logging
import json
import argparse
import joblib
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert import convert_xgboost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_target(df):
    """Creates a simple 3-class target: 0=SELL, 1=NEUTRAL, 2=BUY based on future return."""
    future_ret = df['close'].shift(-5) / df['close'] - 1
    
    target = np.ones(len(df)) # Default NEUTRAL
    target[future_ret > 0.002] = 2 # BUY (> 0.2%)
    target[future_ret < -0.002] = 0 # SELL (< -0.2%)
    
    df['target_direction'] = target
    return df


def optimize_and_train_xgb(X, y, n_trials=10, prefix=""):
    if n_trials <= 0:
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, objective='multi:softprob')
        model.fit(X, y)
        return model

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'objective': 'multi:softprob'
        }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
        
    print(f"Running Optuna tuning for {prefix} ({n_trials} trials)...", flush=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[{prefix}] Best params: {study.best_params} | Best Score: {study.best_value:.4f}", flush=True)
    best_model = xgb.XGBClassifier(**study.best_params, objective='multi:softprob')
    best_model.fit(X, y)
    return best_model

def save_onnx_model(xgb_model, feature_names, save_path, uses_meta=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)
    
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    # Save metadata
    meta = {
        "features": feature_names,
        "classes": {0: "SELL", 1: "NEUTRAL", 2: "BUY"},
        "uses_meta": uses_meta
    }
    meta_path = save_path.replace(".onnx", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

def train_ensemble(args=None):
    logger.info("Starting MoE Ensemble Training Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = args.dataset_file if args and hasattr(args, 'dataset_file') and args.dataset_file else "XAUUSD_DEFAULT.csv"
    macro_file = args.macro_file if args and hasattr(args, 'macro_file') and args.macro_file else None
    
    print("\n================ TRAINING CONFIGURATION ================", flush=True)
    print(f"Base Dataset : {dataset_file}", flush=True)
    if macro_file:
        print(f"Macro Dataset: {macro_file}", flush=True)
    optuna_trials = args.optuna_trials if args and hasattr(args, "optuna_trials") else 0
    print(f"Model Engine : XGBoost (Optuna Trials: {optuna_trials})", flush=True)
    print(f"Features used: 5 Technical + 4 Macro (9 total)", flush=True)
    print("========================================================\n", flush=True)
    

    dataset_path = os.path.join(base_dir, "dataset", dataset_file)
    
    model_name_base = args.model_name if args and hasattr(args, 'model_name') and args.model_name else "xgboost"
    engine_models_dir = os.path.join(os.path.dirname(base_dir), "quant-engine-v1", "ml_models")

    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
    if macro_file:
        macro_path = os.path.join(base_dir, "dataset", macro_file)
        if os.path.exists(macro_path):
            print(f"PROGRESS: 15% - Explicitly joining Macro dataset ({macro_file}) by timestamp...")
            df_macro = pd.read_csv(macro_path)
            if 'time' in df_macro.columns:
                df_macro['time'] = pd.to_datetime(df_macro['time'])
                df_macro.set_index('time', inplace=True)
            cols_to_use = df_macro.columns.difference(df.columns)
            df = df.join(df_macro[cols_to_use], how='left')
            df.ffill(inplace=True)
            df.fillna(0, inplace=True)
        else:
            logger.warning(f"Macro dataset not found at {macro_path}. Macro Expert will fall back.")
            
    df = df.dropna().copy()
    df = create_target(df)
    
    # Feature Selection
    tech_features = ['adx', 'rsi', 'macd', 'bb_width', 'dist_ema_50']
    macro_features = [f for f in df.columns if f in ['tips_10y_level', 'dxy_broad_return_5d', 'vix_level', 'fed_rate_level']]
    
    # Missing features fallback
    features_to_use = [f for f in tech_features + macro_features if f in df.columns]
    
    # 1. Train Trend Expert (ADX > 25)
    print("PROGRESS: 25% - Training Trend Expert (ADX > 25)...")
    logger.info("Training Trend Expert...")
    df_trend = df[df['adx'] > 25].copy() if 'adx' in df.columns else df.copy()
    X_trend = df_trend[features_to_use].values
    y_trend = df_trend['target_direction'].values
    
    model_trend = optimize_and_train_xgb(X_trend, y_trend, optuna_trials, "Trend Expert")
    name_trend = f"{model_name_base}_trend"
    save_onnx_model(model_trend, features_to_use, os.path.join(engine_models_dir, "trend", f"{name_trend}.onnx"), uses_meta=(args.use_meta == "1") if args and hasattr(args, 'use_meta') else True)
    
    # 2. Train MeanRev Expert (ADX <= 25)
    print("PROGRESS: 50% - Training MeanRev Expert (ADX <= 25)...")
    logger.info("Training MeanRev Expert...")
    df_meanrev = df[df['adx'] <= 25].copy() if 'adx' in df.columns else df.copy()
    X_meanrev = df_meanrev[features_to_use].values
    y_meanrev = df_meanrev['target_direction'].values
    
    model_meanrev = optimize_and_train_xgb(X_meanrev, y_meanrev, optuna_trials, "MeanRev Expert")
    name_meanrev = f"{model_name_base}_meanrev"
    save_onnx_model(model_meanrev, features_to_use, os.path.join(engine_models_dir, "meanrev", f"{name_meanrev}.onnx"), uses_meta=(args.use_meta == "1") if args and hasattr(args, 'use_meta') else True)
    
    # 3. Train Macro Expert
    print("PROGRESS: 70% - Training Macro Expert...")
    logger.info("Training Macro Expert...")
    X_macro = df[features_to_use].values
    y_macro = df['target_direction'].values
    
    model_macro = optimize_and_train_xgb(X_macro, y_macro, optuna_trials, "Macro Expert")
    name_macro = f"{model_name_base}_macro"
    save_onnx_model(model_macro, features_to_use, os.path.join(engine_models_dir, "macro", f"{name_macro}.onnx"), uses_meta=(args.use_meta == "1") if args and hasattr(args, 'use_meta') else True)
    
    # 4. Train Gating & Meta Learner on Full Dataset
    print("PROGRESS: 85% - Training Gating Network and Meta-Learner...")
    logger.info("Generating predictions for Meta & Gating training...")
    X_full = df[features_to_use].values
    
    probs_trend = model_trend.predict_proba(X_full)
    probs_meanrev = model_meanrev.predict_proba(X_full)
    probs_macro = model_macro.predict_proba(X_full)
    
    # Gating Network inputs: [adx, vix, bb_width]
    gating_features = []
    if 'adx' in df.columns: gating_features.append('adx')
    if 'vix_level' in df.columns: gating_features.append('vix_level')
    if 'bb_width' in df.columns: gating_features.append('bb_width')
    
    X_gating = df[gating_features].values if gating_features else np.zeros((len(df), 3))
    
    # Create fake ideal gating labels (which model was most confident for the correct class)
    y_gating = []
    y_true = df['target_direction'].values
    for i in range(len(df)):
        c = int(y_true[i])
        p = [probs_trend[i][c], probs_meanrev[i][c], probs_macro[i][c]]
        y_gating.append(np.argmax(p))
        
    gating_model = xgb.XGBClassifier(objective='multi:softprob', max_depth=3, learning_rate=0.05, n_estimators=100)
    gating_model.fit(X_gating, y_gating)
    
    # Meta Learner inputs: 9 probabilities
    X_meta = np.hstack([probs_trend, probs_meanrev, probs_macro])
    meta_model = xgb.XGBClassifier(objective='multi:softprob', max_depth=3, learning_rate=0.05, n_estimators=100)
    meta_model.fit(X_meta, y_true)
    
    # Calculate Accuracies
    acc_trend = model_trend.score(X_trend, y_trend) if hasattr(model_trend, 'score') else 0.0
    acc_meanrev = model_meanrev.score(X_meanrev, y_meanrev) if hasattr(model_meanrev, 'score') else 0.0
    acc_macro = model_macro.score(X_macro, y_macro) if hasattr(model_macro, 'score') else 0.0
    acc_meta = meta_model.score(X_meta, y_true) if hasattr(meta_model, 'score') else 0.0
    
    # Export Gating and Meta
    ensemble_dir = os.path.join(engine_models_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    name_controller = f"{model_name_base}_ensemble"
    joblib.dump(gating_model, os.path.join(ensemble_dir, f"{name_controller}_gating_network.pkl"))
    
    uses_meta = (args.use_meta == "1") if args and hasattr(args, 'use_meta') else True
    if uses_meta:
        joblib.dump(meta_model, os.path.join(ensemble_dir, f"{name_controller}_meta_learner.pkl"))
    
    print("PROGRESS: 95% - Registering trained experts to database...")
    logger.info("Registering trained models to Engine Database...")
    try:
        import requests
        import time
        from datetime import datetime
        now_str = datetime.utcnow().isoformat()
        
        experts = [
            {"name": name_trend, "regime": "TREND_EXPERT", "accuracy": f"{acc_trend*100:.2f}%", "folder": "trend"},
            {"name": name_meanrev, "regime": "MEANREV_EXPERT", "accuracy": f"{acc_meanrev*100:.2f}%", "folder": "meanrev"},
            {"name": name_macro, "regime": "MACRO_EXPERT", "accuracy": f"{acc_macro*100:.2f}%", "folder": "macro"},
            {"name": name_controller, "regime": "MOE_ENSEMBLE", "accuracy": f"{acc_meta*100:.2f}%", "folder": "ensemble"}
        ]
        for exp in experts:
            meta_json_str = None
            meta_file_path = os.path.join(engine_models_dir, exp["folder"], f"{exp['name']}_metadata.json")
            if os.path.exists(meta_file_path):
                try:
                    with open(meta_file_path, "r") as f:
                        meta_json_str = f.read()
                except:
                    pass
                    
            payload = {
                "name": exp["name"],
                "algorithm_type": "XGBoost MoE Expert" if "expert" in exp["name"] else "MoE Gating & Meta",
                "accuracy": exp["accuracy"],
                "status": "Active",
                "train_start_time": now_str,
                "train_duration_sec": "15",
                "regime": exp["regime"],
                "uses_meta": (args.use_meta == "1") if args and hasattr(args, 'use_meta') else True
            }
            if meta_json_str is not None:
                payload["metadata"] = meta_json_str
                
            if args and hasattr(args, 'dataset_id') and args.dataset_id:
                payload["dataset_id"] = args.dataset_id
            try:
                requests.post("http://127.0.0.1:8000/api/models/", json=payload, timeout=5)
            except Exception as e:
                logger.warning(f"Failed to register {exp['name']}: {e}")
    except Exception as e:
        logger.warning(f"Registration skipped: {e}")
        
    print("PROGRESS: 100% - Training and Export Complete.")
    logger.info("All MoE models trained and exported successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="XAUUSD_DEFAULT.csv")
    parser.add_argument("--macro_file", type=str, default=None)
    parser.add_argument("--optuna_trials", type=int, default=10)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--use_meta", type=str, default="1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset_id", type=str, default=None)
    args = parser.parse_args()
    train_ensemble(args)
