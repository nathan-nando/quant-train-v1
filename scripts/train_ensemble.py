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
    df['future_return'] = future_ret
    return df


def get_balanced_weights(y, max_weight=10.0):
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight(class_weight='balanced', y=y)
    return np.clip(sw, 0.1, max_weight)

def optimize_and_train_xgb(X, y, n_trials=10, prefix=""):
    num_classes = len(np.unique(y))
    obj = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
    if n_trials <= 0:
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, objective=obj, tree_method='hist')
        model.fit(X, y)
        return model

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'objective': obj,
            'tree_method': 'hist' # Prevent bad allocation with large weights
        }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        sw_train = get_balanced_weights(y_train)
        sw_val = get_balanced_weights(y_val)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sw_train)
        
        # Use negative unweighted log_loss so Optuna balances raw accuracy
        from sklearn.metrics import log_loss
        y_prob = model.predict_proba(X_val)
        return -log_loss(y_val, y_prob)
        
    print(f"Running Optuna tuning for {prefix} ({n_trials} trials)...", flush=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[{prefix}] Best params: {study.best_params} | Best Score: {study.best_value:.4f}", flush=True)
    best_model = xgb.XGBClassifier(**study.best_params, objective=obj, tree_method='hist')
    best_model.fit(X, y)
    return best_model

def save_onnx_model(xgb_model, feature_names, save_path, uses_meta=False, X_train=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)
    
    with open(save_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    # Compute baseline stats from training data
    baseline_stats = {}
    if X_train is not None:
        import numpy as np
        for i, feat in enumerate(feature_names):
            try:
                col = X_train[:, i] if isinstance(X_train, np.ndarray) else X_train[feat].values
                baseline_stats[feat] = {
                    "mean": float(np.nanmean(col)),
                    "std": float(max(np.nanstd(col), 1e-8))  # floor to prevent div/0
                }
            except Exception:
                pass

    # Save metadata
    meta = {
        "features": feature_names,
        "classes": {0: "SELL", 1: "NEUTRAL", 2: "BUY"},
        "uses_meta": uses_meta,
        "baseline_stats": baseline_stats
    }
    meta_path = save_path.replace(".onnx", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

def train_ensemble(args=None):
    logger.info("Starting MoE Ensemble Training Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = args.dataset_file if args and hasattr(args, 'dataset_file') and args.dataset_file else "XAUUSD_TECHNICAL.csv"
    macro_file = args.macro_file if args and hasattr(args, 'macro_file') and args.macro_file else "XAUUSD_MACRO.csv"
    
    print("\n================ TRAINING CONFIGURATION ================", flush=True)
    print(f"Base Dataset : {dataset_file}", flush=True)
    if macro_file:
        print(f"Macro Dataset: {macro_file}", flush=True)
    optuna_trials = args.optuna_trials if args and hasattr(args, "optuna_trials") else 0
    print(f"Model Engine : XGBoost (Optuna Trials: {optuna_trials})", flush=True)
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
            
    # On-the-fly feature calculation for stationary & missing features (Tasks 5 & 6)
    if 'return_1' not in df.columns: df['return_1'] = df['close'].pct_change(1) * 100
    if 'return_24h' not in df.columns: df['return_24h'] = df['close'].pct_change(24) * 100
    if 'realized_vol_20' not in df.columns: df['realized_vol_20'] = df['return_1'].rolling(20).std()
    if 'session_id' not in df.columns:
        time_s = pd.to_datetime(df.index)
        df['session_id'] = pd.cut(time_s.hour, bins=[0, 8, 13, 21, 24], labels=[0, 1, 2, 3], right=False, include_lowest=True).astype(float)
    if 'tips_10y' in df.columns and 'tips_zscore_20d' not in df.columns:
        df['tips_zscore_20d'] = (df['tips_10y'] - df['tips_10y'].rolling(20).mean()) / df['tips_10y'].rolling(20).std().replace(0, 1)
    if 'fed_rate' in df.columns and 'fed_rate_change_20d' not in df.columns:
        df['fed_rate_change_20d'] = df['fed_rate'].diff(20)
    if 'vix' in df.columns and 'vix_zscore_20d' not in df.columns:
        df['vix_zscore_20d'] = (df['vix'] - df['vix'].rolling(20).mean()) / df['vix'].rolling(20).std().replace(0, 1)
        
    df = df.dropna().copy()
    df = create_target(df)
    
    # 1. Core Features (Golden Era + H1 Liquidity/Time Filters)
    core_features = [
        'adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50',
        'hour_sin', 'hour_cos', 'atr_pct', 'volume'
    ]
    
    # 2. Trend Expert (H1 Swing & Momentum Breakout)
    trend_cols = core_features + [
        'dist_ema_200', 'ema_cross_9_21', 'obv'
    ]
    
    # 3. MeanRev Expert (H1 Boundaries & Reversals)
    meanrev_cols = core_features + [
        'cci', 'bb_position', 'ema_cross_21_50'
    ]
    
    # 4. Macro Expert (Intermarket Context for H1)
    macro_cols = core_features + [
        'dxym_dist_ema_50', 'usoilm_dist_ema_50', 'tips_10y_chg_5d', 'fed_rate_level', 'vix_zscore_20d'
    ]
    
    trend_features = [f for f in trend_cols if f in df.columns]
    meanrev_features = [f for f in meanrev_cols if f in df.columns]
    macro_features = [f for f in macro_cols if f in df.columns]
    
    all_features_to_use = sorted(list(set(trend_features + meanrev_features + macro_features)))
    print(f"Specialized MoE Features: Trend={len(trend_features)}, MeanRev={len(meanrev_features)}, Macro={len(macro_features)} (Total Unique={len(all_features_to_use)})", flush=True)
    
    # 1. Train Trend Expert (ADX > 25)
    print("PROGRESS: 25% - Training Trend Expert (ADX > 25)...")
    logger.info("Training Trend Expert...")
    df_trend = df[df['adx'] > 25].copy() if 'adx' in df.columns else df.copy()
    X_trend = df_trend[trend_features].values
    y_trend = df_trend['target_direction'].values
    
    model_trend = optimize_and_train_xgb(X_trend, y_trend, optuna_trials, "Trend Expert")
    name_trend = f"{model_name_base}_trend"
    uses_meta = (args.use_meta == "1") if args and hasattr(args, 'use_meta') else True
    save_onnx_model(model_trend, trend_features, os.path.join(engine_models_dir, "trend", f"{name_trend}.onnx"), uses_meta=uses_meta, X_train=X_trend)
    
    # 2. Train MeanRev Expert (ADX <= 25)
    print("PROGRESS: 50% - Training MeanRev Expert (ADX <= 25)...")
    logger.info("Training MeanRev Expert...")
    df_meanrev = df[df['adx'] <= 25].copy() if 'adx' in df.columns else df.copy()
    X_meanrev = df_meanrev[meanrev_features].values
    y_meanrev = df_meanrev['target_direction'].values
    
    model_meanrev = optimize_and_train_xgb(X_meanrev, y_meanrev, optuna_trials, "MeanRev Expert")
    name_meanrev = f"{model_name_base}_meanrev"
    save_onnx_model(model_meanrev, meanrev_features, os.path.join(engine_models_dir, "meanrev", f"{name_meanrev}.onnx"), uses_meta=uses_meta, X_train=X_meanrev)
    
    # 3. Train Macro Expert
    print("PROGRESS: 70% - Training Macro Expert...")
    logger.info("Training Macro Expert...")
    X_macro = df[macro_features].values
    y_macro = df['target_direction'].values
    
    model_macro = optimize_and_train_xgb(X_macro, y_macro, optuna_trials, "Macro Expert")
    name_macro = f"{model_name_base}_macro"
    save_onnx_model(model_macro, macro_features, os.path.join(engine_models_dir, "macro", f"{name_macro}.onnx"), uses_meta=uses_meta, X_train=X_macro)
    
    # 4. Train Gating & Meta Learner on Full Dataset
    print("PROGRESS: 85% - Generating OOF Predictions & Training Gating/Meta...")
    logger.info("Generating OOF predictions for Meta & Gating training to prevent data leakage...")
    
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5, gap=5)
    oof_mask = np.zeros(len(df), dtype=bool)
    
    oof_probs_trend = np.full((len(df), 3), 0.33)
    oof_probs_meanrev = np.full((len(df), 3), 0.33)
    oof_probs_macro = np.full((len(df), 3), 0.33)
    
    adx_val = df['adx'].values if 'adx' in df.columns else np.full(len(df), 30)
    y_true = df['target_direction'].values

    X_trend_full = df[trend_features].values
    X_meanrev_full = df[meanrev_features].values
    X_macro_full = df[macro_features].values

    for train_idx, val_idx in tscv.split(df):
        oof_mask[val_idx] = True
        tr_trend_idx = train_idx[adx_val[train_idx] > 25]
        if len(tr_trend_idx) > 0:
            m_t = optimize_and_train_xgb(X_trend_full[tr_trend_idx], y_true[tr_trend_idx], 0, "OOF Trend")
            oof_probs_trend[val_idx] = m_t.predict_proba(X_trend_full[val_idx])
            
        tr_meanrev_idx = train_idx[adx_val[train_idx] <= 25]
        if len(tr_meanrev_idx) > 0:
            m_m = optimize_and_train_xgb(X_meanrev_full[tr_meanrev_idx], y_true[tr_meanrev_idx], 0, "OOF MeanRev")
            oof_probs_meanrev[val_idx] = m_m.predict_proba(X_meanrev_full[val_idx])
            
        m_ma = optimize_and_train_xgb(X_macro_full[train_idx], y_true[train_idx], 0, "OOF Macro")
        oof_probs_macro[val_idx] = m_ma.predict_proba(X_macro_full[val_idx])
    
    # Gating Network inputs
    gating_features = []
    if 'adx' in df.columns: gating_features.append('adx')
    if 'vix_level' in df.columns: gating_features.append('vix_level')
    if 'bb_width' in df.columns: gating_features.append('bb_width')
    X_gating_base = df[gating_features].values if gating_features else np.zeros((len(df), 3))
    
    hmm_features = np.full((len(df), 4), 0.25)
    hmm_path = os.path.join(engine_models_dir, f"{model_name_base}_hmm.pkl")
    if not os.path.exists(hmm_path): hmm_path = os.path.join(engine_models_dir, "base_hmm.pkl")
    if os.path.exists(hmm_path):
        try:
            hmm_data = joblib.load(hmm_path)
            hmm_model = hmm_data['model']
            hmm_scaler = hmm_data.get('scaler', None)
            hmm_feat_names = hmm_data['features']
            X_hmm = df[hmm_feat_names].values
            if hmm_scaler is not None:
                X_hmm = hmm_scaler.transform(X_hmm)
            
            hmm_probs_list = []
            for i in range(len(X_hmm)):
                start_idx = max(0, i - 9)
                X_seq = X_hmm[start_idx:i+1]
                probs_seq = hmm_model.predict_proba(X_seq)
                hmm_probs_list.append(probs_seq[-1])
            hmm_features = np.array(hmm_probs_list)
        except Exception as e:
            logger.warning(f"Failed to generate HMM features for gating: {e}")
            
    X_gating = np.hstack([hmm_features, X_gating_base])
    
    # Create domain-specialized accuracy gating labels
    # Label = expert that assigned highest probability to the actual true outcome, with domain priors
    y_gating = []
    adx_arr = df['adx'].values if 'adx' in df.columns else np.full(len(df), 20.0)
    vix_arr = df['vix_level'].values if 'vix_level' in df.columns else np.full(len(df), 15.0)
    
    for i in range(len(df)):
        true_lbl = int(y_true[i])
        p_trend = oof_probs_trend[i][true_lbl]
        p_meanrev = oof_probs_meanrev[i][true_lbl]
        p_macro = oof_probs_macro[i][true_lbl]
        
        # Clean domain regime labeling without HMM state distortion:
        # 1. Volatility Crisis / High VIX -> Macro Expert (2)
        # 2. Strong Trending market (ADX > 26) -> Trend Expert (0)
        # 3. Ranging / Consolidation market (ADX < 22) -> MeanRev Expert (1)
        # 4. Transition zone (22 <= ADX <= 26) -> Best predicting expert
        if vix_arr[i] > 23.0:
            best_exp = 2 # Macro Expert
        elif adx_arr[i] > 26.0:
            best_exp = 0 # Trend Expert
        elif adx_arr[i] < 22.0:
            best_exp = 1 # MeanRev Expert
        else:
            best_exp = int(np.argmax([p_trend, p_meanrev, p_macro]))
        y_gating.append(best_exp)
        
    y_gating_arr = np.array(y_gating)
    gating_model = xgb.XGBClassifier(objective='multi:softprob', max_depth=3, learning_rate=0.05, n_estimators=100, tree_method='hist')
    sw_gating = get_balanced_weights(y_gating_arr[oof_mask])
    gating_model.fit(X_gating[oof_mask], y_gating_arr[oof_mask], sample_weight=sw_gating)
    
    if uses_meta:
        print("PROGRESS: 90% - Training Meta Learners for each Expert...")
        
        def train_and_save_meta(oof_probs, expert_name, save_folder, save_name, feat_list, X_feat_full):
            conf = np.max(oof_probs, axis=1)
            pred_classes = np.argmax(oof_probs, axis=1)
            
            # CRITICAL FIX: Only train Meta-Learner on NON-NEUTRAL predictions (BUY or SELL) from strictly OOF slices.
            active_idx = np.where((pred_classes != 1) & oof_mask)[0]
            
            if len(active_idx) < 10:
                logger.warning(f"Not enough active trades to train Meta Learner for {expert_name}. Falling back to all OOF rows.")
                active_idx = np.where(oof_mask)[0]
                
            X_meta_active = np.hstack([X_feat_full[active_idx], conf[active_idx].reshape(-1, 1)])
            y_meta_active = (pred_classes[active_idx] == y_true[active_idx]).astype(int)
            meta_features_names = feat_list + ["primary_confidence"]
            
            meta_model = optimize_and_train_xgb(X_meta_active, y_meta_active, 0, f"Meta {expert_name}")
            save_onnx_model(meta_model, meta_features_names, os.path.join(engine_models_dir, save_folder, f"{save_name}_meta.onnx"), uses_meta=False, X_train=X_meta_active)

        train_and_save_meta(oof_probs_trend, "Trend", "trend", name_trend, trend_features, X_trend_full)
        train_and_save_meta(oof_probs_meanrev, "MeanRev", "meanrev", name_meanrev, meanrev_features, X_meanrev_full)
        train_and_save_meta(oof_probs_macro, "Macro", "macro", name_macro, macro_features, X_macro_full)
        
        # Train Ensemble Meta-Learner (Combiner)
        print("PROGRESS: 92% - Training Ensemble Meta-Learner...")
        X_meta_ensemble = np.hstack([oof_probs_trend, oof_probs_meanrev, oof_probs_macro])
        ensemble_meta_model = LogisticRegression(max_iter=1000)
        ensemble_meta_model.fit(X_meta_ensemble[oof_mask], y_true[oof_mask])
        
        ensemble_dir = os.path.join(engine_models_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        name_controller = f"{model_name_base}_ensemble"
        joblib.dump(ensemble_meta_model, os.path.join(ensemble_dir, f"{name_controller}_meta_learner.pkl"))
    
    # Calculate Accuracies (using domain regime subset on OOF predictions for realistic metrics)
    from sklearn.metrics import accuracy_score
    trend_mask = oof_mask & (adx_val > 25)
    meanrev_mask = oof_mask & (adx_val <= 25)
    
    acc_trend = accuracy_score(y_true[trend_mask], np.argmax(oof_probs_trend[trend_mask], axis=1)) if np.sum(trend_mask) > 0 else 0.50
    acc_meanrev = accuracy_score(y_true[meanrev_mask], np.argmax(oof_probs_meanrev[meanrev_mask], axis=1)) if np.sum(meanrev_mask) > 0 else 0.50
    acc_macro = accuracy_score(y_true[oof_mask], np.argmax(oof_probs_macro[oof_mask], axis=1))
    acc_ensemble = accuracy_score(y_gating_arr[oof_mask], gating_model.predict(X_gating[oof_mask]))
    
    # Export Gating Network
    ensemble_dir = os.path.join(engine_models_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    name_controller = f"{model_name_base}_ensemble"
    joblib.dump(gating_model, os.path.join(ensemble_dir, f"{name_controller}_gating_network.pkl"))
    
    print("PROGRESS: 95% - Registering trained experts to database...")
    logger.info("Registering trained models to Engine Database...")
    try:
        import requests
        from datetime import datetime
        now_str = datetime.utcnow().isoformat()
        
        experts = [
            {"name": name_trend, "regime": "TREND_EXPERT", "accuracy": f"{acc_trend*100:.2f}%", "folder": "trend"},
            {"name": name_meanrev, "regime": "MEANREV_EXPERT", "accuracy": f"{acc_meanrev*100:.2f}%", "folder": "meanrev"},
            {"name": name_macro, "regime": "MACRO_EXPERT", "accuracy": f"{acc_macro*100:.2f}%", "folder": "macro"},
            {"name": name_controller, "regime": "MOE_ENSEMBLE", "accuracy": f"{acc_ensemble*100:.2f}%", "folder": "ensemble"}
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
                "algorithm_type": "XGBoost MoE Expert" if "expert" in exp["name"] else "MoE Gating Network",
                "accuracy": exp["accuracy"],
                "status": "Active",
                "train_start_time": now_str,
                "train_duration_sec": "15",
                "regime": exp["regime"],
                "uses_meta": uses_meta
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
    parser.add_argument("--dataset_file", type=str, default="XAUUSD_TECHNICAL.csv")
    parser.add_argument("--macro_file", type=str, default="XAUUSD_MACRO.csv")
    parser.add_argument("--optuna_trials", type=int, default=10)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--use_meta", type=str, default="1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset_id", type=str, default=None)
    args = parser.parse_args()
    train_ensemble(args)
