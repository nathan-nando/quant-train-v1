
import numpy as np
import joblib

meta = joblib.load('../quant-engine-v1/ml_models/ensemble/xgboost_ensemble_meta_learner.pkl')
import onnxruntime as ort
import pandas as pd

df_tech = pd.read_csv('dataset/XAUUSD_TECHNICAL.csv')
df_macro = pd.read_csv('dataset/XAUUSD_MACRO.csv')
cols_to_use = df_macro.columns.difference(df_tech.columns).tolist() + ['time']
df = pd.merge(df_tech, df_macro[cols_to_use], on='time', how='left')

trend_feats = ['adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50', 'hour_sin', 'hour_cos', 'atr_pct', 'volume', 'dist_ema_200', 'ema_cross_9_21', 'obv']
meanrev_feats = ['adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50', 'hour_sin', 'hour_cos', 'atr_pct', 'volume', 'cci', 'bb_position', 'ema_cross_21_50']
macro_feats = ['adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50', 'hour_sin', 'hour_cos', 'atr_pct', 'volume', 'dxym_dist_ema_50', 'usoilm_dist_ema_50', 'tips_10y_chg_5d', 'fed_rate_level', 'vix'] # dummy

t_sess = ort.InferenceSession('../quant-engine-v1/ml_models/trend/xgboost_trend.onnx')
mr_sess = ort.InferenceSession('../quant-engine-v1/ml_models/meanrev/xgboost_meanrev.onnx')
ma_sess = ort.InferenceSession('../quant-engine-v1/ml_models/macro/xgboost_macro.onnx')

X_t = df[trend_feats].astype(np.float32).values
X_mr = df[meanrev_feats].astype(np.float32).values
X_ma = df[macro_feats].astype(np.float32).values

p_t = t_sess.run(None, {'float_input': X_t})[1]
p_mr = mr_sess.run(None, {'float_input': X_mr})[1]
p_ma = ma_sess.run(None, {'float_input': X_ma})[1]

def to_prob(p):
    if isinstance(p, np.ndarray): return p
    res = np.zeros((len(p), 3))
    for i, d in enumerate(p):
        res[i, 0] = d.get(0, 0.0)
        res[i, 1] = d.get(1, 0.0)
        res[i, 2] = d.get(2, 0.0)
    return res

pt = to_prob(p_t)
pm = to_prob(p_mr)
pma = to_prob(p_ma)

X_meta = np.hstack([pt, pm, pma])
meta_probs = meta.predict_proba(X_meta)

sorted_p = np.sort(meta_probs, axis=1)
margins = sorted_p[:, 2] - sorted_p[:, 1]
print('Margins summary:')
print(pd.Series(margins).describe())

print('\nTotal trades for M=0.1, C=0.35:')
valid_c = np.max(meta_probs, axis=1) >= 0.35
valid_m = margins >= 0.1
print(np.sum(valid_c & valid_m))

