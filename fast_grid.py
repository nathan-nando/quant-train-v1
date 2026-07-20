
import pandas as pd
import numpy as np
import onnxruntime as ort
import sys
import os

def main():
    print('Loading Data...', flush=True)
    df_tech = pd.read_csv('dataset/SIMULATION_LATEST.csv')
    df_tech['time'] = pd.to_datetime(df_tech['time'])
    df_macro = pd.read_csv('dataset/SIMULATION_LATEST_MACRO.csv')
    df_macro['time'] = pd.to_datetime(df_macro['time'])

    cols_to_use = df_macro.columns.difference(df_tech.columns).tolist()
    cols_to_use.append('time')
    df = pd.merge(df_tech, df_macro[cols_to_use], on='time', how='left')

    df['vix_zscore_20d'] = (df['vix'] - df['vix'].rolling(20).mean()) / df['vix'].rolling(20).std().replace(0, 1)
    df = df.dropna().copy()

    trend_feats = ['adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50', 'hour_sin', 'hour_cos', 'atr_pct', 'volume', 'dist_ema_200', 'ema_cross_9_21', 'obv']
    meanrev_feats = ['adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50', 'hour_sin', 'hour_cos', 'atr_pct', 'volume', 'cci', 'bb_position', 'ema_cross_21_50']
    macro_feats = ['adx', 'rsi_14', 'macd_diff', 'bb_width', 'dist_ema_50', 'hour_sin', 'hour_cos', 'atr_pct', 'volume', 'dxym_dist_ema_50', 'usoilm_dist_ema_50', 'tips_10y_chg_5d', 'fed_rate_level', 'vix_zscore_20d']

    print('Loading models...', flush=True)
    t_sess = ort.InferenceSession('../quant-engine-v1/ml_models/trend/v2_trend.onnx')
    mr_sess = ort.InferenceSession('../quant-engine-v1/ml_models/meanrev/v2_meanrev.onnx')
    ma_sess = ort.InferenceSession('../quant-engine-v1/ml_models/macro/v2_macro.onnx')

    t_meta = ort.InferenceSession('../quant-engine-v1/ml_models/trend/v2_trend_meta.onnx')
    mr_meta = ort.InferenceSession('../quant-engine-v1/ml_models/meanrev/v2_meanrev_meta.onnx')
    ma_meta = ort.InferenceSession('../quant-engine-v1/ml_models/macro/v2_macro_meta.onnx')

    print('Inference Base...', flush=True)
    X_t = df[trend_feats].astype(np.float32).values
    X_mr = df[meanrev_feats].astype(np.float32).values
    X_ma = df[macro_feats].astype(np.float32).values

    p_t = t_sess.run(None, {'float_input': X_t})[1]
    p_mr = mr_sess.run(None, {'float_input': X_mr})[1]
    p_ma = ma_sess.run(None, {'float_input': X_ma})[1]
    
    def to_prob_array(probs):
        if isinstance(probs, np.ndarray):
            return probs
        res = np.zeros((len(probs), 3))
        for i, p in enumerate(probs):
            res[i, 0] = p.get(0, 0.0)
            res[i, 1] = p.get(1, 0.0)
            res[i, 2] = p.get(2, 0.0)
        return res
        
    p_t = to_prob_array(p_t)
    p_mr = to_prob_array(p_mr)
    p_ma = to_prob_array(p_ma)

    print('Inference Meta...', flush=True)
    conf_t = np.max(p_t, axis=1).reshape(-1, 1).astype(np.float32)
    conf_mr = np.max(p_mr, axis=1).reshape(-1, 1).astype(np.float32)
    conf_ma = np.max(p_ma, axis=1).reshape(-1, 1).astype(np.float32)
    
    X_meta_t = np.hstack([X_t, conf_t])
    X_meta_mr = np.hstack([X_mr, conf_mr])
    X_meta_ma = np.hstack([X_ma, conf_ma])

    meta_t = to_prob_array(t_meta.run(None, {'float_input': X_meta_t})[1])
    meta_mr = to_prob_array(mr_meta.run(None, {'float_input': X_meta_mr})[1])
    meta_ma = to_prob_array(ma_meta.run(None, {'float_input': X_meta_ma})[1])

    vix = df['vix_zscore_20d'].values
    adx = df['adx'].values

    expert_idx = np.zeros(len(df), dtype=int)
    expert_idx[adx > 26] = 0
    expert_idx[adx <= 26] = 1
    expert_idx[vix > 23] = 2

    # In our MoE predict logic, the meta output is NOT an array of 3 probabilities!
    # Wait, meta learner predicts Win/Loss! It's a BINARY classifier!
    # So meta_t output is [Prob(Loss), Prob(Win)] !!
    # If the meta output is Win/Loss probability, then what is final_p?
    # The actual AlphaEngine logic is:
    # return { 'direction': predicted_class, 'confidence': meta_prob_win, 'meta_probs': base_probs }
    # So the Margin is calculated from meta_probs (base probabilities), and Confidence is meta_prob_win!
    
    final_p = np.zeros_like(p_t)
    final_conf = np.zeros(len(df))
    
    # meta_t[:, 1] is Probability of WIN!
    # And final_p is the base probability array for Margin calculation
    for i in range(len(df)):
        e = expert_idx[i]
        if e == 0:
            final_p[i] = p_t[i]
            final_conf[i] = meta_t[i, 1] if meta_t.shape[1] > 1 else meta_t[i, 0]
        elif e == 1:
            final_p[i] = p_mr[i]
            final_conf[i] = meta_mr[i, 1] if meta_mr.shape[1] > 1 else meta_mr[i, 0]
        else:
            final_p[i] = p_ma[i]
            final_conf[i] = meta_ma[i, 1] if meta_ma.shape[1] > 1 else meta_ma[i, 0]

    margins = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    confs = [0.35, 0.4, 0.45, 0.5]
    
    close = df['close'].values
    
    print('Starting Grid Search...', flush=True)
    for margin in margins:
        for conf in confs:
            balance = 1000.0
            peak = balance
            max_dd = 0.0
            trades = 0
            
            valid_conf = final_conf >= conf
            
            sorted_p = np.sort(final_p, axis=1)
            margin_arr = sorted_p[:, 2] - sorted_p[:, 1]
            valid_margin = margin_arr >= margin
            
            # The direction is the argmax of base probs
            direction = np.argmax(final_p, axis=1)
            
            buy_signal = (direction == 2) & valid_conf & valid_margin
            sell_signal = (direction == 0) & valid_conf & valid_margin
            
            future_ret = np.zeros(len(df))
            future_ret[:-5] = close[5:] / close[:-5] - 1
            
            buy_idx = np.where(buy_signal)[0]
            sell_idx = np.where(sell_signal)[0]
            
            if len(buy_idx) == 0 and len(sell_idx) == 0:
                print(f'M:{margin:.2f} | C:{conf:.2f} => Trades: 0')
                continue
                
            for i in range(len(df)):
                if buy_signal[i]:
                    trades += 1
                    ret = future_ret[i]
                    pnl = balance * 0.01 * (ret / 0.0035) 
                    balance += pnl
                elif sell_signal[i]:
                    trades += 1
                    ret = -future_ret[i]
                    pnl = balance * 0.01 * (ret / 0.0035)
                    balance += pnl
                
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak
                if dd > max_dd:
                    max_dd = dd
            
            wr = 0
            if trades > 0:
                wins = np.sum(future_ret[buy_idx] > 0) + np.sum(future_ret[sell_idx] < 0)
                wr = wins / trades * 100.0
                
            pnl_pct = (balance - 1000.0) / 10.0
            print(f'M:{margin:.2f} | C:{conf:.2f} => Trades: {trades}, WR: {wr:.1f}%, DD: {max_dd*100:.1f}%, PnL: {pnl_pct:.1f}%', flush=True)

if __name__ == '__main__':
    main()

