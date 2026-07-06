import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not installed")

def check_distributions(df):
    print("=== Q1: TARGET DISTRIBUTION ANALYSIS ===")
    future_ret = df['close'].shift(-5) / df['close'] - 1
    df['future_return'] = future_ret
    
    thresholds = [0.0015, 0.0020, 0.0025, 0.0035, 0.0050]
    for thresh in thresholds:
        target = np.ones(len(df)) # 1 = NEUTRAL
        target[future_ret > thresh] = 2 # BUY
        target[future_ret < -thresh] = 0 # SELL
        
        valid_target = pd.Series(target).dropna()
        counts = valid_target.value_counts()
        pcts = valid_target.value_counts(normalize=True) * 100
        print(f"\nThreshold: {thresh*100:.2f}% (e.g. ~{thresh*2000:.1f} pips on $2000 Gold)")
        for cls_idx, cls_name in [(0, "SELL"), (1, "NEUTRAL"), (2, "BUY")]:
            cnt = counts.get(cls_idx, 0)
            pct = pcts.get(cls_idx, 0.0)
            print(f"  {cls_name} (Class {cls_idx}): {cnt:6d} samples ({pct:5.2f}%)")

def check_hmm_bic(df):
    if not HMM_AVAILABLE:
        return
    print("\n=== Q3: HMM BIC STATE SELECTION ANALYSIS ===")
    if 'return_1' not in df.columns:
        df['return_1'] = df['close'].pct_change(1) * 100
    if 'return_5' not in df.columns:
        df['return_5'] = df['close'].pct_change(5) * 100
    df['realized_vol_20'] = df['return_1'].rolling(20).std()
    
    features = ['return_1', 'return_5', 'realized_vol_20']
    if 'adx' in df.columns: features.append('adx')
    if 'volume_ratio' in df.columns: features.append('volume_ratio')
    if 'vix_level' in df.columns: features.append('vix_level')
    
    df_clean = df[features].dropna()
    X = df_clean.values
    print(f"Fitting HMM on {len(X)} samples using features: {features}")
    
    results = []
    for n in range(2, 6):
        t0 = time.time()
        model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X)
        log_prob = model.score(X) * len(X) # score returns log likelihood per sample
        n_features = X.shape[1]
        n_params = n * (n - 1) + n * n_features + n * n_features * (n_features + 1) / 2
        bic = -2 * log_prob + n_params * np.log(len(X))
        aic = -2 * log_prob + 2 * n_params
        elapsed = time.time() - t0
        print(f"  States: {n} | Log-Likelihood: {log_prob:12.2f} | BIC: {bic:12.2f} | AIC: {aic:12.2f} | Converged: {model.monitor_.converged} ({elapsed:.1f}s)")
        results.append((n, bic, log_prob))
        
    best_n = min(results, key=lambda x: x[1])[0]
    print(f"\n=> Optimal number of states according to BIC: {best_n}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tech_path = os.path.join(base_dir, "dataset", "XAUUSD_TECHNICAL.csv")
    if os.path.exists(tech_path):
        print(f"Loading {tech_path}...")
        df = pd.read_csv(tech_path)
        check_distributions(df)
        check_hmm_bic(df)
    else:
        print(f"File not found: {tech_path}")
