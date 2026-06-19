import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_correlation_analysis(csv_path="c:/code/quant-v1/quant-train-v1/dataset/XAUUSD_H1_features.csv"):
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return
        
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # Exclude non-feature columns
    exclude_cols = ['time', 'regime', 'future_return', 'target', 'sl_pips', 'tp_pips', 'atr_14', 'close_lag_1']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nAnalyzing {len(features)} features...")
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr().abs()
    
    # Find highly correlated pairs
    threshold = 0.90
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                high_corr_pairs.append((col1, col2, corr_val))
                
    # Sort by highest correlation
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n=== HIGHLY CORRELATED PAIRS (Threshold > {threshold}) ===")
    for col1, col2, corr_val in high_corr_pairs:
        print(f"{col1} <--> {col2}: {corr_val:.4f}")
        
    # Generate Heatmap
    plt.figure(figsize=(24, 20))
    sns.heatmap(df[features].corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    os.makedirs('c:/code/quant-v1/quant-train-v1/analysis', exist_ok=True)
    heatmap_path = 'c:/code/quant-v1/quant-train-v1/analysis/feature_correlation_heatmap.png'
    plt.savefig(heatmap_path)
    print(f"\nSaved correlation heatmap to {heatmap_path}")
    
    # Check for dead features
    print("\n=== DEAD OR CONSTANT FEATURES ===")
    for col in features:
        if df[col].nunique() <= 1:
            print(f"DEAD FEATURE: {col} (Unique values: {df[col].nunique()})")
            
if __name__ == "__main__":
    run_correlation_analysis()
