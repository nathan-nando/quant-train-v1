import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

class FactorEvaluator:
    """
    Evaluates alpha factors using Information Coefficient (IC).
    IC is the rank correlation (Spearman) between the factor value at time t 
    and the forward return from time t to t+n.
    """
    
    @staticmethod
    def compute_forward_returns(df: pd.DataFrame, periods: list = [1, 5, 15, 60], price_col: str = 'close') -> pd.DataFrame:
        """
        Compute forward returns for multiple periods.
        """
        returns_df = pd.DataFrame(index=df.index)
        for p in periods:
            # Future return = (Price[t+p] - Price[t]) / Price[t]
            returns_df[f'fwd_ret_{p}'] = df[price_col].shift(-p) / df[price_col] - 1
            
        return returns_df

    @staticmethod
    def calculate_ic(df_features: pd.DataFrame, df_returns: pd.DataFrame, factor_cols: list) -> pd.DataFrame:
        """
        Calculate Spearman Rank Correlation (IC) for each factor against each forward return horizon.
        """
        ic_results = []
        
        # Align indices
        df_merged = pd.concat([df_features[factor_cols], df_returns], axis=1).dropna()
        
        for factor in factor_cols:
            for ret_col in df_returns.columns:
                ic, p_val = spearmanr(df_merged[factor], df_merged[ret_col])
                ic_results.append({
                    'factor': factor,
                    'horizon': ret_col,
                    'ic': ic,
                    'p_value': p_val,
                    'is_significant': p_val < 0.05
                })
                
        return pd.DataFrame(ic_results)

    @staticmethod
    def calculate_rolling_ic(df_features: pd.DataFrame, df_returns: pd.DataFrame, factor: str, ret_col: str, window: int = 500) -> pd.Series:
        """
        Calculate rolling IC to see factor stability over time.
        """
        df_merged = pd.concat([df_features[factor], df_returns[ret_col]], axis=1).dropna()
        
        def _rolling_spearman(df_chunk):
            if len(df_chunk) < 10:
                return np.nan
            return spearmanr(df_chunk.iloc[:, 0], df_chunk.iloc[:, 1])[0]
            
        # Using a rolling window apply is slow but correct for rank correlation
        rolling_ic = df_merged.rolling(window).apply(
            lambda x: spearmanr(df_merged.loc[x.index, factor], df_merged.loc[x.index, ret_col])[0] if len(x) >= 10 else np.nan, 
            raw=False
        )
        return rolling_ic.iloc[:, 0]
        
    @staticmethod
    def evaluate_factor(df: pd.DataFrame, factor_col: str, price_col: str = 'close') -> dict:
        """
        Comprehensive factor evaluation returning metrics like Mean IC, IC Std, and IC/IR (Information Ratio).
        """
        returns = FactorEvaluator.compute_forward_returns(df, periods=[1, 5, 15, 60], price_col=price_col)
        ic_df = FactorEvaluator.calculate_ic(df, returns, [factor_col])
        
        # Summarize
        summary = {}
        for _, row in ic_df.iterrows():
            horizon = row['horizon']
            summary[f'ic_{horizon}'] = row['ic']
            summary[f'pval_{horizon}'] = row['p_value']
            
        return summary
