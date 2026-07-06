import pandas as pd
from fredapi import Fred
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "quant-engine-v1", ".env"))

# Try to get FRED_API_KEY from environment
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

DAILY_SERIES = {
    "DFII10": "tips_10y",
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",
    "DFEDTARU": "fed_rate",
    "T10YIE": "breakeven_10y",
    "DTWEXBGS": "dxy_broad",
    "VIXCLS": "vix",
}

MONTHLY_SERIES = {
    "CPIAUCSL": "cpi",
    "PAYEMS": "nfp",
    "UNRATE": "unemployment",
}

import argparse

def fetch_macro_data():
    parser = argparse.ArgumentParser(description="Fetch Macro Data from FRED")
    parser.add_argument("--input", type=str, default="XAUUSD_DEFAULT.csv", help="Input OHLCV dataset filename")
    parser.add_argument("--output", type=str, default="XAUUSD_MACRO.csv", help="Output Macro dataset filename")
    args = parser.parse_args()

    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not set. Cannot fetch macro data.")
        return

    fred = Fred(api_key=FRED_API_KEY)
    
    # Target dataset path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Handle absolute or relative paths
    if os.path.isabs(args.input):
        dataset_path = args.input
    else:
        dataset_path = os.path.join(base_dir, "dataset", args.input)
        
    if os.path.isabs(args.output):
        output_path = args.output
    else:
        output_path = os.path.join(base_dir, "dataset", args.output)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Loading base OHLCV dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    
    macro_df = pd.DataFrame(index=df.index)
    
    # 1. Fetch Daily Series
    for fred_id, name in DAILY_SERIES.items():
        logger.info(f"Fetching {name} ({fred_id})...")
        try:
            series = fred.get_series(fred_id, observation_start=start_date, observation_end=end_date)
            # Reindex to H1, forward fill
            series_reindexed = series.reindex(df.index, method='ffill')
            macro_df[name] = series_reindexed
        except Exception as e:
            logger.error(f"Failed to fetch {fred_id}: {e}")
            
    # 2. Fetch Monthly Series (with 1 month lag to prevent lookahead bias)
    for fred_id, name in MONTHLY_SERIES.items():
        logger.info(f"Fetching {name} ({fred_id})...")
        try:
            # Fetch a bit earlier to accommodate lag
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
            series = fred.get_series(fred_id, observation_start=start_date_obj.strftime('%Y-%m-%d'), observation_end=end_date)
            
            # Lag by 31 days to be safe (data is usually released next month)
            series.index = series.index + pd.Timedelta(days=31)
            
            # Reindex to H1
            series_reindexed = series.reindex(df.index, method='ffill')
            macro_df[name] = series_reindexed
        except Exception as e:
            logger.error(f"Failed to fetch {fred_id}: {e}")
            
    logger.info("Engineering derived macro features...")
    
    # Derived Features (simulating feature_store.py logic)
    if 'tips_10y' in macro_df.columns:
        macro_df['tips_10y_level'] = macro_df['tips_10y']
        macro_df['tips_10y_chg_5d'] = macro_df['tips_10y'].diff(5)
        macro_df['tips_zscore_20d'] = (macro_df['tips_10y'] - macro_df['tips_10y'].rolling(20).mean()) / macro_df['tips_10y'].rolling(20).std().replace(0, 1)

    if 'dxy_broad' in macro_df.columns:
        macro_df['dxy_broad_return_5d'] = macro_df['dxy_broad'].pct_change(5) * 100
        
    if 'vix' in macro_df.columns:
        macro_df['vix_level'] = macro_df['vix']
        macro_df['vix_chg_1d'] = macro_df['vix'].diff(1)
        macro_df['vix_zscore_20d'] = (macro_df['vix'] - macro_df['vix'].rolling(20).mean()) / macro_df['vix'].rolling(20).std().replace(0, 1)
        macro_df['vix_regime'] = pd.cut(macro_df['vix'], bins=[0, 15, 25, 40, 1000], labels=[0, 1, 2, 3], right=False).astype(float)

    if 'yield_10y' in macro_df.columns and 'yield_2y' in macro_df.columns:
        macro_df['yield_spread_10y_2y'] = macro_df['yield_10y'] - macro_df['yield_2y']
        macro_df['yield_curve_inverted'] = (macro_df['yield_spread_10y_2y'] < 0).astype(float)

    if 'yield_10y' in macro_df.columns and 'breakeven_10y' in macro_df.columns:
        macro_df['real_rate_proxy'] = macro_df['yield_10y'] - macro_df['breakeven_10y']

    if 'fed_rate' in macro_df.columns:
        macro_df['fed_rate_level'] = macro_df['fed_rate']
        macro_df['fed_rate_change_20d'] = macro_df['fed_rate'].diff(20)
        macro_df['fed_hiking'] = (macro_df['fed_rate'].diff(1) > 0).astype(float)
        macro_df['fed_cutting'] = (macro_df['fed_rate'].diff(1) < 0).astype(float)

    for col in ['cpi', 'nfp', 'unemployment']:
        if col in macro_df.columns:
            macro_df[f"{col}_level"] = macro_df[col]
            
    # Combine datasets
    final_df = pd.concat([df, macro_df], axis=1)
    
    # Save
    logger.info(f"Saving merged dataset to {output_path}")
    final_df.to_csv(output_path)
    logger.info("Done.")

if __name__ == "__main__":
    fetch_macro_data()
