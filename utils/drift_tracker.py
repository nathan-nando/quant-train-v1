import json
import os
import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureDriftTracker:
    @staticmethod
    def save_feature_importance(regime: str, feature_importances: dict, output_dir: str = "feature_drift"):
        """
        Saves the top feature importances for a given regime training run.
        This allows tracking Feature Importance Drift over time.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{regime}_importance_history.jsonl")
        
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "regime": regime,
            "importances": feature_importances
        }
        
        try:
            with open(filename, 'a') as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Saved feature importance drift record for {regime}")
        except Exception as e:
            logger.error(f"Failed to save feature importance: {e}")

    @staticmethod
    def analyze_drift(regime: str, input_dir: str = "feature_drift", lookback_runs: int = 5):
        """
        Analyzes if top-20 features have drifted over recent training runs.
        """
        filename = os.path.join(input_dir, f"{regime}_importance_history.jsonl")
        if not os.path.exists(filename):
            return None
            
        records = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    
        if len(records) < 2:
            return None
            
        # Compare latest vs average of previous runs
        latest = records[-1]['importances']
        recent_history = records[-(lookback_runs+1):-1]
        
        # Calculate average historical importance
        historical_avg = {}
        counts = {}
        for rec in recent_history:
            for feat, imp in rec['importances'].items():
                historical_avg[feat] = historical_avg.get(feat, 0) + imp
                counts[feat] = counts.get(feat, 0) + 1
                
        for feat in historical_avg:
            historical_avg[feat] /= counts[feat]
            
        # Detect drop-offs
        drift_alerts = []
        for feat, hist_val in historical_avg.items():
            current_val = latest.get(feat, 0.0)
            if current_val < hist_val * 0.5: # 50% drop in importance
                drift_alerts.append({
                    "feature": feat,
                    "historical_avg": hist_val,
                    "current_val": current_val,
                    "drop_pct": (hist_val - current_val) / hist_val * 100
                })
                
        return drift_alerts
