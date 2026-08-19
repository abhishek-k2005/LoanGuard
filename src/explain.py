import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / 'models'


def load_explainer(model):
    """Create SHAP TreeExplainer for LightGBM model"""
    return shap.TreeExplainer(model)


def get_shap_values(explainer, X) -> np.ndarray:
    """Compute SHAP values for input data"""
    return explainer.shap_values(X)


def get_top_drivers(
    shap_values: np.ndarray,
    feature_cols: list,
    feature_values: np.ndarray,
    top_n: int = 5
) -> list:
    """
    Extract top N SHAP drivers for a single prediction
    Returns list of dicts sorted by absolute SHAP value
    """
    drivers = []
    for i, feature in enumerate(feature_cols):
        shap_val = float(shap_values[i])
        feat_val = float(feature_values[i])
        drivers.append({
            'feature':       feature,
            'shap_value':    round(shap_val, 4),
            'feature_value': round(feat_val, 4),
            'abs_shap':      abs(shap_val),
            'direction':     'increases' if shap_val > 0 else 'decreases'
        })

    # Sort by absolute SHAP value descending
    drivers = sorted(drivers, key=lambda x: x['abs_shap'], reverse=True)
    return drivers[:top_n]


def get_global_importance(
    shap_values: np.ndarray,
    feature_cols: list
) -> pd.DataFrame:
    """
    Calculate global feature importance from SHAP values
    """
    importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    return importance


def format_drivers_for_prompt(drivers: list) -> str:
    """
    Format top drivers as structured text for LLM prompt
    """
    text = ""
    for i, d in enumerate(drivers, 1):
        text += f"""
    {i}. Feature: {d['feature']}
       Value: {d['feature_value']}
       SHAP: {d['shap_value']:+.4f} — {d['direction']} default risk
       Importance: {d['abs_shap']:.4f}
"""
    return text