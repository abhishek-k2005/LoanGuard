import numpy as np
import lightgbm as lgb
import json
from pathlib import Path

# Paths
MODEL_DIR = Path(__file__).parent.parent / 'models'


def load_pd_model():
    """Load PD model from disk"""
    model = lgb.Booster(
        model_file=str(MODEL_DIR / 'pd_model.txt')
    )
    return model


def load_feature_cols():
    """Load feature column list"""
    with open(MODEL_DIR / 'feature_cols.json', 'r') as f:
        return json.load(f)


def load_ead_ratio():
    """Load EAD ratio"""
    with open(MODEL_DIR / 'ead_ratio.json', 'r') as f:
        data = json.load(f)
        return data['mean_ead_ratio']


def predict_pd(model, X) -> np.ndarray:
    """Predict probability of default"""
    return model.predict(X)


def predict_expected_loss(
    pd_value: float,
    lgd_value: float,
    loan_amnt: float,
    ead_ratio: float = 0.90
) -> dict:
    """
    Calculate Expected Loss using Basel framework
    EL = PD × LGD × EAD
    """
    ead = loan_amnt * ead_ratio
    el  = pd_value * lgd_value * ead
    el_pct = (el / loan_amnt * 100) if loan_amnt > 0 else 0

    return {
        'pd':               round(pd_value, 4),
        'lgd':              round(lgd_value, 4),
        'ead':              round(ead, 2),
        'expected_loss':    round(el, 2),
        'expected_loss_pct': round(el_pct, 2)
    }


def calculate_min_rate(
    el_pct: float,
    cost_of_funds: float = 0.02,
    operating_cost: float = 0.01,
    profit_margin: float  = 0.005
) -> float:
    """
    Calculate minimum interest rate to make loan profitable
    Rate = Cost of Funds + EL% + Operating Cost + Profit Margin
    """
    min_rate = (
        cost_of_funds +
        el_pct / 100 +
        operating_cost +
        profit_margin
    ) * 100
    return round(min_rate, 2)


def probability_to_score(
    prob: float,
    base_score: int = 600,
    pdo: int = 50
) -> int:
    """
    Convert default probability to credit score (300-850)
    Higher score = lower risk
    """
    prob = max(0.001, min(0.999, prob))
    odds = (1 - prob) / prob
    factor = pdo / np.log(2)
    score = base_score + factor * np.log(odds)
    return int(np.clip(score, 300, 850))


def get_risk_label(pd_value: float) -> str:
    """Convert PD to human readable risk label"""
    if pd_value < 0.10:
        return 'LOW'
    elif pd_value < 0.20:
        return 'MEDIUM'
    elif pd_value < 0.35:
        return 'HIGH'
    else:
        return 'VERY HIGH'