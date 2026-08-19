from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import (
    load_pd_model, load_feature_cols, load_ead_ratio,
    predict_pd, predict_expected_loss, calculate_min_rate,
    probability_to_score, get_risk_label
)
from src.explain import (
    load_explainer, get_shap_values,
    get_top_drivers, format_drivers_for_prompt
)
from src.llm_memo import generate_credit_memo
from src.features import prepare_single_application

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="LoanGuard API",
    description="Loan Portfolio Risk Intelligence System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load models at startup ──────────────────────────────────
print("Loading models...")
pd_model    = load_pd_model()
feature_cols = load_feature_cols()
ead_ratio   = load_ead_ratio()
explainer   = load_explainer(pd_model)
MEAN_LGD    = 0.62
BASE_RATE   = 0.18
print("✅ Models loaded")


# ── Request / Response schemas ──────────────────────────────
class LoanApplication(BaseModel):
    int_rate:              float
    loan_amnt:             float
    annual_inc:            float
    dti:                   float
    fico_range_high:       float
    fico_range_low:        float
    installment:           float
    inq_last_6mths:        Optional[float] = 0.0
    loan_to_income:        Optional[float] = None
    installment_to_income: Optional[float] = None


class PredictResponse(BaseModel):
    pd:                  float
    pd_pct:              float
    risk_label:          str
    credit_score:        int
    expected_loss:       float
    expected_loss_pct:   float
    min_rate:            float
    actual_rate:         float
    is_underpriced:      bool
    decision:            str


class ExplainResponse(BaseModel):
    pd:              float
    risk_label:      str
    top_drivers:     list
    credit_memo:     dict


# ── Helper ──────────────────────────────────────────────────
def prepare_features(app: LoanApplication) -> pd.DataFrame:
    data = app.dict()

    # Engineer ratio features if not provided
    if data['loan_to_income'] is None:
        data['loan_to_income'] = (
            data['loan_amnt'] / (data['annual_inc'] + 1)
        )
    if data['installment_to_income'] is None:
        data['installment_to_income'] = (
            data['installment'] / (data['annual_inc'] / 12 + 1)
        )

    df = pd.DataFrame([data])

    # Keep only model features in correct order
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols]


# ── Routes ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":    "LoanGuard API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/explain", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(application: LoanApplication):
    """
    Predict default probability and expected loss
    for a loan application
    """
    try:
        # Prepare features
        X = prepare_features(application)

        # PD prediction
        pd_value = float(predict_pd(pd_model, X)[0])

        # Expected loss
        el = predict_expected_loss(
            pd_value=pd_value,
            lgd_value=MEAN_LGD,
            loan_amnt=application.loan_amnt,
            ead_ratio=ead_ratio
        )

        # Minimum rate
        min_rate = calculate_min_rate(el['expected_loss_pct'])

        # Credit score
        credit_score = probability_to_score(pd_value)

        # Risk label
        risk_label = get_risk_label(pd_value)

        # Decision
        decision = 'DENY' if pd_value > 0.20 else 'APPROVE'

        return PredictResponse(
            pd=round(pd_value, 4),
            pd_pct=round(pd_value * 100, 2),
            risk_label=risk_label,
            credit_score=credit_score,
            expected_loss=el['expected_loss'],
            expected_loss_pct=el['expected_loss_pct'],
            min_rate=min_rate,
            actual_rate=application.int_rate,
            is_underpriced=application.int_rate < min_rate,
            decision=decision
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain(application: LoanApplication):
    """
    Generate SHAP explanation and LLM credit memo
    for a loan application
    """
    try:
        # Prepare features
        X = prepare_features(application)

        # PD prediction
        pd_value = float(predict_pd(pd_model, X)[0])

        # SHAP values
        shap_vals = get_shap_values(explainer, X)

        # Top drivers
        top_drivers = get_top_drivers(
            shap_values=shap_vals[0],
            feature_cols=feature_cols,
            feature_values=X.values[0],
            top_n=5
        )

        # LLM credit memo
        memo = generate_credit_memo(
            top_drivers=top_drivers,
            predicted_pd=pd_value,
            base_rate=BASE_RATE,
            loan_amnt=application.loan_amnt
        )

        return ExplainResponse(
            pd=round(pd_value, 4),
            risk_label=get_risk_label(pd_value),
            top_drivers=top_drivers,
            credit_memo=memo
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))