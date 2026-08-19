import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


def get_client():
    """Initialize Anthropic client"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    return anthropic.Anthropic(api_key=api_key)


def generate_credit_memo(
    top_drivers: list,
    predicted_pd: float,
    base_rate: float,
    loan_amnt: float
) -> dict:
    """
    Generate regulatory-style adverse action credit memo
    grounded strictly in SHAP attributions

    Args:
        top_drivers: list of dicts from explain.get_top_drivers()
        predicted_pd: float, predicted probability of default
        base_rate: float, portfolio average default rate
        loan_amnt: float, requested loan amount

    Returns:
        dict with decision, reasons, adverse action notice
    """
    # Format drivers for prompt
    drivers_text = ""
    for i, d in enumerate(top_drivers, 1):
        drivers_text += f"""
    {i}. Feature: {d['feature']}
       Customer value: {d['feature_value']}
       Impact: SHAP {d['shap_value']:+.4f} — {d['direction']} default risk
"""

    risk_level = (
        'HIGH' if predicted_pd > 0.25
        else 'ELEVATED' if predicted_pd > 0.15
        else 'MODERATE'
    )

    decision = 'DENIED' if predicted_pd > 0.20 else 'APPROVED'

    prompt = f"""You are a credit risk officer at a financial institution.
A loan application has been reviewed by our risk model.

MODEL OUTPUT:
- Predicted default probability: {predicted_pd*100:.1f}%
- Portfolio average default rate: {base_rate*100:.1f}%
- Risk level: {risk_level}
- Decision: {decision}
- Loan amount requested: ${loan_amnt:,.0f}

TOP RISK DRIVERS (SHAP analysis):
{drivers_text}

FEATURE DEFINITIONS:
- int_rate: loan interest rate (%)
- loan_to_income: loan amount / annual income
- fico_range_high/low: FICO credit score range
- installment_to_income: monthly payment / monthly income
- dti: debt-to-income ratio (%)
- loan_amnt: requested loan amount ($)
- installment: monthly payment ($)
- inq_last_6mths: credit inquiries in last 6 months
- annual_inc: annual income ($)

Generate a credit memo as valid JSON only.
Base ALL explanations strictly on the SHAP values provided.
Do not invent reasons not in the data.
Return ONLY this JSON structure, no other text:

{{
    "decision": "{decision}",
    "predicted_pd_pct": {predicted_pd*100:.1f},
    "risk_level": "{risk_level}",
    "primary_reasons": [
        {{
            "reason_code": "SHORT_CODE",
            "feature": "feature_name",
            "plain_language": "one sentence a customer can understand",
            "severity": "HIGH or MEDIUM or LOW"
        }}
    ],
    "adverse_action_notice": "2-3 sentence plain English explanation for the applicant. Reference specific financial factors. Do not mention AI or model.",
    "analyst_notes": "2-3 sentence technical summary for credit team referencing SHAP values and feature values",
    "recommendation": "specific actionable advice for the applicant to improve future application"
}}"""

    client = get_client()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Clean markdown if present
    if '```json' in response_text:
        response_text = response_text.split(
            '```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        response_text = response_text.split(
            '```')[1].split('```')[0].strip()

    return json.loads(response_text)


def batch_generate_memos(
    customers: list,
    top_drivers_list: list,
    base_rate: float
) -> list:
    """
    Generate memos for multiple customers
    customers: list of dicts with pd, loan_amnt
    top_drivers_list: list of top_drivers for each customer
    """
    memos = []
    for customer, drivers in zip(customers, top_drivers_list):
        try:
            memo = generate_credit_memo(
                top_drivers=drivers,
                predicted_pd=customer['pd'],
                base_rate=base_rate,
                loan_amnt=customer['loan_amnt']
            )
            memos.append(memo)
        except Exception as e:
            memos.append({'error': str(e)})
    return memos