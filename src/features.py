import pandas as pd
import numpy as np


def load_and_clean(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    Load raw Lending Club data and apply base cleaning
    """
    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    # Keep only resolved loans
    resolved = ['Fully Paid', 'Charged Off', 'Default']
    df = df[df['loan_status'].isin(resolved)].copy()

    # Binary target
    df['target'] = df['loan_status'].isin(
        ['Charged Off', 'Default']
    ).astype(int)

    # Drop leakage columns
    leakage_cols = [
        'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv',
        'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries',
        'collection_recovery_fee', 'last_pymnt_d',
        'last_pymnt_amnt', 'next_pymnt_d',
        'last_credit_pull_d', 'last_fico_range_high',
        'last_fico_range_low', 'debt_settlement_flag'
    ]
    df = df.drop(
        columns=[c for c in leakage_cols if c in df.columns]
    )

    # Drop high missing columns
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    df = df.drop(columns=high_missing)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw dataframe
    """
    df = df.copy()

    # Date features
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['issue_year'] = df['issue_d'].dt.year
    df['issue_month'] = df['issue_d'].dt.month

    # Credit history length
    df['earliest_cr_line'] = pd.to_datetime(
        df['earliest_cr_line'], format='%b-%Y', errors='coerce'
    )
    df['credit_history_years'] = (
        df['issue_d'] - df['earliest_cr_line']
    ).dt.days / 365

    # Ratio features
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['installment_to_income'] = (
        df['installment'] / (df['annual_inc'] / 12 + 1)
    )

    # Binary flags
    df['high_utilization'] = (df['revol_util'] > 80).astype(int)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)
    df['has_delinq'] = (df['delinq_2yrs'] > 0).astype(int)
    df['short_credit_history'] = (
        df['credit_history_years'] < 3
    ).astype(int)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Return list of model-ready feature columns
    """
    candidates = [
        'int_rate', 'loan_to_income', 'fico_range_high',
        'fico_range_low', 'installment_to_income', 'dti',
        'loan_amnt', 'installment', 'inq_last_6mths',
        'annual_inc', 'revol_util', 'open_acc',
        'delinq_2yrs', 'pub_rec', 'credit_history_years',
        'high_utilization', 'has_delinq', 'has_pub_rec',
        'term', 'issue_year'
    ]
    return [f for f in candidates if f in df.columns]


def prepare_single_application(application: dict) -> pd.DataFrame:
    """
    Convert a single loan application dict to model-ready dataframe
    Used by FastAPI /predict endpoint
    """
    df = pd.DataFrame([application])

    # Engineer features
    feature_cols = [
        'int_rate', 'loan_to_income', 'fico_range_high',
        'fico_range_low', 'installment_to_income', 'dti',
        'loan_amnt', 'installment', 'inq_last_6mths', 'annual_inc'
    ]

    # Fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]