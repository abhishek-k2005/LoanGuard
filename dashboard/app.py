import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import requests
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import (
    load_pd_model, load_feature_cols, load_ead_ratio,
    predict_pd, predict_expected_loss, calculate_min_rate,
    probability_to_score, get_risk_label
)
from src.explain import (
    load_explainer, get_shap_values, get_top_drivers
)
from src.llm_memo import generate_credit_memo

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load models once ─────────────────────────────────────────
@st.cache_resource
def load_models():
    model        = load_pd_model()
    feature_cols = load_feature_cols()
    ead_ratio    = load_ead_ratio()
    explainer    = load_explainer(model)
    return model, feature_cols, ead_ratio, explainer

@st.cache_data
def load_portfolio_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_path, 'data', 'df_model_with_pricing.csv')
    df = pd.read_csv(csv_path)
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['issue_year'] = df['issue_d'].dt.year
    return df

model, feature_cols, ead_ratio, explainer = load_models()
df = load_portfolio_data()

MEAN_LGD  = 0.62
BASE_RATE = 0.18

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("LoanGuard")
st.sidebar.caption("Loan Portfolio Risk Intelligence")

tab = st.sidebar.radio(
    "Navigation",
    ["🏠 Portfolio Overview",
     "🔍 Loan Analyzer",
     "📈 Vintage & Stress",
     "🤖 Model Performance"]
)

# ════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO OVERVIEW
# ════════════════════════════════════════════════════════════
if tab == "🏠 Portfolio Overview":
    st.title("🏠 Portfolio Overview")
    st.caption("Lending Club loan portfolio — key risk metrics")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Loans",
        f"{len(df):,}",
        help="Total resolved loans in portfolio"
    )
    col2.metric(
        "Default Rate",
        f"{df['target'].mean()*100:.1f}%",
        help="Actual historical default rate"
    )
    col3.metric(
        "Mean Expected Loss",
        f"${df['expected_loss'].mean():,.0f}",
        help="Average expected loss per loan"
    )
    col4.metric(
        "Underpriced Loans",
        f"{df['is_underpriced'].mean()*100:.1f}%",
        help="Loans where actual rate < minimum required rate"
    )

    st.divider()

    # Default rate by grade
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Rate by Loan Grade")
        if 'grade' in df.columns:
            grade_stats = df.groupby('grade')['target'].mean() * 100
            grade_stats = grade_stats.sort_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(grade_stats.index, grade_stats.values,
                         color=['#2ecc71','#a8e6cf','#ffd93d',
                                '#ff9a3c','#ff6b6b','#c0392b','#8e44ad'])
            ax.set_xlabel('Loan Grade')
            ax.set_ylabel('Default Rate (%)')
            ax.set_title('Default Rate by Grade')
            for bar, val in zip(bars, grade_stats.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.3,
                       f'{val:.1f}%', ha='center', fontsize=9)
            st.pyplot(fig)
            plt.close()

    with col2:
        st.subheader("Default Rate by Issue Year")
        year_stats = df.groupby('issue_year')['target'].mean() * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(year_stats.index, year_stats.values,
                marker='o', color='coral', linewidth=2)
        ax.fill_between(year_stats.index, year_stats.values,
                       alpha=0.2, color='coral')
        ax.set_xlabel('Issue Year')
        ax.set_ylabel('Default Rate (%)')
        ax.set_title('Default Rate by Vintage Year')
        ax.axhline(year_stats.mean(), color='red',
                  linestyle='--', alpha=0.7,
                  label=f'Mean = {year_stats.mean():.1f}%')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Expected loss distribution
    st.subheader("Expected Loss Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(df['expected_loss'], bins=50,
                color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Expected Loss ($)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Expected Loss Distribution')
    axes[0].axvline(df['expected_loss'].mean(), color='red',
                   linestyle='--',
                   label=f"Mean = ${df['expected_loss'].mean():,.0f}")
    axes[0].legend()

    axes[1].hist(df['int_rate'] - df['min_rate'], bins=50,
                color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1.5,
                   label='Break-even')
    axes[1].set_xlabel('Rate Difference (Actual - Minimum Required)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Rate Adequacy Distribution')
    axes[1].legend()

    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 2 — LOAN ANALYZER
# ════════════════════════════════════════════════════════════
elif tab == "🔍 Loan Analyzer":
    st.title("🔍 Loan Analyzer")
    st.caption("Input a loan application to get risk assessment and explanation")

    # Input form
    st.subheader("Loan Application Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt   = st.number_input("Loan Amount ($)", 1000, 40000, 15000, 500)
        int_rate    = st.number_input("Interest Rate (%)", 5.0, 30.0, 13.0, 0.5)
        annual_inc  = st.number_input("Annual Income ($)", 10000, 500000, 65000, 1000)

    with col2:
        dti              = st.number_input("DTI (%)", 0.0, 50.0, 18.0, 0.5)
        fico_range_high  = st.number_input("FICO High", 580, 850, 710, 5)
        fico_range_low   = st.number_input("FICO Low", 580, 850, 705, 5)

    with col3:
        installment     = st.number_input("Monthly Installment ($)", 50, 2000, 450, 10)
        inq_last_6mths  = st.number_input("Credit Inquiries (6 months)", 0, 10, 1, 1)

    # Engineered features
    loan_to_income        = loan_amnt / (annual_inc + 1)
    installment_to_income = installment / (annual_inc / 12 + 1)

    st.divider()

    if st.button("🔍 Analyze Loan Application", type="primary"):
        with st.spinner("Analyzing application..."):

            # Prepare input
            X = pd.DataFrame([{
                'int_rate':              int_rate,
                'loan_to_income':        loan_to_income,
                'fico_range_high':       fico_range_high,
                'fico_range_low':        fico_range_low,
                'installment_to_income': installment_to_income,
                'dti':                   dti,
                'loan_amnt':             loan_amnt,
                'installment':           installment,
                'inq_last_6mths':        inq_last_6mths,
                'annual_inc':            annual_inc
            }])[feature_cols]

            # Predict
            pd_value     = float(predict_pd(model, X)[0])
            el           = predict_expected_loss(pd_value, MEAN_LGD,
                                                  loan_amnt, ead_ratio)
            min_rate     = calculate_min_rate(el['expected_loss_pct'])
            credit_score = probability_to_score(pd_value)
            risk_label   = get_risk_label(pd_value)
            decision     = 'DENY' if pd_value > 0.20 else 'APPROVE'

        # Results
        st.subheader("Risk Assessment")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Default Probability", f"{pd_value*100:.1f}%")
        col2.metric("Credit Score", credit_score)
        col3.metric("Expected Loss", f"${el['expected_loss']:,.0f}")
        col4.metric("Min Required Rate", f"{min_rate:.2f}%")

        # Decision banner
        if decision == 'DENY':
            st.error(f"🚫 Decision: DENY — Risk level {risk_label} exceeds threshold")
        else:
            st.success(f"✅ Decision: APPROVE — Risk level {risk_label}")

        # Pricing analysis
        rate_diff = int_rate - min_rate
        if rate_diff < 0:
            st.warning(f"⚠️ Loan is UNDERPRICED by {abs(rate_diff):.2f}% — "
                      f"actual rate {int_rate}% < minimum required {min_rate:.2f}%")
        else:
            st.info(f"✅ Loan is adequately priced — "
                   f"margin of {rate_diff:.2f}% above minimum required rate")

        st.divider()

        # SHAP explanation
        st.subheader("Risk Factor Analysis")

        with st.spinner("Computing SHAP values..."):
            shap_vals  = get_shap_values(explainer, X)
            top_drivers = get_top_drivers(
                shap_values=shap_vals[0],
                feature_cols=feature_cols,
                feature_values=X.values[0],
                top_n=5
            )

        # SHAP waterfall
        col1, col2 = st.columns([3, 2])

        with col1:
            shap_explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X.values[0],
                feature_names=feature_cols
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_explanation, show=False)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Top Risk Drivers**")
            for d in top_drivers:
                direction = "🔴" if d['shap_value'] > 0 else "🟢"
                st.markdown(
                    f"{direction} **{d['feature']}** = {d['feature_value']:.3f}  \n"
                    f"Impact: {d['shap_value']:+.4f}"
                )

        st.divider()

        # LLM Credit Memo
        st.subheader("📋 AI-Generated Credit Memo")

        with st.spinner("Generating credit memo via Claude AI..."):
            memo = generate_credit_memo(
                top_drivers=top_drivers,
                predicted_pd=pd_value,
                base_rate=BASE_RATE,
                loan_amnt=loan_amnt
            )

        # Display memo
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Adverse Action Notice**")
            st.info(memo.get('adverse_action_notice', ''))

            st.markdown("**Primary Denial Reasons**")
            for reason in memo.get('primary_reasons', []):
                severity_color = {
                    'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'
                }.get(reason.get('severity', ''), '⚪')
                st.markdown(
                    f"{severity_color} **{reason.get('reason_code', '')}**  \n"
                    f"{reason.get('plain_language', '')}"
                )

        with col2:
            st.markdown("**Analyst Notes**")
            st.warning(memo.get('analyst_notes', ''))

            st.markdown("**Recommendation for Applicant**")
            st.success(memo.get('recommendation', ''))

        # Raw JSON for technical users
        with st.expander("View Raw JSON Response"):
            st.json(memo)

# ════════════════════════════════════════════════════════════
# TAB 3 — VINTAGE & STRESS
# ════════════════════════════════════════════════════════════
elif tab == "📈 Vintage & Stress":
    st.title("📈 Vintage Analysis & Stress Testing")

    # Load stress results
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stress_df  = pd.read_csv(os.path.join(base_path, 'data', 'stress_test_results.csv'))
        vintage_df = pd.read_csv(os.path.join(base_path, 'data', 'vintage_analysis.csv'))
        has_data   = True
    except FileNotFoundError:
        has_data = False
        st.warning("Run notebook 06_vintage_stress.ipynb first")

    if has_data:
        # Vintage analysis
        st.subheader("Vintage Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(vintage_df['issue_year'],
                   vintage_df['actual_default_rate_pct'],
                   marker='o', color='coral', linewidth=2,
                   label='Actual Default Rate')
            ax.plot(vintage_df['issue_year'],
                   vintage_df['mean_pd_pct'],
                   marker='s', color='steelblue',
                   linewidth=2, linestyle='--',
                   label='Predicted PD')
            ax.set_xlabel('Issue Year')
            ax.set_ylabel('Rate (%)')
            ax.set_title('Actual vs Predicted by Vintage')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(vintage_df['issue_year'],
                  vintage_df['total_loans'],
                  color='steelblue', alpha=0.8)
            ax.set_xlabel('Issue Year')
            ax.set_ylabel('Number of Loans')
            ax.set_title('Loan Volume by Vintage Year')
            st.pyplot(fig)
            plt.close()

        # Key findings
        best_year  = vintage_df.loc[
            vintage_df['actual_default_rate_pct'].idxmin(),
            'issue_year'
        ]
        worst_year = vintage_df.loc[
            vintage_df['actual_default_rate_pct'].idxmax(),
            'issue_year'
        ]
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Vintage", str(best_year),
                   f"{vintage_df['actual_default_rate_pct'].min():.1f}% default")
        col2.metric("Worst Vintage", str(worst_year),
                   f"{vintage_df['actual_default_rate_pct'].max():.1f}% default")
        col3.metric("Vintage Range",
                   f"{vintage_df['actual_default_rate_pct'].max() / vintage_df['actual_default_rate_pct'].min():.1f}x",
                   "worst vs best")

        st.divider()

        # Stress testing
        st.subheader("Stress Testing Results")

        col1, col2, col3 = st.columns(3)
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

        for i, (_, row) in enumerate(stress_df.iterrows()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{row['Scenario']}**")
                st.metric(
                    "Total EL",
                    f"${row['Total EL ($M)']:.1f}M",
                    f"+{row['EL Increase %']:.1f}%" if row['EL Increase %'] > 0 else "Baseline"
                )
                st.caption(row['Description'])

        # Stress chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        scenario_names = stress_df['Scenario'].tolist()

        axes[0].bar(scenario_names,
                   stress_df['Total EL ($M)'],
                   color=colors)
        axes[0].set_title('Total Expected Loss by Scenario',
                          fontweight='bold')
        axes[0].set_ylabel('Expected Loss ($M)')

        axes[1].bar(scenario_names,
                   stress_df['EL % Portfolio'],
                   color=colors)
        axes[1].axhline(5, color='red', linestyle='--',
                       label='5% Capital Threshold')
        axes[1].set_title('EL as % of Portfolio',
                          fontweight='bold')
        axes[1].set_ylabel('EL (%)')
        axes[1].legend()

        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif tab == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", "0.729")
    col2.metric("Gini",    "0.458")
    col3.metric("KS Stat", "0.338")
    col4.metric("Avg Precision", "0.405")

    st.divider()

    # Load and show saved plots
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    st.subheader("ROC & Precision-Recall Curves")
    try:
        from PIL import Image
        img = Image.open(os.path.join(base_path, 'notebooks', 'images', '06_roc_pr_curves.png'))
        st.image(img, use_column_width=True)
    except Exception:
        st.info("Run notebook 03_pd_model.ipynb to generate plots")

    st.subheader("Model Comparison")
    try:
        img = Image.open(os.path.join(base_path, 'notebooks', 'images', '09_model_comparison.png'))
        st.image(img, use_column_width=True)
    except Exception:
        st.info("Run notebook 03b_model_comparison.ipynb to generate plots")

    st.subheader("SHAP Feature Importance")
    try:
        img = Image.open(os.path.join(base_path, 'notebooks', 'images', '21_shap_importance.png'))
        st.image(img, use_column_width=True)
    except Exception:
        st.info("Run notebook 07_shap_llm.ipynb to generate plots")

    st.subheader("SHAP Summary Plot")
    try:
        img = Image.open(os.path.join(base_path, 'notebooks', 'images', '22_shap_summary.png'))
        st.image(img, use_column_width=True)
    except Exception:
        st.info("Run notebook 07_shap_llm.ipynb to generate plots")

    # Why LightGBM
    st.divider()
    st.subheader("Why LightGBM?")
    st.markdown("""
| Factor | Logistic Regression | Random Forest | XGBoost | **LightGBM** |
|--------|--------------------:|-------------:|--------:|-------------:|
| Gini   | ~0.38               | ~0.42        | ~0.44   | **0.458**    |
| Speed  | Fast                | Slow         | Medium  | **Fastest**  |
| SHAP   | Linear only         | Approximate  | Exact   | **Exact**    |
| Missing values | Manual | Manual | Native | **Native** |

LightGBM uses leaf-wise tree growth which captures complex
non-linear feature interactions missed by logistic regression,
while being 3-5x faster than XGBoost on tabular data.
Native SHAP support enables exact Shapley value computation
essential for the regulatory explanation layer.
    """)