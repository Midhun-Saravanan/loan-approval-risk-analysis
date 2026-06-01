"""
=============================================================================
LOAN APPROVAL RISK ANALYSIS — STREAMLIT DASHBOARD
=============================================================================
A professional banking analytics platform for loan risk assessment,
borrower segmentation, and approval analysis.

Run:  streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from typing import Dict, Optional, Any

# ── Local modules ──
from utils import (
    detect_columns, format_currency, format_percentage,
    get_numeric_columns, get_categorical_columns, calculate_statistics,
    get_column_info, segment_borrowers,
)
from data_cleaning import (
    analyze_missing_values, clean_dataset, get_cleaning_summary,
)
from visualization import (
    plot_income_distribution, plot_loan_amount_distribution,
    plot_credit_score_distribution, plot_approval_counts,
    plot_gender_approval, plot_correlation_heatmap,
    plot_risk_segments, plot_risk_summary_bar,
    plot_box_plot, plot_feature_importance, plot_confusion_matrix,
    plot_approval_by_category, plot_debt_vs_approval,
    plot_scatter_with_trend, plot_risk_gauge,
)
from ml_model import build_and_evaluate, predict_single
from report_generator import (
    generate_text_report, generate_pdf_report,
    generate_recommendations, generate_executive_summary,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIGURATION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Loan Approval Risk Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CUSTOM CSS — BANKING THEME                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Global Styles ── */
html, body, div[class*="css"], .stMarkdown, p, label, input, button, select, textarea {
    font-family: 'Inter', sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* ── App Background ── */
.stApp {
    background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%) !important;
    color: #f8fafc !important;
}

/* ── Sidebar Customization ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #e2e8f0;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    transition: all 0.2s ease !important;
    padding: 6px 10px !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

/* ── Main Container Padding ── */
.main .block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* ── Metric Cards (Sleek Glassmorphic with Neon Glow) ── */
.metric-card {
    background: rgba(30, 41, 59, 0.45) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 22px 20px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-5px) scale(1.02) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), 0 0 15px rgba(59, 130, 246, 0.25) !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
}
/* Neon glow lines on card left border */
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(180deg, #3b82f6, #6366f1) !important;
}
.metric-card.accent-green::before  { background: linear-gradient(180deg, #10b981, #059669) !important; }
.metric-card.accent-coral::before  { background: linear-gradient(180deg, #f43f5e, #e11d48) !important; }
.metric-card.accent-cyan::before   { background: linear-gradient(180deg, #06b6d4, #0891b2) !important; }

.metric-card.accent-green .metric-value { color: #34d399 !important; }
.metric-card.accent-coral .metric-value { color: #f87171 !important; }
.metric-card.accent-cyan .metric-value  { color: #22d3ee !important; }

.metric-card .metric-label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 6px !important;
}
.metric-card .metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
}
.metric-card .metric-delta {
    font-size: 0.85rem !important;
    color: #34d399 !important;
    font-weight: 600 !important;
    margin-top: 4px !important;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #ffffff !important;
    padding-bottom: 12px !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.08) !important;
    margin-bottom: 24px !important;
    position: relative;
    display: block !important;
}
.section-header::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0; width: 60px; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    border-radius: 2px !important;
}

/* ── Info/Alert Boxes ── */
.info-box, .warning-box, .success-box, .danger-box {
    padding: 16px 20px !important;
    border-radius: 12px !important;
    margin-bottom: 20px !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid transparent !important;
}
.info-box {
    background: rgba(59, 130, 246, 0.1) !important;
    border-color: rgba(59, 130, 246, 0.2) !important;
    color: #93c5fd !important;
}
.warning-box {
    background: rgba(245, 158, 11, 0.1) !important;
    border-color: rgba(245, 158, 11, 0.2) !important;
    color: #fde047 !important;
}
.success-box {
    background: rgba(16, 185, 129, 0.1) !important;
    border-color: rgba(16, 185, 129, 0.2) !important;
    color: #6ee7b7 !important;
}
.danger-box {
    background: rgba(239, 68, 68, 0.1) !important;
    border-color: rgba(239, 68, 68, 0.2) !important;
    color: #fca5a5 !important;
}

/* ── Insight Cards ── */
.insight-card {
    background: rgba(30, 41, 59, 0.3) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.2s ease !important;
}
.insight-card:hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
    transform: translateY(-2px) !important;
}
.insight-card h4 {
    color: #ffffff !important;
    font-size: 1.05rem !important;
    margin-top: 0 !important;
    margin-bottom: 8px !important;
}
.insight-card p {
    color: #94a3b8 !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    padding: 40px 36px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 32px !important;
    position: relative;
    overflow: hidden;
}
/* Glowing indicator or background element */
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(244, 63, 94, 0) 70%) !important;
    pointer-events: none;
}
.hero-banner h1 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 10px 0 !important;
    color: #ffffff !important;
}
.hero-banner p {
    font-size: 1.05rem !important;
    color: #94a3b8 !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* ── Interactive Elements (Buttons, Tabs, Inputs) ── */
/* Primary Buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.25) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35), 0 0 12px rgba(99, 102, 241, 0.3) !important;
}
div.stButton > button:first-child:active {
    transform: translateY(1px) !important;
}

/* Download Buttons */
div.stDownloadButton > button:first-child {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
div.stDownloadButton > button:first-child:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px) !important;
}

/* Form controls, selectboxes, and text inputs */
div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}
input[type="text"], input[type="number"], div[data-baseweb="input"] input {
    background-color: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

/* File Uploader area styling */
section[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.2) !important;
    border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
    background: rgba(59, 130, 246, 0.05) !important;
}

/* Tab Overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(15, 23, 42, 0.4) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.stTabs [data-baseweb="tab"] {
    height: 38px !important;
    border-radius: 8px !important;
    padding: 0 16px !important;
    background-color: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background-color: rgba(255, 255, 255, 0.05) !important;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(59, 130, 246, 0.15) !important;
    color: #60a5fa !important;
}
.stTabs [data-baseweb="tab-highlight-slip"] {
    background-color: #3b82f6 !important;
    height: 2px !important;
}

/* Dataframe customization */
.stDataFrame {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Footer Styling */
.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    padding: 30px 0 10px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

/* ── Prediction Result Cards ── */
.prediction-approved {
    background: rgba(16, 185, 129, 0.15) !important;
    border: 2px solid rgba(16, 185, 129, 0.4) !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    margin: 16px 0 !important;
    text-align: center;
}
.prediction-approved h3 {
    color: #34d399 !important;
    font-size: 1.6rem !important;
    margin: 0 0 8px 0 !important;
}
.prediction-approved p {
    color: #6ee7b7 !important;
    font-size: 1rem !important;
    margin: 0 !important;
}
.prediction-denied {
    background: rgba(239, 68, 68, 0.15) !important;
    border: 2px solid rgba(239, 68, 68, 0.4) !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    margin: 16px 0 !important;
    text-align: center;
}
.prediction-denied h3 {
    color: #f87171 !important;
    font-size: 1.6rem !important;
    margin: 0 0 8px 0 !important;
}
.prediction-denied p {
    color: #fca5a5 !important;
    font-size: 1rem !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def metric_card(label: str, value: str, delta: str = "", accent: str = ""):
    """Render a custom metric card via HTML."""
    cls = f"metric-card {accent}"
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def get_working_df() -> Optional[pd.DataFrame]:
    """Return cleaned df if available, else raw uploaded df, else None."""
    if "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df
    if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
        return st.session_state.uploaded_df
    return None


def get_col(role: str) -> Optional[str]:
    """Get mapped column name for a role from session_state."""
    mapping = st.session_state.get("column_mapping", {})
    return mapping.get(role)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR NAVIGATION                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## 🏦 Loan Risk Analytics")
    st.markdown(
        '<p style="font-size:0.82rem;opacity:0.7;margin-top:-8px">'
        'Professional Banking Analytics Platform</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "📁 Upload & Preview",
            "🧹 Data Cleaning",
            "📈 Exploratory Analysis",
            "🔗 Correlation Analysis",
            "⚠️ Risk Segmentation",
            "💡 Loan Insights",
            "🤖 ML Predictions",
            "📋 Recommendations",
            "📥 Download Reports",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Search & Filter ──
    df_active = get_working_df()
    if df_active is not None:
        st.markdown("##### 🔍 Quick Filters")
        id_col = get_col("applicant_id")
        if id_col and id_col in df_active.columns:
            search_id = st.text_input("Search by ID", placeholder="e.g. LP000001")
        else:
            search_id = None

        if "Risk_Category" in df_active.columns:
            risk_filter = st.multiselect(
                "Filter by Risk", ["Low", "Medium", "High"],
                default=["Low", "Medium", "High"],
            )
        else:
            risk_filter = None

        status_col = get_col("loan_status")
        if status_col and status_col in df_active.columns:
            statuses = df_active[status_col].dropna().unique().tolist()
            status_filter = st.multiselect(
                "Filter by Status", statuses, default=statuses,
            )
        else:
            status_filter = None
    else:
        search_id = None
        risk_filter = None
        status_filter = None

    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem;opacity:0.5;text-align:center">'
        '© 2025 Loan Risk Analytics<br>Built with Streamlit & Python</p>',
        unsafe_allow_html=True,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  APPLY SIDEBAR FILTERS                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar search/filter to the DataFrame."""
    filtered = df.copy()

    # ID search
    if search_id:
        id_col = get_col("applicant_id")
        if id_col and id_col in filtered.columns:
            filtered = filtered[
                filtered[id_col].astype(str).str.contains(search_id, case=False, na=False)
            ]

    # Risk filter
    if risk_filter is not None and "Risk_Category" in filtered.columns:
        filtered = filtered[filtered["Risk_Category"].isin(risk_filter)]

    # Status filter
    if status_filter is not None:
        s_col = get_col("loan_status")
        if s_col and s_col in filtered.columns:
            filtered = filtered[filtered[s_col].isin(status_filter)]

    return filtered


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: DASHBOARD OVERVIEW                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if page == "📊 Dashboard Overview":
    # Hero banner
    st.markdown(
        '<div class="hero-banner">'
        '<h1>🏦 Loan Approval Risk Analysis</h1>'
        '<p>Comprehensive banking analytics for loan risk assessment, '
        'borrower segmentation, and approval insights.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    df = get_working_df()
    if df is not None:
        df = apply_filters(df)
        col_map = st.session_state.get("column_mapping", {})

        # Key metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Applicants", f"{len(df):,}", "", "")
        with c2:
            status_c = col_map.get("loan_status")
            if status_c and status_c in df.columns:
                approved = df[status_c].value_counts()
                top = approved.index[0] if len(approved) > 0 else "N/A"
                rate = approved.iloc[0] / len(df) * 100 if len(df) > 0 else 0
                metric_card("Approval Rate", f"{rate:.1f}%", f"Most common: {top}", "accent-green")
            else:
                metric_card("Approval Rate", "N/A", "", "accent-green")
        with c3:
            credit_c = col_map.get("credit_score")
            if credit_c and credit_c in df.columns:
                avg_cr = pd.to_numeric(df[credit_c], errors="coerce").mean()
                metric_card("Avg Credit Score", f"{avg_cr:.1f}" if not pd.isna(avg_cr) else "N/A",
                            "", "accent-cyan")
            else:
                metric_card("Avg Credit Score", "N/A", "", "accent-cyan")
        with c4:
            loan_c = col_map.get("loan_amount")
            if loan_c and loan_c in df.columns:
                avg_loan = pd.to_numeric(df[loan_c], errors="coerce").mean()
                metric_card("Avg Loan Amount", format_currency(avg_loan) if not pd.isna(avg_loan) else "N/A",
                            "", "accent-coral")
            else:
                metric_card("Avg Loan Amount", "N/A", "", "accent-coral")

        st.markdown("<br>", unsafe_allow_html=True)

        # Quick charts row
        col_a, col_b = st.columns(2)
        with col_a:
            if status_c and status_c in df.columns:
                fig = plot_approval_counts(df, status_c, "Loan Status Distribution")
                st.plotly_chart(fig, width='stretch')
        with col_b:
            if "Risk_Category" in df.columns:
                risk_data = st.session_state.get("risk_summary", {})
                if risk_data:
                    fig = plot_risk_segments(risk_data)
                    st.plotly_chart(fig, width='stretch')
            else:
                income_c = col_map.get("income")
                if income_c and income_c in df.columns:
                    fig = plot_income_distribution(df, income_c)
                    st.pyplot(fig)

        # Data summary
        with st.expander("📊 Dataset Quick Stats", expanded=False):
            st.dataframe(get_column_info(df), width='stretch', hide_index=True)
    else:
        st.markdown(
            '<div class="info-box">'
            '📁 <strong>Welcome!</strong> Upload a loan dataset from the '
            '<strong>Upload & Preview</strong> page to begin your analysis.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### How to Get Started")
        st.markdown("""
        1. **Upload** a CSV or Excel loan dataset via the sidebar → *Upload & Preview*
        2. **Clean** the data in the *Data Cleaning* section
        3. **Explore** patterns in *Exploratory Analysis* and *Correlation Analysis*
        4. **Segment** borrowers by risk in *Risk Segmentation*
        5. **Predict** outcomes with the *ML Predictions* module
        6. **Download** reports from the *Download Reports* section
        """)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: UPLOAD & PREVIEW                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "📁 Upload & Preview":
    section_header("📁 Upload & Preview Dataset")

    uploaded = st.file_uploader(
        "Upload a loan dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV, XLSX, and XLS files. Max 200MB.",
    )

    if uploaded is not None:
        try:
            # Read file
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.session_state.uploaded_df = df
            st.session_state.cleaned_df = None  # Reset
            st.session_state.risk_summary = None
            st.session_state.ml_results = None

            st.markdown(
                '<div class="success-box">✅ <strong>File uploaded successfully!</strong></div>',
                unsafe_allow_html=True,
            )

            # Dataset info metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Rows", f"{df.shape[0]:,}")
            with c2:
                metric_card("Columns", str(df.shape[1]), "", "accent-cyan")
            with c3:
                missing_pct = df.isnull().mean().mean() * 100
                metric_card("Missing Values", f"{missing_pct:.1f}%", "", "accent-coral")

            # Preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df.head(10), width='stretch', hide_index=True)

            # Column info
            with st.expander("📊 Column Details", expanded=True):
                st.dataframe(get_column_info(df), width='stretch', hide_index=True)

            # ── Column mapping ──
            st.markdown("---")
            st.markdown("#### 🔗 Column Mapping")
            st.markdown(
                '<div class="info-box">'
                'The system auto-detects common column names. '
                'Use the dropdowns below to correct any mappings.'
                '</div>',
                unsafe_allow_html=True,
            )

            auto_map = detect_columns(df)
            cols_list = ["(None)"] + df.columns.tolist()

            mapping_roles = [
                ("income",            "💰 Income Column"),
                ("coapplicant_income","💰 Co-applicant Income"),
                ("loan_amount",       "🏦 Loan Amount Column"),
                ("credit_score",      "📊 Credit Score / History"),
                ("loan_status",       "✅ Loan Status (Target)"),
                ("employment",        "👔 Employment Column"),
                ("debt",              "📈 Debt / DTI Column"),
                ("applicant_id",      "🆔 Applicant ID Column"),
                ("gender",            "👤 Gender Column"),
                ("age",               "🎂 Age Column"),
                ("education",         "🎓 Education Column"),
                ("property",          "🏠 Property / Housing"),
                ("marital_status",    "💍 Marital / Dependents"),
                ("term",              "📅 Loan Term Column"),
                ("interest_rate",     "💲 Interest Rate Column"),
                ("purpose",           "🎯 Loan Purpose Column"),
            ]

            final_mapping: Dict[str, Optional[str]] = {}
            cols_grid = st.columns(3)

            for idx, (role, label) in enumerate(mapping_roles):
                with cols_grid[idx % 3]:
                    detected = auto_map.get(role)
                    default_idx = cols_list.index(detected) if detected in cols_list else 0
                    choice = st.selectbox(label, cols_list, index=default_idx, key=f"map_{role}")
                    final_mapping[role] = choice if choice != "(None)" else None

            st.session_state.column_mapping = final_mapping
            st.markdown(
                '<div class="success-box">✅ Column mapping saved.</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: DATA CLEANING                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "🧹 Data Cleaning":
    section_header("🧹 Data Cleaning Module")

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        # Missing value analysis
        st.markdown("#### 📊 Missing Value Analysis")
        mv = analyze_missing_values(df)
        has_missing = mv["Missing_Count"].sum() > 0

        # Color-code missing values
        def color_missing(val):
            if isinstance(val, (int, float)) and val > 0:
                return "background-color: #fdeaea; color: #ef476f; font-weight: 600"
            return ""

        st.dataframe(
            mv.style.map(color_missing, subset=["Missing_Count", "Missing_Percentage"]),
            width='stretch', hide_index=True,
        )

        if has_missing:
            st.markdown(
                '<div class="warning-box">⚠️ Missing values detected. '
                'Select a strategy and clean the data below.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="success-box">✅ No missing values in the dataset!</div>',
                unsafe_allow_html=True,
            )

        # Strategy selector
        st.markdown("#### 🛠️ Cleaning Strategy")
        strategy = st.selectbox(
            "Choose how to handle missing values",
            ["auto (recommended)", "median", "mean", "drop"],
            help="'auto' uses median for numbers and mode for categories.",
        )
        strategy_key = strategy.split(" ")[0]

        # Clean button
        if st.button("🧹 Clean Dataset", type="primary", width='stretch'):
            with st.spinner("Cleaning data..."):
                cleaned_df, report = clean_dataset(df, strategy=strategy_key)
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_report = report

                summary_msgs = get_cleaning_summary(df, cleaned_df, report)

            st.markdown(
                '<div class="success-box">✅ <strong>Data cleaning completed!</strong></div>',
                unsafe_allow_html=True,
            )

            # Summary
            st.markdown("#### 📝 Cleaning Summary")
            for msg in summary_msgs:
                st.markdown(f"- {msg}")

            # Before/After
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before Cleaning**")
                c1, c2 = st.columns(2)
                with c1:
                    metric_card("Rows", f"{df.shape[0]:,}")
                with c2:
                    metric_card("Missing", f"{df.isnull().sum().sum():,}", "", "accent-coral")
            with col2:
                st.markdown("**After Cleaning**")
                c1, c2 = st.columns(2)
                with c1:
                    metric_card("Rows", f"{cleaned_df.shape[0]:,}", "", "accent-green")
                with c2:
                    metric_card("Missing", f"{cleaned_df.isnull().sum().sum():,}", "", "accent-green")

            st.markdown("#### 🔍 Cleaned Data Preview")
            st.dataframe(cleaned_df.head(10), width='stretch', hide_index=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: EXPLORATORY ANALYSIS                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "📈 Exploratory Analysis":
    section_header("📈 Exploratory Data Analysis")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = apply_filters(df)
        col_map = st.session_state.get("column_mapping", {})

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distributions", "✅ Approval Analysis",
            "📂 Categorical", "📦 Box Plots",
        ])

        # ── TAB 1: Distributions ──
        with tab1:
            col_a, col_b = st.columns(2)
            income_c = col_map.get("income")
            loan_c = col_map.get("loan_amount")
            credit_c = col_map.get("credit_score")

            with col_a:
                if income_c and income_c in df.columns:
                    fig = plot_income_distribution(df, income_c)
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ Income column not mapped.")

            with col_b:
                if loan_c and loan_c in df.columns:
                    fig = plot_loan_amount_distribution(df, loan_c)
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ Loan amount column not mapped.")

            if credit_c and credit_c in df.columns:
                fig = plot_credit_score_distribution(df, credit_c)
                st.pyplot(fig)

        # ── TAB 2: Approval Analysis ──
        with tab2:
            status_c = col_map.get("loan_status")
            if status_c and status_c in df.columns:
                col_a, col_b = st.columns(2)
                with col_a:
                    fig = plot_approval_counts(df, status_c)
                    st.plotly_chart(fig, width='stretch')

                with col_b:
                    gender_c = col_map.get("gender")
                    if gender_c and gender_c in df.columns:
                        fig = plot_gender_approval(df, gender_c, status_c)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("ℹ️ Gender column not mapped for cross-analysis.")

                # Employment-wise
                emp_c = col_map.get("employment")
                if emp_c and emp_c in df.columns:
                    fig = plot_approval_by_category(df, emp_c, status_c,
                                                    f"Approval by {emp_c}")
                    st.plotly_chart(fig, width='stretch')

                # Education-wise
                edu_c = col_map.get("education")
                if edu_c and edu_c in df.columns:
                    fig = plot_approval_by_category(df, edu_c, status_c,
                                                    f"Approval by {edu_c}")
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("ℹ️ Loan status column not mapped.")

        # ── TAB 3: Categorical ──
        with tab3:
            cat_cols = get_categorical_columns(df)
            if cat_cols:
                selected_cat = st.selectbox("Select categorical column", cat_cols)
                if selected_cat:
                    counts = df[selected_cat].value_counts().head(15)
                    fig = go.Figure(go.Bar(
                        x=counts.values, y=counts.index.astype(str),
                        orientation="h",
                        marker_color=[
                            "#0f3460", "#00b4d8", "#e94560", "#06d6a0",
                            "#ffd166", "#845ec2", "#ff6f91", "#008f7a",
                        ][:len(counts)],
                    ))
                    fig.update_layout(
                        title=f"Distribution of {selected_cat}",
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("No categorical columns found.")

        # ── TAB 4: Box Plots ──
        with tab4:
            num_cols = get_numeric_columns(df)
            if num_cols:
                selected_num = st.selectbox("Select numeric column", num_cols)
                group_options = ["(None)"] + get_categorical_columns(df)
                group_by = st.selectbox("Group by", group_options)
                grp = group_by if group_by != "(None)" else None
                fig = plot_box_plot(df, selected_num, grp)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No numeric columns found.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: CORRELATION ANALYSIS                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "🔗 Correlation Analysis":
    section_header("🔗 Correlation Analysis")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        else:
            # Heatmap
            fig = plot_correlation_heatmap(df)
            st.pyplot(fig)

            # Top correlations
            st.markdown("#### 🏆 Top Correlations")
            corr = numeric_df.corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    pairs.append({
                        "Feature 1": corr.columns[i],
                        "Feature 2": corr.columns[j],
                        "Correlation": round(corr.iloc[i, j], 4),
                        "Strength": "Strong" if abs(corr.iloc[i, j]) > 0.5
                                     else "Moderate" if abs(corr.iloc[i, j]) > 0.3
                                     else "Weak",
                    })
            pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
            st.dataframe(pairs_df.head(15), width='stretch', hide_index=True)

            # Scatter plots for top pairs
            if len(pairs_df) > 0:
                st.markdown("#### 🔍 Scatter Plot - Top Correlated Pair")
                top_pair = pairs_df.iloc[0]
                fig = plot_scatter_with_trend(
                    df, top_pair["Feature 1"], top_pair["Feature 2"],
                    title=f'{top_pair["Feature 1"]} vs {top_pair["Feature 2"]} '
                          f'(r = {top_pair["Correlation"]:.3f})',
                )
                st.plotly_chart(fig, width='stretch')


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: RISK SEGMENTATION                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "⚠️ Risk Segmentation":
    section_header("⚠️ Borrower Risk Segmentation")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        col_map = st.session_state.get("column_mapping", {})

        if st.button("🔄 Run Risk Segmentation", type="primary", width='stretch'):
            with st.spinner("Classifying borrowers..."):
                df_risk, summary = segment_borrowers(df, col_map)
                st.session_state.cleaned_df = df_risk  # Update with risk column
                st.session_state.risk_summary = summary

        risk_data = st.session_state.get("risk_summary")
        if risk_data:
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Total Borrowers", f"{risk_data['total']:,}")
            with c2:
                metric_card("Low Risk", f"{risk_data['low']:,} ({risk_data['low_pct']:.1f}%)",
                            "", "accent-green")
            with c3:
                metric_card("Medium Risk", f"{risk_data['medium']:,} ({risk_data['medium_pct']:.1f}%)",
                            "", "accent-cyan")
            with c4:
                metric_card("High Risk", f"{risk_data['high']:,} ({risk_data['high_pct']:.1f}%)",
                            "", "accent-coral")

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            col_a, col_b = st.columns(2)
            with col_a:
                fig = plot_risk_segments(risk_data)
                st.plotly_chart(fig, width='stretch')
            with col_b:
                fig = plot_risk_summary_bar(risk_data)
                st.plotly_chart(fig, width='stretch')

            # Risk gauge
            risk_score = risk_data.get("high_pct", 0) + risk_data.get("medium_pct", 0) * 0.3
            fig = plot_risk_gauge(min(100, risk_score), "Overall Portfolio Risk Score")
            st.plotly_chart(fig, width='stretch')

            # High-risk borrowers table
            active_df = get_working_df()
            if active_df is not None and "Risk_Category" in active_df.columns:
                with st.expander("🔴 High-Risk Borrowers", expanded=False):
                    high_risk = active_df[active_df["Risk_Category"] == "High"]
                    if len(high_risk) > 0:
                        st.dataframe(high_risk.head(50), width='stretch', hide_index=True)
                    else:
                        st.success("No high-risk borrowers identified.")
        else:
            st.markdown(
                '<div class="info-box">Click <strong>Run Risk Segmentation</strong> '
                'to classify borrowers into risk categories.</div>',
                unsafe_allow_html=True,
            )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: LOAN INSIGHTS                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "💡 Loan Insights":
    section_header("💡 Loan Approval Insights")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = apply_filters(df)
        col_map = st.session_state.get("column_mapping", {})
        status_c = col_map.get("loan_status")
        income_c = col_map.get("income")
        loan_c = col_map.get("loan_amount")
        credit_c = col_map.get("credit_score")

        # ── Approval vs Rejection ──
        if status_c and status_c in df.columns:
            st.markdown("#### ✅ Approval vs Rejection Analysis")
            counts = df[status_c].value_counts()
            col_a, col_b = st.columns([1, 2])
            with col_a:
                for val, cnt in counts.items():
                    pct = cnt / len(df) * 100
                    st.markdown(
                        f'<div class="insight-card"><h4>{val}</h4>'
                        f'<p>{cnt:,} applicants ({pct:.1f}%)</p></div>',
                        unsafe_allow_html=True,
                    )
            with col_b:
                fig = plot_approval_counts(df, status_c)
                st.plotly_chart(fig, width='stretch')

        # ── Income bracket analysis ──
        if income_c and income_c in df.columns and status_c and status_c in df.columns:
            st.markdown("#### 💰 Approval by Income Bracket")
            try:
                inc = pd.to_numeric(df[income_c], errors="coerce")
                df_temp = df.copy()
                df_temp["Income_Bracket"] = pd.cut(
                    inc, bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
                )
                fig = plot_approval_by_category(df_temp, "Income_Bracket", status_c,
                                                "Approval Rate by Income Bracket")
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.info(f"Could not create income brackets: {e}")

        # ── Debt vs approval ──
        debt_c = col_map.get("debt")
        if debt_c and debt_c in df.columns and status_c and status_c in df.columns:
            st.markdown("#### 📈 Debt vs Approval")
            fig = plot_debt_vs_approval(df, debt_c, status_c)
            st.plotly_chart(fig, width='stretch')

        # ── Key findings ──
        st.markdown("#### 🔍 Key Findings")
        findings = []
        if income_c and income_c in df.columns:
            stats = calculate_statistics(df, income_c)
            findings.append(
                f"Average income is {format_currency(stats['mean'])} "
                f"(median: {format_currency(stats['median'])})"
            )
        if loan_c and loan_c in df.columns:
            stats = calculate_statistics(df, loan_c)
            findings.append(
                f"Average loan amount is {format_currency(stats['mean'])} "
                f"with a range of {format_currency(stats['min'])} – {format_currency(stats['max'])}"
            )
        if credit_c and credit_c in df.columns:
            credit_data = pd.to_numeric(df[credit_c], errors="coerce").dropna()
            if credit_data.nunique() > 2:
                findings.append(
                    f"Credit scores range from {credit_data.min():.0f} to {credit_data.max():.0f} "
                    f"(avg: {credit_data.mean():.0f})"
                )
            else:
                good_credit = (credit_data == 1).sum()
                findings.append(
                    f"{good_credit} ({good_credit / len(credit_data) * 100:.1f}%) applicants "
                    f"have positive credit history"
                )

        for f in findings:
            st.markdown(
                f'<div class="insight-card"><p>📌 {f}</p></div>',
                unsafe_allow_html=True,
            )

        if not findings:
            st.info("Map more columns to unlock additional insights.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: ML PREDICTIONS                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "🤖 ML Predictions":
    section_header("🤖 Machine Learning Predictions")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        col_map = st.session_state.get("column_mapping", {})

        # Model configuration
        st.markdown("#### ⚙️ Model Configuration")
        col_a, col_b = st.columns(2)
        with col_a:
            model_type = st.selectbox(
                "Select Model",
                ["Random Forest", "Logistic Regression"],
                help="Random Forest is generally more accurate; Logistic Regression is more interpretable.",
            )
        with col_b:
            # Target column
            all_cols = df.columns.tolist()
            status_c = col_map.get("loan_status")
            default_target_idx = all_cols.index(status_c) if status_c and status_c in all_cols else 0
            target_col = st.selectbox("Target Column (what to predict)", all_cols,
                                      index=default_target_idx)

        # Feature selection
        num_cols = get_numeric_columns(df)
        available_features = [c for c in num_cols if c != target_col]
        selected_features = st.multiselect(
            "Select Features",
            available_features,
            default=available_features[:8] if len(available_features) > 8 else available_features,
            help="Select the numeric columns to use as input features.",
        )

        # Train button
        if st.button("🚀 Train Model", type="primary", width='stretch'):
            if len(selected_features) < 1:
                st.error("Please select at least 1 feature.")
            else:
                try:
                    with st.spinner("Training model..."):
                        model_key = "random_forest" if model_type == "Random Forest" else "logistic_regression"
                        results = build_and_evaluate(
                            df, target_col, selected_features, model_key,
                        )
                        st.session_state.ml_results = results

                    st.markdown(
                        '<div class="success-box">✅ <strong>Model trained successfully!</strong></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")

        # Display results
        results = st.session_state.get("ml_results")
        if results:
            metrics = results["metrics"]

            # Metrics cards
            st.markdown("#### 📊 Model Performance")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Accuracy", f"{metrics['accuracy']:.1f}%", "", "accent-green")
            with c2:
                metric_card("Precision", f"{metrics['precision']:.1f}%", "", "accent-cyan")
            with c3:
                metric_card("Recall", f"{metrics['recall']:.1f}%", "", "")
            with c4:
                metric_card("F1 Score", f"{metrics['f1']:.1f}%", "", "accent-coral")

            c1, c2 = st.columns(2)
            with c1:
                metric_card("Training Samples", f"{results['train_size']:,}")
            with c2:
                metric_card("Testing Samples", f"{results['test_size']:,}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Confusion Matrix")
                fig = plot_confusion_matrix(
                    metrics["confusion_matrix"],
                    labels=[str(c) for c in results["classes"]],
                )
                st.pyplot(fig)
            with col_b:
                st.markdown("#### Feature Importance")
                fig = plot_feature_importance(
                    results["importances"], results["feature_names"],
                )
                st.pyplot(fig)

            # Classification report
            with st.expander("📋 Full Classification Report"):
                st.code(metrics["classification_report"])

            # ── Single prediction ──
            st.markdown("---")
            st.markdown("#### 🔮 Predict Single Applicant")
            st.markdown(
                '<div class="info-box">Enter values for each feature to predict loan approval.</div>',
                unsafe_allow_html=True,
            )

            input_data = {}
            pred_cols = st.columns(min(4, len(results["feature_names"])))
            for idx, feat in enumerate(results["feature_names"]):
                with pred_cols[idx % len(pred_cols)]:
                    input_data[feat] = st.number_input(
                        feat, value=float(df[feat].median()) if feat in df.columns else 0.0,
                        key=f"pred_{feat}",
                    )

            if st.button("🔮 Predict", width='stretch'):
                pred = predict_single(
                    results["model"], results["scaler"],
                    input_data, results["feature_names"],
                    results.get("label_encoder"),
                )
                if "error" in pred:
                    st.error(f"Prediction error: {pred['error']}")
                else:
                    confidence = pred.get("confidence", 0)
                    label = pred.get("label", "Unknown")
                    # Determine if approved: check common positive labels
                    is_approved = str(label).strip().upper() in (
                        "Y", "YES", "1", "APPROVED", "ACCEPTED", "TRUE"
                    )
                    conf_text = f"Confidence: {confidence:.1f}%" if confidence else ""
                    if is_approved:
                        st.markdown(
                            f'<div class="prediction-approved">'
                            f'<h3>\u2705 APPROVED</h3>'
                            f'<p>Prediction: <strong>{label}</strong> {conf_text}</p></div>',
                            unsafe_allow_html=True,
                        )
                        st.balloons()
                    else:
                        st.markdown(
                            f'<div class="prediction-denied">'
                            f'<h3>\u274c DENIED</h3>'
                            f'<p>Prediction: <strong>{label}</strong> {conf_text}</p></div>',
                            unsafe_allow_html=True,
                        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: RECOMMENDATIONS                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "📋 Recommendations":
    section_header("📋 Insights & Recommendations")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        col_map = st.session_state.get("column_mapping", {})
        risk_data = st.session_state.get("risk_summary")

        # Executive summary
        st.markdown("#### 📝 Executive Summary")
        summary = generate_executive_summary(df, col_map, risk_data)
        st.markdown(
            f'<div class="insight-card"><p>{summary}</p></div>',
            unsafe_allow_html=True,
        )

        # Recommendations
        st.markdown("#### 💡 Lending Recommendations")
        recs = generate_recommendations(df, col_map, risk_data)
        for rec in recs:
            st.markdown(
                f'<div class="insight-card"><p>{rec}</p></div>',
                unsafe_allow_html=True,
            )

        # Credit risk summary
        st.markdown("#### 📊 Credit Risk Summary")
        credit_c = col_map.get("credit_score")
        if credit_c and credit_c in df.columns:
            credit_data = pd.to_numeric(df[credit_c], errors="coerce").dropna()
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Avg Score", f"{credit_data.mean():.1f}", "", "accent-cyan")
            with c2:
                if credit_data.nunique() > 2:
                    low = (credit_data < 580).sum()
                    metric_card("High Risk (<580)", f"{low}", "", "accent-coral")
                else:
                    bad = (credit_data == 0).sum()
                    metric_card("Bad Credit History", f"{bad}", "", "accent-coral")
            with c3:
                if credit_data.nunique() > 2:
                    high = (credit_data >= 700).sum()
                    metric_card("Low Risk (≥700)", f"{high}", "", "accent-green")
                else:
                    good = (credit_data == 1).sum()
                    metric_card("Good Credit History", f"{good}", "", "accent-green")
        else:
            st.info("Map a credit score column for detailed credit risk analysis.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE: DOWNLOAD REPORTS                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

elif page == "📥 Download Reports":
    section_header("📥 Download Reports")

    df = get_working_df()
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        col_map = st.session_state.get("column_mapping", {})
        risk_data = st.session_state.get("risk_summary")
        ml_results = st.session_state.get("ml_results")
        cleaning_report = st.session_state.get("cleaning_report")

        # Build analysis results dict
        analysis_results = {
            "dataset_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "numeric_cols": len(get_numeric_columns(df)),
                "categorical_cols": len(get_categorical_columns(df)),
            },
        }
        if cleaning_report:
            msgs = get_cleaning_summary(
                st.session_state.get("uploaded_df", df), df, cleaning_report
            )
            analysis_results["cleaning"] = {"messages": msgs}
        if risk_data:
            analysis_results["risk_summary"] = risk_data
        if ml_results:
            analysis_results["ml_metrics"] = {
                "model_type": ml_results.get("model_type", "N/A"),
                **ml_results.get("metrics", {}),
            }
        recs = generate_recommendations(df, col_map, risk_data)
        analysis_results["recommendations"] = recs

        st.markdown("#### Available Downloads")

        # Generate report text upfront
        text_report = generate_text_report(analysis_results)

        c1, c2, c3 = st.columns(3)

        # ── Download cleaned dataset ──
        with c1:
            st.markdown(
                '<div class="insight-card"><h4>📊 Cleaned Dataset</h4>'
                '<p>Download the cleaned dataset as CSV.</p></div>',
                unsafe_allow_html=True,
            )
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name="cleaned_loan_data.csv",
                mime="text/csv",
                width='stretch',
            )

        # ── Download text report ──
        with c2:
            st.markdown(
                '<div class="insight-card"><h4>📝 Analysis Report</h4>'
                '<p>Full text analysis report.</p></div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Report",
                data=text_report.encode("utf-8"),
                file_name="loan_analysis_report.txt",
                mime="text/plain",
                width='stretch',
            )

        # ── Download PDF report ──
        with c3:
            st.markdown(
                '<div class="insight-card"><h4>📕 PDF Summary</h4>'
                '<p>Professional PDF report.</p></div>',
                unsafe_allow_html=True,
            )
            # Generate PDF upfront so the download button doesn't need a
            # nested button (which causes Streamlit rerun/overwrite bugs).
            with st.spinner("Preparing PDF..."):
                pdf_bytes = generate_pdf_report(analysis_results)
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="loan_risk_report.pdf",
                    mime="application/pdf",
                    width='stretch',
                )
            else:
                st.error("PDF generation failed.")

        # Preview report
        with st.expander("📄 Preview Report"):
            st.code(text_report)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FOOTER                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

st.markdown(
    '<div class="footer-text">'
    '🏦 Loan Approval Risk Analysis Platform • Built with Python & Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
