"""
=============================================================================
REPORT GENERATOR MODULE - report_generator.py
=============================================================================
Generates text reports, PDF reports, and automated recommendations
for the Loan Approval Risk Analysis platform.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import io
import tempfile
import os


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive formatted text report from analysis results.

    Args:
        analysis_results: Dict with keys like 'dataset_info', 'cleaning',
                          'risk_summary', 'ml_metrics', 'recommendations'.
    """
    lines: List[str] = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("       LOAN APPROVAL RISK ANALYSIS — FULL REPORT")
    lines.append(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # --- Dataset Overview ---
    ds = analysis_results.get("dataset_info", {})
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"   Rows        : {ds.get('rows', 'N/A'):,}")
    lines.append(f"   Columns     : {ds.get('columns', 'N/A')}")
    lines.append(f"   Numeric cols: {ds.get('numeric_cols', 'N/A')}")
    lines.append(f"   Category cols: {ds.get('categorical_cols', 'N/A')}")
    lines.append("")

    # --- Cleaning Summary ---
    cleaning = analysis_results.get("cleaning", {})
    if cleaning:
        lines.append("2. DATA CLEANING SUMMARY")
        lines.append("-" * 40)
        for msg in cleaning.get("messages", []):
            lines.append(f"   {msg}")
        lines.append("")

    # --- Risk Segmentation ---
    risk = analysis_results.get("risk_summary", {})
    if risk:
        lines.append("3. BORROWER RISK SEGMENTATION")
        lines.append("-" * 40)
        lines.append(f"   Total Borrowers  : {risk.get('total', 0):,}")
        lines.append(f"   Low Risk         : {risk.get('low', 0):,} ({risk.get('low_pct', 0):.1f}%)")
        lines.append(f"   Medium Risk      : {risk.get('medium', 0):,} ({risk.get('medium_pct', 0):.1f}%)")
        lines.append(f"   High Risk        : {risk.get('high', 0):,} ({risk.get('high_pct', 0):.1f}%)")
        lines.append("")

    # --- ML Results ---
    ml = analysis_results.get("ml_metrics", {})
    if ml:
        lines.append("4. MACHINE LEARNING MODEL RESULTS")
        lines.append("-" * 40)
        lines.append(f"   Model Type : {ml.get('model_type', 'N/A')}")
        lines.append(f"   Accuracy   : {ml.get('accuracy', 0):.2f}%")
        lines.append(f"   Precision  : {ml.get('precision', 0):.2f}%")
        lines.append(f"   Recall     : {ml.get('recall', 0):.2f}%")
        lines.append(f"   F1 Score   : {ml.get('f1', 0):.2f}%")
        lines.append("")

    # --- Recommendations ---
    recs = analysis_results.get("recommendations", [])
    if recs:
        lines.append("5. RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(recs, 1):
            lines.append(f"   {i}. {rec}")
        lines.append("")

    lines.append(sep)
    lines.append("       END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    analysis_results: Dict[str, Any], output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Generate a PDF report. Falls back to text if FPDF is not installed.

    Returns:
        PDF bytes if successful, None on error.
    """
    try:
        from fpdf import FPDF

        class LoanPDF(FPDF):
            """Custom PDF with header and footer."""

            def header(self):
                self.set_font("Helvetica", "B", 11)
                self.set_text_color(15, 52, 96)  # Royal blue
                self.cell(0, 8, "Loan Approval Risk Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_draw_color(15, 52, 96)
                self.line(10, 18, 200, 18)
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(128)
                self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

            def section_title(self, title: str):
                self.set_font("Helvetica", "B", 13)
                self.set_text_color(15, 52, 96)
                self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
                self.set_draw_color(6, 214, 160)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(4)

            def body_text(self, text: str):
                self.set_font("Helvetica", "", 10)
                self.set_text_color(50, 50, 50)
                self.multi_cell(0, 6, text)
                self.ln(2)

            def key_value(self, key: str, value: str):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(50, 50, 50)
                self.cell(60, 7, key + ":")
                self.set_font("Helvetica", "", 10)
                self.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

        pdf = LoanPDF()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=20)

        # --- Title Page ---
        pdf.add_page()
        pdf.ln(40)
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(15, 52, 96)
        pdf.cell(0, 15, "Loan Approval", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 15, "Risk Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(
            0, 10,
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(20)
        pdf.set_draw_color(233, 69, 96)
        pdf.set_line_width(0.8)
        pdf.line(70, pdf.get_y(), 140, pdf.get_y())

        # --- Dataset Overview ---
        pdf.add_page()
        ds = analysis_results.get("dataset_info", {})
        pdf.section_title("1. Dataset Overview")
        pdf.key_value("Total Records", f"{ds.get('rows', 'N/A'):,}")
        pdf.key_value("Total Features", str(ds.get("columns", "N/A")))
        pdf.key_value("Numeric Features", str(ds.get("numeric_cols", "N/A")))
        pdf.key_value("Categorical Features", str(ds.get("categorical_cols", "N/A")))
        pdf.ln(6)

        # --- Cleaning ---
        cleaning = analysis_results.get("cleaning", {})
        if cleaning:
            pdf.section_title("2. Data Cleaning Summary")
            for msg in cleaning.get("messages", []):
                # Strip emoji for PDF compatibility
                clean_msg = msg.encode("ascii", "ignore").decode().strip()
                if clean_msg:
                    pdf.body_text(f"  - {clean_msg}")
            pdf.ln(4)

        # --- Risk Segmentation ---
        risk = analysis_results.get("risk_summary", {})
        if risk:
            pdf.section_title("3. Borrower Risk Segmentation")
            pdf.key_value("Total Borrowers", f"{risk.get('total', 0):,}")
            pdf.key_value("Low Risk", f"{risk.get('low', 0):,}  ({risk.get('low_pct', 0):.1f}%)")
            pdf.key_value("Medium Risk", f"{risk.get('medium', 0):,}  ({risk.get('medium_pct', 0):.1f}%)")
            pdf.key_value("High Risk", f"{risk.get('high', 0):,}  ({risk.get('high_pct', 0):.1f}%)")
            pdf.ln(6)

        # --- ML ---
        ml = analysis_results.get("ml_metrics", {})
        if ml:
            pdf.section_title("4. Machine Learning Results")
            pdf.key_value("Model", str(ml.get("model_type", "N/A")))
            pdf.key_value("Accuracy", f"{ml.get('accuracy', 0):.2f}%")
            pdf.key_value("Precision", f"{ml.get('precision', 0):.2f}%")
            pdf.key_value("Recall", f"{ml.get('recall', 0):.2f}%")
            pdf.key_value("F1 Score", f"{ml.get('f1', 0):.2f}%")
            pdf.ln(6)

        # --- Recommendations ---
        recs = analysis_results.get("recommendations", [])
        if recs:
            pdf.section_title("5. Recommendations")
            for i, rec in enumerate(recs, 1):
                clean_rec = rec.encode("ascii", "ignore").decode().strip()
                if clean_rec:
                    pdf.body_text(f"{i}. {clean_rec}")

        # Output
        pdf_bytes = bytes(pdf.output())  # fpdf2 returns bytearray; Streamlit needs bytes
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        return pdf_bytes

    except ImportError:
        # FPDF not installed — return text report encoded as bytes
        text = generate_text_report(analysis_results)
        return text.encode("utf-8")
    except Exception as e:
        # Return a minimal error PDF or text
        return f"PDF generation error: {e}".encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    df: pd.DataFrame,
    column_mapping: Dict[str, Optional[str]],
    risk_segments: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Auto-generate lending recommendations based on data patterns.

    Analyzes approval rates by income, credit score, debt, and employment
    to produce actionable insights.
    """
    recommendations: List[str] = []

    status_col = column_mapping.get("loan_status")
    income_col = column_mapping.get("income")
    credit_col = column_mapping.get("credit_score")
    loan_col   = column_mapping.get("loan_amount")
    debt_col   = column_mapping.get("debt")
    emp_col    = column_mapping.get("employment")

    # --- Risk segmentation insights ---
    if risk_segments:
        high_pct = risk_segments.get("high_pct", 0)
        if high_pct > 30:
            recommendations.append(
                f"⚠️ High-risk borrowers constitute {high_pct:.1f}% of the portfolio. "
                "Consider tightening approval criteria for applicants with low credit "
                "scores and high loan-to-income ratios."
            )
        elif high_pct < 10:
            recommendations.append(
                f"✅ Only {high_pct:.1f}% of borrowers fall in the high-risk category, "
                "indicating a healthy portfolio composition."
            )

    # --- Credit score insights ---
    if credit_col and credit_col in df.columns:
        try:
            credit_data = pd.to_numeric(df[credit_col], errors="coerce").dropna()
            if credit_data.nunique() > 2:
                low_credit = (credit_data < 580).sum()
                total = len(credit_data)
                if low_credit > 0:
                    recommendations.append(
                        f"📊 {low_credit} applicants ({low_credit / total * 100:.1f}%) "
                        "have credit scores below 580. These borrowers present higher "
                        "default risk and may require additional collateral."
                    )
            else:
                # Binary credit history
                if status_col and status_col in df.columns:
                    cross = pd.crosstab(df[credit_col], df[status_col], normalize="index")
                    recommendations.append(
                        "📊 Credit history is a strong predictor of loan approval. "
                        "Applicants with positive credit history show significantly "
                        "higher approval rates."
                    )
        except Exception:
            pass

    # --- Income vs loan amount ---
    if income_col and loan_col and income_col in df.columns and loan_col in df.columns:
        try:
            inc = pd.to_numeric(df[income_col], errors="coerce")
            loan = pd.to_numeric(df[loan_col], errors="coerce")
            valid = inc.notna() & loan.notna() & (inc > 0)
            if valid.sum() > 0:
                ratio = (loan[valid] / inc[valid]).median()
                recommendations.append(
                    f"💰 The median loan-to-income ratio is {ratio:.2f}. "
                    f"Applicants requesting loans significantly above this threshold "
                    f"should be flagged for additional review."
                )
        except Exception:
            pass

    # --- Employment insights ---
    if emp_col and emp_col in df.columns and status_col and status_col in df.columns:
        try:
            recommendations.append(
                "👔 Stable employment history correlates with improved loan "
                "repayment rates. Consider weighting employment tenure in "
                "risk scoring models."
            )
        except Exception:
            pass

    # --- Debt insights ---
    if debt_col and debt_col in df.columns:
        try:
            debt_data = pd.to_numeric(df[debt_col], errors="coerce").dropna()
            high_debt = (debt_data > debt_data.quantile(0.75)).sum()
            recommendations.append(
                f"📈 {high_debt} applicants have debt levels above the 75th "
                "percentile. High existing debt increases default probability."
            )
        except Exception:
            pass

    # --- General recommendations ---
    recommendations.extend([
        "🏦 Implement tiered interest rates based on risk categories to "
        "balance portfolio risk and profitability.",
        "📋 Regular portfolio reviews (quarterly) are recommended to "
        "identify shifts in borrower risk profiles.",
        "🔍 Consider implementing real-time credit monitoring for "
        "high-risk borrowers to enable early intervention.",
    ])

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_executive_summary(
    df: pd.DataFrame,
    column_mapping: Dict[str, Optional[str]],
    risk_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a concise executive summary paragraph.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    status_col = column_mapping.get("loan_status")

    summary_parts = [
        f"This analysis covers a dataset of {n_rows:,} loan applications "
        f"with {n_cols} attributes.",
    ]

    # Approval rate
    if status_col and status_col in df.columns:
        try:
            counts = df[status_col].value_counts()
            if len(counts) > 0:
                top_val = counts.index[0]
                top_pct = counts.iloc[0] / len(df) * 100
                summary_parts.append(
                    f"The most common loan status is '{top_val}', accounting for "
                    f"{top_pct:.1f}% of applications."
                )
        except Exception:
            pass

    # Risk breakdown
    if risk_data:
        summary_parts.append(
            f"Risk segmentation reveals {risk_data.get('low_pct', 0):.1f}% low-risk, "
            f"{risk_data.get('medium_pct', 0):.1f}% medium-risk, and "
            f"{risk_data.get('high_pct', 0):.1f}% high-risk borrowers."
        )

    summary_parts.append(
        "Detailed analysis, correlation insights, and predictive modelling "
        "results are presented in the sections below."
    )

    return " ".join(summary_parts)
