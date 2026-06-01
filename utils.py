"""
=============================================================================
UTILITY FUNCTIONS MODULE - utils.py
=============================================================================
Provides intelligent column detection, formatting helpers, and general-purpose
utility functions for the Loan Approval Risk Analysis platform.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN KEYWORD MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────
# Maps standard column roles to lists of keyword patterns commonly found
# across various Kaggle loan datasets.

COLUMN_KEYWORDS: Dict[str, List[str]] = {
    "income": [
        "income", "salary", "earnings", "annual_income", "applicant_income",
        "applicantincome", "coincome", "monthly_income", "yearly_income",
        "total_income", "gross_income", "net_income", "person_income",
    ],
    "coapplicant_income": [
        "coapplicant_income", "coapplicantincome", "co_applicant_income",
        "joint_income", "coborrower_income",
    ],
    "loan_amount": [
        "loan_amount", "loanamount", "loan_amt", "requested_amount",
        "amount", "loan_amnt", "funded_amnt", "funded_amount",
        "principal", "loan_size",
    ],
    "credit_score": [
        "credit_score", "creditscore", "fico", "fico_score", "cibil_score",
        "credit_rating", "credit_history", "credithistory", "cibil",
        "fico_range_high", "fico_range_low",
    ],
    "loan_status": [
        "loan_status", "status", "approved", "approval", "default",
        "is_approved", "loan_approved", "loan_default", "bad_loan",
        "is_default", "target", "label", "y", "result",
    ],
    "employment": [
        "employment", "employed", "emp_status", "employment_status",
        "job", "occupation", "self_employed", "emp_length",
        "employment_length", "years_employed", "work_experience",
        "person_emp_length", "employment_type",
    ],
    "debt": [
        "debt", "total_debt", "dti", "debt_to_income",
        "existing_debt", "liabilities", "total_liabilities",
        "revol_bal", "installment", "annual_debt",
    ],
    "applicant_id": [
        "id", "applicant_id", "loan_id", "customer_id", "member_id",
        "application_id", "borrower_id", "account_id", "serial",
    ],
    "gender": [
        "gender", "sex", "applicant_gender",
    ],
    "age": [
        "age", "applicant_age", "person_age", "borrower_age",
    ],
    "education": [
        "education", "degree", "qualification", "education_level",
        "academic", "graduate",
    ],
    "property": [
        "property", "property_area", "property_type", "home_ownership",
        "home_owner", "housing", "residence_type",
    ],
    "marital_status": [
        "married", "marital_status", "dependents", "family_status",
        "spouse", "family_size", "num_dependents",
    ],
    "term": [
        "term", "loan_term", "tenure", "duration", "loan_amount_term",
        "loan_period", "maturity", "months",
    ],
    "interest_rate": [
        "interest_rate", "int_rate", "rate", "apr", "coupon_rate",
        "loan_int_rate", "loan_percent_income",
    ],
    "purpose": [
        "purpose", "loan_purpose", "reason", "loan_type",
        "loan_intent", "loan_category",
    ],
}


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Intelligently detect standard loan-related columns by scanning
    DataFrame column names against known keyword patterns.

    Args:
        df: Input DataFrame to analyze.

    Returns:
        Dictionary mapping standard column roles to detected column names.
        Value is None if no match is found for that role.
    """
    column_mapping: Dict[str, Optional[str]] = {}
    columns_lower = {col: col.lower().strip().replace(" ", "_") for col in df.columns}

    for role, keywords in COLUMN_KEYWORDS.items():
        best_match = None
        best_score = 0

        for original_col, normalized_col in columns_lower.items():
            for keyword in keywords:
                # Exact match gets highest priority
                if normalized_col == keyword:
                    best_match = original_col
                    best_score = 100
                    break
                # Contains match
                elif keyword in normalized_col and len(keyword) > 2:
                    score = len(keyword) / len(normalized_col) * 80
                    if score > best_score:
                        best_match = original_col
                        best_score = score
                # Partial overlap
                elif normalized_col in keyword and len(normalized_col) > 3:
                    score = len(normalized_col) / len(keyword) * 60
                    if score > best_score:
                        best_match = original_col
                        best_score = score

            if best_score == 100:
                break

        column_mapping[role] = best_match if best_score > 20 else None

    return column_mapping


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a numeric value as currency string."""
    try:
        if pd.isna(value):
            return "N/A"
        if abs(value) >= 1_000_000:
            return f"{symbol}{value / 1_000_000:,.2f}M"
        elif abs(value) >= 1_000:
            return f"{symbol}{value / 1_000:,.1f}K"
        else:
            return f"{symbol}{value:,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a numeric value as percentage string."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{value:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric column names from the DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return list of categorical (non-numeric) column names."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def safe_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Safely convert a Series to numeric, coercing errors to NaN.
    Strips whitespace and common currency symbols first.
    """
    try:
        cleaned = series.astype(str).str.strip()
        cleaned = cleaned.str.replace(r"[$,€£₹%]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")
    except Exception:
        return series


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for a single column.

    Returns:
        Dictionary with mean, median, std, min, max, count, missing.
    """
    stats: Dict[str, Any] = {
        "mean": None, "median": None, "std": None,
        "min": None, "max": None, "count": 0, "missing": 0,
    }
    try:
        if column not in df.columns:
            return stats
        col = df[column]
        stats["count"] = int(col.notna().sum())
        stats["missing"] = int(col.isna().sum())
        if pd.api.types.is_numeric_dtype(col):
            stats["mean"] = round(float(col.mean()), 2)
            stats["median"] = round(float(col.median()), 2)
            stats["std"] = round(float(col.std()), 2)
            stats["min"] = round(float(col.min()), 2)
            stats["max"] = round(float(col.max()), 2)
    except Exception:
        pass
    return stats


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary DataFrame with column name, dtype, non-null count,
    null count, null percentage, and unique values for each column.
    """
    info = []
    for col in df.columns:
        info.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-Null": int(df[col].notna().sum()),
            "Null": int(df[col].isna().sum()),
            "Null %": round(df[col].isna().mean() * 100, 2),
            "Unique": int(df[col].nunique()),
        })
    return pd.DataFrame(info)


def classify_risk(
    row: pd.Series,
    col_map: Dict[str, Optional[str]],
    thresholds: Optional[Dict] = None,
) -> str:
    """
    Classify a single borrower row into Low / Medium / High risk.

    Uses credit score, income-to-loan ratio, and debt indicators.
    """
    if thresholds is None:
        thresholds = {
            "credit_high": 700,
            "credit_low": 580,
            "lti_high": 5.0,     # loan-to-income ratio
            "lti_low": 2.5,
        }

    risk_score = 0  # higher = riskier
    factors = 0

    # --- Credit score factor ---
    credit_col = col_map.get("credit_score")
    if credit_col and credit_col in row.index:
        try:
            credit = float(row[credit_col])
            if not np.isnan(credit):
                factors += 1
                # Handle binary credit history (0/1) vs actual scores
                if credit <= 1:
                    risk_score += 0 if credit == 1 else 3
                else:
                    if credit < thresholds["credit_low"]:
                        risk_score += 3
                    elif credit < thresholds["credit_high"]:
                        risk_score += 1
        except (ValueError, TypeError):
            pass

    # --- Loan-to-income ratio factor ---
    income_col = col_map.get("income")
    loan_col = col_map.get("loan_amount")
    if income_col and loan_col and income_col in row.index and loan_col in row.index:
        try:
            income = float(row[income_col])
            loan = float(row[loan_col])
            if income > 0 and not np.isnan(income) and not np.isnan(loan):
                lti = loan / income
                factors += 1
                if lti > thresholds["lti_high"]:
                    risk_score += 3
                elif lti > thresholds["lti_low"]:
                    risk_score += 1
        except (ValueError, TypeError):
            pass

    # --- Debt factor ---
    debt_col = col_map.get("debt")
    if debt_col and debt_col in row.index:
        try:
            debt = float(row[debt_col])
            if not np.isnan(debt):
                factors += 1
                # Normalize: DTI > 40% is high risk
                if debt > 40:
                    risk_score += 3
                elif debt > 20:
                    risk_score += 1
        except (ValueError, TypeError):
            pass

    # --- Employment factor ---
    emp_col = col_map.get("employment")
    if emp_col and emp_col in row.index:
        try:
            emp_val = str(row[emp_col]).strip().lower()
            factors += 1
            if emp_val in ("no", "0", "unemployed", "false", "n"):
                risk_score += 2
        except (ValueError, TypeError):
            pass

    # --- Classification ---
    if factors == 0:
        return "Medium"  # Not enough data to judge
    avg_risk = risk_score / factors
    if avg_risk >= 2.0:
        return "High"
    elif avg_risk >= 1.0:
        return "Medium"
    else:
        return "Low"


def segment_borrowers(
    df: pd.DataFrame, col_map: Dict[str, Optional[str]]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add a 'Risk_Category' column to the DataFrame and return risk summary.

    Returns:
        Tuple of (DataFrame with risk column, summary dict).
    """
    df = df.copy()
    df["Risk_Category"] = df.apply(lambda row: classify_risk(row, col_map), axis=1)

    total = len(df)
    summary = {
        "total": total,
        "low": int((df["Risk_Category"] == "Low").sum()),
        "medium": int((df["Risk_Category"] == "Medium").sum()),
        "high": int((df["Risk_Category"] == "High").sum()),
        "low_pct": round((df["Risk_Category"] == "Low").mean() * 100, 1),
        "medium_pct": round((df["Risk_Category"] == "Medium").mean() * 100, 1),
        "high_pct": round((df["Risk_Category"] == "High").mean() * 100, 1),
    }
    return df, summary
