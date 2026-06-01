"""
=============================================================================
DATA CLEANING MODULE - data_cleaning.py
=============================================================================
Handles missing values, duplicates, data type correction, categorical
encoding, and produces cleaning reports for the analytics dashboard.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values per column.

    Returns:
        DataFrame with columns: Column, Missing_Count, Missing_Percentage, Data_Type
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    dtypes = df.dtypes.astype(str)

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing_count.values,
        "Missing_Percentage": missing_pct.values,
        "Data_Type": dtypes.values,
    })

    # Sort by missing count descending
    summary = summary.sort_values("Missing_Count", ascending=False).reset_index(drop=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame, strategy: str = "auto"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fill or drop missing values based on the chosen strategy.

    Strategies:
        - 'auto'   : Median for numeric, mode for categorical.
        - 'mean'   : Mean for numeric, mode for categorical.
        - 'median' : Median for numeric, mode for categorical.
        - 'drop'   : Drop rows with any missing values.

    Returns:
        Tuple of (cleaned DataFrame, report dict).
    """
    df = df.copy()
    report: Dict[str, Any] = {"strategy": strategy, "changes": {}}

    if strategy == "drop":
        original_len = len(df)
        df = df.dropna()
        report["rows_dropped"] = original_len - len(df)
        return df, report

    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy in ("auto", "median"):
                fill_value = df[col].median()
                method = "median"
            else:
                fill_value = df[col].mean()
                method = "mean"
            df[col] = df[col].fillna(fill_value)
        else:
            # Categorical → fill with mode
            if not df[col].mode().empty:
                fill_value = df[col].mode().iloc[0]
            else:
                fill_value = "Unknown"
            method = "mode"
            df[col] = df[col].fillna(fill_value)

        report["changes"][col] = {
            "filled": int(missing),
            "method": method,
            "fill_value": str(fill_value),
        }

    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the DataFrame.

    Returns:
        Tuple of (deduplicated DataFrame, count of duplicates removed).
    """
    original_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = original_len - len(df)
    return df, removed


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPE CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def fix_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Auto-detect and fix data types (numerics stored as strings, etc.).

    Returns:
        Tuple of (fixed DataFrame, dict of column → new_dtype conversions).
    """
    df = df.copy()
    conversions: Dict[str, str] = {}

    for col in df.columns:
        if df[col].dtype == "object":
            # Try numeric conversion
            numeric_attempt = pd.to_numeric(df[col], errors="coerce")
            non_null_original = df[col].notna().sum()
            non_null_numeric = numeric_attempt.notna().sum()

            # If ≥ 80% of non-null values convert successfully → treat as numeric
            if non_null_original > 0 and non_null_numeric / non_null_original >= 0.8:
                df[col] = numeric_attempt
                conversions[col] = "numeric"
                continue

            # Try datetime conversion
            try:
                dt_attempt = pd.to_datetime(df[col], errors="coerce")
                non_null_dt = dt_attempt.notna().sum()
                if non_null_original > 0 and non_null_dt / non_null_original >= 0.8:
                    df[col] = dt_attempt
                    conversions[col] = "datetime"
                    continue
            except Exception:
                pass

            # Strip whitespace from remaining string columns
            df[col] = df[col].astype(str).str.strip()

    return df, conversions


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Label-encode categorical columns.

    Args:
        df:      Input DataFrame.
        columns: Specific columns to encode. If None, encode all object columns.

    Returns:
        Tuple of (encoded DataFrame, dict of {column: {value: code}}).
    """
    df = df.copy()
    encoding_map: Dict[str, Dict] = {}

    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        unique_vals = df[col].dropna().unique()
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals, key=str))}
        df[col] = df[col].map(mapping)
        encoding_map[col] = mapping

    return df, encoding_map


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CLEANING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(
    df: pd.DataFrame, strategy: str = "auto"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full cleaning pipeline:
        1. Fix data types
        2. Remove duplicates
        3. Handle missing values

    Note: Categorical encoding is NOT applied here because many EDA
    visualizations work better with the original labels. Encoding is
    deferred to the ML module.

    Returns:
        Tuple of (cleaned DataFrame, comprehensive cleaning report).
    """
    report: Dict[str, Any] = {}

    # Step 1: Fix data types
    df, type_conversions = fix_data_types(df)
    report["type_conversions"] = type_conversions

    # Step 2: Remove duplicates
    df, dup_count = remove_duplicates(df)
    report["duplicates_removed"] = dup_count

    # Step 3: Handle missing values
    df, missing_report = handle_missing_values(df, strategy=strategy)
    report["missing_values"] = missing_report

    # Final shape
    report["final_shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_cleaning_summary(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    report: Dict[str, Any],
) -> List[str]:
    """
    Produce a list of human-readable summary sentences describing
    what the cleaning pipeline did.
    """
    summary: List[str] = []

    # Shape changes
    orig_rows, orig_cols = original_df.shape
    clean_rows, clean_cols = cleaned_df.shape
    summary.append(
        f"📊 Original dataset: {orig_rows:,} rows × {orig_cols} columns"
    )
    summary.append(
        f"✅ Cleaned dataset:  {clean_rows:,} rows × {clean_cols} columns"
    )

    # Duplicates
    dups = report.get("duplicates_removed", 0)
    if dups > 0:
        summary.append(f"🗑️ Removed {dups:,} duplicate rows.")
    else:
        summary.append("✅ No duplicate rows found.")

    # Type conversions
    conversions = report.get("type_conversions", {})
    if conversions:
        for col, new_type in conversions.items():
            summary.append(f"🔄 Converted '{col}' → {new_type}")
    else:
        summary.append("✅ All data types are correct.")

    # Missing values
    missing_info = report.get("missing_values", {})
    changes = missing_info.get("changes", {})
    if changes:
        total_filled = sum(c.get("filled", 0) for c in changes.values())
        summary.append(
            f"🩹 Filled {total_filled:,} missing values across {len(changes)} columns."
        )
        for col, detail in changes.items():
            summary.append(
                f"   • {col}: {detail['filled']} values filled with {detail['method']} "
                f"({detail['fill_value']})"
            )
    elif missing_info.get("strategy") == "drop":
        summary.append(
            f"🗑️ Dropped {missing_info.get('rows_dropped', 0):,} rows with missing values."
        )
    else:
        summary.append("✅ No missing values detected.")

    return summary
