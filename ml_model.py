"""
=============================================================================
MACHINE LEARNING MODULE - ml_model.py
=============================================================================
Train, evaluate, and predict with Logistic Regression or Random Forest
models for loan approval classification.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler, Optional[LabelEncoder]]:
    """
    Prepare feature matrix X and target vector y for modelling.

    Steps:
        1. Select feature columns (default: all numeric except target).
        2. Encode target if it is categorical.
        3. Drop rows with NaN in features or target.
        4. Scale features with StandardScaler.

    Returns:
        (X_scaled, y, feature_names, scaler, label_encoder_or_None)
    """
    df = df.copy()

    # --- Target ---
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    y_series = df[target_column].copy()
    le: Optional[LabelEncoder] = None

    # Encode categorical target
    if not pd.api.types.is_numeric_dtype(y_series):
        le = LabelEncoder()
        y_series = pd.Series(le.fit_transform(y_series.astype(str)), index=y_series.index)

    # --- Features ---
    if feature_columns is None:
        # Use all numeric columns except target
        feature_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_column
        ]
    else:
        feature_columns = [c for c in feature_columns if c in df.columns and c != target_column]

    if len(feature_columns) == 0:
        raise ValueError("No valid numeric feature columns available for training.")

    # Encode any non-numeric feature columns
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            col_le = LabelEncoder()
            df[col] = col_le.fit_transform(df[col].astype(str))

    X_df = df[feature_columns]

    # Drop rows where features or target have NaN
    valid_mask = X_df.notna().all(axis=1) & y_series.notna()
    X_df = X_df.loc[valid_mask]
    y_series = y_series.loc[valid_mask]

    if len(X_df) < 10:
        raise ValueError(
            f"Only {len(X_df)} valid samples after removing NaN — need at least 10."
        )

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)
    y = y_series.values.astype(int)

    return X_scaled, y, feature_columns, scaler, le


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray):
    """Train and return a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    """Train and return a Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate a trained classifier.

    Returns dict with accuracy, precision, recall, f1, confusion_matrix,
    classification_report, and predictions.
    """
    y_pred = model.predict(X_test)

    avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"

    results = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
        "predictions": y_pred,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(
    model, feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract feature importances. Works for Random Forest (.feature_importances_)
    and Logistic Regression (abs of .coef_).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    return importances, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    model, scaler: StandardScaler, input_data: Dict[str, float],
    feature_names: List[str], label_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, Any]:
    """
    Make a prediction for a single applicant.

    Args:
        model:          Trained classifier.
        scaler:         Fitted StandardScaler.
        input_data:     Dict of {feature_name: value}.
        feature_names:  Feature names in correct order.
        label_encoder:  Optional LabelEncoder for decoding the prediction.

    Returns:
        Dict with prediction, probability, and decoded label.
    """
    try:
        values = [float(input_data.get(f, 0)) for f in feature_names]
        X = scaler.transform([values])
        pred = model.predict(X)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]

        label = str(pred)
        if label_encoder is not None:
            label = label_encoder.inverse_transform([int(pred)])[0]

        return {
            "prediction": int(pred),
            "label": str(label),
            "probability": proba.tolist() if proba is not None else None,
            "confidence": round(float(max(proba)) * 100, 1) if proba is not None else None,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_and_evaluate(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    model_type: str = "random_forest",
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Full pipeline: prepare → split → train → evaluate.

    Args:
        df:              Input DataFrame (cleaned).
        target_column:   Name of the target column.
        feature_columns: Specific features to use, or None for all numeric.
        model_type:      'random_forest' or 'logistic_regression'.
        test_size:       Fraction held out for testing.

    Returns:
        Dict with model, scaler, label_encoder, metrics, feature_names,
        importances, and split info.
    """
    # Prepare
    X, y, feat_names, scaler, le = prepare_features(
        df, target_column, feature_columns
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train
    if model_type == "logistic_regression":
        model = train_logistic_regression(X_train, y_train)
    else:
        model = train_random_forest(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importance
    importances, names = get_feature_importance(model, feat_names)

    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "metrics": metrics,
        "feature_names": feat_names,
        "importances": importances,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_type": model_type,
        "target_column": target_column,
        "classes": le.classes_.tolist() if le is not None else list(np.unique(y)),
    }
