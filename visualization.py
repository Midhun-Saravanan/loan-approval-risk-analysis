"""
=============================================================================
VISUALIZATION MODULE - visualization.py
=============================================================================
Publication-quality charts with a consistent banking theme for the
Loan Approval Risk Analysis dashboard.
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# BANKING COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "primary_dark":   "#1a1a2e",
    "primary":        "#16213e",
    "primary_light":  "#0f3460",
    "accent":         "#e94560",
    "cyan":           "#00b4d8",
    "cyan_light":     "#90e0ef",
    "cyan_mid":       "#48cae4",
    "success":        "#06d6a0",
    "warning":        "#ffd166",
    "danger":         "#ef476f",
    "text_dark":      "#2d3436",
    "text_light":     "#636e72",
    "bg_light":       "#f8f9fa",
}

RISK_COLORS = {
    "Low":    "#06d6a0",
    "Medium": "#ffd166",
    "High":   "#ef476f",
}

PALETTE_SEQ = [
    COLORS["primary_light"], COLORS["cyan"], COLORS["accent"],
    COLORS["success"], COLORS["warning"], COLORS["danger"],
    "#845ec2", "#ff6f91", "#ffc75f", "#008f7a",
]


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

def set_chart_style():
    """Configure global matplotlib/seaborn style for consistency."""
    plt.rcParams.update({
        "figure.facecolor":    "#0f172a",  # Match radial bg bottom
        "axes.facecolor":      "#0f172a",  # Match radial bg bottom
        "axes.edgecolor":      "#334155",  # Slate 700 border
        "axes.labelcolor":     "#e2e8f0",  # Light text
        "axes.titlesize":      14,
        "axes.titleweight":    "bold",
        "axes.labelsize":      11,
        "xtick.color":         "#94a3b8",  # Gray text
        "ytick.color":         "#94a3b8",
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "grid.color":          "#1e293b",  # Dark slate grids
        "grid.linestyle":      "--",
        "grid.alpha":          0.5,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
        "legend.framealpha":   0.2,
        "legend.facecolor":    "#0f172a",
        "legend.edgecolor":    "#334155",
        "text.color":          "#e2e8f0",
    })
    sns.set_palette(PALETTE_SEQ)

# Apply on import
set_chart_style()


# ─────────────────────────────────────────────────────────────────────────────
# 1. INCOME DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_income_distribution(
    df: pd.DataFrame, income_col: str, title: str = "Income Distribution"
):
    """Histogram + KDE of income with mean/median markers."""
    try:
        data = df[income_col].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data, kde=True, color=COLORS["primary_light"],
                     edgecolor="white", alpha=0.8, ax=ax)
        ax.axvline(data.mean(), color=COLORS["accent"], ls="--", lw=2,
                   label=f"Mean: {data.mean():,.0f}")
        ax.axvline(data.median(), color=COLORS["success"], ls="-.", lw=2,
                   label=f"Median: {data.median():,.0f}")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(income_col)
        ax.set_ylabel("Frequency")
        ax.legend(frameon=True, fancybox=True)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.tight_layout()
        return fig
    except Exception as e:
        return _error_figure(f"Income chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAN AMOUNT DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_loan_amount_distribution(
    df: pd.DataFrame, loan_col: str, title: str = "Loan Amount Distribution"
):
    """Histogram + KDE of loan amounts."""
    try:
        data = df[loan_col].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data, kde=True, color=COLORS["cyan"], edgecolor="white",
                     alpha=0.8, ax=ax)
        ax.axvline(data.mean(), color=COLORS["accent"], ls="--", lw=2,
                   label=f"Mean: {data.mean():,.0f}")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(loan_col)
        ax.set_ylabel("Frequency")
        ax.legend(frameon=True)
        plt.tight_layout()
        return fig
    except Exception as e:
        return _error_figure(f"Loan amount chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CREDIT SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_credit_score_distribution(
    df: pd.DataFrame, credit_col: str,
    title: str = "Credit Score Distribution",
):
    """Histogram with risk-zone coloring."""
    try:
        data = df[credit_col].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))

        # Determine if this is binary (0/1) or actual score
        if data.nunique() <= 5:
            sns.countplot(x=data, palette=[COLORS["danger"], COLORS["success"]],
                          edgecolor="white", ax=ax)
            ax.set_xlabel("Credit History")
        else:
            sns.histplot(data, kde=True, color=COLORS["primary_light"],
                         edgecolor="white", alpha=0.7, ax=ax)
            # Risk zones
            ax.axvspan(data.min(), 580, alpha=0.1, color=COLORS["danger"], label="High Risk (<580)")
            ax.axvspan(580, 700, alpha=0.1, color=COLORS["warning"], label="Medium Risk (580-700)")
            ax.axvspan(700, data.max(), alpha=0.1, color=COLORS["success"], label="Low Risk (>700)")
            ax.legend(frameon=True, fancybox=True)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig
    except Exception as e:
        return _error_figure(f"Credit score chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. APPROVAL STATUS (Donut)
# ─────────────────────────────────────────────────────────────────────────────

def plot_approval_counts(
    df: pd.DataFrame, status_col: str, title: str = "Loan Approval Status"
):
    """Plotly donut chart of approval/rejection counts."""
    try:
        counts = df[status_col].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=counts.index.astype(str),
            values=counts.values,
            hole=0.5,
            marker=dict(colors=[COLORS["success"], COLORS["danger"],
                                COLORS["warning"], COLORS["cyan"]][:len(counts)]),
            textinfo="label+percent",
            textfont_size=13,
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#ffffff")),
            showlegend=True,
            legend=dict(font=dict(color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=60, b=20, l=20, r=20),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Approval chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GENDER-WISE APPROVAL
# ─────────────────────────────────────────────────────────────────────────────

def plot_gender_approval(
    df: pd.DataFrame, gender_col: str, status_col: str,
    title: str = "Gender-wise Loan Approval",
):
    """Grouped bar chart comparing approval across genders."""
    try:
        cross = pd.crosstab(df[gender_col], df[status_col])
        fig = px.bar(
            cross, barmode="group",
            color_discrete_sequence=[COLORS["success"], COLORS["danger"],
                                     COLORS["warning"]],
            title=title,
        )
        fig.update_layout(
            xaxis_title=gender_col,
            yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=60, b=40, l=40, r=20),
            font=dict(color="#e2e8f0"),
            title=dict(font=dict(color="#ffffff", size=16)),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Gender approval chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, title: str = "Feature Correlation Matrix"
):
    """Seaborn heatmap (lower triangle) of numeric column correlations."""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return _error_figure("Need at least 2 numeric columns for correlation.")
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        return _error_figure(f"Heatmap error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. RISK SEGMENT DONUT
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_segments(
    risk_data: Dict[str, Any], title: str = "Borrower Risk Segmentation"
):
    """Donut chart showing Low / Medium / High risk distribution."""
    try:
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        values = [risk_data.get("low", 0), risk_data.get("medium", 0),
                  risk_data.get("high", 0)]
        colors = [RISK_COLORS["Low"], RISK_COLORS["Medium"], RISK_COLORS["High"]]

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.55,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            textfont_size=13,
        )])
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#ffffff")),
            legend=dict(font=dict(color="#e2e8f0")),
            paper_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=60, b=20, l=20, r=20),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Risk segment chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. RISK SUMMARY BAR
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_summary_bar(
    risk_data: Dict[str, Any], title: str = "Risk Category Distribution"
):
    """Horizontal bar chart of risk categories."""
    try:
        categories = ["Low Risk", "Medium Risk", "High Risk"]
        values = [risk_data.get("low", 0), risk_data.get("medium", 0),
                  risk_data.get("high", 0)]
        colors = [RISK_COLORS["Low"], RISK_COLORS["Medium"], RISK_COLORS["High"]]

        fig = go.Figure(go.Bar(
            y=categories, x=values, orientation="h",
            marker_color=colors,
            text=[f"{v} ({v / max(sum(values), 1) * 100:.1f}%)" for v in values],
            textposition="auto",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#ffffff")),
            xaxis_title="Number of Borrowers",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(t=60, b=40, l=100, r=20),
            font=dict(color="#e2e8f0"),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Risk bar error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. BOX PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_box_plot(
    df: pd.DataFrame, column: str, group_col: Optional[str] = None,
    title: Optional[str] = None,
):
    """Box plot for a numeric column, optionally grouped."""
    try:
        t = title or f"Distribution of {column}"
        if group_col and group_col in df.columns:
            fig = px.box(df, x=group_col, y=column, color=group_col,
                         color_discrete_sequence=PALETTE_SEQ, title=t)
        else:
            fig = px.box(df, y=column, color_discrete_sequence=[COLORS["cyan"]],
                         title=t)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=60, b=40, l=40, r=20),
            font=dict(color="#e2e8f0"),
            title=dict(font=dict(color="#ffffff", size=16)),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Box plot error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray, feature_names: List[str],
    title: str = "Feature Importance",
):
    """Horizontal bar chart of model feature importances."""
    try:
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors,
                edgecolor="white", height=0.6)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        return fig
    except Exception as e:
        return _error_figure(f"Feature importance error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray, labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
):
    """Heatmap-style confusion matrix with counts and percentages."""
    try:
        if labels is None:
            labels = [f"Class {i}" for i in range(cm.shape[0])]
        fig, ax = plt.subplots(figsize=(7, 6))
        total = cm.sum()
        annot = np.array([
            [f"{val}\n({val / total * 100:.1f}%)" for val in row]
            for row in cm
        ])
        sns.heatmap(cm, annot=annot, fmt="", xticklabels=labels,
                    yticklabels=labels, cmap="Blues", linewidths=1,
                    linecolor="white", ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        plt.tight_layout()
        return fig
    except Exception as e:
        return _error_figure(f"Confusion matrix error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. APPROVAL BY CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def plot_approval_by_category(
    df: pd.DataFrame, category_col: str, status_col: str,
    title: Optional[str] = None,
):
    """Grouped bar chart of approval status per category."""
    try:
        t = title or f"Loan Approval by {category_col}"
        cross = pd.crosstab(df[category_col], df[status_col])
        fig = px.bar(
            cross, barmode="group",
            color_discrete_sequence=PALETTE_SEQ,
            title=t,
        )
        fig.update_layout(
            xaxis_title=category_col,
            yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=60, b=40, l=40, r=20),
            font=dict(color="#e2e8f0"),
            title=dict(font=dict(color="#ffffff", size=16)),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Category approval error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. DEBT VS APPROVAL
# ─────────────────────────────────────────────────────────────────────────────

def plot_debt_vs_approval(
    df: pd.DataFrame, debt_col: str, status_col: str,
    title: str = "Debt vs Loan Approval",
):
    """Violin/box plot comparing debt levels by approval status."""
    try:
        fig = px.violin(
            df, x=status_col, y=debt_col, color=status_col,
            box=True, points="outliers",
            color_discrete_sequence=[COLORS["success"], COLORS["danger"]],
            title=title,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(t=60, b=40, l=40, r=20),
            font=dict(color="#e2e8f0"),
            title=dict(font=dict(color="#ffffff", size=16)),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Debt vs approval error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. SCATTER WITH TREND
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_with_trend(
    df: pd.DataFrame, x_col: str, y_col: str,
    color_col: Optional[str] = None, title: Optional[str] = None,
):
    """Plotly scatter with OLS trendline."""
    try:
        t = title or f"{y_col} vs {x_col}"
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            trendline="ols" if color_col is None else None,
            color_discrete_sequence=PALETTE_SEQ,
            title=t, opacity=0.6,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(t=60, b=40, l=40, r=20),
            font=dict(color="#e2e8f0"),
            title=dict(font=dict(color="#ffffff", size=16)),
        )
        return fig
    except Exception:
        # Fallback without trendline (statsmodels may not be installed)
        try:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                color_discrete_sequence=PALETTE_SEQ,
                title=title or f"{y_col} vs {x_col}", opacity=0.6,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                font=dict(color="#e2e8f0"),
                title=dict(font=dict(color="#ffffff", size=16)),
            )
            return fig
        except Exception as e:
            return _error_plotly(f"Scatter error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 15. RISK GAUGE
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_gauge(
    risk_score: float, title: str = "Overall Portfolio Risk"
):
    """Gauge/indicator chart showing overall risk level (0-100)."""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            title={"text": title, "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": COLORS["primary_light"]},
                "steps": [
                    {"range": [0, 33], "color": RISK_COLORS["Low"]},
                    {"range": [33, 66], "color": RISK_COLORS["Medium"]},
                    {"range": [66, 100], "color": RISK_COLORS["High"]},
                ],
                "threshold": {
                    "line": {"color": COLORS["accent"], "width": 4},
                    "thickness": 0.8,
                    "value": risk_score,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(t=50, b=20, l=30, r=30),
            font=dict(color="#e2e8f0"),
        )
        return fig
    except Exception as e:
        return _error_plotly(f"Gauge error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ERROR FALLBACK FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _error_figure(msg: str):
    """Return a matplotlib figure displaying an error message."""
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.text(0.5, 0.5, f"⚠️ {msg}", ha="center", va="center",
            fontsize=12, color=COLORS["danger"], transform=ax.transAxes)
    ax.axis("off")
    plt.tight_layout()
    return fig


def _error_plotly(msg: str):
    """Return a Plotly figure displaying an error message."""
    fig = go.Figure()
    fig.add_annotation(text=f"⚠️ {msg}", showarrow=False,
                       font=dict(size=14, color=COLORS["danger"]),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(color="#e2e8f0"),
    )
    return fig
