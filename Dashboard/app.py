"""
Return Risk Analytics Dashboard
================================
Production-ready Streamlit application for ML internship project.
Predicts product return risk using Logistic Regression and provides
interactive business insights.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR      = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"

DATASET_PATH        = DATA_DIR  / "final_scored_dataset.csv"
MODEL_PATH          = MODELS_DIR / "return_prediction_model.pkl"
FEATURES_PATH       = MODELS_DIR / "model_features.pkl"
HIGH_RISK_PATH      = REPORTS_DIR / "High_Risk_Orders.csv"
MODEL_COMPARE_PATH  = REPORTS_DIR / "model_comparison.csv"

XGB_COMPARISON_ROW = {
    "Model": "XGBoost",
    "Accuracy": 0.6967,
    "Precision": 0.3958,
    "Recall": 0.2346,
    "F1": 0.2946,
    "ROC_AUC": 0.6482,
}

FIG_PATHS = {
    "return_distribution":    FIGURES_DIR / "return_distribution.png",
    "rating_distribution":    FIGURES_DIR / "rating_distribution.png",
    "price_histogram":        FIGURES_DIR / "price_histogram.png",
    "price_vs_return_status": FIGURES_DIR / "price_vs_return_status.png",
    "rating_vs_return":       FIGURES_DIR / "rating_vs_return.png",
    "correlation_matrix":     FIGURES_DIR / "correlation_matrix.png",
    "feature_importance":     FIGURES_DIR / "feature_importance.png",
    "lr_confusion_matrix":    FIGURES_DIR / "lr_confusion_matrix.png",
    "rf_confusion_matrix":    FIGURES_DIR / "rf_confusion_matrix.png",
    "xgb_confusion_matrix":   FIGURES_DIR / "xgb_confusion_matrix.png",
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Return Risk Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MINIMAL TARGETED STYLES
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* KPI cards */
    [data-testid="metric-container"] {
        background: var(--background-color, #f8f9fa);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 10px;
        padding: 16px 20px;
    }
    /* Insight cards */
    .insight-card {
        border-left: 4px solid #4F8EF7;
        background: rgba(79, 142, 247, 0.06);
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .insight-card strong { font-size: 0.95rem; }
    .insight-card p { margin: 4px 0 0; font-size: 0.85rem; opacity: 0.8; }
    /* Section headers */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.5;
        margin-bottom: 4px;
    }
    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 6px;
    }
    .risk-low    { background: #d4edda; color: #155724; }
    .risk-medium { background: #fff3cd; color: #856404; }
    .risk-high   { background: #f8d7da; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DATA LOADERS  (cached)
# ─────────────────────────────────────────────

@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        return None


@st.cache_data
def load_high_risk() -> pd.DataFrame | None:
    try:
        return pd.read_csv(HIGH_RISK_PATH)
    except FileNotFoundError:
        return None


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    try:
        comparison = pd.read_csv(MODEL_COMPARE_PATH)
        if "Model" not in comparison.columns and "model" in comparison.columns:
            comparison = comparison.rename(columns={"model": "Model"})

        if "Model" in comparison.columns and "XGBoost" not in comparison["Model"].astype(str).tolist():
            comparison = pd.concat([comparison, pd.DataFrame([XGB_COMPARISON_ROW])], ignore_index=True)

        return comparison
    except FileNotFoundError:
        return None


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_resource
def load_features() -> list[str] | None:
    try:
        return joblib.load(FEATURES_PATH)
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📦 Return Risk Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        options=[
            "🏠 Executive Overview",
            "📊 EDA Insights",
            "🎯 Prediction Center",
            "⚠️ High-Risk Orders",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("ML Internship Project · Logistic Regression")
    st.caption("Model: Return Risk Classifier")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Executive Overview":
    # Title block
    st.markdown("## Return Risk Analytics Dashboard")
    st.markdown(
        "Business intelligence platform for product return prediction "
        "and proactive order management."
    )
    st.markdown("---")

    # ── Load data ──
    df = load_dataset()
    hr = load_high_risk()
    mc = load_model_comparison()

    # ── KPI ROW ──
    st.markdown('<p class="section-label">Key Performance Indicators</p>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    if df is not None:
        total_orders  = len(df)
        return_col    = next((c for c in ["return_status", "returned", "Return_Status"] if c in df.columns), None)
        return_rate   = round(df[return_col].mean() * 100, 1) if return_col else "N/A"
        price_col     = next((c for c in ["price", "Price"] if c in df.columns), None)
        avg_price     = f"₹{df[price_col].mean():,.0f}" if price_col else "N/A"
    else:
        total_orders = return_rate = avg_price = "N/A"

    high_risk_count = len(hr) if hr is not None else "N/A"

    with kpi1:
        st.metric("Total Orders", f"{total_orders:,}" if isinstance(total_orders, int) else total_orders)
    with kpi2:
        st.metric("Return Rate", f"{return_rate}%" if isinstance(return_rate, float) else return_rate)
    with kpi3:
        st.metric("High-Risk Orders", f"{high_risk_count:,}" if isinstance(high_risk_count, int) else high_risk_count)
    with kpi4:
        st.metric("Avg. Order Price", avg_price)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts + Model Comparison side by side ──
    chart_col, table_col = st.columns([1.2, 1], gap="large")

    with chart_col:
        st.markdown('<p class="section-label">Risk Category Distribution</p>', unsafe_allow_html=True)
        risk_col = next((c for c in ["risk_category", "Risk_Category", "risk_level"] if df is not None and c in df.columns), None)
        if df is not None and risk_col:
            risk_counts = df[risk_col].value_counts().reset_index()
            risk_counts.columns = ["Category", "Count"]
            color_map = {"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"}
            fig = px.pie(
                risk_counts,
                names="Category",
                values="Count",
                hole=0.55,
                color="Category",
                color_discrete_map=color_map,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(
                showlegend=True,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk category column not found in dataset. Check column names.")

    with table_col:
        st.markdown('<p class="section-label">Model Performance Comparison</p>', unsafe_allow_html=True)
        if mc is not None:
            # Ensure numeric metrics are parsed correctly
            mc_display = mc.copy()
            # Accept either 'Model' or 'model' as the name column
            if "model" in mc_display.columns and "Model" not in mc_display.columns:
                mc_display = mc_display.rename(columns={"model": "Model"})

            metric_cols = [c for c in mc_display.columns if c != "Model"]
            if metric_cols:
                mc_display[metric_cols] = mc_display[metric_cols].apply(pd.to_numeric, errors="coerce")

            # Styled dataframe
            st.dataframe(
                mc_display.style.format(precision=3).highlight_max(
                    axis=0, props="background-color: rgba(79,142,247,0.15); font-weight: bold;"
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Grouped bar chart for metric comparison
            try:
                melted = mc_display.melt(id_vars=["Model"], value_vars=metric_cols, var_name="Metric", value_name="Value")
                fig_cmp = px.bar(
                    melted,
                    x="Model",
                    y="Value",
                    color="Metric",
                    barmode="group",
                    height=320,
                )
                fig_cmp.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_cmp, use_container_width=True)
            except Exception:
                # Fallback: if plotting fails, silently continue (table still shown)
                pass

            # Download button for the full comparison
            csv_bytes = mc_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Model Comparison CSV",
                data=csv_bytes,
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        else:
            st.warning("model_comparison.csv not found in reports/")

    st.markdown("---")

    # ── Business Insights ──
    st.markdown('<p class="section-label">Business Insights</p>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)

    insights = [
        ("⭐ Rating & Returns",    "Low product ratings (≤ 2 stars) are the strongest predictor of return probability. Monitoring rating trends enables early intervention."),
        ("🏆 Best Model",          "Logistic Regression delivered the best balance of precision and interpretability across all evaluated classifiers, making it ideal for production deployment."),
        ("🚨 Risk Identification", f"{high_risk_count:,} high-risk orders identified, enabling targeted outreach before returns are initiated." if isinstance(high_risk_count, int) else "High-risk orders flagged for proactive intervention."),
    ]
    for col, (title, body) in zip([ic1, ic2, ic3], insights):
        with col:
            st.markdown(
                f'<div class="insight-card"><strong>{title}</strong><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project Summary ──
    st.markdown('<p class="section-label">Project Summary</p>', unsafe_allow_html=True)
    with st.expander("About this project", expanded=True):
        st.markdown(
            """
            This dashboard is the analytical front-end for a supervised Machine Learning pipeline
            built to predict product return risk at the order level.

            **Pipeline overview:** Raw order data was cleaned and enriched with engineered features
            (low-rating flags, price buckets), then used to train Logistic Regression, Random Forest,
            and XGBoost classifiers. The best-performing model was selected based on F1-score and
            AUC-ROC, and deployed here for real-time inference.

            **Business value:** By surfacing high-risk orders before they are returned, operations
            and customer-success teams can intervene proactively — reducing return costs and improving
            customer satisfaction.
            """
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — EDA INSIGHTS
# ═══════════════════════════════════════════════════════════════════

elif page == "📊 EDA Insights":
    st.markdown("## Exploratory Data Analysis")
    st.markdown("Visual analysis of the training dataset, feature distributions, and model diagnostics.")
    st.markdown("---")

    def show_image(key: str, caption: str = "") -> None:
        path = FIG_PATHS.get(key)
        if path and path.exists():
            st.image(str(path), use_container_width=True, caption=caption)
        else:
            st.info(f"Figure not found: `{path.name if path else key}`")

    # ── Row 1: Return & Rating ──
    st.markdown('<p class="section-label">Target & Rating Distributions</p>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        show_image("return_distribution")
    with r1c2:
        show_image("rating_distribution")
    st.caption(
        "**Left:** Class balance between returned and non-returned orders. "
        "**Right:** Distribution of customer ratings across the dataset."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Price ──
    st.markdown('<p class="section-label">Price Analysis</p>', unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        show_image("price_histogram")
    with r2c2:
        show_image("price_vs_return_status")
    st.caption(
        "**Left:** Price distribution across all orders. "
        "**Right:** Price comparison between returned and non-returned orders."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Rating vs Return & Correlation ──
    st.markdown('<p class="section-label">Feature Relationships</p>', unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2, gap="medium")
    with r3c1:
        show_image("rating_vs_return")
    with r3c2:
        show_image("correlation_matrix")
    st.caption(
        "**Left:** Average return rate segmented by customer rating. "
        "**Right:** Pearson correlation heatmap across all numerical features."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance ──
    st.markdown('<p class="section-label">Feature Importance</p>', unsafe_allow_html=True)
    show_image("feature_importance")
    st.caption("Relative importance of each feature in the Logistic Regression model (absolute coefficient values).")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrices ──
    st.markdown('<p class="section-label">Model Confusion Matrices</p>', unsafe_allow_html=True)
    cm1, cm2, cm3 = st.columns(3, gap="medium")
    with cm1:
        st.markdown("**Logistic Regression**")
        show_image("lr_confusion_matrix")
    with cm2:
        st.markdown("**Random Forest**")
        show_image("rf_confusion_matrix")
    with cm3:
        st.markdown("**XGBoost**")
        show_image("xgb_confusion_matrix")
    st.caption(
        "Confusion matrices for all three classifiers evaluated during model selection. "
        "Logistic Regression was selected for deployment based on overall performance."
    )


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION CENTER
# ═══════════════════════════════════════════════════════════════════

elif page == "🎯 Prediction Center":
    st.markdown("## Prediction Center")
    st.markdown("Enter order details to predict return risk using the trained Logistic Regression model.")
    st.markdown("---")

    model    = load_model()
    features = load_features()

    if model is None:
        st.error(
            "**Model file not found.** "
            f"Expected at: `{MODEL_PATH}`. "
            "Ensure the model has been trained and saved correctly."
        )
        st.stop()

    if features is None:
        st.error(
            "**Feature list file not found.** "
            f"Expected at: `{FEATURES_PATH}`. "
            "Ensure `model_features.pkl` was saved during training."
        )
        st.stop()

    # ── Input form ──
    st.markdown('<p class="section-label">Order Details</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        price = st.number_input(
            "Price (₹)",
            min_value=0.0,
            max_value=100_000.0,
            value=500.0,
            step=10.0,
            help="Selling price of the product in INR.",
        )
        rating = st.slider(
            "Customer Rating",
            min_value=1,
            max_value=5,
            value=3,
            help="Product rating given by the customer (1 = lowest, 5 = highest).",
        )

    with col_b:
        product_order_count = st.number_input(
            "Product Order Count",
            min_value=1,
            max_value=10_000,
            value=50,
            step=1,
            help="Total number of times this product has been ordered.",
        )
        price_bucket = st.selectbox(
            "Price Bucket",
            options=["Low", "Medium", "High"],
            index=1,
            help="Categorical bucket the product price falls into.",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    if predict_btn:
        # ── Feature Engineering ──
        low_rating_flag         = 1 if rating <= 2 else 0
        price_bucket_Low        = 1 if price_bucket == "Low"    else 0
        price_bucket_Medium     = 1 if price_bucket == "Medium" else 0

        raw_input = {
            "price":                price,
            "rating":               rating,
            "product_order_count":  product_order_count,
            "low_rating_flag":      low_rating_flag,
            "price_bucket_Low":     price_bucket_Low,
            "price_bucket_Medium":  price_bucket_Medium,
        }

        # Build dataframe in exact feature order
        try:
            input_df = pd.DataFrame([{feat: raw_input.get(feat, 0) for feat in features}])
        except Exception as e:
            st.error(f"Error constructing feature vector: {e}")
            st.stop()

        # ── Predict ──
        try:
            prob = float(model.predict_proba(input_df)[0][1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # ── Risk Category ──
        if prob < 0.30:
            risk_cat   = "Low Risk"
            badge_cls  = "risk-low"
            rec        = "No immediate action required. Order appears low-risk — proceed normally."
            bar_color  = "normal"
        elif prob < 0.60:
            risk_cat   = "Medium Risk"
            badge_cls  = "risk-medium"
            rec        = "Monitor customer feedback closely. Consider a follow-up message post-delivery."
            bar_color  = "normal"
        else:
            risk_cat   = "High Risk"
            badge_cls  = "risk-high"
            rec        = "Consider proactive intervention — quality check, personalised outreach, or flexible return policy."
            bar_color  = "normal"

        # ── Display Results ──
        st.markdown("---")
        st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)

        res1, res2 = st.columns([1, 1.4], gap="large")

        with res1:
            st.metric("Return Probability", f"{prob * 100:.1f}%")
            st.markdown(
                f'<span class="risk-badge {badge_cls}">{risk_cat}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(prob, text=f"{prob * 100:.1f}% probability of return")

        with res2:
            st.markdown("**Business Recommendation**")
            st.info(rec)
            with st.expander("Feature vector used for inference"):
                st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — HIGH-RISK ORDERS
# ═══════════════════════════════════════════════════════════════════

elif page == "⚠️ High-Risk Orders":
    st.markdown("## High-Risk Orders")
    st.markdown("Orders flagged with a return probability above the high-risk threshold (> 60%).")
    st.markdown("---")

    hr = load_high_risk()

    if hr is None:
        st.error(
            "**High-Risk Orders file not found.** "
            f"Expected at: `{HIGH_RISK_PATH}`. "
            "Run the scoring pipeline to generate this report."
        )
        st.stop()

    # ── KPIs ──
    st.markdown('<p class="section-label">Summary</p>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)

    score_col = next(
        (c for c in ["return_probability", "risk_score", "predicted_prob", "score"] if c in hr.columns),
        None,
    )
    avg_score = hr[score_col].mean() if score_col else None

    with h1:
        st.metric("Total High-Risk Orders", f"{len(hr):,}")
    with h2:
        st.metric(
            "Avg. Risk Score",
            f"{avg_score * 100:.1f}%" if avg_score is not None else "N/A",
        )
    with h3:
        price_col = next((c for c in ["price", "Price"] if c in hr.columns), None)
        st.metric(
            "Avg. Order Price",
            f"₹{hr[price_col].mean():,.0f}" if price_col else "N/A",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter / Search ──
    st.markdown('<p class="section-label">Filter & Explore</p>', unsafe_allow_html=True)

    search_col, filter_col = st.columns([2, 1], gap="medium")

    with search_col:
        search_text = st.text_input(
            "Search",
            placeholder="Filter by any column value…",
            label_visibility="collapsed",
        )

    with filter_col:
        if score_col:
            min_risk = st.slider(
                "Minimum Risk Score",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.01,
                format="%.2f",
            )
        else:
            min_risk = None

    # Apply filters
    display_df = hr.copy()

    if search_text:
        mask = display_df.apply(
            lambda col: col.astype(str).str.contains(search_text, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    if score_col and min_risk is not None:
        display_df = display_df[display_df[score_col] >= min_risk]

    st.caption(f"Showing {len(display_df):,} of {len(hr):,} orders")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download ──
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Filtered CSV",
        data=csv_bytes,
        file_name="High_Risk_Orders_filtered.csv",
        mime="text/csv",
        type="primary",
    )