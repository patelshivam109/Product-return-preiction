from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Return Risk Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_scored_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "return_prediction_model.pkl"
)

FEATURES_PATH = (
    BASE_DIR
    / "models"
    / "model_features.pkl"
)

REPORTS_PATH = (
    BASE_DIR
    / "reports"
)

FIGURES_PATH = (
    REPORTS_PATH
    / "figures"
)

HIGH_RISK_PATH = (
    REPORTS_PATH
    / "High_Risk_Orders.csv"
)

MODEL_COMPARISON_PATH = (
    REPORTS_PATH
    / "model_comparison.csv"
)

# =====================================================
# STYLING
# =====================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #808080;
        margin-bottom: 1rem;
    }

    .insight-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_high_risk():
    return pd.read_csv(HIGH_RISK_PATH)

@st.cache_data
def load_model_results():
    return pd.read_csv(MODEL_COMPARISON_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)

# =====================================================
# LOAD RESOURCES
# =====================================================

df = load_data()

high_risk_df = load_high_risk()

results_df = load_model_results()

model = load_model()

features = load_features()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📦 Return Risk Analytics")

st.sidebar.markdown(
    """
    Predict product return risk and
    analyze return behavior using
    machine learning insights.
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Overview",
        "📊 EDA Insights",
        "🎯 Prediction Center",
        "⚠️ High-Risk Orders"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Return Prediction Model"
)