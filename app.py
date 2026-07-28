import streamlit as st
import os
import json
import joblib


# Automatically detect the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to your model artifacts
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.json")

# Load model and feature columns safely
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Missing model file at: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Missing feature file at: {FEATURES_PATH}")
        st.stop()
        
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_artifacts()

st.set_page_config(
    page_title="ExoVision AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Space Theme CSS Injection
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0E14;
        color: #E0E6ED;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2A5298 0%, #1E3C72 100%);
        box-shadow: 0px 0px 12px #00D2FF;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🌍 ExoVision AI")
st.sidebar.info("Navigate through the application sections above.")

# Main app entry point redirects to Home or displays Landing Summary
st.title("ExoVision AI Platform")
st.write("Welcome to the ExoVision AI hub. Select a page from the sidebar navigation to get started.")

st.markdown("""
* **1_Home**: Landing page & Overview
* **2_Analyzer**: Main Planet Prediction & SHAP Analysis Engine
* **3_Dashboard**: Mission Dataset & Model Analytics
* **4_About**: Methodology & NASA Kepler KOI Data Details
* **5_Team**: Contributor & Tech Stack Profiles
""")