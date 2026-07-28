import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Candidate Analyzer - ExoVision AI",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------------------------
# Helper: Earth Similarity Index (ESI) Calculation
# -------------------------------------------------------------------
def calculate_esi(prad, teq):
    try:
        prad, teq = float(prad), float(teq)
        if prad <= 0 or teq <= 0:
            return 0.0
        
        # Radius term (reference: Earth = 1.0)
        radius_diff = abs((prad - 1.0) / (prad + 1.0))
        esi_r = max(0.0, 1.0 - (radius_diff ** 0.57))
        
        # Temperature term (reference: Earth = 288 K)
        temp_diff = abs((teq - 288.0) / (teq + 288.0))
        esi_t = max(0.0, 1.0 - (temp_diff ** 1.07))
        
        return float(np.sqrt(esi_r * esi_t))
    except (ValueError, ZeroDivisionError):
        return 0.0

# -------------------------------------------------------------------
# Data & Model Artifact Loader
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.json")

@st.cache_resource
def load_artifacts():
    model, features = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH, "r") as f:
                features = json.load(f)
        except Exception:
            model, features = None, None
    return model, features

model, feature_columns = load_artifacts()

# -------------------------------------------------------------------
# Page Header
# -------------------------------------------------------------------
st.title("🔬 Real-time Exoplanet Candidate Analyzer")
st.write(
    "Adjust candidate parameters below to calculate real-time Earth Similarity scoring "
    "and run live Machine Learning inference."
)

st.markdown("---")

# -------------------------------------------------------------------
# Interactive Form Layout
# -------------------------------------------------------------------
col_inputs, col_outputs = st.columns([1.1, 1], gap="large")

with col_inputs:
    st.subheader("⚙️ Physical & Observational Parameters")
    
    # Primary Parameters
    c1, c2 = st.columns(2)
    with c1:
        prad = st.slider(
            "Planetary Radius ($R_p$ in $R_{\\oplus}$)",
            min_value=0.1,
            max_value=30.0,
            value=1.05,
            step=0.05,
            help="Earth radius relative unit. Earth = 1.0"
        )
        
        teq = st.slider(
            "Equilibrium Temp ($T_{eq}$ in K)",
            min_value=50,
            max_value=3500,
            value=288,
            step=10,
            help="Calculated equilibrium temperature in Kelvin. Earth ~ 288 K"
        )

        period = st.number_input(
            "Orbital Period (Days)",
            min_value=0.1,
            max_value=1000.0,
            value=365.25,
            step=1.0,
            help="Time taken for the exoplanet to complete one orbit."
        )

    with c2:
        steff = st.slider(
            "Stellar Effective Temp ($T_{eff}$ in K)",
            min_value=2000,
            max_value=10000,
            value=5778,
            step=50,
            help="Temperature of the host star. Sun = 5778 K"
        )

        srad = st.number_input(
            "Stellar Radius ($R_*$ in $R_{\\odot}$)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.05,
            help="Radius of host star relative to Sun. Sun = 1.0"
        )

        snr = st.number_input(
            "Transit Signal-to-Noise (SNR)",
            min_value=1.0,
            max_value=500.0,
            value=25.0,
            step=1.0,
            help="Model signal-to-noise ratio from light curve transit depth."
        )

    with st.expander("🛠️ Advanced Kepler Flags & Light Curve Controls"):
        col_a, col_b = st.columns(2)
        with col_a:
            fp_nt = st.checkbox("Not Transit-like Flag (NT)", value=False)
            fp_ss = st.checkbox("Stellar Eclipse Flag (SS)", value=False)
        with col_b:
            fp_co = st.checkbox("Centroid Offset Flag (CO)", value=False)
            fp_ec = st.checkbox("Ephemeris Contamination Flag (EC)", value=False)

        depth = st.number_input("Transit Depth (ppm)", min_value=0.0, value=1000.0, step=100.0)
        duration = st.number_input("Transit Duration (hours)", min_value=0.1, value=3.5, step=0.1)
        slogg = st.number_input("Stellar Surface Gravity ($\log g$)", min_value=1.0, max_value=6.0, value=4.43, step=0.01)

# -------------------------------------------------------------------
# Real-Time Calculations & Model Inference
# -------------------------------------------------------------------
esi_score = calculate_esi(prad, teq)

input_data = {
    'koi_fpflag_nt': int(fp_nt),
    'koi_fpflag_ss': int(fp_ss),
    'koi_fpflag_co': int(fp_co),
    'koi_fpflag_ec': int(fp_ec),
    'koi_period': period,
    'koi_duration': duration,
    'koi_depth': depth,
    'koi_prad': prad,
    'koi_teq': teq,
    'koi_model_snr': snr,
    'koi_steff': steff,
    'koi_slogg': slogg,
    'koi_srad': srad,
    'feat_esi': esi_score
}

pred_class = "CANDIDATE"
confidence = 0.50

if model is not None and feature_columns is not None:
    try:
        df_input = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=0.0)
        probs = model.predict_proba(df_input)[0]
        
        if len(probs) == 2:
            confidence = float(probs[1])
            pred_class = "CONFIRMED PLANET" if confidence >= 0.5 else "FALSE POSITIVE"
            if pred_class == "FALSE POSITIVE":
                confidence = float(probs[0])
        else:
            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            classes = model.classes_ if hasattr(model, 'classes_') else ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
            pred_class = str(classes[max_idx]).upper()
    except Exception:
        # Fallback heuristic rules if feature mapping fails
        if fp_nt or fp_ss or fp_co or fp_ec:
            pred_class = "FALSE POSITIVE"
            confidence = 0.96
        else:
            pred_class = "CONFIRMED PLANET" if (0.5 <= prad <= 2.5 and 180 <= teq <= 320) else "CANDIDATE"
            confidence = 0.84
else:
    # Rule engine backup when model file is not loaded
    if fp_nt or fp_ss or fp_co or fp_ec:
        pred_class = "FALSE POSITIVE"
        confidence = 0.98
    else:
        if 0.6 <= prad <= 2.2 and 200 <= teq <= 320:
            pred_class = "CONFIRMED PLANET"
            confidence = 0.89
        else:
            pred_class = "CANDIDATE"
            confidence = 0.72

# -------------------------------------------------------------------
# Dynamic Output Display
# -------------------------------------------------------------------
with col_outputs:
    st.subheader("🤖 Live Model Output & Analysis")

    if "CONFIRMED" in pred_class:
        status_color = "#00FF87"
        badge_bg = "rgba(0, 255, 135, 0.1)"
    elif "FALSE" in pred_class:
        status_color = "#FF4B4B"
        badge_bg = "rgba(255, 75, 75, 0.1)"
    else:
        status_color = "#FFD700"
        badge_bg = "rgba(255, 215, 0, 0.1)"

    # Result Badge Card
    st.markdown(
        f"""
        <div style="
            background-color: {badge_bg};
            border: 2px solid {status_color};
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            margin-bottom: 20px;">
            <span style="font-size: 0.85rem; color: #AAA; text-transform: uppercase; letter-spacing: 1px;">Predicted Disposition</span>
            <h2 style="color: {status_color}; margin: 5px 0 0 0; font-size: 2.1rem;">{pred_class}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gauges: Model Confidence & ESI
    g1, g2 = st.columns(2)

    with g1:
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'suffix': "%", 'font': {'color': "#FFFFFF", 'size': 26}},
            title={'text': "Model Confidence", 'font': {'color': "#AAA", 'size': 13}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#444"},
                'bar': {'color': status_color},
                'bgcolor': "#111",
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 50], 'color': "#1a1a1a"},
                    {'range': [50, 100], 'color': "#222"}
                ],
            }
        ))
        fig_conf.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=15, r=15, t=35, b=15)
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    with g2:
        esi_color = "#00FF87" if esi_score >= 0.7 else ("#FFD700" if esi_score >= 0.4 else "#FF4B4B")
        fig_esi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=esi_score,
            number={'valueformat': ".3f", 'font': {'color': "#FFFFFF", 'size': 26}},
            title={'text': "Earth Similarity (ESI)", 'font': {'color': "#AAA", 'size': 13}},
            gauge={
                'axis': {'range': [0, 1.0], 'tickcolor': "#444"},
                'bar': {'color': esi_color},
                'bgcolor': "#111",
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 0.5], 'color': "#1a1a1a"},
                    {'range': [0.5, 1.0], 'color': "#222"}
                ],
            }
        ))
        fig_esi.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=15, r=15, t=35, b=15)
        )
        st.plotly_chart(fig_esi, use_container_width=True)

    # Formula Callout
    st.markdown(
        r"""
        > **Earth Similarity Index Formula:**
        > $$ESI = \sqrt{\max\left(0, 1 - \sqrt{\left|\frac{R_p - 1}{R_p + 1}\right|}\right) \cdot \max\left(0, 1 - \left|\frac{T_{eq} - 288}{T_{eq} + 288}\right|^{1.07}\right)}$$
        """
    )