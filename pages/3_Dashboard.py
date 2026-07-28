import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mission Dashboard - ExoVision AI", page_icon="📊", layout="wide")

# -------------------------------------------------------------------
# Helper: Earth Similarity Index (ESI) Calculation
# -------------------------------------------------------------------
def calculate_esi(prad, teq):
    try:
        prad, teq = float(prad), float(teq)
        radius_diff = abs((prad - 1.0) / (prad + 1.0))
        esi_r = max(0.0, 1.0 - (radius_diff ** 0.57))
        
        temp_diff = abs((teq - 288.0) / (teq + 288.0))
        esi_t = max(0.0, 1.0 - (temp_diff ** 1.07))
        
        if esi_r > 0 and esi_t > 0:
            return float(np.sqrt(esi_r * esi_t))
        return 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0

# -------------------------------------------------------------------
# Data & Artifact Loading
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATHS = [
    os.path.join(BASE_DIR, "kepler_dataset.csv"),
    os.path.join(BASE_DIR, "cumulative.csv"),
    os.path.join(BASE_DIR, "koi_data.csv")
]
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.json")

@st.cache_data
def load_dataset():
    target_path = None
    for path in DATA_PATHS:
        if os.path.exists(path):
            target_path = path
            break
            
    if target_path is None:
        return None
        
    df = pd.read_csv(target_path)
    
    # Standardize disposition column name
    if 'koi_disposition' in df.columns:
        df['disposition'] = df['koi_disposition']
    elif 'disposition' not in df.columns:
        df['disposition'] = 'UNKNOWN'

    # Compute ESI if not already present
    if 'feat_esi' not in df.columns and 'koi_prad' in df.columns and 'koi_teq' in df.columns:
        df['feat_esi'] = df.apply(lambda r: calculate_esi(r.get('koi_prad', 1.0), r.get('koi_teq', 288.0)), axis=1)
        
    return df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH, "r") as f:
                features = json.load(f)
            return model, features
        except Exception:
            return None, None
    return None, None

df = load_dataset()
model, feature_columns = load_model()

# -------------------------------------------------------------------
# Main UI Logic
# -------------------------------------------------------------------
st.title("📊 Mission Analytics Dashboard")
st.write("Real-time aggregate data & distributions from the NASA Kepler Objects of Interest (KOI) catalog.")

if df is None:
    st.warning("⚠️ **Dataset CSV File Not Found!**")
    st.info("Please place your NASA dataset CSV file named `kepler_dataset.csv` in your project root directory.")
    st.stop()

st.markdown("---")

# -------------------------------------------------------------------
# Dynamic KPI Header
# -------------------------------------------------------------------
total_records = len(df)
confirmed_count = len(df[df['disposition'].str.upper().str.contains('CONFIRMED', na=False)])
fp_count = len(df[df['disposition'].str.upper().str.contains('FALSE', na=False)])
candidate_count = len(df[df['disposition'].str.upper().str.contains('CANDIDATE', na=False)])

avg_esi = df['feat_esi'].mean() if 'feat_esi' in df.columns else 0.0
avg_temp = df['koi_teq'].mean() if 'koi_teq' in df.columns else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total KOIs Analyzed", f"{total_records:,}")
kpi2.metric("Confirmed Planets", f"{confirmed_count:,}")
kpi3.metric("False Positives", f"{fp_count:,}")
kpi4.metric("Candidates", f"{candidate_count:,}")
kpi5.metric("Avg Earth Similarity", f"{avg_esi:.3f}")

st.markdown("---")

# -------------------------------------------------------------------
# Row 1: Real Visualizations
# -------------------------------------------------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("🪐 Planetary Radius vs. Equilibrium Temperature")
    if 'koi_teq' in df.columns and 'koi_prad' in df.columns:
        # Filter extreme outliers for clear rendering
        scatter_df = df[(df['koi_prad'] > 0) & (df['koi_prad'] < 50) & (df['koi_teq'] < 3500)].copy()
        
        fig_scatter = px.scatter(
            scatter_df,
            x="koi_teq",
            y="koi_prad",
            color="disposition",
            hover_data=['koi_period'] if 'koi_period' in df.columns else None,
            labels={
                "koi_teq": "Equilibrium Temp (K)",
                "koi_prad": "Planet Radius (Earth Radii)",
                "disposition": "Disposition"
            },
            color_discrete_map={
                "CONFIRMED": "#00FF87",
                "FALSE POSITIVE": "#FF4B4B",
                "CANDIDATE": "#FFD700"
            },
            template="plotly_dark",
            opacity=0.7
        )
        # Highlight Habitable Zone Temp Bounds (approx. 180K to 310K)
        fig_scatter.add_vrect(
            x0=180, x1=310,
            fillcolor="#00FF87", opacity=0.15,
            line_width=0,
            annotation_text="Habitable Zone Temp Range",
            annotation_position="top left"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with row1_col2:
    st.subheader("🥧 NASA KOI Disposition Distribution")
    disp_counts = df['disposition'].value_counts().reset_index()
    disp_counts.columns = ['disposition', 'count']
    
    fig_pie = px.pie(
        disp_counts,
        names="disposition",
        values="count",
        color="disposition",
        color_discrete_map={
            "CONFIRMED": "#00FF87",
            "FALSE POSITIVE": "#FF4B4B",
            "CANDIDATE": "#FFD700"
        },
        hole=0.4,
        template="plotly_dark"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------
# Row 2: Real Physical Distributions
# -------------------------------------------------------------------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📏 Actual KOI Planetary Radius Distribution")
    if 'koi_prad' in df.columns:
        # Limit to 30 Earth Radii for readability
        rad_df = df[df['koi_prad'] <= 30].dropna(subset=['koi_prad'])
        
        fig_rad = px.histogram(
            rad_df,
            x="koi_prad",
            color="disposition",
            nbins=50,
            labels={"koi_prad": "Planetary Radius (Earth Radii)", "count": "Frequency"},
            color_discrete_map={
                "CONFIRMED": "#00FF87",
                "FALSE POSITIVE": "#FF4B4B",
                "CANDIDATE": "#FFD700"
            },
            barmode="overlay",
            template="plotly_dark"
        )
        fig_rad.update_traces(opacity=0.75)  # FIXED: update_traces instead of update_layout
        st.plotly_chart(fig_rad, use_container_width=True)

with row2_col2:
    st.subheader("🔥 Actual Equilibrium Temperature Distribution")
    if 'koi_teq' in df.columns:
        temp_df = df.dropna(subset=['koi_teq'])
        
        fig_temp = px.histogram(
            temp_df,
            x="koi_teq",
            color="disposition",
            nbins=50,
            labels={"koi_teq": "Equilibrium Temperature (K)", "count": "Frequency"},
            color_discrete_map={
                "CONFIRMED": "#00FF87",
                "FALSE POSITIVE": "#FF4B4B",
                "CANDIDATE": "#FFD700"
            },
            barmode="overlay",
            template="plotly_dark"
        )
        fig_temp.update_traces(opacity=0.75)  # FIXED: update_traces instead of update_layout
        st.plotly_chart(fig_temp, use_container_width=True)

# -------------------------------------------------------------------
# Row 3: Model Confidence Distribution (Batch Inference)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Model Prediction Confidence Distribution")

if model is not None and feature_columns is not None:
    with st.spinner("Calculating batch prediction probabilities across dataset..."):
        try:
            # Reindex dataset to match exact feature set
            eval_df = df.reindex(columns=feature_columns, fill_value=0.0)
            
            # Run batch probability predictions
            probs = model.predict_proba(eval_df)[:, 1]
            conf_df = pd.DataFrame({"Confidence": probs, "disposition": df['disposition']})
            
            fig_conf = px.histogram(
                conf_df,
                x="Confidence",
                color="disposition",
                nbins=40,
                labels={"Confidence": "Model Confirmation Probability", "count": "Count"},
                color_discrete_map={
                    "CONFIRMED": "#00FF87",
                    "FALSE POSITIVE": "#FF4B4B",
                    "CANDIDATE": "#FFD700"
                },
                template="plotly_dark"
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        except Exception as e:
            st.info("Batch model confidence chart unavailable: Ensure feature column names in your CSV match your trained model.")
else:
    st.info("💡 Place your `best_model.pkl` and `feature_columns.json` in the project root folder to enable batch confidence analytics.")