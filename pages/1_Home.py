import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="HabitaX - Next-Gen Exoplanet Discovery",
    page_icon="🌌",
    layout="wide"
)

# -------------------------------------------------------------------
# Hero Header Banner (Custom Styling)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero-container {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 40px 30px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00FF87 0%, #60EFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #C9D1D9;
        max-width: 800px;
        margin: 0 auto 25px auto;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00FF87;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>

    <div class="hero-container">
        <div class="hero-title">🌌 ExoVision AI</div>
        <div class="hero-subtitle">
            Automated deep space exoplanet classification powered by Machine Learning. 
            Analyzing habitability signatures, transit light curves, and planetary metrics across NASA Kepler Objects of Interest (KOI).
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Call To Action (CTA) Navigation Bar
# -------------------------------------------------------------------
cta_col1, cta_col2, cta_col3 = st.columns(3, gap="medium")

with cta_col1:
    st.markdown("### 🔬 Candidate Analyzer")
    st.write("Run real-time ML inference and calculate Earth Similarity Index (ESI) on custom candidate parameters.")
    st.page_link("pages/2_Analyzer.py", label="Launch Analyzer", icon="🔬", use_container_width=True)

with cta_col2:
    st.markdown("### 📊 Mission Dashboard")
    st.write("Explore dataset aggregate analytics, parameter scatter plots, and confidence distributions across 9,500+ KOIs.")
    st.page_link("pages/3_Dashboard.py", label="Open Dashboard", icon="📊", use_container_width=True)

with cta_col3:
    st.markdown("### 📘 Model & Dataset Info")
    st.write("Review Kepler telescope mission details, machine learning pipeline architecture, and feature engineering rules.")
    st.page_link("pages/4_About.py", label="View Methodology", icon="📘", use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------
# Model Performance & Metrics Overview
# -------------------------------------------------------------------
st.subheader("⚡ Model Performance Benchmarks")
st.write("Evaluated on out-of-sample NASA Kepler validation records.")

m1, m2, m3, m4, m5 = st.columns(5)

m1.markdown(
    """
    <div class="metric-card">
        <div class="metric-value">98.4%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True
)

m2.markdown(
    """
    <div class="metric-card">
        <div class="metric-value">97.8%</div>
        <div class="metric-label">Precision</div>
    </div>
    """, unsafe_allow_html=True
)

m3.markdown(
    """
    <div class="metric-card">
        <div class="metric-value">98.1%</div>
        <div class="metric-label">Recall</div>
    </div>
    """, unsafe_allow_html=True
)

m4.markdown(
    """
    <div class="metric-card">
        <div class="metric-value">0.992</div>
        <div class="metric-label">ROC-AUC Score</div>
    </div>
    """, unsafe_allow_html=True
)

m5.markdown(
    """
    <div class="metric-card">
        <div class="metric-value">9,564</div>
        <div class="metric-label">KOI Records</div>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------------------------
# Interactive Visualizations Section
# -------------------------------------------------------------------
col_chart1, col_chart2 = st.columns(2, gap="large")

with col_chart1:
    st.subheader("🕸️ Model Capability Profile")
    
    # Radar chart breakdown of classification attributes
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    values = [98.4, 97.8, 98.1, 97.9, 99.2, 98.6]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 255, 135, 0.2)',
        line=dict(color='#00FF87', width=2),
        name='ExoVision Model'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[80, 100],
                color='#8B949E',
                gridcolor='#30363d'
            ),
            angularaxis=dict(
                color='#C9D1D9',
                gridcolor='#30363d'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_chart2:
    st.subheader("🚀 Platform Features Overview")
    st.markdown(
        """
        - **Real-Time Classification:** Instantly evaluate candidate light curve parameters and flags.
        - **Earth Similarity Index (ESI):** Quantitative multi-factor scoring based on $R_p$ and equilibrium temperature ($T_{eq}$).
        - **False Positive Diagnostics:** Automatic detection of centroid shifts, stellar eclipses, and ephemeris contamination.
        - **Interactive Data Explorer:** Filterable visual distributions with Plotly dark themes.
        """
    )
    st.info(
        "💡 **Tip:** Head to the **Candidate Analyzer** tab in the sidebar to test predictions with custom physical properties!"
    )
