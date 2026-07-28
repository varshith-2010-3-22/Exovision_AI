import streamlit as st

st.set_page_config(
    page_title="Mission Control - ExoVision AI",
    page_icon="👥",
    layout="wide"
)

# -------------------------------------------------------------------
# Page Header
# -------------------------------------------------------------------
st.title("👥 Mission Control & Developer Profiles")
st.write("Meet the team behind ExoVision AI, building machine learning models for exoplanet discovery.")

st.markdown("---")

# -------------------------------------------------------------------
# Team Grid Layout
# -------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px;">
            <h3 style="color: #00FF87; margin-bottom: 5px;">🚀 Lead AI Engineer & Developer</h3>
            <h4 style="color: #FFF; margin-top: 0;">Varshith</h4>
            <p style="color: #8B949E; font-size: 0.95rem;">Full-Stack & Machine Learning Developer</p>
            <hr style="border-color: #30363d;">
            <p style="color: #C9D1D9; line-height: 1.6;">
                Architected the <b>ExoVision AI</b> pipeline, responsible for data preprocessing, 
                feature engineering, real-time ML inference integration, and Streamlit dashboard design.
            </p>
            <div style="margin-top: 15px;">
                <span style="background: #21262d; border: 1px solid #30363d; color: #60EFFF; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; margin-right: 5px;">Python</span>
                <span style="background: #21262d; border: 1px solid #30363d; color: #00FF87; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; margin-right: 5px;">Scikit-Learn</span>
                <span style="background: #21262d; border: 1px solid #30363d; color: #FFD700; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; margin-right: 5px;">Plotly</span>
                <span style="background: #21262d; border: 1px solid #30363d; color: #FF4B4B; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem;">Streamlit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px;">
            <h3 style="color: #60EFFF; margin-bottom: 5px;">🌌 Mission Vision & Objectives</h3>
            <h4 style="color: #FFF; margin-top: 0;">Automating Deep Space Discovery</h4>
            <p style="color: #8B949E; font-size: 0.95rem;">NASA Open Data Challenge Initiative</p>
            <hr style="border-color: #30363d;">
            <p style="color: #C9D1D9; line-height: 1.6;">
                ExoVision AI aims to streamline human vetting of space telescope light curves, 
                reducing false-positive identification timelines and highlighting potentially habitable worlds across the galaxy.
            </p>
            <div style="margin-top: 15px;">
                <span style="background: #21262d; border: 1px solid #30363d; color: #00FF87; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; margin-right: 5px;">NASA Exoplanet Archive</span>
                <span style="background: #21262d; border: 1px solid #30363d; color: #60EFFF; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem;">Kepler KOI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------------------------------------------------
# Project Links & Resources
# -------------------------------------------------------------------
st.subheader("🔗 Resources & Documentation")

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.page_link("pages/1_Home.py", label="Return to Home", icon="🌌", use_container_width=True)

with col_res2:
    st.page_link("pages/2_Analyzer.py", label="Run Exoplanet Analyzer", icon="🔬", use_container_width=True)

with col_res3:
    st.page_link("pages/3_Dashboard.py", label="View Dashboard Analytics", icon="📊", use_container_width=True)