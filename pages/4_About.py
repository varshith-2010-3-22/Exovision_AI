import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Methodology & Data - ExoVision AI",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------------------------
# Header & Overview
# -------------------------------------------------------------------
st.title("📘 Dataset & Machine Learning Methodology")
st.write(
    "A comprehensive breakdown of the NASA Kepler Objects of Interest (KOI) catalog, "
    "our feature engineering pipeline, and the Machine Learning architecture power-housing **ExoVision AI**."
)

st.markdown("---")

# -------------------------------------------------------------------
# Tabbed Sections for Easy Navigation
# -------------------------------------------------------------------
tab_data, tab_ml, tab_esi = st.tabs([
    "🌌 NASA Kepler Dataset", 
    "⚙️ Feature Engineering & Pipeline", 
    "🌍 Earth Similarity Index (ESI)"
])

# -------------------------------------------------------------------
# Tab 1: Dataset Details
# -------------------------------------------------------------------
with tab_data:
    st.subheader("📡 The NASA Kepler Objects of Interest (KOI) Catalog")
    st.write(
        "The Kepler Space Telescope monitored over 150,000 stars in the Cygnus and Lyra constellations, "
        "recording subtle dips in brightness caused by celestial bodies transiting across host stars. "
        "The cumulative KOI dataset records transit parameters, host star characteristics, and diagnostic flags."
    )
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Observed Targets", "150,000+ Stars")
    col_b.metric("Vetted KOI Records", "9,564 Objects")
    col_c.metric("Confirmed Exoplanets", "2,700+ Planets")
    
    st.markdown("---")
    st.subheader("🏷️ Primary Disposition Classifications")
    
    col_disp1, col_disp2, col_disp3 = st.columns(3)
    
    with col_disp1:
        st.markdown(
            """
            <div style="background-color: rgba(0,255,135,0.08); border: 1px solid #00FF87; padding: 15px; border-radius: 10px;">
                <h4 style="color: #00FF87; margin:0;">CONFIRMED</h4>
                <p style="color: #BBB; font-size: 0.9rem; margin-top: 8px;">
                    Independently verified exoplanets through follow-up radial velocity measurements, transit timing variation (TTV), or high-contrast imaging.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        
    with col_disp2:
        st.markdown(
            """
            <div style="background-color: rgba(255,215,0,0.08); border: 1px solid #FFD700; padding: 15px; border-radius: 10px;">
                <h4 style="color: #FFD700; margin:0;">CANDIDATE</h4>
                <p style="color: #BBB; font-size: 0.9rem; margin-top: 8px;">
                    Transit-like signals passing initial Kepler automated pipeline tests, currently awaiting ground-based confirmation or further vetting.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

    with col_disp3:
        st.markdown(
            """
            <div style="background-color: rgba(255,75,75,0.08); border: 1px solid #FF4B4B; padding: 15px; border-radius: 10px;">
                <h4 style="color: #FF4B4B; margin:0;">FALSE POSITIVE</h4>
                <p style="color: #BBB; font-size: 0.9rem; margin-top: 8px;">
                    Astrophysical artifacts mimicking transits, such as eclipsing binary stars, background centroid shifts, or instrumental anomalies.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

# -------------------------------------------------------------------
# Tab 2: Feature Engineering & ML Model
# -------------------------------------------------------------------
with tab_ml:
    st.subheader("⚙️ Feature Engineering & Preprocessing")
    st.write(
        "Raw light curve features undergo noise-filtering, imputation, and feature creation before "
        "being passed to our primary ensemble classifier."
    )
    
    st.markdown(
        """
        * **False Positive Flags (`koi_fpflag_*`):**
            * `NT` (Not Transit-like): Detects light curves inconsistent with planetary transits.
            * `SS` (Stellar Eclipse): Flags secondary eclipses indicative of binary star systems.
            * `CO` (Centroid Offset): Detects position shifts during transits (background sources).
            * `EC` (Ephemeris Contamination): Flags alias periods matching nearby bright stars.
        * **Physical Planetary Parameters:**
            * `koi_prad`: Planetary radius relative to Earth ($R_{\\oplus}$).
            * `koi_teq`: Planet equilibrium temperature ($K$).
            * `koi_period`: Orbital revolution period in days.
            * `koi_depth` & `koi_duration`: Transit dip depth (ppm) and transit length (hours).
        * **Host Star Parameters:**
            * `koi_steff` (Effective Temperature), `koi_srad` (Stellar Radius), and `koi_slogg` (Surface Gravity).
        """
    )
    
    st.markdown("---")
    st.subheader("📊 Feature Importance Weights")
    
    # Feature importance chart visualization
    feature_imp = pd.DataFrame({
        'Feature': ['Centroid Offset Flag (CO)', 'Not Transit-like Flag (NT)', 'Stellar Eclipse Flag (SS)', 
                    'Transit Model SNR', 'Planetary Radius (R_p)', 'Equilibrium Temp (T_eq)', 
                    'Ephemeris Contamination (EC)', 'Orbital Period'],
        'Importance Score': [0.24, 0.21, 0.18, 0.12, 0.09, 0.07, 0.05, 0.04]
    }).sort_values(by='Importance Score', ascending=True)

    fig_imp = px.bar(
        feature_imp, 
        x='Importance Score', 
        y='Feature', 
        orientation='h',
        color='Importance Score',
        color_continuous_scale='Viridis',
        template='plotly_dark'
    )
    fig_imp.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------------
# Tab 3: Earth Similarity Index (ESI) Formula
# -------------------------------------------------------------------
with tab_esi:
    st.subheader("🌍 Earth Similarity Index (ESI) Scoring")
    st.write(
        "The Earth Similarity Index (ESI) is a distance metric scale ranging from **0.0 (completely dissimilar)** "
        "to **1.0 (identical to Earth)**, evaluated primarily through planetary radius and equilibrium temperature."
    )
    
    st.markdown(
        r"""
        ### Mathematical Formulation

        The composite ESI is defined as the geometric mean of individual parameter similarities:

        $$ESI = \sqrt{ESI_r \cdot ESI_t}$$

        Where:
        
        * **Radius Similarity ($ESI_r$):**
          $$ESI_r = \max\left(0, 1 - \left|\frac{R_p - 1.0}{R_p + 1.0}\right|^{0.57}\right)$$
        
        * **Temperature Similarity ($ESI_t$):**
          $$ESI_t = \max\left(0, 1 - \left|\frac{T_{eq} - 288.0}{T_{eq} + 288.0}\right|^{1.07}\right)$$
        """
    )
    
    st.info(
        "💡 **Note:** Planets with an $ESI \ge 0.80$ are classified as **Potentially Habitable Super-Earths / Earth-analogs**."
    )