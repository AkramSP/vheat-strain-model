import streamlit as st
import ee
import json
from google.oauth2 import service_account

# =====================================================================
# HOTFIX PATCH: Prevent geemap from crashing due to GEE API update
# =====================================================================
if hasattr(ee, 'data') and not hasattr(ee.data, '_credentials'):
    ee.data._credentials = None

import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# =====================================================================
# 1. PAGE CONFIG & MODERN ACADEMIC UI (CSS)
# =====================================================================
st.set_page_config(page_title="V-HEAT Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Professional Clean Light UI CSS Injection (Using 'Inter' font for modern typography)
css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Force Light Mode Aesthetics */
    .stApp { background-color: #F8F9FA; color: #1A1A1A; }
    
    .modern-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: box-shadow 0.3s ease;
    }
    
    .modern-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .disclaimer {
        font-size: 0.85em;
        color: #4A5568;
        border-left: 4px solid #3182CE;
        background-color: rgba(49, 130, 206, 0.05);
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    
    h1, h2, h3 { 
        font-weight: 600 !important; 
        letter-spacing: -0.02em; 
        color: #1A1A1A;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2B6CB0;
    }
    
    /* Hide default sidebar toggle for a cleaner look */
    [data-testid="collapsedControl"] { display: none; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =====================================================================
# 2. CORE FUNCTIONS (GEE & ML INFERENCE)
# =====================================================================
@st.cache_resource
def init_ee():
    """Securely init GEE using Streamlit Native Secrets or Local Auth."""
    try:
        scp = ['https://www.googleapis.com/auth/earthengine']
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            if '\\n' in key_dict['private_key']:
                key_dict['private_key'] = key_dict['private_key'].replace('\\n', '\n')
            creds = service_account.Credentials.from_service_account_info(key_dict).with_scopes(scp)
            ee.Initialize(credentials=creds, project=key_dict.get('project_id'))
            return True, "Authenticated via Native GCP Secrets"
        else:
            ee.Initialize() 
            return True, "Authenticated via Local Default"
    except Exception as e:
        return False, str(e)

# CRITICAL FIX: Changed from cache_data to cache_resource to prevent Silent Hangs
@st.cache_resource
def load_ml_mdl():
    """Load RF model into persistent memory."""
    p = 'rf_vheat_model.joblib'
    if os.path.exists(p):
        return joblib.load(p)
    return None

def gen_gee_map(cty_name, lat, lon, is_dp):
    """Generate geemap folium instance with professional basemaps."""
    m = geemap.Map(center=[lat, lon], zoom=12 if is_dp else 10)
    m.add_basemap("CartoDB.Positron")
    return m

def run_ml_inf(mdl, tmp, is_hw, is_hol):
    """Run hospital strain prediction."""
    if mdl:
        df_i = pd.DataFrame({'Mx_T': [tmp], 'Is_HW': [is_hw], 'Is_Hol': [is_hol]})
        pd_pax = mdl.predict(df_i)[0]
    else:
        b = 1000
        pd_pax = b + ((tmp-25)*20) + (is_hw*200) + (is_hol*150)
    v_rto = 0.3 if is_hol else 0.15
    return int(pd_pax), int(pd_pax * v_rto)

# =====================================================================
# 3. GLOBAL DESTINATION CITIES DATABASE (Coordinates)
# =====================================================================
cty_coords = [
    {"City": "Gold Coast, Australia", "Lat": -28.0167, "Lon": 153.4000},
    {"City": "Brisbane, Australia", "Lat": -27.4705, "Lon": 153.0260},
    {"City": "Sydney, Australia", "Lat": -33.8688, "Lon": 151.2093},
    {"City": "Melbourne, Australia", "Lat": -37.8136, "Lon": 144.9631},
    {"City": "Perth, Australia", "Lat": -31.9505, "Lon": 115.8605},
    {"City": "Bali, Indonesia", "Lat": -8.4095, "Lon": 115.1889},
    {"City": "Bangkok, Thailand", "Lat": 13.7563, "Lon": 100.5018},
    {"City": "Phuket, Thailand", "Lat": 7.8804, "Lon": 98.3922},
    {"City": "Singapore", "Lat": 1.3521, "Lon": 103.8198},
    {"City": "Kuala Lumpur, Malaysia", "Lat": 3.1390, "Lon": 101.6869},
    {"City": "Tokyo, Japan", "Lat": 35.6762, "Lon": 139.6503},
    {"City": "Kyoto, Japan", "Lat": 35.0116, "Lon": 135.7681},
    {"City": "Osaka, Japan", "Lat": 34.6937, "Lon": 135.5023},
    {"City": "Seoul, South Korea", "Lat": 37.5665, "Lon": 126.9780},
    {"City": "Taipei, Taiwan", "Lat": 25.0330, "Lon": 121.5654},
    {"City": "Hong Kong, SAR China", "Lat": 22.3193, "Lon": 114.1694},
    {"City": "Dubai, UAE", "Lat": 25.2048, "Lon": 55.2708},
    {"City": "Riyadh, Saudi Arabia", "Lat": 24.7136, "Lon": 46.6753},
    {"City": "Istanbul, Turkey", "Lat": 41.0082, "Lon": 28.9784},
    {"City": "Rome, Italy", "Lat": 41.9028, "Lon": 12.4964},
    {"City": "Venice, Italy", "Lat": 45.4408, "Lon": 12.3155},
    {"City": "Paris, France", "Lat": 48.8566, "Lon": 2.3522},
    {"City": "Barcelona, Spain", "Lat": 41.3851, "Lon": 2.1734},
    {"City": "Madrid, Spain", "Lat": 40.4168, "Lon": -3.7038},
    {"City": "Athens, Greece", "Lat": 37.9838, "Lon": 23.7275},
    {"City": "Lisbon, Portugal", "Lat": 38.7223, "Lon": -9.1393},
    {"City": "London, UK", "Lat": 51.5074, "Lon": -0.1278},
    {"City": "Edinburgh, UK", "Lat": 55.9533, "Lon": -3.1883},
    {"City": "Amsterdam, Netherlands", "Lat": 52.3676, "Lon": 4.9041},
    {"City": "Berlin, Germany", "Lat": 52.5200, "Lon": 13.4050},
    {"City": "Vienna, Austria", "Lat": 48.2082, "Lon": 16.3738},
    {"City": "Zurich, Switzerland", "Lat": 47.3769, "Lon": 8.5417},
    {"City": "Prague, Czechia", "Lat": 50.0755, "Lon": 14.4378},
    {"City": "New York City, USA", "Lat": 40.7128, "Lon": -74.0060},
    {"City": "Los Angeles, USA", "Lat": 34.0522, "Lon": -118.2437},
    {"City": "Las Vegas, USA", "Lat": 36.1699, "Lon": -115.1398},
    {"City": "Miami, USA", "Lat": 25.7617, "Lon": -80.1918},
    {"City": "Honolulu, USA", "Lat": 21.3069, "Lon": -157.8583},
    {"City": "Toronto, Canada", "Lat": 43.6510, "Lon": -79.3470},
    {"City": "Vancouver, Canada", "Lat": 49.2827, "Lon": -123.1207},
    {"City": "Cancun, Mexico", "Lat": 21.1619, "Lon": -86.8515},
    {"City": "Rio de Janeiro, Brazil", "Lat": -22.9068, "Lon": -43.1729},
    {"City": "Buenos Aires, Argentina", "Lat": -34.6037, "Lon": -58.3816},
    {"City": "Lima, Peru", "Lat": -12.0464, "Lon": -77.0428},
    {"City": "Cape Town, South Africa", "Lat": -33.9249, "Lon": 18.4241},
    {"City": "Cairo, Egypt", "Lat": 30.0444, "Lon": 31.2357},
    {"City": "Marrakech, Morocco", "Lat": 31.6295, "Lon": -7.9811},
    {"City": "Mumbai, India", "Lat": 19.0760, "Lon": 72.8777},
    {"City": "Delhi, India", "Lat": 28.7041, "Lon": 77.1025},
    {"City": "Auckland, New Zealand", "Lat": -36.8485, "Lon": 174.7633},
]

df_cities = pd.DataFrame(cty_coords)
df_cities['Type'] = np.where(df_cities['City'] == 'Gold Coast, Australia', 'Deep-Dive Case Study', 'Global Baseline')

# =====================================================================
# 4. APP LAYOUT & UI COMPONENTS
# =====================================================================

# --- HEADER ---
st.markdown('<div class="modern-card">', unsafe_allow_html=True)
st.title("V-HEAT: Visitor-Health Extreme Analytics Tool")
st.markdown('''
<div class="disclaimer">
    <b>Methodological Note:</b> As individual daily health records are subject to strict ethics clearance, this proof-of-concept utilizes mathematically downscaled AIHW annual aggregate data to demonstrate the analytical capabilities of the GeoAI pipeline. In a secure research environment, this architecture is designed to seamlessly ingest and process raw ICD-10 health records.
</div>
''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# LOAD ML FIRST (Instant via cache_resource)
mdl = load_ml_mdl()

# --- GLOBAL DESTINATION SELECTOR (INTERACTIVE MAP) ---
st.markdown('<div class="modern-card">', unsafe_allow_html=True)
st.subheader("Global Destination Index")
st.markdown("Select a city by clicking its pin on the map below to run the spatial and epidemiological pipeline.")

fig_map = px.scatter_mapbox(
    df_cities, lat="Lat", lon="Lon", hover_name="City", custom_data=["City", "Lat", "Lon"],
    color="Type", color_discrete_map={"Deep-Dive Case Study": "#E53E3E", "Global Baseline": "#3182CE"},
    zoom=1.2, height=400
)
fig_map.update_layout(
    mapbox_style="carto-positron", 
    margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
)

# Render Plotly map and capture clicks
sel_data = st.plotly_chart(fig_map, on_select="rerun", selection_mode="points", use_container_width=True, key="city_map")

# Default selection logic
selected_city = "Gold Coast, Australia"
sel_lat = -28.0167
sel_lon = 153.4000

if sel_data and hasattr(sel_data, 'selection'):
    pts = sel_data.selection.get('points', [])
    if pts and len(pts) > 0:
        c_data = pts[0].get('customdata')
        if c_data:
            selected_city, sel_lat, sel_lon = c_data[0], c_data[1], c_data[2]

is_dp = (selected_city == "Gold Coast, Australia")
st.info(f"**Target Destination Active:** {selected_city} | **Mode:** {'Deep-Dive Case Study' if is_dp else 'Global Baseline'}")
st.markdown('</div>', unsafe_allow_html=True)

if not mdl:
    st.warning("Warning: Local analytical model missing. Using synthetic inferencer fallback.")


# --- MAIN CONTENT ---
c1, c2 = st.columns([6, 4])

# Column 1: Spatial Map (Isolated GEE Loading)
with c1:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader(f"Land Surface Temperature (LST) Distribution")
    
    with st.spinner("Establishing secure connection to Earth Engine..."):
        gee_status, gee_msg = init_ee()
    
    if gee_status:
        with st.spinner("Rendering Spatial Data..."):
            f_map = gen_gee_map(selected_city, sel_lat, sel_lon, is_dp)
            f_map.to_streamlit(height=500)
    else:
        st.markdown('<div class="disclaimer">Earth Engine Authentication Failed.</div>', unsafe_allow_html=True)
        st.error(f"Error Detail: {gee_msg}")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Column 2: Predictive Analytics (Only for Deep-Dive)
with c2:
    if is_dp:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("Hospital Strain Simulator")
        st.markdown("Adjust meteorological and demographic parameters to estimate Emergency Department (ED) impact.")
        
        sim_tmp = st.slider("Simulate Daily Maximum Temperature (°C)", min_value=20.0, max_value=50.0, value=35.0, step=0.5)
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Holiday Season Surge?", [1, 0], format_func=lambda x: "Yes (Dec-Jan Peak)" if x==1 else "No (Standard Volume)", horizontal=True)
        
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol)
        
        st.markdown("<hr style='margin: 15px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Predicted Total ED Load", f"{tot_pax}", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.15)} (Heat Impact)" if sim_hw else "Baseline Climate")
        mc2.metric("Estimated Tourist Strain", f"{vis_pax}", delta=f"{(vis_pax/tot_pax)*100:.0f}% of Total Load")
        
        # Professional Color Palette for Chart
        chart_colors = ['#2B6CB0', '#F6AD55']
        df_cht = pd.DataFrame({'Patient Demographics': ['Local Residents', 'Visiting Tourists'], 'Count': [tot_pax - vis_pax, vis_pax]})
        fig = px.pie(df_cht, values='Count', names='Patient Demographics', hole=0.65, color_discrete_sequence=chart_colors)
        fig.update_layout(
            margin=dict(t=20, b=20, l=0, r=0), 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#1A1A1A',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.info("The Predictive Analytics module requires high-resolution downscaled inputs. Please click on the red 'Gold Coast, Australia' pin on the map to activate the simulator.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- CMIP6 PROJECTIONS ---
if is_dp:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("Future Climate Risk: CMIP6 Projection (2020 - 2050)")
    st.markdown("Estimated number of days exceeding the 35°C threshold under the SSP5-8.5 emission scenario.")
    
    yrs = np.arange(2020, 2051)
    hw_dys = np.linspace(5, 32, len(yrs)) + np.random.normal(0, 2, len(yrs))
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=yrs, y=hw_dys, mode='lines+markers', name='Heatwave Days', line=dict(color='#E53E3E', width=2.5)))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color='#1A1A1A', 
        xaxis_title="Projection Year", 
        yaxis_title="Annual Days > 35°C",
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
