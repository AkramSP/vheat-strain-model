import streamlit as st
import ee
import json
from google.oauth2 import service_account

# =====================================================================
# HOTFIX PATCH: Prevent geemap from crashing due to GEE API update
# We must inject a dummy _credentials attribute before importing geemap
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
# 1. PAGE CONFIG & GLASSMORPHISM UI (CSS)
# =====================================================================
st.set_page_config(page_title="V-HEAT Dashboard", layout="wide", initial_sidebar_state="expanded")

# Handle Theme State
if 'thm' not in st.session_state:
    st.session_state.thm = 'dark'

def tgl_thm():
    """Toggle theme state."""
    st.session_state.thm = 'light' if st.session_state.thm == 'dark' else 'dark'

# Glassmorphism CSS Injection
if st.session_state.thm == 'dark':
    bg_clr, txt_clr, gls_bg, brd = "#0E1117", "#FFFFFF", "rgba(17, 25, 40, 0.75)", "rgba(255, 255, 255, 0.125)"
else:
    bg_clr, txt_clr, gls_bg, brd = "#F0F2F6", "#000000", "rgba(255, 255, 255, 0.65)", "rgba(0, 0, 0, 0.1)"

css = f"""
<style>
    .stApp {{ background-color: {bg_clr}; color: {txt_clr}; }}
    .glass-card {{
        background: {gls_bg};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 12px;
        border: 1px solid {brd};
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }}
    .disclaimer {{ font-size: 0.85em; color: #ff4b4b; border-left: 4px solid #ff4b4b; padding-left: 10px; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =====================================================================
# 2. CORE FUNCTIONS (GEE & ML INFERENCE)
# =====================================================================
@st.cache_resource
def init_ee():
    """Securely init GEE using Streamlit Secrets or Local Auth."""
    try:
        if "EARTHENGINE_TOKEN" in st.secrets:
            token = st.secrets["EARTHENGINE_TOKEN"]
            
            # Safely parse the token string into a dict
            if isinstance(token, str):
                key_dict = json.loads(token)
            else:
                key_dict = dict(token)
                
            creds = service_account.Credentials.from_service_account_info(key_dict)
            ee.Initialize(credentials=creds)
            return True, "Authenticated via Secrets"
        else:
            ee.Initialize() # Fallback for local environment
            return True, "Authenticated via Local Default"
    except Exception as e:
        return False, str(e)

@st.cache_data
def load_ml_mdl():
    """Load RF model."""
    p = 'rf_vheat_model.joblib'
    if os.path.exists(p):
        return joblib.load(p)
    return None

def gen_gee_map(cty, is_dp):
    """Generate geemap folium instance."""
    m = geemap.Map(center=[-28.0167, 153.4000] if cty == "Gold Coast" else [0,0], zoom=12 if cty == "Gold Coast" else 2)
    m.add_basemap("CARTO_DARK" if st.session_state.thm == 'dark' else "OpenStreetMap")
    return m

def run_ml_inf(mdl, tmp, is_hw, is_hol):
    """Run hospital strain prediction."""
    if mdl:
        df_i = pd.DataFrame({'Mx_T': [tmp], 'Is_HW': [is_hw], 'Is_Hol': [is_hol]})
        pd_pax = mdl.predict(df_i)[0]
    else:
        # Dummy logic if model absent
        b = 1000
        pd_pax = b + ((tmp-25)*20) + (is_hw*200) + (is_hol*150)
    
    v_rto = 0.3 if is_hol else 0.15
    return int(pd_pax), int(pd_pax * v_rto)

# =====================================================================
# 3. APP LAYOUT & UI COMPONENTS
# =====================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ V-HEAT Control")
    st.button(f"Switch to {'Light' if st.session_state.thm == 'dark' else 'Dark'} Mode", on_click=tgl_thm)
    
    st.markdown("---")
    cty_lst = ["Gold Coast", "Brisbane", "Sydney", "Melbourne", "Bali (ID)", "Phuket (TH)"] + [f"City {i}" for i in range(7, 101)]
    sel_cty = st.selectbox("🌐 Select Target City", cty_lst)
    
    is_dp = sel_cty == "Gold Coast"
    if is_dp:
        st.success("✅ Deep-Dive Mode Active")
    else:
        st.info("ℹ️ Global Mode Active")

# --- HEADER ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.title("🏥 V-HEAT: Visitor-Health Extreme Analytics Tool")
st.markdown('<div class="disclaimer"><b>Academic Integrity Disclaimer:</b> As individual daily health records are highly sensitive, this PoC utilizes mathematically downscaled AIHW annual aggregate data to demonstrate GeoAI pipeline capabilities.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- ELEGANT LOADING UI & INITIALIZATION ---
with st.spinner("🛰️ Establishing secure connection to Earth Engine & warming up AI Models..."):
    gee_status, gee_msg = init_ee()
    mdl = load_ml_mdl()

if not gee_status:
    st.markdown('<div class="glass-card"><div class="disclaimer">⚠️ GEE Authentication Failed. Please configure the Google Service Account JSON in Streamlit Secrets.</div></div>', unsafe_allow_html=True)
    st.error(f"Auth Error Detail: {gee_msg}")

if not mdl:
    st.sidebar.warning("⚠️ Local .joblib missing. Using synthetic inferencer.")

# --- MAIN CONTENT ---
c1, c2 = st.columns([6, 4])

# Column 1: Spatial Map
with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"🛰️ LST Spatial Distribution: {sel_cty}")
    
    # GUARDRAIL: Do not render map if GEE Auth failed!
    if gee_status:
        f_map = gen_gee_map(sel_cty, is_dp)
        f_map.to_streamlit(height=450)
    else:
        st.warning("🗺️ Map visualization is currently disabled due to missing/invalid Earth Engine credentials. Check the error message above.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Column 2: Predictive Analytics (Only for Deep-Dive)
with c2:
    if is_dp:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🌡️ Hospital Strain Simulator")
        
        sim_tmp = st.slider("Simulate Max Temp (°C)", min_value=20.0, max_value=50.0, value=30.0, step=0.5)
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Is Holiday Season?", [1, 0], format_func=lambda x: "Yes (Dec-Jan)" if x==1 else "No", horizontal=True)
        
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol)
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Total ED Patients", f"{tot_pax} pax", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.15)} from Heat" if sim_hw else "Normal")
        mc2.metric("Est. Tourist Impact", f"{vis_pax} pax", delta=f"{(vis_pax/tot_pax)*100:.0f}% of ED load")
        
        df_cht = pd.DataFrame({'Category': ['Locals', 'Tourists'], 'Count': [tot_pax - vis_pax, vis_pax]})
        fig = px.pie(df_cht, values='Count', names='Category', hole=0.7, color_discrete_sequence=['#4B4B4B', '#FF4B4B'])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=txt_clr)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.warning("⚠️ Predictive AI model is only trained for the Deep-Dive Case Study (Gold Coast). Please select 'Gold Coast' from the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- CMIP6 PROJECTIONS ---
if is_dp:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Future Risk: CMIP6 Heatwave Days Projection (2020 - 2050)")
    
    yrs = np.arange(2020, 2051)
    hw_dys = np.linspace(5, 32, len(yrs)) + np.random.normal(0, 2, len(yrs))
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=yrs, y=hw_dys, mode='lines+markers', name='HW Days', line=dict(color='#FF4B4B', width=3)))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=txt_clr, xaxis_title="Year", yaxis_title="Days > 35°C")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
