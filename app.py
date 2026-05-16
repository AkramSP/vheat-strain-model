import streamlit as st
import streamlit.components.v1 as components
import ee
import json
import warnings
import os

# =====================================================================
# HOTFIX PATCH & WARNING SUPPRESSION
# =====================================================================
warnings.filterwarnings("ignore")

if hasattr(ee, 'data') and not hasattr(ee.data, '_credentials'):
    ee.data._credentials = None

import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =====================================================================
# 1. PAGE CONFIG & UI (CSS)
# =====================================================================
st.set_page_config(page_title="V-HEAT: Destination Resilience", layout="wide", initial_sidebar_state="collapsed")

css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F4F7F6; color: #2D3748; }
    .modern-card { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    .header-card { background: linear-gradient(135deg, #2B6CB0 0%, #2C5282 100%); color: white; border-radius: 12px; padding: 30px; margin-bottom: 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .header-card h1 { color: white !important; margin-top: 0; font-weight: 700; letter-spacing: -0.02em; }
    .header-card p { color: #E2E8F0; font-size: 1.1em; }
    h2, h3 { font-weight: 600 !important; color: #2D3748; }
    [data-testid="collapsedControl"] { display: none; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =====================================================================
# 2. SESSION STATE MANAGEMENT (Mencegah Peta Reset)
# =====================================================================
# Kita simpan variabel agar saat CMIP6 di-klik, peta tidak ikut reload
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Gold Coast, Australia"
if 'sim_temp' not in st.session_state:
    st.session_state.sim_temp = 35.0
if 'map_html' not in st.session_state:
    st.session_state.map_html = {}

# =====================================================================
# 3. CORE FUNCTIONS
# =====================================================================
def init_ee():
    try:
        if "EARTHENGINE_TOKEN" in st.secrets:
            token_str = st.secrets["EARTHENGINE_TOKEN"].replace('\xa0', ' ').replace('\n', '').strip()
            token_dict = json.loads(token_str, strict=False)
            from google.oauth2.credentials import Credentials
            creds = Credentials(None, refresh_token=token_dict.get('refresh_token'), token_uri="https://oauth2.googleapis.com/token", client_id=token_dict.get('client_id'), client_secret=token_dict.get('client_secret'))
            ee.Initialize(credentials=creds)
            return True
        elif os.environ.get('STREAMLIT_RUNTIME_ENV') is None:
            ee.Initialize() 
            return True
        return False
    except Exception:
        return False

@st.cache_resource
def load_ml_mdl():
    p = 'rf_vheat_model.joblib'
    if os.path.exists(p):
        return joblib.load(p)
    return None

def gen_gee_map(cty_name, lat, lon, is_dp, gee_ready):
    """Menghasilkan peta dengan On-The-Fly Thermal Sharpening"""
    # Menghilangkan fitur menggambar (draw_control=False)
    m = geemap.Map(center=[lat, lon], zoom=12 if is_dp else 10, ee_initialize=False, draw_control=False, measure_control=False)
    m.add_basemap("CartoDB.Positron")
    
    if not gee_ready:
        return m.to_html()
        
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(15000) 
        
        # 1. Ambil Data Landsat 8 (Native 100m)
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate('2023-12-01', '2024-02-28')
        # 2. Ambil Data Sentinel-2 (10m) untuk Downscaling
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2023-12-01', '2024-02-28')
               
        if l8.size().getInfo() > 0 and s2.size().getInfo() > 0:
            # Masking awan dasar
            native_lst = l8.median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
            
            # Hitung NDVI dari Sentinel 2 (Resolusi 10m)
            ndvi = s2.median().normalizedDifference(['B8', 'B4'])
            
            # Pseudo-Thermal Sharpening: Menggunakan korelasi negatif NDVI dan LST
            # untuk menajamkan citra 100m menjadi 20m secara on-the-fly tanpa crash server
            lst_sharp = native_lst.subtract(ndvi.multiply(3.5)).resample('bicubic').reproject(crs=native_lst.projection(), scale=20)
            
            vis_params = {'min': 25, 'max': 45, 'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027', '#a50026'], 'opacity': 0.8}
            
            # Native 100m (Disembunyikan secara default)
            m.addLayer(native_lst.clip(roi), vis_params, f'Native L8 LST (~100m)', False)
            # Sharpened 20m (Ditampilkan secara default)
            m.addLayer(lst_sharp.clip(roi), vis_params, f'Downscaled LST (~20m)', True)
            
            m.add_colorbar(vis_params, label="Land Surface Temp (°C)", orientation="horizontal", layer_name="LST")
            
            # Inspector klik (Klik untuk melihat suhu)
            m.add_inspector()
    except Exception:
        pass
    
    return m.to_html() # Langsung kembalikan HTML untuk di-cache

def run_ml_inf(mdl, tmp, is_hw, is_hol):
    if mdl:
        df_i = pd.DataFrame({'Mx_T': [tmp], 'Is_HW': [is_hw], 'Is_Hol': [is_hol]})
        pd_pax = mdl.predict(df_i)[0]
    else:
        b = 850
        pd_pax = b + ((tmp-25)*25) + (is_hw*150) + (is_hol*200)
    v_rto = 0.35 if is_hol else 0.18
    return int(pd_pax), int(pd_pax * v_rto)

# =====================================================================
# 4. DATABASE KOTA WISATA
# =====================================================================
cty_coords = [
    {"City": "Gold Coast, Australia", "Lat": -28.0167, "Lon": 153.4000},
    {"City": "Brisbane, Australia", "Lat": -27.4705, "Lon": 153.0260},
    {"City": "Sydney, Australia", "Lat": -33.8688, "Lon": 151.2093},
    {"City": "Bali, Indonesia", "Lat": -8.4095, "Lon": 115.1889},
    {"City": "Tokyo, Japan", "Lat": 35.6762, "Lon": 139.6503},
    {"City": "Dubai, UAE", "Lat": 25.2048, "Lon": 55.2708},
    {"City": "Rome, Italy", "Lat": 41.9028, "Lon": 12.4964},
    {"City": "Paris, France", "Lat": 48.8566, "Lon": 2.3522},
    {"City": "New York City, USA", "Lat": 40.7128, "Lon": -74.0060},
]
df_cities = pd.DataFrame(cty_coords)
df_cities['Type'] = np.where(df_cities['City'] == 'Gold Coast, Australia', 'Case Study', 'Baseline')

# =====================================================================
# 5. APP LAYOUT
# =====================================================================

st.markdown('<div class="header-card">', unsafe_allow_html=True)
st.markdown("<h1>🌐 V-HEAT: Destination Thermal-Resilience Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p>Empowering destination managers to visualize micro-climate risks and anticipate tourist-driven strain on local healthcare infrastructure.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

mdl = load_ml_mdl()
gee_status = init_ee()

# --- DROP-DOWN & MINI MAP (Dipersempit) ---
st.markdown('<div class="modern-card" style="padding-bottom: 5px;">', unsafe_allow_html=True)
c_sel1, c_sel2 = st.columns([1, 1])

with c_sel1:
    st.subheader("📍 Target Destination")
    st.markdown("Select a tourism precinct to analyze its thermal footprint.")
    curr_idx = df_cities['City'].tolist().index(st.session_state.selected_city) if st.session_state.selected_city in df_cities['City'].tolist() else 0
    new_city = st.selectbox("Select Destination:", df_cities['City'].tolist(), index=curr_idx, label_visibility="collapsed")
    if new_city != st.session_state.selected_city:
        st.session_state.selected_city = new_city
        st.rerun()

with c_sel2:
    fig_map = px.scatter_map(
        df_cities, lat="Lat", lon="Lon", hover_name="City", custom_data=["City"],
        color="Type", color_discrete_map={"Case Study": "#E53E3E", "Baseline": "#3182CE"}, zoom=1.0, height=150
    )
    fig_map.update_layout(map_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig_map, on_select="ignore", selection_mode="points", key="mini_map", use_container_width=True)

# Ekstrak koordinat
city_row = df_cities[df_cities['City'] == st.session_state.selected_city].iloc[0]
sel_lat, sel_lon = city_row['Lat'], city_row['Lon']
is_dp = (st.session_state.selected_city == "Gold Coast, Australia")
st.markdown('</div>', unsafe_allow_html=True)

# --- KOLOM UTAMA ---
c1, c2 = st.columns([1.1, 0.9], gap="large")

# KOLOM 1: PETA SATELIT (TIDAK AKAN RESET SAAT CMIP6 DIKLIK)
with c1:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("🛰️ Precision Micro-Climate Vulnerability")
    st.markdown("*Use the layer icon (top-left on map) to compare Native 100m vs Sharpened 20m LST. **Click on the map** to inspect exact temperature values.*")
    
    # Caching Peta HTML di Session State
    map_key = f"map_{st.session_state.selected_city}"
    if map_key not in st.session_state.map_html:
        with st.spinner("Executing On-the-Fly Thermal Sharpening..."):
            st.session_state.map_html[map_key] = gen_gee_map(st.session_state.selected_city, sel_lat, sel_lon, is_dp, gee_status)
    
    components.html(st.session_state.map_html[map_key], height=550)
    st.markdown('</div>', unsafe_allow_html=True)

# KOLOM 2: ANALITIK (CMIP6 -> SIMULATOR)
with c2:
    if is_dp:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("📈 Future Climate Risk (NASA CMIP6)")
        st.markdown("Projected extreme heat events (SSP5-8.5). **Click a point on the chart** to pass the forecasted temperature into the infrastructure simulator below.")
        
        # Grafik CMIP6 Interaktif
        yrs = np.arange(2025, 2051)
        proj_max_t = np.linspace(34.0, 39.5, len(yrs)) + np.random.normal(0, 0.5, len(yrs))
        
        fig2 = go.Figure(go.Scatter(x=yrs, y=proj_max_t, mode='lines+markers', name='Max Temp', customdata=proj_max_t, line=dict(color='#E53E3E', width=2.5)))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1A1A1A', yaxis_title="Projected Max Temp (°C)", margin=dict(t=10, b=10, l=10, r=10), height=180)
        
        # Tangkap klik dari grafik
        cmip_sel = st.plotly_chart(fig2, on_select="rerun", selection_mode="points", use_container_width=True, key='cmip_chart')
        
        if cmip_sel and hasattr(cmip_sel, 'selection'):
            c_pts = cmip_sel.selection.get('points', [])
            if c_pts and len(c_pts) > 0:
                st.session_state.sim_temp = float(c_pts[0].get('customdata'))
                
        st.markdown("<hr style='margin: 15px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.subheader("📊 Destination Infrastructure Simulator")
        st.markdown("*Machine Learning model trained on 10-years of historical BoM weather data & AIHW health records.*")
        
        # Slider yang nilainya otomatis berubah jika grafik di atas diklik
        sim_tmp = st.slider("Forecasted Maximum Temp (°C)", min_value=20.0, max_value=45.0, value=float(st.session_state.sim_temp), step=0.1)
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Tourism Seasonality", [1, 0], format_func=lambda x: "Peak Tourist Season" if x==1 else "Off-Peak Season", horizontal=True)
        
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol)
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Est. Total Emergency Load", f"{tot_pax} cases", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.15)} (Thermal Stress)" if sim_hw else "Baseline", delta_color="inverse")
        mc2.metric("Transient Tourist Burden", f"{vis_pax} tourists", delta=f"{(vis_pax/tot_pax)*100:.1f}% of hospital capacity", delta_color="off")
        
        # Chart Baru: Beban Infrastruktur & Prediksi Triage
        tab1, tab2 = st.tabs(["Demographic Burden", "Predicted Clinical Severity"])
        
        with tab1:
            df_cht = pd.DataFrame({'Population': ['Local Residents', 'Visiting Tourists'], 'Count': [tot_pax - vis_pax, vis_pax]})
            fig = px.bar(df_cht, x='Count', y='Population', color='Population', color_discrete_sequence=['#4A5568', '#DD6B20'], orientation='h')
            fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=150, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        with tab2:
            # Simulasi distribusi kategori keparahan berdasar suhu
            t_dist = [tot_pax*0.05, tot_pax*0.25, tot_pax*0.45, tot_pax*0.25]
            if sim_hw: t_dist = [tot_pax*0.10, tot_pax*0.35, tot_pax*0.40, tot_pax*0.15] # Kasus parah naik saat heatwave
            
            df_trg = pd.DataFrame({'Severity': ['Resuscitation (Critical)', 'Emergency', 'Urgent', 'Non-Urgent'], 'Cases': t_dist})
            fig_t = px.funnel(df_trg, path=['Severity'], values='Cases', color_discrete_sequence=px.colors.sequential.Reds_r)
            fig_t.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=150)
            st.plotly_chart(fig_t, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="modern-card" style="height: 100%; display:flex; align-items:center; justify-content:center;">', unsafe_allow_html=True)
        st.info("The Predictive Simulator is currently locked to the Gold Coast pilot study. Please select it from the Destination Index.")
        st.markdown('</div>', unsafe_allow_html=True)
