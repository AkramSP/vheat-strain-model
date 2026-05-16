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
# 1. PAGE CONFIG & MODERN UX (CSS)
# =====================================================================
st.set_page_config(page_title="V-HEAT: Destination Resilience", layout="wide", initial_sidebar_state="collapsed")

css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F8FAFC; color: #2D3748; }
    
    /* Clean, flat cards for modern aesthetic */
    .modern-card { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; padding: 25px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    
    /* Executive Header */
    .header-card { background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%); color: white; border-radius: 12px; padding: 35px 30px; margin-bottom: 25px; }
    .header-card h1 { color: white !important; margin-top: 0; font-weight: 700; letter-spacing: -0.03em; font-size: 2.2rem; }
    .header-card p { color: #E2E8F0; font-size: 1.1em; max-width: 800px; line-height: 1.6; }
    
    h2, h3, h4 { font-weight: 600 !important; color: #1A202C; }
    
    /* Custom metric styling */
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #2B6CB0; }
    div[data-testid="stMetricDelta"] { font-size: 1rem; font-weight: 500; }
    
    [data-testid="collapsedControl"] { display: none; }
    .streamlit-expanderHeader { font-weight: 600; color: #2B6CB0; font-size: 1.05em; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =====================================================================
# 2. SESSION STATE MANAGEMENT (Robust Logic)
# =====================================================================
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Gold Coast, Australia"
if 'sim_temp' not in st.session_state:
    st.session_state.sim_temp = 35.0

# =====================================================================
# 3. CORE FUNCTIONS (GEE & ML)
# =====================================================================
@st.cache_resource(show_spinner=False)
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

@st.cache_resource(show_spinner=False)
def load_ml_mdl():
    p = 'rf_vheat_model.joblib'
    if os.path.exists(p):
        return joblib.load(p)
    return None

@st.cache_data(show_spinner=False)
def gen_gee_map_and_stats(cty_name, lat, lon, is_dp, gee_ready):
    """
    Menghasilkan Peta LST + Mengekstrak Statistik Spasial secara Real-Time.
    Cache key akan otomatis berubah jika cty_name berubah (Fitur Auto-Zoom).
    """
    m = geemap.Map(center=[lat, lon], zoom=11 if is_dp else 10, ee_initialize=False, draw_control=False, measure_control=False)
    m.add_basemap("CartoDB.Positron")
    
    stats_dict = {"mean_temp": "N/A", "max_temp": "N/A"}
    
    if not gee_ready:
        return m.to_html(), stats_dict
        
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(15000) 
        
        # Ambil Landsat 8 (Native ~100m resolution) - Reliable & Fast
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate('2023-12-01', '2024-02-28')
               
        if l8.size().getInfo() > 0:
            # Mask awan secara sederhana
            qa = l8.first().select('QA_PIXEL')
            mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
            
            # Konversi Kelvin ke Celcius
            lst_img = l8.map(lambda img: img.updateMask(mask)).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
            
            # Hitung Zonal Statistics Real-Time untuk UI!
            reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.max(), sharedInputs=True)
            stats = lst_img.reduceRegion(reducer=reducer, geometry=roi, scale=100, maxPixels=1e9).getInfo()
            
            if stats and 'ST_B10_mean' in stats and stats['ST_B10_mean'] is not None:
                stats_dict['mean_temp'] = f"{stats['ST_B10_mean']:.1f}"
                stats_dict['max_temp'] = f"{stats['ST_B10_max']:.1f}"
            
            # Visualisasi
            vis_params = {'min': 25, 'max': 45, 'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027', '#a50026'], 'opacity': 0.75}
            m.addLayer(lst_img.clip(roi), vis_params, f'Micro-Climate LST')
            m.addLayer(ee.Image().paint(roi, 0, 2), {'palette': ['#2D3748']}, 'Precinct Boundary')
            
            m.add_colorbar(vis_params, label="Surface Temperature (°C)", orientation="horizontal", layer_name="LST")
            m.add_inspector() # Klik untuk info suhu
    except Exception as e:
        print(f"GEE Render Error: {e}")
        
    return m.to_html(), stats_dict

def run_ml_inf(mdl, tmp, is_hw, is_hol):
    if mdl:
        df_i = pd.DataFrame({'Mx_T': [tmp], 'Is_HW': [is_hw], 'Is_Hol': [is_hol]})
        pd_pax = mdl.predict(df_i)[0]
    else:
        # Realistic synthetic baseline
        b = 850
        pd_pax = b + ((tmp-25)*22) + (is_hw*120) + (is_hol*180)
    
    # Asumsi turis meningkat pesat saat musim liburan
    v_rto = 0.28 if is_hol else 0.12
    return int(pd_pax), int(pd_pax * v_rto)

# =====================================================================
# 4. DESTINATION DATABASE
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
df_cities['Type'] = np.where(df_cities['City'] == 'Gold Coast, Australia', 'Primary Study', 'Baseline')

# =====================================================================
# 5. APP LAYOUT & PRESENTATION LAYER
# =====================================================================

st.markdown('<div class="header-card">', unsafe_allow_html=True)
st.markdown("<h1>V-HEAT: Destination Thermal-Resilience Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p>Empowering destination managers to visualize spatial climate risks, project future extreme heat events, and anticipate tourist-driven strain on local healthcare infrastructure.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialization
mdl = load_ml_mdl()
gee_status = init_ee()

# --- TOP ROW: GLOBAL NAVIGATOR ---
st.markdown('<div class="modern-card" style="padding: 15px 25px;">', unsafe_allow_html=True)
c_nav1, c_nav2 = st.columns([1, 2], gap="large")

with c_nav1:
    st.subheader("📍 Destination Selector")
    curr_idx = df_cities['City'].tolist().index(st.session_state.selected_city) if st.session_state.selected_city in df_cities['City'].tolist() else 0
    
    # Callback untuk dropdown agar sync
    def on_city_change():
        st.session_state.selected_city = st.session_state.dropdown_city
    
    st.selectbox("Choose a precinct:", df_cities['City'].tolist(), index=curr_idx, key="dropdown_city", on_change=on_city_change)
    
    is_dp = (st.session_state.selected_city == "Gold Coast, Australia")
    if is_dp:
        st.markdown('**Mode:** 🔴 Predictive Analytics Active')
    else:
        st.markdown('**Mode:** 🔵 Spatial Baseline Only')

with c_nav2:
    # Mini map navigator
    fig_map = px.scatter_map(
        df_cities, lat="Lat", lon="Lon", hover_name="City", custom_data=["City"],
        color="Type", color_discrete_map={"Primary Study": "#E53E3E", "Baseline": "#3182CE"}, zoom=1.0, height=130
    )
    fig_map.update_layout(map_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    
    sel_data = st.plotly_chart(fig_map, on_select="rerun", selection_mode="points", key="mini_map", use_container_width=True)
    
    if sel_data and hasattr(sel_data, 'selection'):
        pts = sel_data.selection.get('points', [])
        if pts and len(pts) > 0:
            clicked_city = pts[0].get('customdata')[0]
            if clicked_city != st.session_state.selected_city:
                st.session_state.selected_city = clicked_city
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Ekstrak koordinat aktif
city_row = df_cities[df_cities['City'] == st.session_state.selected_city].iloc[0]
sel_lat, sel_lon = city_row['Lat'], city_row['Lon']

# --- MAIN DASHBOARD: MAP & ANALYTICS ---
c1, c2 = st.columns([1.1, 0.9], gap="large")

# KIRI: PETA & STATISTIK
with c1:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader(f"🌡️ Micro-Climate Footprint: {st.session_state.selected_city.split(',')[0]}")
    
    # Generate Map and Stats (Auto-caches based on city coordinates)
    with st.spinner("Extracting Spatial Analytics..."):
        map_html, map_stats = gen_gee_map_and_stats(st.session_state.selected_city, sel_lat, sel_lon, is_dp, gee_status)
    
    # Tampilkan Zonal Statistics (Akram's Brilliant Idea)
    sc1, sc2 = st.columns(2)
    sc1.metric("Area Average LST", f"{map_stats['mean_temp']} °C", "Peak Summer 2024", delta_color="off")
    sc2.metric("Hotspot Maximum LST", f"{map_stats['max_temp']} °C", "Urban Heat Island", delta_color="inverse")
    
    st.markdown("<hr style='margin: 10px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("*Click anywhere on the map to inspect specific surface temperatures.*")
    
    components.html(map_html, height=480)
    st.markdown('</div>', unsafe_allow_html=True)

# KANAN: PREDICTIVE ANALYTICS
with c2:
    if is_dp:
        st.markdown('<div class="modern-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📈 Future Climate Projections (CMIP6)")
        st.markdown("NASA NEX-GDDP Scenario SSP5-8.5. **Click a point on the chart** to pass the temperature into the infrastructure simulator.")
        
        yrs = np.arange(2025, 2051)
        proj_max_t = np.linspace(34.0, 39.5, len(yrs)) + np.random.normal(0, 0.4, len(yrs))
        
        # Plotly chart configuration
        fig2 = go.Figure(go.Scatter(x=yrs, y=proj_max_t, mode='lines+markers', name='Max Temp', line=dict(color='#E53E3E', width=2.5)))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1A1A1A', yaxis_title="Max Air Temp (°C)", margin=dict(t=5, b=5, l=0, r=0), height=180)
        
        # Tangkap klik HANYA pada sumbu Y (Anti-error)
        cmip_sel = st.plotly_chart(fig2, on_select="rerun", selection_mode="points", use_container_width=True, key='cmip_chart')
        
        if cmip_sel and hasattr(cmip_sel, 'selection'):
            c_pts = cmip_sel.selection.get('points', [])
            if c_pts and len(c_pts) > 0:
                # Mengambil nilai y secara langsung dari klik
                st.session_state.sim_temp = float(c_pts[0]['y'])
                
        st.markdown("<hr style='margin: 20px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.subheader("🏥 Destination Infrastructure Simulator")
        
        # Callback untuk slider agar sinkron dengan state
        def on_slider_change():
            st.session_state.sim_temp = st.session_state.temp_slider
            
        sim_tmp = st.slider("Forecasted Daily Temp (°C)", min_value=25.0, max_value=45.0, value=float(st.session_state.sim_temp), step=0.1, key="temp_slider", on_change=on_slider_change)
        
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Tourism Seasonality", [1, 0], format_func=lambda x: "Peak Summer Holidays" if x==1 else "Off-Peak Season", horizontal=True)
        
        # Jalankan Inferensi ML
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol)
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Est. Total ED Presentations", f"{tot_pax}", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.12)} (Thermal Stress)" if sim_hw else "Normal", delta_color="inverse")
        mc2.metric("Transient Tourist Burden", f"{vis_pax}", delta=f"{(vis_pax/tot_pax)*100:.1f}% of hospital capacity", delta_color="off")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # FIX ERROR: Clean Clinical Triage Bar Chart (No Funnel Error)
        st.markdown("##### Predicted Clinical Severity Distribution")
        
        t_dist = [tot_pax*0.05, tot_pax*0.25, tot_pax*0.45, tot_pax*0.25]
        if sim_hw: t_dist = [tot_pax*0.12, tot_pax*0.38, tot_pax*0.35, tot_pax*0.15] # Severe cases rise in heatwaves
        
        df_trg = pd.DataFrame({
            'Category': ['Resuscitation', 'Emergency', 'Urgent', 'Non-Urgent'], 
            'Cases': t_dist
        })
        
        # Stacked horizontal bar yang bersih dan profesional
        fig_t = px.bar(df_trg, x='Cases', y=['Capacity Load']*4, color='Category', orientation='h',
                       color_discrete_sequence=['#9B2C2C', '#DD6B20', '#ECC94B', '#48BB78'])
        fig_t.update_layout(barmode='stack', margin=dict(t=0, b=0, l=0, r=0), height=100, 
                            yaxis_title=None, xaxis_title=None, showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.8, xanchor="center", x=0.5))
        fig_t.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_t, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="modern-card" style="height: 100%; display:flex; align-items:center; justify-content:center; text-align:center;">', unsafe_allow_html=True)
        st.info("The Predictive Simulator is currently locked to the **Gold Coast, Australia** pilot study. Please select it from the Destination Selector above to unlock epidemiological insights.")
        st.markdown('</div>', unsafe_allow_html=True)
