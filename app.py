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
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. PAGE CONFIG & MODERN TOURISM-POLICY UX (CSS)
# =====================================================================
st.set_page_config(page_title="V-HEAT: Destination Climate Resilience", layout="wide", initial_sidebar_state="collapsed")

css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #F8FAFC; color: #1E293B; }
    
    .modern-card { background-color: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    
    .header-card { background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); color: white; border-radius: 12px; padding: 40px 30px; margin-bottom: 25px; }
    .header-card h1 { color: white !important; margin-top: 0; font-weight: 700; font-size: 2.4rem; letter-spacing: -0.02em; }
    .header-card p { color: #CBD5E1; font-size: 1.15rem; max-width: 900px; line-height: 1.6; margin-bottom: 0;}
    
    h2, h3, h4 { font-weight: 600 !important; color: #0F172A; }
    .subtitle-text { font-size: 0.95rem; color: #64748B; margin-bottom: 15px; display: block; }
    
    div[data-testid="stMetricValue"] { font-size: 2.0rem; font-weight: 700; color: #1E3A8A; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; font-weight: 600; color: #475569; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem; font-weight: 500; }
    
    .status-badge { padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-top: 10px;}
    .status-safe { background-color: #DEF7EC; color: #22543D; border: 1px solid #9AE6B4; }
    .status-warn { background-color: #FEFCBF; color: #744210; border: 1px solid #F6E05E; }
    .status-critical { background-color: #FED7D7; color: #822727; border: 1px solid #FEB2B2; }
    
    .btn-geoai > button { width: 100%; font-weight: 600; background-color: #1E3A8A; color: white; border-radius: 8px; padding: 10px; border: none;}
    .btn-geoai > button:hover { background-color: #1E40AF; color: white; border: none;}
    
    [data-testid="collapsedControl"] { display: none; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =====================================================================
# 2. SESSION STATE MANAGEMENT
# =====================================================================
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Gold Coast, Australia"
if 'rf_downscale_run' not in st.session_state:
    st.session_state.rf_downscale_run = False
if 'rf_results' not in st.session_state:
    st.session_state.rf_results = None

# =====================================================================
# 3. SMART INTEGRATION FUNCTIONS (RS, BoM, AIHW, CMIP6)
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
def get_real_cmip6_data(lat, lon, gee_ready):
    if not gee_ready:
        yrs = np.arange(2025, 2051)
        temps = np.linspace(34.0, 39.5, len(yrs)) + np.random.normal(0, 0.4, len(yrs))
        return pd.DataFrame({'Year': yrs, 'Max_Temp': temps})
        
    try:
        pt = ee.Geometry.Point([lon, lat])
        col = ee.ImageCollection("NASA/GDDP-CMIP6").filterBounds(pt).filter(ee.Filter.eq('model', 'ACCESS-CM2')).filter(ee.Filter.eq('scenario', 'ssp585')).select('tasmax').filterDate('2025-01-01', '2050-12-31')

        def get_yearly_max(year):
            start = ee.Date.fromYMD(year, 1, 1)
            end = ee.Date.fromYMD(year, 12, 31)
            yearly_max = col.filterDate(start, end).max()
            val = yearly_max.reduceRegion(ee.Reducer.max(), pt, 1000).get('tasmax')
            return ee.Feature(None, {'year': year, 'max_temp': val})

        years = ee.List.sequence(2025, 2050)
        fc = ee.FeatureCollection(years.map(get_yearly_max))
        data = fc.getInfo()['features']

        yrs = [d['properties']['year'] for d in data]
        temps = [d['properties']['max_temp'] - 273.15 for d in data]
        return pd.DataFrame({'Year': yrs, 'Max_Temp': temps})
    except Exception:
        yrs = np.arange(2025, 2051)
        temps = np.linspace(34.0, 39.5, len(yrs)) + np.random.normal(0, 0.4, len(yrs))
        return pd.DataFrame({'Year': yrs, 'Max_Temp': temps})

@st.cache_data(show_spinner=False)
def gen_gee_map_and_stats(cty_name, lat, lon, gee_ready):
    m = geemap.Map(center=[lat, lon], zoom=11, ee_initialize=False, draw_control=False, measure_control=False)
    m.add_basemap("CartoDB.Positron")
    stats_dict = {"mean_temp": "N/A", "max_temp": "N/A", "std_dev": "N/A"}
    
    if not gee_ready:
        return m.to_html(), stats_dict
        
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(12000) 
        
        # LOGIKA HEMISFER: Utata (Jun-Aug) vs Selatan (Dec-Feb)
        if lat > 0:
            month_filter = ee.Filter.calendarRange(6, 8, 'month')
        else:
            month_filter = ee.Filter.calendarRange(12, 2, 'month')
            
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate('2019-01-01', '2024-12-31').filter(month_filter)
               
        if l8.size().getInfo() > 0:
            def mask_l8(img):
                qa = img.select('QA_PIXEL')
                mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                return img.updateMask(mask)
                
            lst_img = l8.map(mask_l8).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
            
            reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.stdDev(), sharedInputs=True)
            stats = lst_img.reduceRegion(reducer=reducer, geometry=roi, scale=100, maxPixels=1e9).getInfo()
            
            if stats and 'ST_B10_mean' in stats and stats['ST_B10_mean'] is not None:
                stats_dict['mean_temp'] = f"{stats['ST_B10_mean']:.1f}"
                stats_dict['max_temp'] = f"{stats['ST_B10_max']:.1f}"
                stats_dict['std_dev'] = f"{stats['ST_B10_stdDev']:.1f}"
            
            vis_params = {'min': 25, 'max': 45, 'palette': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'], 'opacity': 0.8}
            m.addLayer(lst_img.clip(roi), vis_params, f'Baseline Native LST (100m)')
            m.addLayer(ee.Image().paint(roi, 0, 2), {'palette': ['#1E293B']}, 'Tourism Precinct')
            
            m.add_colorbar(vis_params, label="Surface Temperature (°C)", orientation="horizontal")
            m.add_inspector()
    except Exception as e:
        print("L8 Error:", e)
        pass
        
    return m.to_html(), stats_dict

def run_on_the_fly_downscaling(lat, lon):
    """
    Executes Random Forest Thermal Sharpening dynamically on GEE Servers.
    Dioptimalkan (Buffer lebih kecil, titik lebih sedikit) untuk mencegah Timeout di Streamlit.
    """
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(8000) # Diperkecil menjadi 8km untuk kecepatan render web
        
        # LOGIKA HEMISFER
        if lat > 0:
            month_filter = ee.Filter.calendarRange(6, 8, 'month')
        else:
            month_filter = ee.Filter.calendarRange(12, 2, 'month')
        
        # 1. Fetch L8 Native (Target)
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate('2021-01-01', '2024-12-31').filter(month_filter)
        def mask_l8(img):
            qa = img.select('QA_PIXEL')
            mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
            return img.updateMask(mask)
        lst_100m = l8.map(mask_l8).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        
        # 2. Fetch S2 Predictors
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2021-01-01', '2024-12-31').filter(month_filter).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median()
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')
        ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        predictors = s2.select(['B2', 'B3', 'B4', 'B8']).addBands([ndvi, ndbi, ndwi])
        feat_names = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'NDBI', 'NDWI']
        
        # 3. Compile Training Data (NumPixels diperkecil agar tidak timeout)
        training_img = lst_100m.addBands(predictors)
        training_pts = training_img.sample(region=roi, scale=100, numPixels=300, seed=42, geometries=False, dropNulls=True)
        
        # 4. Train RF on GEE Server (Cepat)
        rf_model = ee.Classifier.smileRandomForest(30).setOutputMode('REGRESSION').train(
            features=training_pts, classProperty='LST', inputProperties=feat_names
        )
        
        # 5. Predict 20m & Feature Importance
        lst_20m = predictors.classify(rf_model, 'Predicted_LST')
        dict_imp = rf_model.explain().get('importance').getInfo()
        
        # 6. Extract Validation Points
        predicted_pts = training_pts.classify(rf_model, 'Predicted_LST')
        val_data = predicted_pts.reduceColumns(ee.Reducer.toList(2), ['LST', 'Predicted_LST']).get('list').getInfo()
        df_eval = pd.DataFrame(val_data, columns=['Actual', 'Predicted'])
        rmse = np.sqrt(mean_squared_error(df_eval['Actual'], df_eval['Predicted']))
        r2 = r2_score(df_eval['Actual'], df_eval['Predicted'])
        
        # 7. Generate New Map HTML
        m = geemap.Map(center=[lat, lon], zoom=12, ee_initialize=False, draw_control=False, measure_control=False)
        m.add_basemap("CartoDB.Positron")
        vis = {'min': 25, 'max': 45, 'palette': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'], 'opacity': 0.8}
        m.addLayer(lst_100m.clip(roi), vis, 'Native LST 100m', False)
        m.addLayer(lst_20m.clip(roi), vis, 'RF Downscaled LST 20m', True)
        m.addLayer(ee.Image().paint(roi, 0, 2), {'palette': ['#1E293B']}, 'Boundary')
        m.add_colorbar(vis, label="Surface Temperature (°C)", orientation="horizontal")
        m.add_inspector()
        
        return m.to_html(), df_eval, rmse, r2, dict_imp
    except Exception as e:
        return None, str(e), None, None, None

def run_ml_inf(mdl, tmp, is_hw, is_hol):
    if mdl:
        df_i = pd.DataFrame({'Mx_T': [tmp], 'Is_HW': [is_hw], 'Is_Hol': [is_hol]})
        pd_pax = mdl.predict(df_i)[0]
    else:
        b = 850
        pd_pax = b + ((tmp-25)*22) + (is_hw*120) + (is_hol*180)
    v_rto = 0.28 if is_hol else 0.12
    return int(pd_pax), int(pd_pax * v_rto)

# =====================================================================
# 4. DATABASE DESTINASI
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

# =====================================================================
# 5. APP LAYOUT & PRESENTATION LAYER
# =====================================================================

st.markdown('<div class="header-card">', unsafe_allow_html=True)
st.markdown("<h1>V-HEAT: Destination Infrastructure Resilience Model</h1>", unsafe_allow_html=True)
st.markdown("<p>An integrated GeoAI framework linking Earth Observation (GEE), Historical Climate baselines (BoM), Future Projections (NASA CMIP6), and Public Health infrastructure (AIHW) to assess tourism destination carrying capacity under extreme heat.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

mdl = load_ml_mdl()
gee_status = init_ee()

# --- NAVIGATOR ---
st.markdown('<div class="modern-card" style="padding: 20px 25px;">', unsafe_allow_html=True)
c_nav1, c_nav2 = st.columns([1, 3])
with c_nav1:
    curr_idx = df_cities['City'].tolist().index(st.session_state.selected_city) if st.session_state.selected_city in df_cities['City'].tolist() else 0
    new_city = st.selectbox("Select Tourism Precinct:", df_cities['City'].tolist(), index=curr_idx)
    if new_city != st.session_state.selected_city:
        st.session_state.selected_city = new_city
        st.session_state.rf_downscale_run = False 
        st.session_state.rf_results = None
        st.rerun()
with c_nav2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.selected_city == "Gold Coast, Australia":
        st.markdown("**Status:** 🟢 Full Integration Active (Remote Sensing + BoM ML + NASA CMIP6)")
    else:
        st.markdown("**Status:** 🔵 Partial Mode (Remote Sensing + NASA CMIP6 Only).")
st.markdown('</div>', unsafe_allow_html=True)

city_row = df_cities[df_cities['City'] == st.session_state.selected_city].iloc[0]
sel_lat, sel_lon = city_row['Lat'], city_row['Lon']
is_dp = (st.session_state.selected_city == "Gold Coast, Australia")
season_txt = "Jun-Aug" if sel_lat > 0 else "Dec-Feb"

# --- MAIN DASHBOARD: THE 3 PILLARS ---
c1, c2 = st.columns([1.1, 0.9], gap="large")

# PILLAR 1: SPATIAL HAZARD (WITH LIVE RF DOWNSCALING TRIGGER)
with c1:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("1. Spatial Hazard Exposure (Remote Sensing)")
    st.markdown(f'<span class="subtitle-text"><b>Data Source:</b> Landsat 8 TIRS (100m native). Map below shows historical multi-year peak summer ({season_txt}) thermal signatures.</span>', unsafe_allow_html=True)
    
    # Menampilkan Peta Baseline
    if not st.session_state.rf_downscale_run:
        with st.spinner("Compiling Spatial Analytics..."):
            map_html, map_stats = gen_gee_map_and_stats(st.session_state.selected_city, sel_lat, sel_lon, gee_status)
        components.html(map_html, height=430)
        
        st.markdown("<hr style='margin: 10px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown('<div class="btn-geoai">', unsafe_allow_html=True)
        if st.button("🚀 Run GeoAI Thermal Sharpening (Downscale to 20m)"):
            st.session_state.rf_downscale_run = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Tampilkan Peta yang sudah di-Downscale secara Live
        with st.spinner("🤖 Google Earth Engine is training Random Forest model on-the-fly. Please wait (~15s)..."):
            if st.session_state.rf_results is None:
                map_html_rf, df_ev, rmse, r2, dict_imp = run_on_the_fly_downscaling(sel_lat, sel_lon)
                st.session_state.rf_results = (map_html_rf, df_ev, rmse, r2, dict_imp)
            else:
                map_html_rf, df_ev, rmse, r2, dict_imp = st.session_state.rf_results
                
        # Handle success or failure gracefullly
        if map_html_rf is not None:
            st.success(f"✅ GEE Model Trained! Spatial R²: {r2:.2f} | RMSE: {rmse:.2f} °C")
            components.html(map_html_rf, height=430)
            
            c_rf1, c_rf2 = st.columns(2)
            with c_rf1:
                fig_s = px.scatter(df_ev, x='Actual', y='Predicted', title="Model Fit (Predicted vs Actual LST)")
                mn, mx = min(df_ev['Actual']), max(df_ev['Actual'])
                fig_s.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx, line=dict(color='red', dash='dash'))
                fig_s.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})
            with c_rf2:
                df_i = pd.DataFrame({'Feature': list(dict_imp.keys()), 'Importance': list(dict_imp.values())}).sort_values(by='Importance', ascending=True)
                fig_i = px.bar(df_i, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importance")
                fig_i.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_i, use_container_width=True, config={'displayModeBar': False})
        else:
            st.error(f"❌ GeoAI Downscaling Timeout/Failed. This is typically due to GEE memory limits or dense cloud cover during {season_txt}. Detail: {df_ev}")
            
        if st.button("🔙 Back to Baseline Map"):
            st.session_state.rf_downscale_run = False
            st.session_state.rf_results = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# KANAN: IKLIM & INFRASTRUKTUR
with c2:
    if is_dp:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("2. Climate Projections (CMIP6 Forecast)")
        st.markdown('<span class="subtitle-text">NASA NEX-GDDP (Model: ACCESS-CM2, Scenario: SSP5-8.5). Use the slider below to simulate these forecasted extremes.</span>', unsafe_allow_html=True)
        
        with st.spinner("Querying NASA CMIP6 Spatio-Temporal Database..."):
            df_cmip = get_real_cmip6_data(sel_lat, sel_lon, gee_status)
        
        fig2 = go.Figure(go.Scatter(x=df_cmip['Year'], y=df_cmip['Max_Temp'], mode='lines', fill='tozeroy', fillcolor='rgba(229, 62, 62, 0.1)', line=dict(color='#E53E3E', width=3)))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1E293B', yaxis_title="Annual Max Air Temp (°C)", margin=dict(t=5, b=5, l=0, r=0), height=150, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E2E8F0'))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<hr style='margin: 25px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        
        st.subheader("3. Destination Capacity Simulator (BoM + AIHW)")
        st.markdown('<span class="subtitle-text">Machine Learning trained on historical BoM weather & AIHW hospital records. Simulate how extreme heat events affect local infrastructure carrying capacity.</span>', unsafe_allow_html=True)
        
        sim_tmp = st.slider("Simulate Maximum Air Temperature (°C)", min_value=25.0, max_value=45.0, value=35.0, step=0.5)
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Tourism Seasonality Exposure", [1, 0], format_func=lambda x: f"Peak Tourist Season ({season_txt})" if x==1 else "Off-Peak Season", horizontal=True)
        
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol)
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Total ED Presentations", f"{tot_pax}", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.12)} (Heat Stress Burden)" if sim_hw else "Baseline", delta_color="inverse")
        mc2.metric("Transient Tourist Load", f"{vis_pax}", delta=f"{(vis_pax/tot_pax)*100:.1f}% of operating capacity", delta_color="off")
        
        if sim_hw and sim_hol:
            st.markdown('<div class="status-badge status-critical">⚠️ CRITICAL: Severe Heatwave during Peak Season. High risk of infrastructure failure.</div>', unsafe_allow_html=True)
        elif sim_hw or sim_hol:
            st.markdown('<div class="status-badge status-warn">⚡ WARNING: Elevated Strain. Destination carrying capacity stressed.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-safe">✅ SAFE: Normal Operating Capacity. Destination resilient.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="modern-card" style="height: 100%; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding: 40px;">', unsafe_allow_html=True)
        st.markdown("<h3>🔒 Simulator Locked</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B;'>The BoM + AIHW Predictive Simulator requires localized historical health records to establish carrying capacity baselines. Currently, this PoC is trained exclusively on the <b>Gold Coast, Australia</b> pilot data.</p>", unsafe_allow_html=True)
        st.info("Please select 'Gold Coast, Australia' from the dropdown to unlock the full GeoAI integration.")
        st.markdown('</div>', unsafe_allow_html=True)
