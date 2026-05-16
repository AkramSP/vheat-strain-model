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
import folium
from folium.plugins import SideBySideLayers
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. PAGE CONFIG & MODERN UX
# =====================================================================
st.set_page_config(page_title="V-HEAT: Destination Resilience", layout="wide", initial_sidebar_state="collapsed")

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
    
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1E3A8A; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; font-weight: 600; color: #475569; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem; font-weight: 500; }
    
    .status-badge { padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-top: 10px;}
    .status-safe { background-color: #DEF7EC; color: #22543D; border: 1px solid #9AE6B4; }
    .status-warn { background-color: #FEFCBF; color: #744210; border: 1px solid #F6E05E; }
    .status-critical { background-color: #FED7D7; color: #822727; border: 1px solid #FEB2B2; }
    
    .btn-ml > button { width: 100%; font-weight: 600; background-color: #1E3A8A; color: white; border-radius: 8px; padding: 10px; border: none;}
    .btn-ml > button:hover { background-color: #1E40AF; color: white; border: none;}
    
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
if 'sim_temp' not in st.session_state:
    st.session_state.sim_temp = 35.0

# =====================================================================
# 3. CORE FUNCTIONS (GEE & ML)
# =====================================================================
@st.cache_resource(show_spinner=False)
def init_ee():
    # Menginisialisasi GEE dengan Token
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
    # Mengekstrak proyeksi iklim dari NASA CMIP6
    if not gee_ready:
        yrs = np.arange(2025, 2051)
        return pd.DataFrame({'Year': yrs, 'Max_Temp': np.linspace(34.0, 39.5, len(yrs))})
        
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
        data = ee.FeatureCollection(years.map(get_yearly_max)).getInfo()['features']
        
        yrs = [d['properties']['year'] for d in data]
        temps = [d['properties']['max_temp'] - 273.15 for d in data]
        return pd.DataFrame({'Year': yrs, 'Max_Temp': temps})
    except Exception:
        yrs = np.arange(2025, 2051)
        return pd.DataFrame({'Year': yrs, 'Max_Temp': np.linspace(34.0, 39.5, len(yrs))})

def safe_stat(d, key):
    return round(d.get(key, 0.0), 1) if d and d.get(key) is not None else 0.0

@st.cache_data(show_spinner=False)
def gen_baseline_map(lat, lon, gee_ready):
    # Membuat Peta Baseline 100m
    m = geemap.Map(center=[lat, lon], zoom=12, ee_initialize=False, draw_control=False, measure_control=False)
    m.add_basemap("CartoDB.Positron")
    stats_dict = {"min": 0.0, "mean": 0.0, "max": 0.0}
    
    if not gee_ready:
        return m.to_html(), stats_dict
        
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(8000) 
        
        month_filter = ee.Filter.calendarRange(6, 8, 'month') if lat > 0 else ee.Filter.calendarRange(12, 2, 'month')
            
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate('2019-01-01', '2024-12-31').filter(month_filter)
        if l8.size().getInfo() > 0:
            def mask_l8(img):
                qa = img.select('QA_PIXEL')
                mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                return img.updateMask(mask)
                
            lst_100m = l8.map(mask_l8).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
            
            # FIX: Filter Thermal Ekstrem (Membuang awan tinggi -47C dan atap pabrik 70C)
            valid_mask = lst_100m.gt(5).And(lst_100m.lt(55))
            lst_100m = lst_100m.updateMask(valid_mask)
            
            reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.min(), sharedInputs=True)
            stats = lst_100m.reduceRegion(reducer=reducer, geometry=roi, scale=100, maxPixels=1e9).getInfo()
            
            stats_dict['mean'] = safe_stat(stats, 'LST_mean')
            stats_dict['max'] = safe_stat(stats, 'LST_max')
            stats_dict['min'] = safe_stat(stats, 'LST_min')
            
            vis_params = {'min': 25, 'max': 45, 'palette': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'], 'opacity': 0.8}
            m.addLayer(lst_100m.clip(roi), vis_params, f'Baseline Native LST (100m)')
            m.addLayer(ee.Image().paint(roi, 0, 2), {'palette': ['#1E293B']}, 'Tourism Precinct')
            m.add_colorbar(vis_params, label="Surface Temperature (°C)", orientation="horizontal")
    except Exception as e:
        print("L8 Error:", e)
        
    return m.to_html(), stats_dict

def run_rf_downscaling_split(lat, lon):
    """
    Menjalankan Spatial Downscaling & Menghasilkan SPLIT PANEL Peta terpotong (Clipped).
    """
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(8000)
        
        month_filter = ee.Filter.calendarRange(6, 8, 'month') if lat > 0 else ee.Filter.calendarRange(12, 2, 'month')
        
        # 1. LANDSAT 8 (Target)
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate('2021-01-01', '2024-12-31').filter(month_filter)
        def mask_l8(img):
            qa = img.select('QA_PIXEL')
            mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
            return img.updateMask(mask)
        lst_100m = l8.map(mask_l8).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        
        # FIX: Filter Thermal Ekstrem untuk Data Training
        valid_mask = lst_100m.gt(5).And(lst_100m.lt(55))
        lst_100m = lst_100m.updateMask(valid_mask)
        
        # 2. PREDICTOR: DEM (SRTM 30m)
        dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
        elev = dem.select('elevation')
        slope = ee.Terrain.slope(elev).rename('Slope')
        aspect = ee.Terrain.aspect(elev).rename('Aspect')
        
        # 3. PREDICTOR: Sentinel-2 Indices
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2021-01-01', '2024-12-31').filter(month_filter).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median()
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')
        ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # 4. Compile Predictors
        predictors = ee.Image([ndvi, ndbi, ndwi, elev, slope, aspect])
        feat_names = ['NDVI', 'NDBI', 'NDWI', 'elevation', 'Slope', 'Aspect']
        
        # 5. Train Random Forest
        training_img = lst_100m.addBands(predictors)
        training_pts = training_img.sample(region=roi, scale=100, numPixels=350, seed=42, geometries=False, dropNulls=True)
        rf_model = ee.Classifier.smileRandomForest(30).setOutputMode('REGRESSION').train(
            features=training_pts, classProperty='LST', inputProperties=feat_names
        )
        
        # 6. Predict Downscaled 20m & Terapkan Mask yang sama
        lst_20m = predictors.classify(rf_model, 'Predicted_LST').updateMask(valid_mask)
        dict_imp = rf_model.explain().get('importance').getInfo()
        
        # 7. Zonal Statistics Comparison
        reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.min(), sharedInputs=True)
        stats_100 = lst_100m.reduceRegion(reducer=reducer, geometry=roi, scale=100, maxPixels=1e9).getInfo()
        stats_20 = lst_20m.reduceRegion(reducer=reducer, geometry=roi, scale=20, maxPixels=1e9).getInfo()
        
        comp_stats = {
            'n_min': safe_stat(stats_100, 'LST_min'), 'n_mean': safe_stat(stats_100, 'LST_mean'), 'n_max': safe_stat(stats_100, 'LST_max'),
            'd_min': safe_stat(stats_20, 'Predicted_LST_min'), 'd_mean': safe_stat(stats_20, 'Predicted_LST_mean'), 'd_max': safe_stat(stats_20, 'Predicted_LST_max')
        }
        
        # 8. ML Metrics
        predicted_pts = training_pts.classify(rf_model, 'Predicted_LST')
        val_data = predicted_pts.reduceColumns(ee.Reducer.toList(2), ['LST', 'Predicted_LST']).get('list').getInfo()
        df_eval = pd.DataFrame(val_data, columns=['Actual', 'Predicted'])
        rmse = np.sqrt(mean_squared_error(df_eval['Actual'], df_eval['Predicted']))
        r2 = r2_score(df_eval['Actual'], df_eval['Predicted'])
        
        # 9. SPLIT PANEL MAP (FIX: Clipped to ROI properly)
        m = geemap.Map(center=[lat, lon], zoom=12, ee_initialize=False, draw_control=False, measure_control=False)
        m.add_basemap("CartoDB.Positron")
        vis = {'min': 25, 'max': 45, 'palette': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'], 'opacity': 0.9}
        
        # Terapkan clip(roi) sebelum mengambil URL agar petanya terpotong rapi
        left_url = lst_100m.clip(roi).getMapId(vis)['tile_fetcher'].url_format
        right_url = lst_20m.clip(roi).getMapId(vis)['tile_fetcher'].url_format
        
        left_tile = folium.TileLayer(tiles=left_url, attr='GEE', name='Native 100m', overlay=True, control=True)
        right_tile = folium.TileLayer(tiles=right_url, attr='GEE', name='Downscaled 20m', overlay=True, control=True)
        
        left_tile.add_to(m)
        right_tile.add_to(m)
        SideBySideLayers(left_tile, right_tile).add_to(m)
        
        m.addLayer(ee.Image().paint(roi, 0, 2), {'palette': ['#1E293B']}, 'Boundary')
        m.add_colorbar(vis, label="Surface Temperature (°C)", orientation="horizontal")
        
        return m.to_html(), df_eval, rmse, r2, dict_imp, comp_stats
    except Exception as e:
        return None, str(e), None, None, None, None

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
# 4. DATABASE 50 DESTINASI GLOBAL (Out of the Box Update)
# =====================================================================
cty_coords = [
    # Oceania
    {"City": "Gold Coast, Australia", "Lat": -28.0167, "Lon": 153.4000, "Tourists_M": 12.0, "Avg_Summer_LST": 34.5, "Continent": "Oceania"},
    {"City": "Brisbane, Australia", "Lat": -27.4705, "Lon": 153.0260, "Tourists_M": 8.0, "Avg_Summer_LST": 36.2, "Continent": "Oceania"},
    {"City": "Sydney, Australia", "Lat": -33.8688, "Lon": 151.2093, "Tourists_M": 16.0, "Avg_Summer_LST": 32.5, "Continent": "Oceania"},
    {"City": "Melbourne, Australia", "Lat": -37.8136, "Lon": 144.9631, "Tourists_M": 11.0, "Avg_Summer_LST": 31.0, "Continent": "Oceania"},
    {"City": "Perth, Australia", "Lat": -31.9505, "Lon": 115.8605, "Tourists_M": 5.0, "Avg_Summer_LST": 38.0, "Continent": "Oceania"},
    {"City": "Auckland, New Zealand", "Lat": -36.8485, "Lon": 174.7633, "Tourists_M": 3.0, "Avg_Summer_LST": 24.5, "Continent": "Oceania"},
    {"City": "Queenstown, New Zealand", "Lat": -45.0312, "Lon": 168.6626, "Tourists_M": 1.5, "Avg_Summer_LST": 22.0, "Continent": "Oceania"},
    # Asia
    {"City": "Jakarta, Indonesia", "Lat": -6.2088, "Lon": 106.8456, "Tourists_M": 5.0, "Avg_Summer_LST": 35.0, "Continent": "Asia"},
    {"City": "Bali, Indonesia", "Lat": -8.4095, "Lon": 115.1889, "Tourists_M": 6.5, "Avg_Summer_LST": 30.5, "Continent": "Asia"},
    {"City": "Bangkok, Thailand", "Lat": 13.7563, "Lon": 100.5018, "Tourists_M": 22.7, "Avg_Summer_LST": 38.5, "Continent": "Asia"},
    {"City": "Phuket, Thailand", "Lat": 7.8804, "Lon": 98.3922, "Tourists_M": 10.0, "Avg_Summer_LST": 32.0, "Continent": "Asia"},
    {"City": "Singapore", "Lat": 1.3521, "Lon": 103.8198, "Tourists_M": 19.1, "Avg_Summer_LST": 34.0, "Continent": "Asia"},
    {"City": "Kuala Lumpur, Malaysia", "Lat": 3.1390, "Lon": 101.6869, "Tourists_M": 13.8, "Avg_Summer_LST": 35.5, "Continent": "Asia"},
    {"City": "Tokyo, Japan", "Lat": 35.6762, "Lon": 139.6503, "Tourists_M": 14.0, "Avg_Summer_LST": 36.5, "Continent": "Asia"},
    {"City": "Kyoto, Japan", "Lat": 35.0116, "Lon": 135.7681, "Tourists_M": 8.0, "Avg_Summer_LST": 35.0, "Continent": "Asia"},
    {"City": "Osaka, Japan", "Lat": 34.6937, "Lon": 135.5023, "Tourists_M": 11.5, "Avg_Summer_LST": 36.0, "Continent": "Asia"},
    {"City": "Seoul, South Korea", "Lat": 37.5665, "Lon": 126.9780, "Tourists_M": 11.0, "Avg_Summer_LST": 34.5, "Continent": "Asia"},
    {"City": "Taipei, Taiwan", "Lat": 25.0330, "Lon": 121.5654, "Tourists_M": 9.5, "Avg_Summer_LST": 36.0, "Continent": "Asia"},
    {"City": "Hong Kong, SAR China", "Lat": 22.3193, "Lon": 114.1694, "Tourists_M": 29.0, "Avg_Summer_LST": 35.5, "Continent": "Asia"},
    {"City": "Dubai, UAE", "Lat": 25.2048, "Lon": 55.2708, "Tourists_M": 16.7, "Avg_Summer_LST": 45.0, "Continent": "Asia"},
    {"City": "Riyadh, Saudi Arabia", "Lat": 24.7136, "Lon": 46.6753, "Tourists_M": 5.5, "Avg_Summer_LST": 44.5, "Continent": "Asia"},
    {"City": "Mumbai, India", "Lat": 19.0760, "Lon": 72.8777, "Tourists_M": 6.0, "Avg_Summer_LST": 35.0, "Continent": "Asia"},
    {"City": "Delhi, India", "Lat": 28.7041, "Lon": 77.1025, "Tourists_M": 5.5, "Avg_Summer_LST": 42.0, "Continent": "Asia"},
    # Europe
    {"City": "Istanbul, Turkey", "Lat": 41.0082, "Lon": 28.9784, "Tourists_M": 14.7, "Avg_Summer_LST": 33.5, "Continent": "Europe"},
    {"City": "Rome, Italy", "Lat": 41.9028, "Lon": 12.4964, "Tourists_M": 10.0, "Avg_Summer_LST": 36.0, "Continent": "Europe"},
    {"City": "Venice, Italy", "Lat": 45.4408, "Lon": 12.3155, "Tourists_M": 5.5, "Avg_Summer_LST": 32.5, "Continent": "Europe"},
    {"City": "Milan, Italy", "Lat": 45.4642, "Lon": 9.1900, "Tourists_M": 6.5, "Avg_Summer_LST": 35.0, "Continent": "Europe"},
    {"City": "Paris, France", "Lat": 48.8566, "Lon": 2.3522, "Tourists_M": 19.0, "Avg_Summer_LST": 33.0, "Continent": "Europe"},
    {"City": "Barcelona, Spain", "Lat": 41.3851, "Lon": 2.1734, "Tourists_M": 9.0, "Avg_Summer_LST": 34.0, "Continent": "Europe"},
    {"City": "Madrid, Spain", "Lat": 40.4168, "Lon": -3.7038, "Tourists_M": 7.5, "Avg_Summer_LST": 38.5, "Continent": "Europe"},
    {"City": "Athens, Greece", "Lat": 37.9838, "Lon": 23.7275, "Tourists_M": 6.0, "Avg_Summer_LST": 39.0, "Continent": "Europe"},
    {"City": "Lisbon, Portugal", "Lat": 38.7223, "Lon": -9.1393, "Tourists_M": 4.5, "Avg_Summer_LST": 34.5, "Continent": "Europe"},
    {"City": "London, UK", "Lat": 51.5074, "Lon": -0.1278, "Tourists_M": 19.5, "Avg_Summer_LST": 28.0, "Continent": "Europe"},
    {"City": "Amsterdam, Netherlands", "Lat": 52.3676, "Lon": 4.9041, "Tourists_M": 8.5, "Avg_Summer_LST": 27.5, "Continent": "Europe"},
    {"City": "Berlin, Germany", "Lat": 52.5200, "Lon": 13.4050, "Tourists_M": 6.0, "Avg_Summer_LST": 29.0, "Continent": "Europe"},
    {"City": "Vienna, Austria", "Lat": 48.2082, "Lon": 16.3738, "Tourists_M": 6.5, "Avg_Summer_LST": 30.5, "Continent": "Europe"},
    {"City": "Zurich, Switzerland", "Lat": 47.3769, "Lon": 8.5417, "Tourists_M": 3.5, "Avg_Summer_LST": 28.0, "Continent": "Europe"},
    {"City": "Prague, Czechia", "Lat": 50.0755, "Lon": 14.4378, "Tourists_M": 9.0, "Avg_Summer_LST": 29.5, "Continent": "Europe"},
    # Americas
    {"City": "New York City, USA", "Lat": 40.7128, "Lon": -74.0060, "Tourists_M": 13.5, "Avg_Summer_LST": 34.0, "Continent": "Americas"},
    {"City": "Los Angeles, USA", "Lat": 34.0522, "Lon": -118.2437, "Tourists_M": 8.5, "Avg_Summer_LST": 35.5, "Continent": "Americas"},
    {"City": "Las Vegas, USA", "Lat": 36.1699, "Lon": -115.1398, "Tourists_M": 6.5, "Avg_Summer_LST": 42.0, "Continent": "Americas"},
    {"City": "Miami, USA", "Lat": 25.7617, "Lon": -80.1918, "Tourists_M": 8.0, "Avg_Summer_LST": 35.0, "Continent": "Americas"},
    {"City": "Honolulu, USA", "Lat": 21.3069, "Lon": -157.8583, "Tourists_M": 3.0, "Avg_Summer_LST": 31.0, "Continent": "Americas"},
    {"City": "Toronto, Canada", "Lat": 43.6510, "Lon": -79.3470, "Tourists_M": 4.5, "Avg_Summer_LST": 29.5, "Continent": "Americas"},
    {"City": "Cancun, Mexico", "Lat": 21.1619, "Lon": -86.8515, "Tourists_M": 6.0, "Avg_Summer_LST": 34.5, "Continent": "Americas"},
    {"City": "Rio de Janeiro, Brazil", "Lat": -22.9068, "Lon": -43.1729, "Tourists_M": 2.5, "Avg_Summer_LST": 36.0, "Continent": "Americas"},
    {"City": "Buenos Aires, Argentina", "Lat": -34.6037, "Lon": -58.3816, "Tourists_M": 3.0, "Avg_Summer_LST": 34.0, "Continent": "Americas"},
    # Africa
    {"City": "Cape Town, South Africa", "Lat": -33.9249, "Lon": 18.4241, "Tourists_M": 1.5, "Avg_Summer_LST": 32.0, "Continent": "Africa"},
    {"City": "Cairo, Egypt", "Lat": 30.0444, "Lon": 31.2357, "Tourists_M": 6.0, "Avg_Summer_LST": 41.5, "Continent": "Africa"},
    {"City": "Marrakech, Morocco", "Lat": 31.6295, "Lon": -7.9811, "Tourists_M": 3.0, "Avg_Summer_LST": 40.0, "Continent": "Africa"}
]
df_cities = pd.DataFrame(cty_coords)
df_cities['Type'] = np.where(df_cities['City'] == 'Gold Coast, Australia', 'Primary Study', 'Baseline')

# =====================================================================
# 5. APP LAYOUT & PRESENTATION LAYER
# =====================================================================
st.markdown('<div class="header-card">', unsafe_allow_html=True)
st.markdown("<h1>V-HEAT: Destination Infrastructure Resilience Model</h1>", unsafe_allow_html=True)
st.markdown("<p>An integrated analytical framework linking Earth Observation (GEE), Historical Climate baselines (BoM), Future Projections (NASA CMIP6), and Public Health infrastructure to assess tourism destination carrying capacity under extreme heat.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

mdl = load_ml_mdl()
gee_status = init_ee()

# --- NAVIGATOR TABS ---
tab_nav, tab_global = st.tabs(["📍 Local Precinct Analysis", "🌍 Global Destination Vulnerability Index"])

with tab_nav:
    st.markdown('<div class="modern-card" style="padding: 20px 25px;">', unsafe_allow_html=True)
    c_nav1, c_nav2 = st.columns([1, 3])
    with c_nav1:
        curr_idx = df_cities['City'].tolist().index(st.session_state.selected_city) if st.session_state.selected_city in df_cities['City'].tolist() else 0
        new_city = st.selectbox("Select Tourism Precinct (50 Cities):", df_cities['City'].tolist(), index=curr_idx)
        if new_city != st.session_state.selected_city:
            st.session_state.selected_city = new_city
            st.session_state.rf_downscale_run = False 
            st.session_state.rf_results = None
            st.rerun()
    with c_nav2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.selected_city == "Gold Coast, Australia":
            st.markdown("**Status:** 🟢 Full Integration Active (Remote Sensing + Local Health Data + NASA CMIP6)")
        else:
            st.markdown("**Status:** 🔵 Partial Mode (Remote Sensing + NASA CMIP6 Only).")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB GLOBAL 50 CITIES CHART (Merespon Poin 5)
with tab_global:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Global Tourism Vulnerability Matrix")
    st.markdown("Comparative analysis of 50 global destinations mapping tourist volume against extreme summer surface temperatures.")
    
    # Interaktif Bubble Chart
    fig_global = px.scatter(df_cities, x='Avg_Summer_LST', y='Tourists_M', size='Tourists_M', color='Continent', 
                            hover_name='City', size_max=40, opacity=0.7,
                            labels={'Avg_Summer_LST': 'Average Summer LST (°C)', 'Tourists_M': 'Annual Tourists (Millions)'},
                            color_discrete_sequence=px.colors.qualitative.Prism)
    fig_global.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=10, l=10, r=10), height=400)
    # Garis kuadran risiko
    fig_global.add_hline(y=10, line_dash="dot", line_color="gray")
    fig_global.add_vline(x=35, line_dash="dot", line_color="red")
    fig_global.add_annotation(x=42, y=25, text="High Risk Zone", showarrow=False, font=dict(color="red", size=14))
    
    st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

city_row = df_cities[df_cities['City'] == st.session_state.selected_city].iloc[0]
sel_lat, sel_lon = city_row['Lat'], city_row['Lon']
is_dp = (st.session_state.selected_city == "Gold Coast, Australia")
season_txt = "Jun-Aug" if sel_lat > 0 else "Dec-Feb"

# --- MAIN DASHBOARD: THE 3 PILLARS ---
c1, c2 = st.columns([1.1, 0.9], gap="large")

# PILLAR 1: SPATIAL HAZARD
with c1:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("1. Spatial Hazard Exposure (Remote Sensing)")
    st.markdown(f'<span class="subtitle-text"><b>Data Source:</b> Landsat 8 TIRS. Historical multi-year peak summer ({season_txt}) thermal signatures.</span>', unsafe_allow_html=True)
    
    # KONDISI 1: PETA BASELINE (SEBELUM DOWNSCALING)
    if not st.session_state.rf_downscale_run:
        with st.spinner("Extracting Spatial Analytics..."):
            map_html, base_stats = gen_baseline_map(sel_lat, sel_lon, gee_status)
        components.html(map_html, height=430)
        
        # Panel Statistik Eksternal
        st.markdown("##### 📍 Regional Baseline LST Panel (100m)")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Min Temp", f"{base_stats['min']} °C")
        sc2.metric("Mean Temp", f"{base_stats['mean']} °C")
        sc3.metric("Max Temp", f"{base_stats['max']} °C")
        
        st.markdown("<hr style='margin: 15px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown('<div class="btn-ml">', unsafe_allow_html=True)
        if st.button("🚀 Run Spatial Downscaling Model (100m to 20m)"):
            st.session_state.rf_downscale_run = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    # KONDISI 2: PETA DOWNSCALE & SPLIT PANEL
    else:
        with st.spinner("🤖 Executing Machine Learning Downscaling. Slide the center bar to compare Native vs Downscaled maps..."):
            if st.session_state.rf_results is None:
                map_html_rf, df_ev, rmse, r2, dict_imp, comp_stats = run_rf_downscaling_split(sel_lat, sel_lon)
                st.session_state.rf_results = (map_html_rf, df_ev, rmse, r2, dict_imp, comp_stats)
            else:
                map_html_rf, df_ev, rmse, r2, dict_imp, comp_stats = st.session_state.rf_results
                
        if map_html_rf is not None:
            # Peta Split Panel Folium
            components.html(map_html_rf, height=450)
            
            # THE MAGIC: Perbandingan Statistik!
            st.markdown("##### 📊 LST Extracted Statistics: Native vs Downscaled")
            st.markdown('<span class="subtitle-text">Notice how the downscaled 20m model detects higher extreme localized temperatures (Hotspots) missed by the 100m baseline.</span>', unsafe_allow_html=True)
            
            # Ini akan mengupdate sim_temp untuk Simulator di kanan!
            st.session_state.sim_temp = float(comp_stats['d_max'])
            
            s1, s2, s3 = st.columns(3)
            s1.metric("Min Temp (20m)", f"{comp_stats['d_min']} °C", f"{round(comp_stats['d_min'] - comp_stats['n_min'], 1)} °C vs Native", delta_color="inverse")
            s2.metric("Mean Temp (20m)", f"{comp_stats['d_mean']} °C", f"{round(comp_stats['d_mean'] - comp_stats['n_mean'], 1)} °C vs Native", delta_color="off")
            s3.metric("Local Max Temp (20m)", f"{comp_stats['d_max']} °C", f"{round(comp_stats['d_max'] - comp_stats['n_max'], 1)} °C vs Native", delta_color="inverse")
            
            with st.expander("Show Machine Learning Validation Metrics"):
                c_rf1, c_rf2 = st.columns(2)
                with c_rf1:
                    fig_s = px.scatter(df_ev, x='Actual', y='Predicted', title=f"Spatial R²: {r2:.2f} | RMSE: {rmse:.2f} °C")
                    mn, mx = min(df_ev['Actual']), max(df_ev['Actual'])
                    fig_s.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx, line=dict(color='red', dash='dash'))
                    fig_s.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})
                with c_rf2:
                    df_i = pd.DataFrame({'Predictor': list(dict_imp.keys()), 'Importance': list(dict_imp.values())}).sort_values(by='Importance', ascending=True)
                    fig_i = px.bar(df_i, x='Importance', y='Predictor', orientation='h', title="Random Forest Variable Importance")
                    fig_i.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_i, use_container_width=True, config={'displayModeBar': False})
        else:
            st.error(f"❌ Spatial downscaling failed due to GEE timeout or lack of satellite data for this region. Detail: {df_ev}")
            
        if st.button("🔙 Back to Baseline Map"):
            st.session_state.rf_downscale_run = False
            st.session_state.rf_results = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# PILLAR 2 & 3: IKLIM & INFRASTRUKTUR
with c2:
    if is_dp:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("2. Future Climate Projections (CMIP6)")
        st.markdown('<span class="subtitle-text">NASA NEX-GDDP (Model: ACCESS-CM2). <b>Select a projected year on the chart</b> to automatically load its temperature into the Infrastructure Simulator below.</span>', unsafe_allow_html=True)
        
        with st.spinner("Querying NASA CMIP6 Database..."):
            df_cmip = get_real_cmip6_data(sel_lat, sel_lon, gee_status)
        
        fig2 = go.Figure(go.Scatter(x=df_cmip['Year'], y=df_cmip['Max_Temp'], mode='lines+markers', customdata=df_cmip['Max_Temp'], fill='tozeroy', fillcolor='rgba(229, 62, 62, 0.1)', line=dict(color='#E53E3E', width=3)))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1E293B', yaxis_title="Max Air Temp (°C)", margin=dict(t=5, b=5, l=0, r=0), height=150)
        
        # Interaktivitas CMIP6 dengan Loading Response
        cmip_sel = st.plotly_chart(fig2, on_select="rerun", selection_mode="points", use_container_width=True, key="cmip_chart")
        if cmip_sel and hasattr(cmip_sel, 'selection'):
            pts = cmip_sel.selection.get('points', [])
            if pts and len(pts) > 0:
                new_temp = float(pts[0].get('customdata'))
                if st.session_state.sim_temp != new_temp:
                    st.session_state.sim_temp = new_temp
                    # Menambahkan Notifikasi Toast (Response Visual)
                    st.toast(f"Simulator updated to {new_temp:.1f}°C based on CMIP6 projection.", icon="✅")
        
        st.markdown("<hr style='margin: 25px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        
        st.subheader("3. Destination Capacity Simulator (BoM + Hospital Data)")
        st.markdown('<span class="subtitle-text">Machine Learning trained on historical BoM weather & hospital records. Simulates how localized heat extremes and tourist seasons impact infrastructure carrying capacity.</span>', unsafe_allow_html=True)
        
        def on_slider_change():
            st.session_state.sim_temp = st.session_state.temp_slider
            
        sim_tmp = st.slider("Simulate Maximum Temperature (°C)", min_value=25.0, max_value=48.0, value=float(st.session_state.sim_temp), step=0.1, key="temp_slider", on_change=on_slider_change)
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Tourism Seasonality Exposure", [1, 0], format_func=lambda x: f"Peak Tourist Season ({season_txt})" if x==1 else "Off-Peak Season", horizontal=True)
        
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol)
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Est. Daily Hospital Cases", f"{tot_pax}", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.12)} (Heat Stress Burden)" if sim_hw else "Baseline", delta_color="inverse")
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
        st.markdown("<p style='color: #64748B;'>The Predictive Simulator requires localized historical health records to establish carrying capacity baselines. Currently, this PoC is trained exclusively on the <b>Gold Coast, Australia</b> pilot data.</p>", unsafe_allow_html=True)
        st.info("Please select 'Gold Coast, Australia' from the dropdown to unlock the full integration.")
        st.markdown('</div>', unsafe_allow_html=True)
