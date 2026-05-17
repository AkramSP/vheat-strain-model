import streamlit as st
import ee
import json
import warnings
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import folium
from folium.plugins import SideBySideLayers
from streamlit_folium import st_folium
import google.generativeai as genai
import branca.colormap as cm # Added for Map Legend

# =====================================================================
# HOTFIX PATCH & WARNING SUPPRESSION
# =====================================================================
warnings.filterwarnings("ignore")

if hasattr(ee, 'data') and not hasattr(ee.data, '_credentials'):
    ee.data._credentials = None

# =====================================================================
# 1. PAGE CONFIG & MODERN UX (CSS)
# =====================================================================
st.set_page_config(page_title="V-HEAT: Destination Resilience", layout="wide", initial_sidebar_state="collapsed")

css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; scroll-behavior: smooth; }
    .stApp { background-color: #F8FAFC; color: #1E293B; }
    
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .modern-card { 
        background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E2E8F0; padding: 25px; margin-bottom: 20px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        animation: slideUpFade 0.4s ease-out forwards;
        transition: all 0.3s ease;
    }
    .header-card { 
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); color: white; border-radius: 8px; padding: 40px 30px; margin-bottom: 25px; 
        animation: slideUpFade 0.3s ease-out forwards;
    }
    .header-card h1 { color: white !important; margin-top: 0; font-weight: 700; font-size: 2.2rem; letter-spacing: -0.02em; }
    .header-card p { color: #CBD5E1; font-size: 1.1rem; max-width: 900px; line-height: 1.6; margin-bottom: 0;}
    
    h2, h3, h4 { font-weight: 600 !important; color: #0F172A; }
    .subtitle-text { font-size: 0.9rem; color: #64748B; margin-bottom: 15px; display: block; line-height: 1.5; }
    
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1E3A8A; transition: color 0.3s ease; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricDelta"] { font-size: 0.85rem; font-weight: 500; }
    
    .status-badge { padding: 12px 16px; border-radius: 4px; font-weight: 700; font-size: 0.9rem; display: block; margin-top: 15px; width: 100%; text-align: center; text-transform: uppercase; letter-spacing: 0.05em; transition: all 0.3s ease;}
    .status-safe { background-color: #F0FDF4; color: #047857; border: 1px solid #A7F3D0; }
    .status-warn { background-color: #FEF9C3; color: #B45309; border: 1px solid #FDE047; }
    .status-critical { background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    
    .sim-panel { background-color: #EFF6FF; border-left: 4px solid #4F46E5; padding: 12px 16px; border-radius: 0 4px 4px 0; font-size: 0.9rem; color: #3730A3; margin-bottom: 15px; animation: slideUpFade 0.3s ease-out forwards; }
    .sim-panel-manual { background-color: #F8FAFC; border-left: 4px solid #94A3B8; padding: 12px 16px; border-radius: 0 4px 4px 0; font-size: 0.9rem; color: #475569; margin-bottom: 15px; animation: slideUpFade 0.3s ease-out forwards; }
    
    .integration-full { background-color: #F0FDF4; border-left: 4px solid #10B981; padding: 10px 16px; border-radius: 0 4px 4px 0; font-size: 0.95rem; color: #047857; margin-top: 25px; animation: slideUpFade 0.3s ease-out forwards; }
    .integration-partial { background-color: #F8FAFC; border-left: 4px solid #94A3B8; padding: 10px 16px; border-radius: 0 4px 4px 0; font-size: 0.95rem; color: #475569; margin-top: 25px; animation: slideUpFade 0.3s ease-out forwards; }
    
    .btn-ml > button { width: 100%; font-weight: 600; background-color: #1E3A8A; color: white; border-radius: 4px; padding: 10px; border: none; transition: all 0.2s ease;}
    .btn-ml > button:hover { background-color: #1E40AF; color: white; border: none; transform: scale(1.01);}
    
    .ai-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 20px; font-size: 0.95rem; line-height: 1.6; color: #334155; }
    .inspector-box { background-color: #FFFFFF; border-left: 3px solid #0F172A; padding: 10px 15px; margin-bottom: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-size: 0.9rem; color: #1E293B;}
    
    /* Out of the Box: Fix for oversized Leaflet attribution font */
    .leaflet-control-attribution { font-size: 0.65rem !important; color: #94A3B8 !important; background: rgba(255,255,255,0.7) !important; padding: 0 5px !important;}
    .leaflet-control-attribution a { color: #64748B !important; }
    
    [data-testid="collapsedControl"] { display: none; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# =====================================================================
# 2. SESSION STATE MANAGEMENT
# =====================================================================
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Gold Coast, Australia"
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2019-2025 Multi-Year Composite"
if 'rf_downscale_run' not in st.session_state:
    st.session_state.rf_downscale_run = False
if 'sim_temp' not in st.session_state:
    st.session_state.sim_temp = 35.0
if 'temp_slider' not in st.session_state:
    st.session_state.temp_slider = 35.0
if 'sim_year_label' not in st.session_state:
    st.session_state.sim_year_label = "Composite Baseline"
if 'temp_source' not in st.session_state:
    st.session_state.temp_source = "Default"
if 'cmip_chart_key' not in st.session_state:
    st.session_state.cmip_chart_key = 0
if 'run_ai' not in st.session_state:
    st.session_state.run_ai = False

# =====================================================================
# 3. CORE FUNCTIONS (GEE, ML & LLM)
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

def get_ai_policy_insights(city, temp, year, status, tourist_pct, lst_max):
    """Generates dynamic policy recommendations using a two-step LLM chain with mandatory citations."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            
            local_context = "General sustainable tourism resilience protocols."
            try:
                model_search = genai.GenerativeModel('gemini-2.5-flash')
                search_prompt = f"Search the web for recent climate adaptation strategies, sustainable tourism frameworks, or heatwave response plans implemented in {city}. Provide a concise factual summary of the specific local policies."
                search_response = model_search.generate_content(search_prompt, tools="google_search")
                if search_response and search_response.text:
                    local_context = search_response.text
            except Exception as search_err:
                pass # Bypass grounding if API fails, proceed with fundamental knowledge
            
            model_synth = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
            synth_prompt = f"""
            Act as an expert Sustainable Tourism and Public Health Policy Advisor.
            Analyze the following climate and infrastructure scenario for the destination: {city}.
            
            SCENARIO DATA:
            - Target Forecast Year: {year}
            - Forecasted Max Air Temperature: {temp}°C
            - Localized Urban Heat Island (LST Peak): {lst_max}°C
            - Hospital Infrastructure Status: {status}
            - Tourist Burden on Emergency Departments: {tourist_pct}% of operational capacity.
            
            LOCAL CONTEXT (Retrieved from web search if available):
            {local_context}
            
            TASK:
            1. Provide exactly 3 concise, highly actionable policy recommendations (bullet points) for the local destination management organization (DMO) and city council to mitigate this specific level of tourism-related hospital strain.
            2. Integrate the provided LOCAL CONTEXT to ensure the recommendations are relevant to {city}.
            3. CRITICAL: Align the tone and strategic focus directly with the published research paradigms of Professor Susanne Becken (expert in Sustainable Tourism and Climate Change Adaptation). Emphasize "systemic destination resilience", "visitor vulnerability", and "reputational risks".
            4. MANDATORY CITATION: You MUST conclude your response with a "Key Reference" section containing at least one markdown hyperlink citing a specific, real-world research paper authored by Professor Susanne Becken that supports your strategies (e.g., `[Becken, S. (Year). Title. Journal.](URL)`). Search your knowledge base to ensure the paper exists.
            5. Use a professional, academic, and authoritative tone. Do not use emojis. Limit the response to 150 words (excluding citations).
            """
            final_response = model_synth.generate_content(synth_prompt)
            return final_response.text
        else:
            return "Generative AI policy advisor requires a valid Gemini API Key. Please configure GEMINI_API_KEY in Streamlit Secrets."
    except Exception as e:
        return f"AI Service Initialization Error. Please verify API quotas or connection. Log: {str(e)}"

@st.cache_data(show_spinner=False)
def get_real_cmip6_data(lat, lon, gee_ready):
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

def get_date_range(year_selection):
    if year_selection == "2019-2025 Multi-Year Composite":
        return '2019-01-01', '2025-12-31'
    else:
        return f'{year_selection}-01-01', f'{year_selection}-12-31'

# Modified to return EE objects instead of HTML to allow native Streamlit-Folium integration
@st.cache_resource(show_spinner=False)
def get_ee_baseline_data(lat, lon, year_selection, gee_ready):
    if not gee_ready: return None, None, None
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(8000) 
        month_filter = ee.Filter.calendarRange(6, 8, 'month') if lat > 0 else ee.Filter.calendarRange(12, 2, 'month')
        start_date, end_date = get_date_range(year_selection)
            
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date).filter(month_filter)
        if l8.size().getInfo() == 0: return None, None, None
        
        def mask_l8(img):
            qa = img.select('QA_PIXEL')
            mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
            return img.updateMask(mask)
            
        lst_100m = l8.map(mask_l8).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        valid_mask = lst_100m.gt(5).And(lst_100m.lt(55))
        lst_100m = lst_100m.updateMask(valid_mask)
        
        reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.min(), sharedInputs=True).combine(ee.Reducer.median(), sharedInputs=True)
        stats = lst_100m.reduceRegion(reducer=reducer, geometry=roi, scale=100, maxPixels=1e9).getInfo()
        
        stats_dict = {
            'mean': safe_stat(stats, 'LST_mean'), 'median': safe_stat(stats, 'LST_median'),
            'max': safe_stat(stats, 'LST_max'), 'min': safe_stat(stats, 'LST_min')
        }
        return lst_100m, stats_dict, roi
    except Exception:
        return None, None, None

@st.cache_resource(show_spinner=False)
def get_ee_downscaled_data(lat, lon, year_selection, gee_ready):
    if not gee_ready: return None, None, None, None, None, None, None, None
    try:
        pt = ee.Geometry.Point([lon, lat])
        roi = pt.buffer(8000)
        month_filter = ee.Filter.calendarRange(6, 8, 'month') if lat > 0 else ee.Filter.calendarRange(12, 2, 'month')
        start_date, end_date = get_date_range(year_selection)
        
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date).filter(month_filter)
        def mask_l8(img):
            qa = img.select('QA_PIXEL')
            mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
            return img.updateMask(mask)
        lst_100m = l8.map(mask_l8).median().select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        valid_mask = lst_100m.gt(5).And(lst_100m.lt(55))
        lst_100m = lst_100m.updateMask(valid_mask)
        
        dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
        elev = dem.select('elevation')
        slope = ee.Terrain.slope(elev).rename('Slope')
        aspect = ee.Terrain.aspect(elev).rename('Aspect')
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(start_date, end_date).filter(month_filter).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median()
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')
        ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        predictors = ee.Image([ndvi, ndbi, ndwi, elev, slope, aspect])
        feat_names = ['NDVI', 'NDBI', 'NDWI', 'elevation', 'Slope', 'Aspect']
        
        training_pts = lst_100m.addBands(predictors).sample(region=roi, scale=100, numPixels=350, seed=42, geometries=False, dropNulls=True)
        rf_model = ee.Classifier.smileRandomForest(30).setOutputMode('REGRESSION').train(features=training_pts, classProperty='LST', inputProperties=feat_names)
        lst_20m = predictors.classify(rf_model, 'Predicted_LST').updateMask(valid_mask)
        dict_imp = rf_model.explain().get('importance').getInfo()
        
        reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.min(), sharedInputs=True).combine(ee.Reducer.median(), sharedInputs=True)
        stats_100 = lst_100m.reduceRegion(reducer=reducer, geometry=roi, scale=100, maxPixels=1e9).getInfo()
        stats_20 = lst_20m.reduceRegion(reducer=reducer, geometry=roi, scale=20, maxPixels=1e9).getInfo()
        
        comp_stats = {
            'n_min': safe_stat(stats_100, 'LST_min'), 'n_mean': safe_stat(stats_100, 'LST_mean'), 'n_median': safe_stat(stats_100, 'LST_median'), 'n_max': safe_stat(stats_100, 'LST_max'),
            'd_min': safe_stat(stats_20, 'Predicted_LST_min'), 'd_mean': safe_stat(stats_20, 'Predicted_LST_mean'), 'd_median': safe_stat(stats_20, 'Predicted_LST_median'), 'd_max': safe_stat(stats_20, 'Predicted_LST_max')
        }
        
        val_data = training_pts.classify(rf_model, 'Predicted_LST').reduceColumns(ee.Reducer.toList(2), ['LST', 'Predicted_LST']).get('list').getInfo()
        df_eval = pd.DataFrame(val_data, columns=['Actual', 'Predicted'])
        rmse = np.sqrt(mean_squared_error(df_eval['Actual'], df_eval['Predicted']))
        r2 = r2_score(df_eval['Actual'], df_eval['Predicted'])
        
        return lst_100m, lst_20m, df_eval, rmse, r2, dict_imp, comp_stats, roi
    except Exception as e:
        return None, None, None, None, None, None, None, None

def run_ml_inf(mdl, tmp, is_hw, is_hol, scale_factor=1.0):
    if mdl:
        df_i = pd.DataFrame({'Mx_T': [tmp], 'Is_HW': [is_hw], 'Is_Hol': [is_hol]})
        pd_pax = mdl.predict(df_i)[0] * scale_factor
    else:
        b = 850 * scale_factor
        pd_pax = b + ((tmp-25)*22*scale_factor) + (is_hw*120*scale_factor) + (is_hol*180*scale_factor)
    v_rto = 0.28 if is_hol else 0.12
    return int(pd_pax), int(pd_pax * v_rto)

# =====================================================================
# 4. DESTINATION DATABASE
# =====================================================================
cty_coords = [
    {"City": "Gold Coast, Australia", "Lat": -28.0167, "Lon": 153.4000, "Tourists_M": 12.0, "Avg_Summer_LST": 34.5, "Continent": "Oceania"},
    {"City": "Brisbane, Australia", "Lat": -27.4705, "Lon": 153.0260, "Tourists_M": 8.0, "Avg_Summer_LST": 36.2, "Continent": "Oceania"},
    {"City": "Sydney, Australia", "Lat": -33.8688, "Lon": 151.2093, "Tourists_M": 16.0, "Avg_Summer_LST": 32.5, "Continent": "Oceania"},
    {"City": "Melbourne, Australia", "Lat": -37.8136, "Lon": 144.9631, "Tourists_M": 11.0, "Avg_Summer_LST": 31.0, "Continent": "Oceania"},
    {"City": "Perth, Australia", "Lat": -31.9505, "Lon": 115.8605, "Tourists_M": 5.0, "Avg_Summer_LST": 38.0, "Continent": "Oceania"},
    {"City": "Adelaide, Australia", "Lat": -34.9285, "Lon": 138.6007, "Tourists_M": 3.0, "Avg_Summer_LST": 36.5, "Continent": "Oceania"},
    {"City": "Canberra, Australia", "Lat": -35.2809, "Lon": 149.1300, "Tourists_M": 2.5, "Avg_Summer_LST": 32.0, "Continent": "Oceania"},
    {"City": "Hobart, Australia", "Lat": -42.8821, "Lon": 147.3272, "Tourists_M": 1.5, "Avg_Summer_LST": 25.0, "Continent": "Oceania"},
    {"City": "Darwin, Australia", "Lat": -12.4634, "Lon": 130.8456, "Tourists_M": 1.0, "Avg_Summer_LST": 33.0, "Continent": "Oceania"},
    {"City": "Cairns, Australia", "Lat": -16.9186, "Lon": 145.7781, "Tourists_M": 2.8, "Avg_Summer_LST": 31.5, "Continent": "Oceania"},
    {"City": "Auckland, New Zealand", "Lat": -36.8485, "Lon": 174.7633, "Tourists_M": 3.0, "Avg_Summer_LST": 24.5, "Continent": "Oceania"},
    {"City": "Queenstown, New Zealand", "Lat": -45.0312, "Lon": 168.6626, "Tourists_M": 1.5, "Avg_Summer_LST": 22.0, "Continent": "Oceania"},
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
    {"City": "New York City, USA", "Lat": 40.7128, "Lon": -74.0060, "Tourists_M": 13.5, "Avg_Summer_LST": 34.0, "Continent": "Americas"},
    {"City": "Los Angeles, USA", "Lat": 34.0522, "Lon": -118.2437, "Tourists_M": 8.5, "Avg_Summer_LST": 35.5, "Continent": "Americas"},
    {"City": "Las Vegas, USA", "Lat": 36.1699, "Lon": -115.1398, "Tourists_M": 6.5, "Avg_Summer_LST": 42.0, "Continent": "Americas"},
    {"City": "Miami, USA", "Lat": 25.7617, "Lon": -80.1918, "Tourists_M": 8.0, "Avg_Summer_LST": 35.0, "Continent": "Americas"},
    {"City": "Honolulu, USA", "Lat": 21.3069, "Lon": -157.8583, "Tourists_M": 3.0, "Avg_Summer_LST": 31.0, "Continent": "Americas"},
    {"City": "Toronto, Canada", "Lat": 43.6510, "Lon": -79.3470, "Tourists_M": 4.5, "Avg_Summer_LST": 29.5, "Continent": "Americas"},
    {"City": "Cancun, Mexico", "Lat": 21.1619, "Lon": -86.8515, "Tourists_M": 6.0, "Avg_Summer_LST": 34.5, "Continent": "Americas"},
    {"City": "Rio de Janeiro, Brazil", "Lat": -22.9068, "Lon": -43.1729, "Tourists_M": 2.5, "Avg_Summer_LST": 36.0, "Continent": "Americas"},
    {"City": "Buenos Aires, Argentina", "Lat": -34.6037, "Lon": -58.3816, "Tourists_M": 3.0, "Avg_Summer_LST": 34.0, "Continent": "Americas"},
    {"City": "Cape Town, South Africa", "Lat": -33.9249, "Lon": 18.4241, "Tourists_M": 1.5, "Avg_Summer_LST": 32.0, "Continent": "Africa"},
    {"City": "Cairo, Egypt", "Lat": 30.0444, "Lon": 31.2357, "Tourists_M": 6.0, "Avg_Summer_LST": 41.5, "Continent": "Africa"},
    {"City": "Marrakech, Morocco", "Lat": 31.6295, "Lon": -7.9811, "Tourists_M": 3.0, "Avg_Summer_LST": 40.0, "Continent": "Africa"}
]
df_cities = pd.DataFrame(cty_coords)
df_cities['Type'] = np.where(df_cities['City'].str.contains('Australia'), 'Primary Integration (AU)', 'Global Observation')

# =====================================================================
# 5. APP LAYOUT & PRESENTATION LAYER
# =====================================================================
st.markdown('<div class="header-card">', unsafe_allow_html=True)
st.markdown("<h1>V-HEAT: Destination Infrastructure Resilience Model</h1>", unsafe_allow_html=True)
st.markdown("<p>An integrated analytical framework linking Earth Observation, Historical Climate baselines, Future Projections, and Public Health infrastructure to assess tourism destination carrying capacity under extreme heat.</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("How to Use This Dashboard"):
    st.markdown("""
    **Step 1: Select a Destination**
    Use the dropdown to select from 50 global tourism precincts. Note that the hospital simulation module is strictly unlocked for Australian cities to ensure predictive integrity.
    
    **Step 2: Explore Spatial Hazards**
    Review the historical Land Surface Temperature (LST) baseline. You may run the **Spatial Downscaling Model** to sharpen the thermal imagery from 100m to 20m, revealing granular urban heat hotspots. Click anywhere on the map to activate the live coordinate inspector.
    
    **Step 3: Analyze Future Projections & Simulate Constraints**
    Click on a specific year within the **NASA CMIP6 Chart** to inject that year's forecasted maximum temperature directly into the hospital simulator. Alternatively, utilize the manual slider to test hypothetical climate thresholds.
    
    **Step 4: AI Strategic Advisor**
    Once the parameters are set, generate a dynamic policy brief utilizing Large Language Models to receive actionable mitigation strategies tailored to the localized constraints.
    """)

with st.expander("Architecture, Methodology & Transparent Data Sources"):
    st.markdown("""
    **Core Data Sources & Hyperlinks:**
    * **Public Health Data:** [Australian Institute of Health and Welfare (AIHW) ED API](https://myhospitalsapi.aihw.gov.au/api/v1/measure-downloads/myh-ed).
    * **Historical Climate Data:** [Bureau of Meteorology (BoM) AWS SILO Open Data](https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/index.html).
    * **Spatial Data:** Landsat 8 TIRS & Sentinel-2 Harmonized via Google Earth Engine API.
    * **Climate Forecasts:** NASA NEX-GDDP-CMIP6 (Scenario SSP5-8.5). Extracts Annual Maximum Near-Surface Air Temperature (tasmax) projections.
    
    **Analytical Pipeline Notes:**
    * **Spatial Downscaling:** Transforms native 100m thermal resolution to 20m utilizing Machine Learning (Random Forest) driven by Sentinel-2 predictors (NDVI, NDBI, DEM).
    * **Zonal Statistics Clarification:** The values displayed are spatial aggregates (Mean, Median, Max) calculated over the defined tourism precinct boundary, derived from a temporal composite image (median values over the selected summer months) to ensure a gap-free representation.
    * **Baseline Hospital Data:** The underlying health capacity model is anchored to **2022 records**, deliberately selected as a stable, post-pandemic representative period to accurately gauge standard hospital carrying capacity without COVID-19 lockdown anomalies.
    * **Demographic Scaling Factor (Proof of Concept Assumption):** Because this model is trained exclusively on the Gold Coast (12 Million annual tourists), simulating capacity for other Australian cities utilizes a synthetic scaling multiplier. This compares local tourist volume against the Gold Coast baseline to proportionally scale the hospital bed footprint. 
    * *Tourist volumes for the global matrix are synthesized estimates benchmarking the Mastercard Global Destination Cities Index.*
    
    **Definition of Extreme Heat (The 35°C Threshold):**
    * In operational climatology, the Bureau of Meteorology (BoM) defines severe heat events using the Excess Heat Factor (EHF), a complex index comparing 3-day average temperatures against local 95th percentile historical baselines.
    * For the purpose of this interactive computational prototype, a standardized absolute threshold of **35°C** is utilized. This temperature is widely recognized by Australian occupational health and safety protocols (e.g., Safe Work Australia) as a critical trigger point for acute thermal stress and physiological vulnerability.
    
    **Resilience Status & Clinical Triage Assumptions:**
    * **SAFE:** Triggered when the maximum temperature is below the extreme heat threshold (<35°C) during off-peak seasons.
    * **WARNING:** Triggered by EITHER an extreme heat event (≥35°C) OR peak tourist season volume.
    * **CRITICAL:** Triggered when BOTH an extreme heat event (≥35°C) AND peak tourist season occur simultaneously, presenting a severe risk of infrastructure failure.
    * **Baseline vs Heat Stress Burden:** "Baseline" refers to the expected daily emergency presentations under normal thermal conditions. The delta (Heat Stress Burden) represents the excess patient influx directly attributable to temperatures exceeding 35°C.
    * **Clinical Triage Shift:** During severe heat events (≥35°C), the model assumes a proportional shift in clinical severity. Presentations requiring 'Resuscitation' and 'Emergency' care increase by approximately 7-13% due to acute thermal stress, while 'Non-Urgent' cases relatively decrease.
    """)

mdl = load_ml_mdl()
gee_status = init_ee()

# --- STATE SYNCHRONIZATION CALLBACKS ---
def on_city_change():
    st.session_state.selected_city = st.session_state.dropdown_city
    st.session_state.rf_downscale_run = False 
    st.session_state.rf_results = None
    st.session_state.temp_source = "Default"
    st.session_state.run_ai = False

def on_year_change():
    st.session_state.selected_year = st.session_state.dropdown_year
    st.session_state.rf_downscale_run = False 
    st.session_state.rf_results = None
    st.session_state.temp_source = "Default"
    st.session_state.run_ai = False

def on_slider_change():
    st.session_state.sim_temp = st.session_state.temp_slider
    st.session_state.temp_source = "Manual"
    st.session_state.sim_year_label = "Manual"
    st.session_state.cmip_chart_key += 1 
    st.session_state.run_ai = False

def on_downscale_click():
    st.session_state.rf_downscale_run = True
    st.session_state.sim_year_label = "Custom Baseline"
    st.session_state.run_ai = False

def on_back_click():
    st.session_state.rf_downscale_run = False
    st.session_state.rf_results = None
    st.session_state.temp_source = "Default"
    st.session_state.run_ai = False

# --- MAIN NAVIGATOR TABS ---
tab_nav, tab_global = st.tabs(["Local Precinct Analysis", "Global Destination Vulnerability Index"])

with tab_nav:
    st.markdown('<div class="modern-card" style="padding: 20px 25px;">', unsafe_allow_html=True)
    c_nav1, c_nav2 = st.columns([1, 3])
    with c_nav1:
        curr_idx = df_cities['City'].tolist().index(st.session_state.selected_city) if st.session_state.selected_city in df_cities['City'].tolist() else 0
        st.selectbox("Select Tourism Precinct (50 Cities):", df_cities['City'].tolist(), index=curr_idx, key="dropdown_city", on_change=on_city_change)
    with c_nav2:
        if "Australia" in st.session_state.selected_city:
            st.markdown('<div class="integration-full"><b>Status:</b> Full Integration Active (Remote Sensing + Local Health Data + NASA CMIP6 Forecast)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="integration-partial"><b>Status:</b> Partial Integration (Remote Sensing + NASA CMIP6 Forecast Only). Health simulator locked.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_global:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### Global Tourism Vulnerability Matrix")
    st.markdown("Comparative analysis mapping tourist volume against extreme summer surface temperatures. Data synthesized for prototype demonstration.")
    
    fig_global = px.scatter(df_cities, x='Avg_Summer_LST', y='Tourists_M', size='Tourists_M', color='Type', 
                            hover_name='City', size_max=40, opacity=0.8,
                            labels={'Avg_Summer_LST': 'Average Summer LST (°C)', 'Tourists_M': 'Annual Tourists (Millions)'},
                            color_discrete_map={"Primary Integration (AU)": "#1E3A8A", "Global Observation": "#94A3B8"})
    fig_global.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=10, l=10, r=10), height=400)
    fig_global.add_hline(y=10, line_dash="dot", line_color="gray")
    fig_global.add_vline(x=35, line_dash="dot", line_color="#E53E3E")
    fig_global.add_annotation(x=42, y=25, text="High Risk Zone", showarrow=False, font=dict(color="#E53E3E", size=14, weight="bold"))
    st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

city_row = df_cities[df_cities['City'] == st.session_state.selected_city].iloc[0]
sel_lat, sel_lon = city_row['Lat'], city_row['Lon']

is_dp = ("Australia" in st.session_state.selected_city)
season_txt = "Jun-Aug" if sel_lat > 0 else "Dec-Feb"

current_lst_max = 0.0
current_status = "SAFE"
current_tourist_pct = 0.0

# --- THE 3 PILLARS (REBALANCED LAYOUT) ---
c1, c2 = st.columns([1.2, 0.8], gap="large")

with c1:
    st.markdown('<div class="modern-card" style="height: 100%;">', unsafe_allow_html=True)
    
    row_title, row_slider = st.columns([2, 1])
    with row_title:
        st.subheader("1. Spatial Hazard Exposure")
    with row_slider:
        yr_options = ["2019-2025 Multi-Year Composite", "2025", "2024", "2023", "2022", "2021", "2020", "2019"]
        yr_idx = yr_options.index(st.session_state.selected_year)
        st.selectbox("Select Temporal Baseline:", yr_options, index=yr_idx, key="dropdown_year", on_change=on_year_change)

    st.markdown(f'<span class="subtitle-text"><b>Data Source:</b> Landsat 8 TIRS. Peak summer ({season_txt}) thermal signatures for {st.session_state.selected_year}.</span>', unsafe_allow_html=True)
    
    # Pure Folium Map Rendering
    m_base = folium.Map(location=[sel_lat, sel_lon], zoom_start=12, control_scale=True)
    folium.TileLayer('CartoDB positron', name="Light Map", control=False, attr="CARTO").add_to(m_base)
    
    # Inject CSS directly into Folium iframe to minimize attribution text size
    m_base.get_root().html.add_child(folium.Element("<style>.leaflet-control-attribution { font-size: 8px !important; color: #94A3B8 !important; background: transparent !important; } .leaflet-control-attribution a { color: #94A3B8 !important; }</style>"))
    
    # Render Interactive Legend (25 - 50 C)
    colormap = cm.LinearColormap(colors=['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'], vmin=25, vmax=50)
    colormap.caption = 'Surface Temperature (Celsius)'
    m_base.add_child(colormap)
    
    if not st.session_state.rf_downscale_run:
        with st.spinner(f"Extracting Spatial Analytics for {st.session_state.selected_year}..."):
            lst_100m, base_stats, roi = get_ee_baseline_data(sel_lat, sel_lon, st.session_state.selected_year, gee_status)
            
        if lst_100m is not None:
            vis_params = {'min': 25, 'max': 50, 'palette': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']}
            map_id_dict = ee.Image(lst_100m.clip(roi)).getMapId(vis_params)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format, attr='Map Data &copy; Google Earth Engine', name='Baseline LST (100m)', overlay=True, opacity=0.8
            ).add_to(m_base)
            
            # Map Rendering with Streamlit-Folium (enabling click inspector without geemap overhead)
            map_output = st_folium(m_base, height=430, use_container_width=True, returned_objects=["last_clicked"])
            
            # Live Inspector Logic
            if map_output and map_output.get("last_clicked"):
                c_lat = map_output["last_clicked"]["lat"]
                c_lon = map_output["last_clicked"]["lng"]
                c_pt = ee.Geometry.Point([c_lon, c_lat])
                try:
                    val = lst_100m.sample(c_pt, scale=100).first().getInfo()
                    if val:
                        v_str = f"{val['properties']['LST']:.1f}"
                        st.markdown(f'<div class="inspector-box"><b>Coordinates:</b> {c_lat:.4f}, {c_lon:.4f} &nbsp;|&nbsp; <b>Baseline LST:</b> {v_str} °C</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="inspector-box" style="color: #94A3B8;"><b>Coordinates:</b> {c_lat:.4f}, {c_lon:.4f} &nbsp;|&nbsp; No data at this location.</div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div class="inspector-box" style="color: #94A3B8;"><b>Coordinates:</b> {c_lat:.4f}, {c_lon:.4f} &nbsp;|&nbsp; Error retrieving data.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="inspector-box" style="color: #94A3B8;"><b>Coordinates:</b> Waiting for map click... &nbsp;|&nbsp; <b>Baseline LST:</b> ---</div>', unsafe_allow_html=True)

            st.markdown("##### Regional Baseline LST Panel (100m)")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Coolest Area", f"{base_stats['min']} °C")
            sc2.metric("Median Temp", f"{base_stats['median']} °C")
            sc3.metric("Average Temp", f"{base_stats['mean']} °C")
            sc4.metric("Peak Hotspot", f"{base_stats['max']} °C")
            
            current_lst_max = float(base_stats['max'])
            
            if base_stats['mean'] > 0 and st.session_state.temp_source in ["Default", "Native LST"]: 
                st.session_state.temp_slider = float(base_stats['mean'])
                st.session_state.sim_temp = float(base_stats['mean'])
                st.session_state.temp_source = "Native LST"
            
            st.markdown("<hr style='margin: 15px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
            st.markdown('<div class="btn-ml">', unsafe_allow_html=True)
            st.button("Run Spatial Downscaling Model (~15s)", on_click=on_downscale_click)
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        with st.spinner("Executing Machine Learning Downscaling (~15s). Processing high-res predictors..."):
            # Robust State Fix: using .get() to prevent AttributeError on callback re-runs
            if st.session_state.get('rf_results') is None:
                lst_100m, lst_20m, df_eval, rmse, r2, dict_imp, comp_stats, roi = get_ee_downscaled_data(sel_lat, sel_lon, st.session_state.selected_year, gee_status)
                st.session_state.rf_results = (lst_100m, lst_20m, df_eval, rmse, r2, dict_imp, comp_stats, roi)
            else:
                lst_100m, lst_20m, df_eval, rmse, r2, dict_imp, comp_stats, roi = st.session_state.get('rf_results')
                
        if lst_100m is not None:
            vis_params = {'min': 25, 'max': 50, 'palette': ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']}
            left_id = ee.Image(lst_100m.clip(roi)).getMapId(vis_params)
            right_id = ee.Image(lst_20m.clip(roi)).getMapId(vis_params)
            
            left_tile = folium.TileLayer(tiles=left_id['tile_fetcher'].url_format, attr='GEE', name='Native 100m', overlay=True, opacity=0.9).add_to(m_base)
            right_tile = folium.TileLayer(tiles=right_id['tile_fetcher'].url_format, attr='GEE', name='Downscaled 20m', overlay=True, opacity=0.9).add_to(m_base)
            SideBySideLayers(left_tile, right_tile).add_to(m_base)
            
            map_output = st_folium(m_base, height=450, use_container_width=True, returned_objects=["last_clicked"])
            
            # Live Inspector Logic (Comparing both layers)
            if map_output and map_output.get("last_clicked"):
                c_lat = map_output["last_clicked"]["lat"]
                c_lon = map_output["last_clicked"]["lng"]
                c_pt = ee.Geometry.Point([c_lon, c_lat])
                try:
                    val_100 = lst_100m.sample(c_pt, scale=100).first().getInfo()
                    val_20 = lst_20m.sample(c_pt, scale=20).first().getInfo()
                    v1_str = f"{val_100['properties']['LST']:.1f}" if val_100 else "N/A"
                    v2_str = f"{val_20['properties']['Predicted_LST']:.1f}" if val_20 else "N/A"
                    st.markdown(f'<div class="inspector-box"><b>Coordinates:</b> {c_lat:.4f}, {c_lon:.4f} &nbsp;|&nbsp; <b>Native:</b> {v1_str} °C &nbsp;|&nbsp; <b>Downscaled:</b> {v2_str} °C</div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown(f'<div class="inspector-box" style="color: #94A3B8;"><b>Coordinates:</b> {c_lat:.4f}, {c_lon:.4f} &nbsp;|&nbsp; Error retrieving data.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="inspector-box" style="color: #94A3B8;"><b>Coordinates:</b> Waiting for map click... &nbsp;|&nbsp; <b>Native:</b> --- &nbsp;|&nbsp; <b>Downscaled:</b> ---</div>', unsafe_allow_html=True)
            
            st.markdown("##### LST Extracted Statistics: Native vs Downscaled")
            st.markdown('<span class="subtitle-text">Notice how the downscaled 20m model may detect lower or higher extreme localized temperatures (Hotspots) missed by the 100m baseline, while smoothing anomalous out-of-bounds pixels.</span>', unsafe_allow_html=True)
            
            current_lst_max = float(comp_stats['d_max'])
            
            if st.session_state.temp_source in ["Default", "Native LST", "Downscaled LST"]:
                st.session_state.temp_slider = float(comp_stats['d_mean'])
                st.session_state.sim_temp = float(comp_stats['d_mean'])
                st.session_state.temp_source = "Downscaled LST"
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Coolest (20m)", f"{comp_stats['d_min']} °C", f"{round(comp_stats['d_min'] - comp_stats['n_min'], 1)} °C vs Native", delta_color="inverse")
            s2.metric("Median (20m)", f"{comp_stats['d_median']} °C", f"{round(comp_stats['d_median'] - comp_stats['n_median'], 1)} °C vs Native", delta_color="off")
            s3.metric("Average (20m)", f"{comp_stats['d_mean']} °C", f"{round(comp_stats['d_mean'] - comp_stats['n_mean'], 1)} °C vs Native", delta_color="off")
            s4.metric("Hotspot (20m)", f"{comp_stats['d_max']} °C", f"{round(comp_stats['d_max'] - comp_stats['n_max'], 1)} °C vs Native", delta_color="inverse")
            
            with st.expander("Show Machine Learning Validation Metrics & Limitations"):
                st.markdown("<i>Note: In Random Forest spatial regression, extreme thermal outliers in 100m native pixels might be smoothed down in 20m predictions if unsupported by the predictors (NDVI/NDBI/DEM). This is a known Remote Sensing phenomenon called Regression to the Mean.</i>", unsafe_allow_html=True)
                c_rf1, c_rf2 = st.columns(2)
                with c_rf1:
                    fig_s = px.scatter(df_eval, x='Actual', y='Predicted', title=f"Spatial R²: {r2:.2f} | RMSE: {rmse:.2f} °C")
                    mn, mx = min(df_eval['Actual']), max(df_eval['Actual'])
                    fig_s.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx, line=dict(color='red', dash='dash'))
                    fig_s.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})
                with c_rf2:
                    df_i = pd.DataFrame({'Predictor': list(dict_imp.keys()), 'Importance': list(dict_imp.values())}).sort_values(by='Importance', ascending=True)
                    fig_i = px.bar(df_i, x='Importance', y='Predictor', orientation='h', title="Random Forest Variable Importance")
                    fig_i.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_i, use_container_width=True, config={'displayModeBar': False})
        else:
            st.error("Spatial downscaling failed due to GEE timeout or lack of satellite data for this region.")
            
        if st.button("Back to Baseline Map", on_click=on_back_click):
            pass

    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("2. Future Climate Projections (CMIP6)")
    st.markdown('<span class="subtitle-text">NASA NEX-GDDP (Model: ACCESS-CM2). <b>Select a projected year on the chart</b> to automatically load its temperature into the Infrastructure Simulator below.</span>', unsafe_allow_html=True)
    
    with st.spinner("Querying NASA CMIP6 Database..."):
        df_cmip = get_real_cmip6_data(sel_lat, sel_lon, gee_status)
    
    fig2 = go.Figure(go.Scatter(x=df_cmip['Year'], y=df_cmip['Max_Temp'], mode='lines+markers', customdata=df_cmip['Max_Temp'], fill='tozeroy', fillcolor='rgba(229, 62, 62, 0.1)', line=dict(color='#E53E3E', width=3)))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1E293B', yaxis_title="Max Air Temp (°C)", margin=dict(t=5, b=5, l=0, r=0), height=150)
    
    cmip_sel = st.plotly_chart(fig2, on_select="rerun", selection_mode="points", use_container_width=True, key=f"cmip_chart_{st.session_state.cmip_chart_key}")
    if cmip_sel and hasattr(cmip_sel, 'selection'):
        pts = cmip_sel.selection.get('points', [])
        if pts and len(pts) > 0:
            new_temp = float(pts[0].get('customdata'))
            new_year = int(pts[0].get('x'))
            if st.session_state.sim_temp != new_temp or st.session_state.temp_source != "CMIP6":
                st.session_state.temp_slider = new_temp
                st.session_state.sim_temp = new_temp
                st.session_state.sim_year_label = str(new_year)
                st.session_state.temp_source = "CMIP6"
                st.session_state.run_ai = False
                st.rerun() 
    st.markdown('</div>', unsafe_allow_html=True)
    
    if is_dp:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.subheader("3. Destination Capacity Simulator")
        st.markdown('<span class="subtitle-text">Machine Learning trained on historical BoM weather & hospital records. Simulates how localized heat extremes and tourist seasons impact infrastructure carrying capacity.</span>', unsafe_allow_html=True)
        
        if st.session_state.temp_source == "CMIP6":
            st.markdown(f'<div class="sim-panel"><b>Simulation Active:</b> Applying CMIP6 forecast for the year <b>{st.session_state.sim_year_label}</b> at <b>{st.session_state.sim_temp:.1f}°C</b>.</div>', unsafe_allow_html=True)
        elif st.session_state.temp_source == "Manual":
            st.markdown(f'<div class="sim-panel-manual"><b>Manual Mode:</b> Custom temperature applied. CMIP6 selection cleared.</div>', unsafe_allow_html=True)
        elif st.session_state.temp_source == "Native LST":
            st.markdown(f'<div class="sim-panel-manual"><b>Baseline Mode:</b> Average temp from Native LST (100m) used (<b>{st.session_state.sim_temp:.1f}°C</b>).</div>', unsafe_allow_html=True)
        elif st.session_state.temp_source == "Downscaled LST":
            st.markdown(f'<div class="sim-panel"><b>Machine Learning Mode:</b> Average temp from Downscaled LST (20m) used (<b>{st.session_state.sim_temp:.1f}°C</b>).</div>', unsafe_allow_html=True)

        sim_tmp = st.slider("Forecasted Maximum Temperature (°C)", min_value=15.0, max_value=55.0, key="temp_slider", on_change=on_slider_change)
        sim_hw = 1 if sim_tmp >= 35.0 else 0
        sim_hol = st.radio("Tourism Seasonality Exposure", [1, 0], format_func=lambda x: f"Peak Tourist Season ({season_txt})" if x==1 else "Off-Peak Season", horizontal=True, on_change=lambda: st.session_state.update(run_ai=False))
        
        city_tourists = city_row['Tourists_M']
        demographic_scale = city_tourists / 12.0
        
        tot_pax, vis_pax = run_ml_inf(mdl, sim_tmp, sim_hw, sim_hol, scale_factor=demographic_scale)
        current_tourist_pct = round((vis_pax/tot_pax)*100, 1) if tot_pax > 0 else 0
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Est. Daily Hospital Cases", f"{tot_pax}", delta=f"{'+' if sim_hw else ''}{int(tot_pax*0.12)} (Severe Heat Burden)" if sim_hw else "Baseline", delta_color="inverse")
        mc2.metric("Transient Tourist Load", f"{vis_pax}", delta=f"{current_tourist_pct}% of operating capacity", delta_color="off")
        
        st.markdown("##### Predicted Clinical Severity Distribution")
        
        t_dist = [tot_pax*0.05, tot_pax*0.25, tot_pax*0.45, tot_pax*0.25]
        if sim_hw: 
            t_dist = [tot_pax*0.12, tot_pax*0.38, tot_pax*0.35, tot_pax*0.15] 
            
        df_trg = pd.DataFrame({'Category': ['Resuscitation', 'Emergency', 'Urgent', 'Non-Urgent'], 'Cases': t_dist})
        
        fig_t = px.bar(df_trg, x='Cases', y=['Capacity Load']*4, color='Category', orientation='h', color_discrete_sequence=['#9B2C2C', '#DD6B20', '#ECC94B', '#48BB78'])
        fig_t.update_layout(barmode='stack', margin=dict(t=10, b=0, l=0, r=0), height=110, yaxis_title=None, xaxis_title=None, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.8, xanchor="center", x=0.5))
        fig_t.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_t, use_container_width=True, config={'displayModeBar': False})
        
        if sim_hw and sim_hol:
            current_status = "CRITICAL"
            st.markdown('<div class="status-badge status-critical">[ CRITICAL ] Severe Heat Event during Peak Season. High risk of infrastructure failure.</div>', unsafe_allow_html=True)
        elif sim_hw or sim_hol:
            current_status = "WARNING"
            st.markdown('<div class="status-badge status-warn">[ WARNING ] Elevated Strain. Destination carrying capacity stressed.</div>', unsafe_allow_html=True)
        else:
            current_status = "SAFE"
            st.markdown('<div class="status-badge status-safe">[ SAFE ] Normal Operating Capacity. Destination resilient.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="modern-card" style="height: 100%; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding: 40px;">', unsafe_allow_html=True)
        st.markdown("<h3>Simulator Locked</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B;'>The Predictive Simulator requires localized historical health records to establish carrying capacity baselines. Currently, this prototype is trained and unlocked for <b>all major cities in Australia</b>.</p>", unsafe_allow_html=True)
        st.info("Please select an Australian city from the dropdown to unlock the full integration.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PILLAR 4: AI STRATEGIC POLICY ADVISOR ---
if is_dp:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.subheader("4. AI-Driven Strategic Policy Advisor")
    st.markdown('<span class="subtitle-text">Generative AI analysis based on the current spatial, climatic, and infrastructure parameters to provide actionable mitigation strategies for destination management.</span>', unsafe_allow_html=True)
    
    col_ai1, col_ai2 = st.columns([1, 3], gap="large")
    
    with col_ai1:
        st.markdown("##### Reputation Risk Assessment")
        if current_status == "CRITICAL":
            st.markdown('<div class="status-badge status-critical" style="margin-top:0;">[ HIGH RISK ]</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #64748B; margin-top: 10px;'>Potential for negative international press, increased travel insurance claims, and decline in future bookings due to overwhelmed local amenities.</p>", unsafe_allow_html=True)
        elif current_status == "WARNING":
            st.markdown('<div class="status-badge status-warn" style="margin-top:0;">[ MODERATE RISK ]</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #64748B; margin-top: 10px;'>Noticeable decline in visitor satisfaction. Minor infrastructure delays expected. Preventative communications advised.</p>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-safe" style="margin-top:0;">[ LOW RISK ]</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.9rem; color: #64748B; margin-top: 10px;'>Destination operating optimally. High probability of positive visitor experience and sustained tourism reputation.</p>", unsafe_allow_html=True)
            
        st.markdown('<div class="btn-ml" style="margin-top: 20px;">', unsafe_allow_html=True)
        if st.button("Generate AI Policy Brief (~5s)"):
            st.session_state.run_ai = True
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ai2:
        st.markdown("##### Actionable Mitigation Strategies")
        if st.session_state.get('run_ai', False):
            with st.spinner("Analyzing current variables via Google Gemini LLM (~5s)..."):
                target_year = st.session_state.sim_year_label if st.session_state.sim_year_label not in ["Manual", "Composite Baseline"] else "Current Baseline"
                ai_response = get_ai_policy_insights(
                    city=st.session_state.selected_city, temp=st.session_state.sim_temp, year=target_year, status=current_status, tourist_pct=current_tourist_pct, lst_max=current_lst_max
                )
                st.markdown(f'<div class="ai-box">{ai_response}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ai-box" style="color: #94A3B8; text-align: center; padding: 40px;"><i>Click "Generate AI Policy Brief" to formulate dynamic, context-aware policy recommendations based on the currently selected simulation scenario.</i></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; margin-top: 50px; color: #94A3B8; font-size: 0.75rem; font-family: \'Inter\', sans-serif;">Disclaimer: The AI-generated insights and spatial models provided by the V-HEAT tool are for supplementary analytical context only and should not replace primary clinical, meteorological, or authoritative local data.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; margin-top: 10px; margin-bottom: 20px; color: #64748B; font-size: 0.85rem; font-family: \'Inter\', sans-serif;">Developed by Akram Sripandam Prihanantya, 2026.</div>', unsafe_allow_html=True)
