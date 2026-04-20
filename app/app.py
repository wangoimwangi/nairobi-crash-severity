# ============================================================
# app.py
# Accident Severity Classification System (ASCS)
# ============================================================
# SYSTEM OVERVIEW:
# This application classifies road traffic accident severity as HIGH or LOW using a Balanced Random Forest model trained  on the RTA Addis Ababa dataset as a proxy for Nairobi's emergency dispatch environment.
# Dispatcher inputs 7 fields → model derives features → outputs HIGH (ALS) or LOW (BLS) severity classification.
# ============================================================

import os
import streamlit as st
import pandas as pd
from datetime import datetime

# Local modules
from predictor import predict, get_temporal_features, get_weather
from hospitals import get_addis_area, get_hospitals

# ── 1. PAGE CONFIGURATION ────────────────────────────────────

st.set_page_config(
    page_title="Accident Severity Classification",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ── 2. LOAD EXTERNAL CSS ─────────────────────────────────────
# CSS is kept in a separate style.css file for maintainability.
# os.path.dirname(__file__) ensures the path works both locally and on Streamlit Cloud deployment.

def load_css(path):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "style.css"))


# ── 3. SESSION STATE ─────────────────────────────────────────
# Stores the last 5 classifications made in the current session.
# Streamlit re-runs the script on every interaction, so session state is used to persist data across re-runs.

if 'history' not in st.session_state:
    st.session_state.history = []


# ── 4. LIVE TEMPORAL & WEATHER DATA ──────────────────────────
# Temporal features (hour, day, rush hour, night, weekend) are auto-derived from the system clock - no dispatcher input needed.
# Weather is fetched from the Open-Meteo API for Nairobi.
# Both are passed to the model as engineered features.

temporal        = get_temporal_features()
now             = datetime.now()
time_str        = now.strftime("%H:%M")
day_str         = now.strftime("%A, %d %B %Y")
current_weather = get_weather()

# Map weather codes to display labels

weather_str = {
    'Raining'    : '🌧️ Rain',
    'Cloudy'     : '☁️ Cloudy',
    'Fog or mist': '🌫️ Fog',
    'Normal'     : '☀️ Clear'
}.get(current_weather, '☀️ Clear')

# Build operational condition flags for the info bar

flags = []
if temporal.get('Is_night'):     flags.append("🌙 Night")
if temporal.get('Is_rush_hour'): flags.append("🚦 Rush hour")
if temporal.get('Is_weekend'):   flags.append(" Weekend")
flag_str = " · ".join(flags) if flags else "Normal conditions"


# ── 5. PAGE HEADER ───────────────────────────────────────────
# Custom HTML header - styled via .main-header in style.css.

st.markdown(f"""
<div class="main-header">
    <div class="agency-tag">Incident Triage Unit</div>
    <h1>Accident Severity Classification System</h1>
    <p class="header-subtitle">
        Emergency Dispatch Decision Support · Nairobi County · ML-Powered Analysis
    </p>
</div>
""", unsafe_allow_html=True)


# ── 6. AUTO-INFO BAR ─────────────────────────────────────────
# Displays live time, date, weather and conditions.
# The pulsing cyan dot signals the system is active.
# These values feed into the model automatically - the dispatcher does not need to enter time or weather manually.

st.markdown(f"""
<div class="auto-info">
    <span>
        <strong>{time_str}</strong>
        &nbsp;·&nbsp; {day_str}
        &nbsp;·&nbsp; {weather_str}
        &nbsp;·&nbsp; {flag_str}
    </span>
</div>
""", unsafe_allow_html=True)


# ── 7. TWO-COLUMN LAYOUT ─────────────────────────────────────
# Left column  : Dispatcher inputs (7 fields from caller report)
# Right column : Classification result, hospital alert, risk factors, and session log — all output panels

col_input, col_result = st.columns([1, 1.2], gap="large")


# ════════════════════════════════════════════════════════════
# LEFT COLUMN - INCIDENT INPUT
# The dispatcher enters details reported by the caller at the accident scene. 
# All 7 fields map to features in the trained Balanced Random Forest model pipeline.
# ════════════════════════════════════════════════════════════

with col_input:
    st.subheader("Incident Details")
    st.markdown("*Enter details from the caller report*")

    # ── LOCATION ─────────────────────────────────────────────
# Nairobi area is mapped to the land-use category used in the RTA training dataset (via hospitals.py).
# Categories: Office areas, Residential areas, Outside rural areas, Industrial areas. This satisfies FR iv.

    st.markdown("**Location**")
    nairobi_area = st.selectbox(
        "Area of Accident",
        options=[
            "CBD", "Upper Hill", "Westlands", "Parklands",
            "Mombasa Road", "Langata/Ngong Road/Southern Bypass",
            "Thika Road/Kasarani", "Waiyaki Way", "Limuru Road",
            "Outer Ring Road", "Jogoo Road",
            "Eastleigh/Jogoo Road", "Karen", "Kilimani",
            "Lavington", "South B/C", "Gigiri/Runda",
            "Industrial Area", "Embakasi/JKIA", "Ruiru/Juja",
            "Dagoretti", "Kibera/Kawangware", "Other/Unknown"
        ],
        help="Select the nearest area to the accident location."
    )
    st.caption("Select the nearest area if exact location is not listed")

    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)


    # ── CRASH DYNAMICS ───────────────────────────────────────
    # Collision type, vehicle type, number of vehicles, and casualties are direct model input features that inform the severity classification.

    st.markdown("**Crash Dynamics**")

    col_a, col_b = st.columns(2)
    with col_a:
        collision_type = st.selectbox(
            "Type of Collision",
            options=["Head-on", "Rear-end", "Rollover",
                     "Hit pedestrian", "Side impact", "Other"]
        )
        num_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=1, max_value=20, value=2
        )
    with col_b:
        vehicle_type = st.selectbox(
            "Type of Vehicle",
            options=["Car/Saloon", "Matatu/Minibus",
                     "Motorcycle/Boda Boda", "Lorry/Truck",
                     "Bus", "Pickup/SUV", "Other"]
        )
        num_casualties = st.number_input(
            "Estimated Casualties",
            min_value=0, max_value=50, value=1
        )

    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)


    # ── CAUSE OF ACCIDENT ────────────────────────────────────
    # Select Unknown if the caller cannot confirm the cause.
    # This maps to the cause_of_accident feature in the pipeline.

    st.markdown("**Primary Cause of Accident**")
    cause_of_accident = st.selectbox(
        "Cause of Accident",
        options=["Unknown", "Overspeeding", "Overtaking",
                 "Changing lanes unsafely", "Drunk driving",
                 "Mechanical failure", "Other"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)


    # ── PEDESTRIAN INVOLVEMENT ───────────────────────────────
    # Pedestrian involvement is a binary feature that significantly increases predicted severity probability.

    st.markdown("**Pedestrian Involvement**")
    pedestrian_involved = st.radio(
        "Is a pedestrian involved?",
        options=["No", "Yes"],
        horizontal=True
    )

    st.markdown("<div style='margin: 1.2rem 0;'></div>", unsafe_allow_html=True)


    # ── CLASSIFY BUTTON ──────────────────────────────────────
    # Triggers the full prediction pipeline: inputs → feature engineering → model inference → result
    classify_clicked = st.button(
        "CLASSIFY SEVERITY",
        use_container_width=True,
        type="primary"
    )


# ════════════════════════════════════════════════════════════
# RIGHT COLUMN - CLASSIFICATION OUTPUT
# All output panels live here: severity result, recommended action, hospital alert, risk factors, and session log.
# ════════════════════════════════════════════════════════════

with col_result:
    st.subheader("Classification Result")

    if classify_clicked:

# ── MODEL PREDICTION ─────────────────────────────────
# Step 1: Map Nairobi area to Addis Ababa dataset equivalent (proxy dataset mapping)
# Step 2: Look up nearest hospitals for the area
# Step 3: Run the prediction pipeline - the model internally derives all engineered features from the 7 dispatcher inputs + temporal +  weather data fetched automatically
        
        addis_area = get_addis_area(nairobi_area)
        hospitals  = get_hospitals(nairobi_area)

        result = predict(
            area_addis          = addis_area,
            nairobi_area        = nairobi_area, 
            vehicle_type        = vehicle_type,
            collision_type      = collision_type,
            num_vehicles        = int(num_vehicles),
            num_casualties      = int(num_casualties),
            pedestrian_involved = (pedestrian_involved == "Yes"),
            cause_of_accident   = cause_of_accident
        )

        severity     = result['severity']
        confidence   = result['confidence']
        risk_factors = result['risk_factors']
        weather_used = result['weather']
        is_high      = severity == 'HIGH'


# ── SEVERITY RESULT PANEL ─────────────────────────────
# HIGH → ALS (Advanced Life Support): paramedics, trauma team, critical care protocol
# LOW  → BLS (Basic Life Support): standard ambulance
# The confidence score reflects the model's probability estimate. Threshold is set at 0.40 (optimised on the validation set using F2-score to prioritise recall minimising under-triage risk).

        if is_high:
            st.markdown(f"""
<div class="result-high">
    <p class="severity-text-high">🔴 HIGH SEVERITY</p>
    <p class="action-text"> DISPATCH ADVANCED LIFE SUPPORT (ALS)</p>
    <p class="confidence-text">Model confidence: {confidence}%</p>
    <div class="conf-track">
        <div class="conf-fill-high" style="width:{confidence}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-low">
    <p class="severity-text-low">🟢 LOW SEVERITY</p>
    <p class="action-text"> DISPATCH BASIC LIFE SUPPORT (BLS)</p>
    <p class="confidence-text"> Model confidence: {confidence}%</p>
    <div class="conf-track">
        <div class="conf-fill-low" style="width:{confidence}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)


 # ── HOSPITAL ALERT ────────────────────────────────────
# Nearest trauma centres are looked up from a static mapping table in hospitals.py, keyed by Nairobi area.
# Primary = closest major trauma facility.
# Secondary = backup if primary is unavailable.

        st.markdown(f"""
<div class="hospital-box">
    <strong> Alert Nearest Trauma Centre - {nairobi_area}</strong>
    <span>Primary: </span><b>{hospitals['primary']}</b><br>
    <span>Secondary: </span><b>{hospitals['secondary']}</b>
</div>
""", unsafe_allow_html=True)


# ── CONTRIBUTING RISK FACTORS ─────────────────────────
# Risk factors are derived from the input combination and highlight why this incident was classified at this severity level.
# Useful for dispatcher awareness.

        with st.expander("Contributing Risk Factors"):
            for factor in risk_factors:
                st.markdown(f"• {factor}")
            st.caption(
                f"Environment: {weather_used} conditions at time of report"
            )


# ── UPDATE SESSION HISTORY ────────────────────────────
# Keeps the last 5 classifications for the dispatcher to review within the current session.
        st.session_state.history.insert(0, {
            'Time'      : now.strftime("%H:%M"),
            'Area'      : nairobi_area,
            'Severity'  : severity,
            'Confidence': f"{confidence}%",
            'Action'    : 'ALS' if is_high else 'BLS'
        })
        st.session_state.history = st.session_state.history[:5]

    else:

# ── AWAITING STATE ────────────────────────────────────
# Shown before any classification is made.

        st.markdown("""
<div class="awaiting-box">
    <div style="font-size:2.5rem;margin-bottom:1rem"> </div>
    <div style="font-weight:600;color:#94a3b8;font-size:1.1rem">
        Awaiting Incident Report
    </div>
    <p>Fill in details on the left and click <b>Classify Severity</b></p>
</div>
""", unsafe_allow_html=True)


# ── SESSION LOG ───────────────────────────────────────────
# Visible on the right side — whether or not a classification has been run in this session.

    st.markdown("<div style='margin: 2rem 0 0.5rem 0;'></div>",
                unsafe_allow_html=True)

    with st.expander(" View Recent Classifications Log",
                     expanded=False):
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.markdown(
                '<div class="no-history-text">'
                'No incidents classified in this session yet.'
                '</div>',
                unsafe_allow_html=True
            )