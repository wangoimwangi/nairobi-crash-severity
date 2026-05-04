# ============================================================
# app.py
# Accident Severity Classification System (ASCS)
# ============================================================
# SYSTEM OVERVIEW:
# Classifies road traffic accident severity as HIGH or LOW using
# a Balanced Random Forest model trained on the RTA Addis Ababa
# dataset as a proxy for Nairobi's emergency dispatch environment.
# Dispatcher inputs 7 fields → model derives 28 features →
# outputs HIGH (ALS) or LOW (BLS) severity classification.
# ============================================================

import os
import streamlit as st
import pandas as pd
from datetime import datetime

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

def load_css(path):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "style.css"))


# ── 3. SESSION STATE ─────────────────────────────────────────

if 'history' not in st.session_state:
    st.session_state.history = []


# ── 4. LIVE TEMPORAL & WEATHER DATA ──────────────────────────

temporal        = get_temporal_features()
now             = datetime.now()
time_str        = now.strftime("%H:%M")
day_str         = now.strftime("%A, %d %B %Y")
current_weather = get_weather()

weather_str = {
    'Raining'    : 'Rain',
    'Cloudy'     : ' Cloudy',
    'Fog or mist': ' Fog',
    'Normal'     : '☀️ Clear'
}.get(current_weather, '☀️ Clear')

flags = []
if temporal.get('Is_night'):     flags.append(" Night")
if temporal.get('Is_rush_hour'): flags.append(" Rush hour")
if temporal.get('Is_weekend'):   flags.append(" Weekend")
flag_str = " · ".join(flags) if flags else "Normal conditions"


# ── 5. PAGE HEADER ───────────────────────────────────────────

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

col_input, col_result = st.columns([1, 1.2], gap="large")


# ════════════════════════════════════════════════════════════
# LEFT COLUMN — INCIDENT INPUT
#
# Default values reflect a minor incident profile (LOW):
#   Karen (residential), Rear-end, Car/Saloon, 1 vehicle,
#   0 casualties, Unknown cause, no pedestrian.
#
# Mirrors real dispatcher workflow start with least severe
# assumptions and escalate as the caller provides detail.
# ════════════════════════════════════════════════════════════

with col_input:
    st.subheader("Incident Details")
    st.markdown("*Enter details from the caller report*")

    # ── LOCATION ─────────────────────────────────────────────
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
        index=12,  # Karen — residential, LOW-profile default
        help="Select the nearest area to the accident location."
    )
    st.caption("Select the nearest area if exact location is not listed")

    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)

    # ── CRASH DYNAMICS ───────────────────────────────────────
    st.markdown("**Crash Dynamics**")

    col_a, col_b = st.columns(2)
    with col_a:
        collision_type = st.selectbox(
            "Type of Collision",
            options=["Head-on", "Rear-end", "Rollover",
                     "Hit pedestrian", "Side impact", "Other"],
            index=1   # Rear-end default
        )
        num_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=1, max_value=20, value=1
        )
    with col_b:
        vehicle_type = st.selectbox(
            "Type of Vehicle",
            options=["Car/Saloon", "Matatu/Minibus",
                     "Motorcycle/Boda Boda", "Lorry/Truck",
                     "Bus", "Pickup/SUV", "Other"],
            index=0   # Car/Saloon default
        )
        num_casualties = st.number_input(
            "Estimated Casualties",
            min_value=0, max_value=50, value=0
        )

    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)

    # ── CAUSE OF ACCIDENT ────────────────────────────────────
    st.markdown("**Primary Cause of Accident**")
    cause_of_accident = st.selectbox(
        "Cause of Accident",
        options=["Unknown", "Overspeeding", "Overtaking",
                 "Changing lanes unsafely", "Drunk driving",
                 "Mechanical failure", "Other"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)

    # ── PEDESTRIAN INVOLVEMENT ───────────────────────────────
    st.markdown("**Pedestrian Involvement**")
    pedestrian_involved = st.radio(
        "Is a pedestrian involved?",
        options=["No", "Yes"],
        index=0,
        horizontal=True
    )

    st.markdown("<div style='margin: 1.2rem 0;'></div>", unsafe_allow_html=True)

    # ── CLASSIFY BUTTON ──────────────────────────────────────
    classify_clicked = st.button(
        "CLASSIFY SEVERITY",
        use_container_width=True,
        type="primary"
    )


# ════════════════════════════════════════════════════════════
# RIGHT COLUMN — CLASSIFICATION OUTPUT
# ════════════════════════════════════════════════════════════

with col_result:
    st.subheader("Classification Result")

    if classify_clicked:

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

        severity      = result['severity']
        confidence    = result['confidence']
        risk_factors  = result['risk_factors']
        weather_used  = result['weather']
        is_borderline = result['is_borderline']
        is_high       = severity == 'HIGH'

        # ── SEVERITY RESULT PANEL ─────────────────────────────
        # HIGH → ALS (Advanced Life Support)
        # LOW  → BLS (Basic Life Support)
        # Threshold: 0.40 (F2-optimised — maximises recall,
        # minimises under-triage in emergency dispatch context)

        if is_high:
            st.markdown(f"""
<div class="result-high">
    <p class="severity-text-high">🔴 HIGH SEVERITY</p>
    <p class="action-text">DISPATCH ADVANCED LIFE SUPPORT (ALS)</p>
    <p class="confidence-text">Model confidence: {confidence}%</p>
    <div class="conf-track">
        <div class="conf-fill-high" style="width:{confidence}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            severity_label = (
                " LOW SEVERITY — BORDERLINE"
                if is_borderline else
                "🟢 LOW SEVERITY"
            )
            st.markdown(f"""
<div class="result-low">
    <p class="severity-text-low">{severity_label}</p>
    <p class="action-text">DISPATCH BASIC LIFE SUPPORT (BLS)</p>
    <p class="confidence-text">Model confidence: {confidence}%</p>
    <div class="conf-track">
        <div class="conf-fill-low" style="width:{confidence}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── HOSPITAL ALERT ────────────────────────────────────
        st.markdown(f"""
<div class="hospital-box">
    <strong> Alert Nearest Trauma Centre — {nairobi_area}</strong>
    <span>Primary: </span><b>{hospitals['primary']}</b><br>
    <span>Secondary: </span><b>{hospitals['secondary']}</b>
</div>
""", unsafe_allow_html=True)

        # ── CONTRIBUTING RISK FACTORS ─────────────────────────
        # Explains WHY the model produced this classification.
        # HIGH: inputs that elevated severity probability.
        # LOW:  inputs that kept probability below threshold.
        # Contextual: auto-derived time and weather signals.

        with st.expander(" Contributing Risk Factors", expanded=True):
            for factor in risk_factors:
                st.markdown(f"• {factor}")
            st.caption(
                f"Context: {weather_used} · "
                f"{'Night-time' if temporal.get('Is_night') else 'Daytime'} · "
                f"{'Rush hour' if temporal.get('Is_rush_hour') else 'Off-peak'}"
            )

        # ── UPDATE SESSION HISTORY ────────────────────────────
        st.session_state.history.insert(0, {
            'Time'      : now.strftime("%H:%M"),
            'Area'      : nairobi_area,
            'Severity'  : severity,
            'Confidence': f"{confidence}%",
            'Action'    : 'ALS' if is_high else 'BLS'
        })
        st.session_state.history = st.session_state.history[:5]

    else:

        st.markdown("""
<div class="awaiting-box">
    <div style="font-size:2.5rem;margin-bottom:1rem"></div>
    <div style="font-weight:600;color:#94a3b8;font-size:1.1rem">
        Awaiting Incident Report
    </div>
    <p>Fill in details on the left and click <b>Classify Severity</b></p>
</div>
""", unsafe_allow_html=True)

    # ── SESSION LOG ───────────────────────────────────────────
    st.markdown("<div style='margin: 2rem 0 0.5rem 0;'></div>",
                unsafe_allow_html=True)

    with st.expander(" View Recent Classifications Log", expanded=False):
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.markdown(
                '<div class="no-history-text">'
                'No incidents classified in this session yet.'
                '</div>',
                unsafe_allow_html=True
            )