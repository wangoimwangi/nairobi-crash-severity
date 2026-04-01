# ============================================================
# app.py
# Accident Severity Classification System
# Emergency Dispatch Decision Support — Nairobi
# Streamlit web application
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from predictor import predict, get_temporal_features, get_weather_conditions
from hospitals import get_addis_area, get_hospitals

# ---- Page configuration ----
st.set_page_config(
    page_title="Accident Severity Classification",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        background-color: #1a1a2e;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3748;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.6rem;
        margin: 0;
    }
    .main-header p {
        color: #a0aec0;
        margin: 0.2rem 0 0 0;
        font-size: 0.9rem;
    }
    .auto-info {
        background-color: #1a202c;
        border-left: 4px solid #4a90e2;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 1rem;
    }
    .result-high {
        background-color: #2d1515;
        border: 2px solid #fc8181;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-low {
        background-color: #1a2d1a;
        border: 2px solid #68d391;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .severity-text-high {
        color: #fc8181;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .severity-text-low {
        color: #68d391;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .action-text {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        color: #e2e8f0;
    }
    .confidence-text {
        color: #a0aec0;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .hospital-box {
        background-color: #1a2d3d;
        border: 1px solid #2b6cb0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        color: #bee3f8;
    }
</style>
""", unsafe_allow_html=True)

# ---- Initialise session state ----
if 'history' not in st.session_state:
    st.session_state.history = []

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🚨 Accident Severity Classification System</h1>
    <p>Emergency Dispatch Decision Support — Nairobi County</p>
</div>
""", unsafe_allow_html=True)

# ---- Auto-filled info bar ----
temporal      = get_temporal_features()
current_weather = get_weather_conditions()
now           = datetime.now()
time_str      = now.strftime("%H:%M")
day_str       = now.strftime("%A, %d %B %Y")

auto_flags = []
if temporal['Is_night']:
    auto_flags.append("🌙 Night time")
if temporal['Is_rush_hour']:
    auto_flags.append("🚗 Rush hour")
if temporal['Is_weekend']:
    auto_flags.append("📅 Weekend")

weather_emoji = {
    'Raining'    : '🌧️ Raining',
    'Cloudy'     : '☁️ Cloudy',
    'Fog or mist': '🌫️ Foggy',
    'Normal'     : '☀️ Clear'
}.get(current_weather, '☀️ Clear')

auto_flags.append(weather_emoji)
flag_str = " · ".join(auto_flags)

st.markdown(f"""
<div class="auto-info">
    ⏱️ <strong>{time_str}</strong> · {day_str} · {flag_str}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Temporal and weather features auto-filled
</div>
""", unsafe_allow_html=True)

# ---- Main layout ----
col_input, col_result = st.columns([1, 1], gap="large")

# ============================================================
# LEFT COLUMN — Incident Input
# ============================================================
with col_input:
    st.markdown("### 📋 Incident Details")
    st.markdown("*Enter details from the caller report*")

    # ---- Location ----
    st.markdown("**📍 Location**")
    nairobi_area = st.selectbox(
        "Area of Accident",
        options=[
            "CBD", "Westlands", "Upper Hill",
            "Industrial Area", "Langata/Ngong Road",
            "Kasarani/Thika Road", "Eastleigh", "Karen",
            "Embakasi/JKIA", "Mombasa Road",
            "Parklands", "South B/C"
        ],
        help="Select the area where the accident occurred"
    )

    st.markdown("---")

    # ---- Crash Dynamics ----
    st.markdown("**💥 Crash Dynamics**")

    col_a, col_b = st.columns(2)
    with col_a:
        collision_type = st.selectbox(
            "Type of Collision",
            options=[
                "Head-on", "Rear-end", "Rollover",
                "Hit pedestrian", "Side impact", "Other"
            ],
            help="How did the collision happen?"
        )
    with col_b:
        vehicle_type = st.selectbox(
            "Type of Vehicle",
            options=[
                "Car/Saloon", "Matatu/Minibus",
                "Motorcycle/Boda Boda", "Lorry/Truck",
                "Bus", "Pickup/SUV", "Other"
            ],
            help="Primary vehicle involved"
        )

    col_c, col_d = st.columns(2)
    with col_c:
        num_vehicles = st.number_input(
            "Number of Vehicles",
            min_value=1, max_value=20,
            value=2, step=1,
            help="Total vehicles involved"
        )
    with col_d:
        num_casualties = st.number_input(
            "Estimated Casualties",
            min_value=0, max_value=50,
            value=1, step=1,
            help="Injured or deceased persons"
        )

    st.markdown("---")

    # ---- Pedestrian ----
    st.markdown("**🚶 Pedestrian Involvement**")
    pedestrian_involved = st.radio(
        "Is a pedestrian involved?",
        options=["No", "Yes"],
        horizontal=True,
        help="Is anyone lying in the road or struck outside a vehicle?"
    )
    pedestrian_bool = pedestrian_involved == "Yes"

    st.markdown("---")

    # ---- Classify button ----
    classify_clicked = st.button(
        "🔍 CLASSIFY SEVERITY",
        use_container_width=True,
        type="primary"
    )

# ============================================================
# RIGHT COLUMN — Result
# ============================================================
with col_result:
    st.markdown("### 📊 Classification Result")

    if classify_clicked:
        addis_area = get_addis_area(nairobi_area)
        hospitals  = get_hospitals(nairobi_area)

        result = predict(
            area_addis          = addis_area,
            vehicle_type        = vehicle_type,
            collision_type      = collision_type,
            num_vehicles        = int(num_vehicles),
            num_casualties      = int(num_casualties),
            pedestrian_involved = pedestrian_bool
        )

        severity     = result['severity']
        confidence   = result['confidence']
        risk_factors = result['risk_factors']

        # ---- Severity display ----
        if severity == 'HIGH':
            st.markdown(f"""
<div class="result-high">
    <p class="severity-text-high">🔴 HIGH SEVERITY</p>
    <p class="action-text">⚡ Dispatch ALS Immediately</p>
    <p class="confidence-text">Confidence: {confidence}%</p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-low">
    <p class="severity-text-low">🟢 LOW SEVERITY</p>
    <p class="action-text">🚔 Standard Police Response</p>
    <p class="confidence-text">Confidence: {confidence}%</p>
</div>
""", unsafe_allow_html=True)

        # ---- Hospital lookup ----
        st.markdown(f"""
<div class="hospital-box">
    <strong>🏥 Nearest Trauma Centers — {nairobi_area}</strong><br>
    Primary &nbsp;: {hospitals['primary']}<br>
    Secondary : {hospitals['secondary']}
</div>
""", unsafe_allow_html=True)

        # ---- Key risk factors ----
        with st.expander("⚠️ Key Risk Factors (click to expand)"):
            for factor in risk_factors:
                st.markdown(f"• {factor}")
            st.caption(
                "Based on collision type, vehicle mass, "
                "casualties and temporal context"
            )

        # ---- Add to session history ----
        st.session_state.history.insert(0, {
            'Time'      : now.strftime("%H:%M"),
            'Area'      : nairobi_area,
            'Severity'  : severity,
            'Confidence': f"{confidence}%",
            'Action'    : 'ALS' if severity == 'HIGH' else 'Standard'
        })
        st.session_state.history = st.session_state.history[:5]

    else:
        st.info(
            "👈 Fill in the incident details and click "
            "**CLASSIFY SEVERITY** to get a result."
        )

# ============================================================
# RECENT CLASSIFICATIONS
# ============================================================
st.markdown("---")
st.markdown("### 🕐 Recent Classifications (Current Session)")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.caption("No classifications yet this session.")