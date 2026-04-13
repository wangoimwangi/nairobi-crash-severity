import streamlit as st
import pandas as pd
import os
from datetime import datetime
from predictor import predict, get_temporal_features, get_weather
from hospitals import get_addis_area, get_hospitals

# 1. Page Configuration
st.set_page_config(
    page_title="Accident Severity Classification",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Load CSS
def load_css(path):
    # Add encoding="utf-8" here
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# 3. Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# 4. Live Data & Temporal Logic
temporal = get_temporal_features()
now = datetime.now()
time_str = now.strftime("%H:%M")
day_str = now.strftime("%A, %d %B %Y")
current_weather = get_weather()

weather_icons = {
    'Raining': '🌧️ Rain',
    'Cloudy': '☁️ Cloudy',
    'Fog or mist': '🌫️ Fog',
    'Normal': '☀️ Clear'
}
weather_str = weather_icons.get(current_weather, '☀️ Clear')

flags = []
if temporal.get('Is_night'):     flags.append("🌙 Night")
if temporal.get('Is_rush_hour'): flags.append("🚦 Rush hour")
if temporal.get('Is_weekend'):   flags.append("📅 Weekend")
flag_str = " · ".join(flags) if flags else "Normal conditions"

# 5. UI Header (Optimized for Multi-line)
st.markdown(f"""
<div class="main-header">
        <div class="agency-tag">Incident Triage Unit</div>
        <h1>Accident Severity Classification System</h1>
         <p class="header-subtitle">Emergency Dispatch Decision Support · Nairobi County · ML-Powered Analysis</p>
</div>
""", unsafe_allow_html=True)


# 6. Auto-info Bar (Simplified for left-alignment)
st.markdown(f"""
<div class="auto-info">
    <span>
        <strong>{time_str}</strong> &nbsp;·&nbsp; {day_str} &nbsp;·&nbsp; {weather_str} &nbsp;·&nbsp; {flag_str}
    </span>
</div>
""", unsafe_allow_html=True)

# 7. Main Layout Columns
col_input, col_result = st.columns([1, 1.2], gap="large")

# --- LEFT COLUMN: INPUT ---
with col_input:
    st.subheader("Incident Details")
    st.markdown("*Enter details from the caller report*")

    # Section 1: Location
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
        help="Select the nearest area."
    )
    st.caption("Select the nearest area if exact location is not listed")

    # CUSTOM SPACING (Location > Crash Dynamics)
    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)

    # Section 2: Crash Dynamics
    st.markdown("**Crash Dynamics**")
    col_a, col_b = st.columns(2)
    with col_a:
        collision_type = st.selectbox(
            "Type of Collision",
            options=["Head-on", "Rear-end", "Rollover", "Hit pedestrian", "Side impact", "Other"]
        )
        num_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=20, value=2)

    with col_b:
        vehicle_type = st.selectbox(
            "Type of Vehicle",
            options=["Car/Saloon", "Matatu/Minibus", "Motorcycle/Boda Boda", "Lorry/Truck", "Bus", "Pickup/SUV", "Other"]
        )
        num_casualties = st.number_input("Estimated Casualties", min_value=0, max_value=50, value=1)

    # CUSTOM SPACING (Crash Dynamics > Cause)
    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)

    # Section 3: Cause
    st.markdown("**Primary Cause of Accident**")
    cause_of_accident = st.selectbox(
        "Cause of Accident",
        options=["Unknown", "Overspeeding", "Overtaking", "Changing lanes unsafely", "Drunk driving", "Mechanical failure", "Other"],
        label_visibility="collapsed"
    )

    # CUSTOM SPACING (Cause > Pedestrian)
    st.markdown("<div style='margin: 1.8rem 0;'></div>", unsafe_allow_html=True)

    # Section 4: Pedestrian
    st.markdown("**Pedestrian Involvement**")
    pedestrian_involved = st.radio("Is a pedestrian involved?", options=["No", "Yes"], horizontal=True)
    
    # CUSTOM SPACING (Pedestrian > Button)
    st.markdown("<div style='margin: 1.2rem 0;'></div>", unsafe_allow_html=True)

    classify_clicked = st.button("CLASSIFY SEVERITY", use_container_width=True, type="primary")

# --- RIGHT COLUMN: RESULTS ---
with col_result:
    st.subheader("Classification Result")

    if classify_clicked:
        # 1. Model Prediction Logic
        addis_area = get_addis_area(nairobi_area)
        hospitals  = get_hospitals(nairobi_area)

        result = predict(
            area_addis=addis_area,
            vehicle_type=vehicle_type,
            collision_type=collision_type,
            num_vehicles=int(num_vehicles),
            num_casualties=int(num_casualties),
            pedestrian_involved=(pedestrian_involved == "Yes"),
            cause_of_accident=cause_of_accident
        )


        # 2. Display Severity Result
        if result['severity'] == 'HIGH':
            st.markdown(f"""
            <div class="result-high">
                <p class="severity-text-high">🔴 HIGH SEVERITY</p>
                <p class="action-text"> DISPATCH ADVANCED LIFE SUPPORT (ALS)</p>
                <p class="confidence-text">Model confidence: {result['confidence']}%</p>
                <div class="conf-track"><div class="conf-fill-high" style="width:{result['confidence']}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <p class="severity-text-low">🟢 LOW SEVERITY</p>
                <p class="action-text"> DISPATCH BASIC LIFE SUPPORT (BLS)</p>
                <p class="confidence-text">Model confidence: {result['confidence']}%</p>
                <div class="conf-track"><div class="conf-fill-low" style="width:{result['confidence']}%"></div></div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Hospital Info

        st.markdown(f"""
        <div class="hospital-box">
            <strong>Nearest Trauma Centre — {nairobi_area}</strong>
            <span>Primary: </span><b>{hospitals['primary']}</b><br>
            <span>Secondary: </span><b>{hospitals['secondary']}</b>
        </div>
        """, unsafe_allow_html=True)

        # 4. Risk Factors Expander
        with st.expander("Contributing Risk Factors"):
            for factor in result['risk_factors']:
                st.markdown(f"• {factor}")
            st.caption(f"Environment: {result['weather']} conditions at time of report")

        # 5. Update Session History
        st.session_state.history.insert(0, {
            'Time': now.strftime("%H:%M"),
            'Area': nairobi_area,
            'Severity': result['severity'],
            'Confidence': f"{result['confidence']}%",
            'Action': 'ALS' if result['severity'] == 'HIGH' else 'BLS'
        })
        st.session_state.history = st.session_state.history[:5]

    else:
        st.markdown("""
        <div class="awaiting-box">
            <div style="font-size:2.5rem;margin-bottom:1rem"></div>
            <div style="font-weight:600;color:#94a3b8;font-size:1.1rem">Awaiting Incident Report</div>
            <p>Fill in details on the left and click <b>Classify Severity</b></p>
        </div>
        """, unsafe_allow_html=True)

# --- 8. Footer / History Section (Collapsible) ---
st.markdown("<div style='margin: 3rem 0 1rem 0;'></div>", unsafe_allow_html=True)

# Use an expander to keep the UI simple
with st.expander("View Recent Classifications Log", expanded=False):
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.markdown('<div class="no-history-text">No incidents classified in this session yet.</div>', unsafe_allow_html=True)


