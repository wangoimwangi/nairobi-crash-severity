# ============================================================
# predictor.py
# Feature hydration and prediction logic
# Takes 7 dispatcher inputs + auto-fills remaining 21 features
# Passes complete 28-feature vector to trained RF model
# ============================================================

import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime
from pytz import timezone
import json
import os

# ---- Load model and metadata (cached - loads once only) ----
MODEL_PATH    = os.path.join(os.path.dirname(__file__),
                              '..', 'models', 'best_rf_compressed.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__),
                              '..', 'models', 'model_metadata.json')

@st.cache_resource
def load_model():
    import gzip
    import pickle
    with gzip.open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_metadata():
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

model    = load_model()
metadata = load_metadata()

THRESHOLD = metadata['optimal_thresholds']['Balanced Random Forest']

# Nairobi timezone — East Africa Time (UTC+3)
NAIROBI_TZ = timezone('Africa/Nairobi')


# ---- Per-area coordinates for location-specific weather ----
AREA_COORDINATES = {
    "CBD"                                : (-1.2833, 36.8167),
    "Upper Hill"                         : (-1.2978, 36.8178),
    "Westlands"                          : (-1.2636, 36.8022),
    "Parklands"                          : (-1.2600, 36.8200),
    "Mombasa Road"                       : (-1.3300, 36.8500),
    "Langata/Ngong Road/Southern Bypass" : (-1.3500, 36.7500),
    "Thika Road/Kasarani"                : (-1.2167, 36.8833),
    "Waiyaki Way"                        : (-1.2600, 36.7700),
    "Limuru Road"                        : (-1.2100, 36.7600),
    "Outer Ring Road"                    : (-1.2667, 36.8833),
    "Jogoo Road"                         : (-1.2833, 36.8500),
    "Eastleigh/Jogoo Road"               : (-1.2700, 36.8450),
    "Karen"                              : (-1.3500, 36.7167),
    "Kilimani"                           : (-1.2900, 36.7900),
    "Lavington"                          : (-1.2800, 36.7700),
    "South B/C"                          : (-1.3167, 36.8333),
    "Gigiri/Runda"                       : (-1.2167, 36.7833),
    "Industrial Area"                    : (-1.3100, 36.8400),
    "Embakasi/JKIA"                      : (-1.3192, 36.9275),
    "Ruiru/Juja"                         : (-1.1500, 36.9500),
    "Dagoretti"                          : (-1.3000, 36.7500),
    "Kibera/Kawangware"                  : (-1.3133, 36.7817),
    "Other/Unknown"                      : (-1.2921, 36.8219),
}


# ---- Modal defaults ----
MODAL_DEFAULTS = {
    'Age_band_of_driver'     : '31-50',
    'Sex_of_driver'          : 'Male',
    'Educational_level'      : 'High school',
    'Vehicle_driver_relation': 'Employee',
    'Driving_experience'     : 'Above 10yr',
    'Owner_of_vehicle'       : 'Organization',
    'Service_year_of_vehicle': '2-5yrs',
    'Defect_of_vehicle'      : 'No defect',
    'Lanes_or_Medians'       : 'Double carriageway (median)',
    'Road_allignment'        : 'Tangent',
    'Road_surface_type'      : 'Asphalt roads',
    'Road_surface_conditions': 'Dry',
    'Weather_conditions'     : 'Normal',
    'Vehicle_movement'       : 'Going straight',
    'Pedestrian_movement'    : 'Not a Pedestrian',
    'Cause_of_accident'      : 'No distancing',
    'Types_of_Junction'      : 'Y Shape',
}


# ---- Vehicle type mapping ----
VEHICLE_MAPPING = {
    'Matatu/Minibus'      : 'Public (> 45 seats)',
    'Car/Saloon'          : 'Automobile',
    'Motorcycle/Boda Boda': 'Motorcycle',
    'Lorry/Truck'         : 'Lorry (41?100Q)',
    'Bus'                 : 'Public (> 45 seats)',
    'Pickup/SUV'          : 'Pick up upto 10Q',
    'Other'               : 'Other'
}


# ---- Collision type mapping ----
COLLISION_MAPPING = {
    'Head-on'        : 'Rollover',
    'Rear-end'       : 'Rear-end',
    'Rollover'       : 'Rollover',
    'Hit pedestrian' : 'Collision with pedestrians',
    'Side impact'    : 'Vehicle with vehicle collision',
    'Other'          : 'Other'
}

HIGH_SEVERITY_COLLISIONS = {'Head-on', 'Rollover', 'Hit pedestrian'}

# ---- Cause of accident mapping ----
CAUSE_MAPPING = {
    'Unknown'                    : 'No distancing',
    'Overspeeding'               : 'Overspeed',
    'Overtaking'                 : 'Overtaking',
    'Changing lanes unsafely'    : 'Changing lane to the right',
    'Did not yield to pedestrian': 'No priority to pedestrian',
    'Drunk driving'              : 'Drunk driving',
    'Mechanical failure'         : 'Defect of vehicle',
    'Other'                      : 'Other'
}


def _clinical_override(collision_type, vehicle_type, num_vehicles,
                        num_casualties, cause_of_accident):
    """
    Clinical override rule — applied on top of the ML model score.

    When multiple extreme HIGH-severity signals are simultaneously
    present, the combination is unambiguously HIGH regardless of
    the model probability. This mirrors real-world clinical triage
    practice (Manchester Triage System, SALT triage) where hard
    override criteria exist alongside scoring models.

    Override fires when ALL of the following are true:
      - High-energy collision: Rollover or Head-on
      - Heavy or high-occupancy vehicle: Lorry, Bus, or Matatu
      - Mass casualty threshold: 3+ casualties
      - Multi-vehicle: 3+ vehicles involved
    """
    is_high_energy   = collision_type in {'Rollover', 'Head-on'}
    is_heavy_vehicle = vehicle_type in {'Lorry/Truck', 'Matatu/Minibus', 'Bus'}
    is_mass_casualty = num_casualties >= 3
    is_multi_vehicle = num_vehicles >= 3

    return is_high_energy and is_heavy_vehicle and is_mass_casualty and is_multi_vehicle


@st.cache_data(ttl=300)
def get_weather(nairobi_area: str = "Other/Unknown") -> str:
    """
    Fetch live weather for the specific Nairobi area using Open-Meteo API.
    Requires BOTH precipitation > 0 AND a rain weathercode to report Rain.
    Cached 5 minutes per area. Falls back to Normal if unavailable.
    """
    lat, lon = AREA_COORDINATES.get(nairobi_area, (-1.2921, 36.8219))
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=precipitation,weathercode"
        f"&timezone=Africa/Nairobi"
    )
    try:
        response = requests.get(url, timeout=5)
        data     = response.json()
        precip   = data['current']['precipitation']
        code     = data['current']['weathercode']

        rain_codes  = {51, 53, 55, 61, 63, 65, 80, 81, 82}
        fog_codes   = {45, 48}
        cloud_codes = {71, 73, 75, 77, 3}

        if precip > 0 and code in rain_codes:
            return 'Raining'
        elif code in fog_codes:
            return 'Fog or mist'
        elif code in cloud_codes:
            return 'Cloudy'
        else:
            return 'Normal'
    except Exception:
        return 'Normal'


def get_temporal_features() -> dict:
    """
    Auto-derive temporal features from Nairobi local clock (EAT = UTC+3).
    No caching — always reads current time fresh.
    """
    now          = datetime.now(NAIROBI_TZ)
    hour         = now.hour
    day_of_week  = now.strftime('%A')
    is_night     = 1 if (hour >= 20 or hour <= 5) else 0
    is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
    is_weekend   = 1 if day_of_week in ['Saturday', 'Sunday'] else 0

    if 6 <= hour <= 18:
        light = 'Daylight'
    elif 19 <= hour <= 20:
        light = 'Darkness - lights lit'
    else:
        light = 'Darkness - no lighting'

    return {
        'Day_of_week'     : day_of_week,
        'Hour_of_day'     : hour,
        'Is_night'        : is_night,
        'Is_rush_hour'    : is_rush_hour,
        'Is_weekend'      : is_weekend,
        'Light_conditions': light
    }


def hydrate_features(area_addis, nairobi_area, vehicle_type, collision_type,
                     num_vehicles, num_casualties,
                     pedestrian_involved, cause_of_accident):
    """Build complete 28-feature vector from 7 dispatcher inputs."""
    features = MODAL_DEFAULTS.copy()
    features['Weather_conditions'] = get_weather(nairobi_area)
    temporal = get_temporal_features()
    features.update(temporal)

    features['Area_accident_occured']       = area_addis
    features['Type_of_vehicle']             = VEHICLE_MAPPING.get(vehicle_type, 'Automobile')
    features['Type_of_collision']           = COLLISION_MAPPING.get(collision_type, 'Other')
    features['Number_of_vehicles_involved'] = num_vehicles
    features['Number_of_casualties']        = num_casualties
    features['Cause_of_accident']           = CAUSE_MAPPING.get(cause_of_accident, 'No distancing')

    if pedestrian_involved:
        features['Pedestrian_movement'] = 'Crossing from driver\'s nearside'
    else:
        features['Pedestrian_movement'] = 'Not a Pedestrian'

    column_order = [
        'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
        'Educational_level', 'Vehicle_driver_relation',
        'Driving_experience', 'Type_of_vehicle', 'Owner_of_vehicle',
        'Service_year_of_vehicle', 'Defect_of_vehicle',
        'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment',
        'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions',
        'Light_conditions', 'Weather_conditions', 'Type_of_collision',
        'Number_of_vehicles_involved', 'Number_of_casualties',
        'Vehicle_movement', 'Pedestrian_movement', 'Cause_of_accident',
        'Hour_of_day', 'Is_night', 'Is_rush_hour', 'Is_weekend'
    ]

    df = pd.DataFrame([features])[column_order]
    return df


def predict(area_addis, nairobi_area, vehicle_type, collision_type,
            num_vehicles, num_casualties,
            pedestrian_involved, cause_of_accident):
    """
    Run prediction on 7 dispatcher inputs.

    Classification pipeline:
    1. ML model produces probability score
    2. Clinical override checked — if extreme multi-signal HIGH
       combination present, severity forced to HIGH regardless
       of model probability (rule-based safety net)
    3. Risk factors assembled to match and explain the final
       severity classification — always consistent with result

    Returns severity, confidence, risk_factors, is_borderline, weather.
    """

    df = hydrate_features(
        area_addis, nairobi_area, vehicle_type, collision_type,
        num_vehicles, num_casualties,
        pedestrian_involved, cause_of_accident
    )

    proba           = model.predict_proba(df)[0][1]
    current_weather = get_weather(nairobi_area)
    temporal        = get_temporal_features()

    # ---- Apply clinical override ----
    override = _clinical_override(
        collision_type, vehicle_type,
        num_vehicles, num_casualties, cause_of_accident
    )

    severity   = 'HIGH' if (proba >= THRESHOLD or override) else 'LOW'
    confidence = round(proba * 100, 1)

    # ---- Build clinical factor lists ----
    clinical_high = []
    clinical_low  = []
    contextual    = []

    # HIGH-signal clinical inputs
    if collision_type in HIGH_SEVERITY_COLLISIONS:
        clinical_high.append(
            f"{collision_type} — maximum kinetic energy transfer, high entrapment risk"
        )
    if pedestrian_involved:
        clinical_high.append(
            "Pedestrian involved — unprotected road user, high trauma probability"
        )
    if vehicle_type == 'Lorry/Truck':
        clinical_high.append(
            "Heavy goods vehicle — high mass multiplies impact force"
        )
    if vehicle_type in ['Matatu/Minibus', 'Bus']:
        clinical_high.append(
            "Public service vehicle — high occupancy increases casualty risk"
        )
    if num_casualties >= 5:
        clinical_high.append(
            f"{num_casualties} casualties — mass casualty threshold exceeded"
        )
    elif num_casualties >= 3:
        clinical_high.append(
            f"{num_casualties} casualties — exceeds single BLS unit capacity"
        )
    elif num_casualties >= 1 and pedestrian_involved:
        clinical_high.append(
            f"{num_casualties} casualty with pedestrian involvement — elevated injury severity"
        )
    if num_vehicles >= 3:
        clinical_high.append(
            f"{num_vehicles} vehicles — multi-vehicle high-energy crash"
        )
    if cause_of_accident == 'Overspeeding':
        clinical_high.append(
            "Overspeeding — kinetic energy scales with square of velocity"
        )
    if cause_of_accident == 'Drunk driving':
        clinical_high.append(
            "Impaired driver — unpredictable behaviour, delayed braking"
        )
    if cause_of_accident == 'Overtaking':
        clinical_high.append(
            "Overtaking manoeuvre — elevated frontal collision risk"
        )

    # LOW-signal clinical inputs
    if collision_type == 'Rear-end':
        clinical_low.append(
            "Rear-end collision — lower energy transfer than frontal impact"
        )
    if collision_type == 'Side impact':
        clinical_low.append(
            "Side impact — vehicle-to-vehicle contact without head-on force"
        )
    if num_casualties == 0:
        clinical_low.append(
            "No casualties reported — incident below injury threshold"
        )
    elif num_casualties <= 2 and not pedestrian_involved:
        clinical_low.append(
            f"{num_casualties} casualty - within single BLS unit response capacity"
        )
    if num_vehicles == 1:
        clinical_low.append(
            "Single vehicle - contained incident, no multi-vehicle energy transfer"
        )
    if vehicle_type == 'Car/Saloon':
        clinical_low.append(
            "Passenger car - standard crumple zone and restraint systems present"
        )
    if cause_of_accident == 'Unknown':
        clinical_low.append(
            "Cause unconfirmed - no high-energy trigger reported by caller"
        )
    if not pedestrian_involved:
        clinical_low.append(
            "No pedestrian involvement - all parties have vehicle protection"
        )

    # Contextual signals
    if temporal['Is_night']:
        contextual.append(
            "Night-time - reduced visibility elevates injury severity risk"
        )
    if temporal['Is_rush_hour']:
        contextual.append(
            "Rush hour - high traffic density increases multi-vehicle risk"
        )
    if current_weather == 'Raining':
        contextual.append("Rain - reduced road grip and stopping distance")
    elif current_weather == 'Fog or mist':
        contextual.append("Fog — severely reduced visibility at scene")

    # ---- Assemble final risk factors ----
    is_borderline_high = THRESHOLD <= proba < 0.55

    if severity == 'HIGH':
        if clinical_high:
            if is_borderline_high and clinical_low and not override:
                # Borderline HIGH: show dominant HIGH signal + top
                # mitigating LOW factor for dispatcher awareness
                risk_factors = (clinical_high[:2] + clinical_low[:1] + contextual)[:3]
            else:
                risk_factors = (clinical_high + contextual)[:3]
        else:
            # No single dominant HIGH signal — model or override classified
            # HIGH from combined pattern. Show what IS present specifically.
            present = []
            present.append(
                f"{collision_type} collision — combined incident pattern exceeds threshold"
            )
            present.append(
                f"{vehicle_type} involved — vehicle type contributes to severity profile"
            )
            if contextual:
                present.extend(contextual)
            risk_factors = present[:3]
    else:
        # LOW — only show LOW factors, never HIGH signals
        if clinical_low:
            risk_factors = (clinical_low + contextual)[:3]
        else:
            risk_factors = [
                "No pedestrian involvement — primary LOW-severity indicator",
                "Incident probability below dispatch threshold"
            ]

    # Weather display label
    weather_display = {
        'Raining'    : 'Rain',
        'Cloudy'     : 'Cloudy',
        'Fog or mist': 'Fog',
        'Normal'     : 'Clear'
    }.get(current_weather, 'Clear')

    return {
        'severity'     : severity,
        'confidence'   : confidence,
        'probability'  : proba,
        'risk_factors' : risk_factors,
        'weather'      : weather_display,
        'is_borderline': 0.35 <= proba < THRESHOLD and not override
    }