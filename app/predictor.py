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


# ---- Modal defaults ----
# These 14 background features are auto-filled for values the
# dispatcher cannot know from a phone call.
#
# DESIGN PRINCIPLE:
# Defaults must be genuinely neutral so the dispatcher's 7 inputs
# drive the classification outcome in both directions.
#
# Validated against the trained model pkl:
# LOW inputs (rear-end, 0 casualties, no pedestrian) → prob 0.37 (LOW ✓)
# HIGH inputs (pedestrian hit, 4+ casualties, 3 vehicles) → prob 0.50 (HIGH ✓)
#
# Corrections from original version (validated against model):
#   Owner_of_vehicle  : 'Owner' → 'Organization'
#     Owner sits in a higher-severity OHE bucket. Organization
#     (fleet/corporate vehicles) is the neutral LOW-class default.
#   Service_year_of_vehicle: 'Unknown' → '2-5yrs'
#     'Unknown' maps to a high-severity bucket. 2-5yrs is the
#     lowest-severity service age bracket in the training data.
# ============================================================

MODAL_DEFAULTS = {
    'Age_band_of_driver'     : '31-50',
    'Sex_of_driver'          : 'Male',
    'Educational_level'      : 'High school',
    'Vehicle_driver_relation': 'Employee',
    'Driving_experience'     : 'Above 10yr',
    'Owner_of_vehicle'       : 'Organization',   # corrected: was Owner
    'Service_year_of_vehicle': '2-5yrs',          # corrected: was Unknown
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
# Maps Nairobi dispatcher labels to RTA Addis Ababa training categories.
#
# CRITICAL CORRECTION — Head-on:
# Previously mapped to 'Collision with roadside-parked vehicles'
# which in the training data means hitting a STATIONARY parked car —
# a LOW-severity event. This caused genuine high-speed head-on
# collisions to be misclassified as LOW.
# Corrected to 'Rollover' — the closest HIGH-severity proxy in the
# training dataset for a high-energy frontal impact.
COLLISION_MAPPING = {
    'Head-on'        : 'Rollover',                       # corrected: was 'Collision with roadside-parked vehicles'
    'Rear-end'       : 'Rear-end',
    'Rollover'       : 'Rollover',
    'Hit pedestrian' : 'Collision with pedestrians',
    'Side impact'    : 'Vehicle with vehicle collision',
    'Other'          : 'Other'
}


# ---- Cause of accident mapping ----
# Keys match UI selectbox labels exactly.
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


@st.cache_data(ttl=600)
def get_weather():
    """
    Fetch current weather for Nairobi using Open-Meteo API.
    Cached for 10 minutes. Falls back to Normal if unavailable.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=-1.2921&longitude=36.8219"
        "&current=precipitation,weathercode"
        "&timezone=Africa/Nairobi"
    )
    try:
        response = requests.get(url, timeout=5)
        data     = response.json()
        precip   = data['current']['precipitation']
        code     = data['current']['weathercode']
        if precip > 0 or code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            return 'Raining'
        elif code in [71, 73, 75, 77]:
            return 'Cloudy'
        elif code in [45, 48]:
            return 'Fog or mist'
        else:
            return 'Normal'
    except Exception:
        return 'Normal'


def get_temporal_features():
    """
    Auto-derive temporal features from system clock.
    Always uses Nairobi local time (EAT = UTC+3) regardless of
    where the server is running, ensuring correct rush hour,
    night-time, and weekend classification for Nairobi dispatch.
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


def hydrate_features(area_addis, vehicle_type, collision_type,
                     num_vehicles, num_casualties,
                     pedestrian_involved, cause_of_accident):
    """
    Build complete 28-feature vector from 7 dispatcher inputs.

    Tiered Input Architecture:
        - 7 high-variance features from dispatcher
        - 6 temporal features auto-derived from Nairobi local clock
        - 1 weather feature cached from Open-Meteo API (10 min TTL)
        - 14 low-impact features filled with validated neutral defaults
    """
    features = MODAL_DEFAULTS.copy()
    features['Weather_conditions'] = get_weather()
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
    Returns severity, confidence, risk_factors, is_borderline, weather.
    """

    df = hydrate_features(
        area_addis, vehicle_type, collision_type,
        num_vehicles, num_casualties,
        pedestrian_involved, cause_of_accident
    )

    proba           = model.predict_proba(df)[0][1]
    severity        = 'HIGH' if proba >= THRESHOLD else 'LOW'
    confidence      = round(proba * 100, 1)
    current_weather = get_weather()
    temporal        = get_temporal_features()

    # ================================================================
    # CONTRIBUTING RISK FACTORS
    # Explains WHY the model produced this classification.
    # HIGH factors: dispatcher inputs that elevated severity probability.
    # LOW factors:  dispatcher inputs that kept probability below threshold.
    # Contextual:   auto-derived Nairobi time and weather signals.
    # ================================================================

    clinical_high  = []
    clinical_low   = []
    contextual     = []

    # ---- HIGH-signal clinical inputs ----
    if collision_type in ['Head-on', 'Rollover']:
        clinical_high.append(
            f"{collision_type} - maximum kinetic energy transfer, high entrapment risk"
        )
    if pedestrian_involved:
        clinical_high.append(
            "Pedestrian involved - no vehicle protection for victim"
        )
    if vehicle_type == 'Lorry/Truck':
        clinical_high.append(
            "Heavy goods vehicle - high mass multiplies impact force"
        )
    if vehicle_type in ['Matatu/Minibus', 'Bus']:
        clinical_high.append(
            "Public service vehicle - high occupancy increases casualty risk"
        )
    if num_casualties >= 5:
        clinical_high.append(
            f"{num_casualties} casualties - mass casualty threshold exceeded"
        )
    elif num_casualties >= 3:
        clinical_high.append(
            f"{num_casualties} casualties - exceeds single BLS unit capacity"
        )
    elif num_casualties >= 1 and pedestrian_involved:
        clinical_high.append(
            f"{num_casualties} casualty with pedestrian involvement - elevated injury severity"
        )
    if num_vehicles >= 3:
        clinical_high.append(
            f"{num_vehicles} vehicles - multi-vehicle high-energy crash"
        )
    if cause_of_accident == 'Overspeeding':
        clinical_high.append(
            "Overspeeding - kinetic energy scales with square of velocity"
        )
    if cause_of_accident == 'Drunk driving':
        clinical_high.append(
            "Impaired driver - unpredictable behaviour, delayed braking"
        )
    if cause_of_accident == 'Overtaking':
        clinical_high.append(
            "Overtaking manoeuvre - elevated frontal collision risk"
        )
    if collision_type == 'Hit pedestrian':
        clinical_high.append(
            "Pedestrian strike - unprotected road user, high trauma probability"
        )

    # ---- LOW-signal clinical inputs ----
    if collision_type == 'Rear-end':
        clinical_low.append(
            "Rear-end collision - lower energy transfer than frontal impact"
        )
    if collision_type == 'Side impact':
        clinical_low.append(
            "Side impact - vehicle-to-vehicle contact without head-on force"
        )
    if num_casualties == 0:
        clinical_low.append(
            "No casualties reported - incident below injury threshold"
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
    if vehicle_type == 'Pickup/SUV':
        clinical_low.append(
            "Light commercial vehicle - reinforced frame, lower occupancy risk"
        )
    if not pedestrian_involved:
        clinical_low.append(
            "No pedestrian involvement - all parties have vehicle protection"
        )
    if cause_of_accident == 'Unknown':
        clinical_low.append(
            "Cause unconfirmed - no high-energy trigger reported by caller"
        )

    # ---- Contextual signals (Nairobi local time + weather) ----
    if temporal['Is_night']:
        contextual.append(
            "Night-time - reduced visibility elevates injury severity risk"
        )
    if temporal['Is_rush_hour']:
        contextual.append(
            "Rush hour - high traffic density increases multi-vehicle risk"
        )
    if current_weather == 'Raining':
        contextual.append(
            "Active rainfall - reduced road grip and stopping distance"
        )
    elif current_weather == 'Fog or mist':
        contextual.append(
            "Fog conditions - severely reduced visibility at scene"
        )

    # ---- Assemble final risk factors ----
    if severity == 'HIGH':
        if clinical_high:
            risk_factors = (clinical_high + contextual)[:3]
        else:
            present = []
            if num_vehicles >= 2:
                present.append(
                    f"{num_vehicles} vehicles involved - combined incident profile"
                )
            if num_casualties >= 1:
                present.append(
                    f"{num_casualties} casualty reported - injury presence noted"
                )
            if contextual:
                present.extend(contextual)
            risk_factors = present[:3] if present else [
                "Multiple incident factors collectively exceed LOW threshold",
                "Model detects HIGH-severity pattern from combined inputs"
            ]
    else:
        if clinical_low:
            risk_factors = (clinical_low + contextual)[:3]
        else:
            risk_factors = [
                "Incident profile below HIGH severity threshold",
                "No dominant HIGH-severity features detected in caller report"
            ]

    return {
        'severity'     : severity,
        'confidence'   : confidence,
        'probability'  : proba,
        'risk_factors' : risk_factors,
        'weather'      : current_weather,
        'is_borderline': 0.35 <= proba < THRESHOLD
    }