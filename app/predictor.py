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
import joblib
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


# ---- Modal defaults ----
MODAL_DEFAULTS = {
    'Age_band_of_driver'     : '18-30',
    'Sex_of_driver'          : 'Male',
    'Educational_level'      : 'Junior high school',
    'Vehicle_driver_relation': 'Owner',
    'Driving_experience'     : '5-10yr',
    'Owner_of_vehicle'       : 'Owner',
    'Service_year_of_vehicle': 'Above 10yr',
    'Defect_of_vehicle'      : 'No defect',
    'Lanes_or_Medians'       : 'Undivided Two way',
    'Road_allignment'        : 'Tangent',
    'Road_surface_type'      : 'Asphalt roads',
    'Road_surface_conditions': 'Dry',
    'Weather_conditions'     : 'Normal',
    'Vehicle_movement'       : 'Going straight',
    'Pedestrian_movement'    : 'Not a Pedestrian',
    'Cause_of_accident'      : 'No distancing',
    'Types_of_Junction'      : 'No junction'
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
    'Head-on'        : 'Collision with roadside-parked vehicles',
    'Rear-end'       : 'Rear-end',
    'Rollover'       : 'Rollover',
    'Hit pedestrian' : 'Collision with pedestrians',
    'Side impact'    : 'Other',
    'Other'          : 'Other'
}


# ---- Cause of accident mapping ----
CAUSE_MAPPING = {
    'Unknown'                    : 'No distancing',
    'Overspeeding'               : 'Overspeed',
    'Overtaking'                 : 'Overtaking',
    'Changing lanes unsafely'    : 'Changing lane to the right',
    'Did not yield to pedestrian': 'No priority to pedestrian',
    'Drunk/Impaired driving'     : 'Drunk driving',
    'Mechanical failure'         : 'Defect of vehicle',
    'Other'                      : 'Other'
}


@st.cache_data(ttl=600)
def get_weather():
    """
Fetch current weather for Nairobi using Open-Meteo API. Cached for 10 minutes - not called on every click.
No API key required. Falls back to Normal if API unavailable.
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
    """Auto-derive temporal features from system clock."""
    now          = datetime.now()
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
    - 6 temporal features auto-derived from system clock
    - 1 weather feature cached from Open-Meteo API (10 min TTL)
    - 14 low-impact features filled with modal defaults
    """

    # ---- Start with modal defaults ----
    features = MODAL_DEFAULTS.copy()

    # ---- Auto-retrieve weather (cached — fast) ----
    features['Weather_conditions'] = get_weather()

    # ---- Auto-fill temporal features ----
    temporal = get_temporal_features()
    features.update(temporal)

    # ---- Apply dispatcher inputs ----
    features['Area_accident_occured']       = area_addis
    features['Type_of_vehicle']             = VEHICLE_MAPPING.get(
        vehicle_type, 'Automobile'
    )
    features['Type_of_collision']           = COLLISION_MAPPING.get(
        collision_type, 'Other'
    )
    features['Number_of_vehicles_involved'] = num_vehicles
    features['Number_of_casualties']        = num_casualties
    features['Cause_of_accident']           = CAUSE_MAPPING.get(
        cause_of_accident, 'No distancing'
    )

    # ---- Pedestrian movement ----
    if pedestrian_involved:
        features['Pedestrian_movement'] = 'Crossing from driver\'s nearside'
    else:
        features['Pedestrian_movement'] = 'Not a Pedestrian'

    # ---- Build dataframe in correct column order ----
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
Run prediction on 7 dispatcher inputs. Returns severity, confidence, and top risk factors.
    """

    # ---- Build feature vector ----
    df = hydrate_features(
        area_addis, vehicle_type, collision_type,
        num_vehicles, num_casualties,
        pedestrian_involved, cause_of_accident
    )

    # ---- Get probability from model ----
    proba      = model.predict_proba(df)[0][1]
    severity   = 'HIGH' if proba >= THRESHOLD else 'LOW'
    confidence = round(proba * 100, 1)
    current_weather = get_weather()
    temporal        = get_temporal_features()

# ---- Identify which factors are driving the classification ----
# Risk factors are split into two tiers:
# - Clinical factors: directly reported by dispatcher (high signal)
# - Contextual factors: auto-derived from time and weather

    clinical_factors    = []
    contextual_factors  = []

    # Clinical inputs — directly reported by dispatcher
    if collision_type == 'Head-on':
        clinical_factors.append(
            "Head-on collision - maximum energy transfer"
        )
    if collision_type == 'Rollover':
        clinical_factors.append(
            "Rollover - high injury and entrapment risk"
        )
    if pedestrian_involved:
        clinical_factors.append(
            "Pedestrian involved - zero vehicle protection"
        )
    if vehicle_type == 'Lorry/Truck':
        clinical_factors.append(
            "Heavy vehicle - high mass impact force"
        )
    if num_casualties >= 3:
        clinical_factors.append(
            f"{num_casualties} casualties - mass casualty event"
        )
    if num_vehicles >= 3:
        clinical_factors.append(
            f"{num_vehicles} vehicles - high energy crash"
        )
    if cause_of_accident == 'Overspeeding':
        clinical_factors.append(
            "Overspeeding - high kinetic energy at impact"
        )
    if cause_of_accident == 'Drunk/Impaired driving':
        clinical_factors.append(
            "Impaired driver - unpredictable behaviour"
        )
    if cause_of_accident == 'Overtaking':
        clinical_factors.append(
            "Overtaking - elevated head-on collision risk"
        )

    # Contextual inputs - auto-derived from system clock and weather API
    if temporal['Is_night']:
        contextual_factors.append("Night time - reduced visibility")
    if temporal['Is_rush_hour']:
        contextual_factors.append("Rush hour - high traffic density")
    if current_weather == 'Raining':
        contextual_factors.append(
            "Raining - reduced road grip and visibility"
        )
    elif current_weather == 'Fog or mist':
        contextual_factors.append(
            "Fog or mist - severely reduced visibility"
        )

    if severity == 'HIGH':
        # Prioritise clinical factors first, then contextual.
        if clinical_factors:
            risk_factors = clinical_factors + contextual_factors
        else:
            risk_factors = [
                f"Area risk pattern - {nairobi_area} corridor has "
                f"elevated historical severity in training data",
            ] + contextual_factors
            if not contextual_factors:
                risk_factors.append(
                    "No single dominant input factor - model uses "
                    "combined area, time and incident pattern"
                )

    else:
# LOW classification
# Borderline cases (proba 30-40%) get escalation-aware notes so the dispatcher remains alert despite the LOW result.
 # Confident LOW cases get factors explaining the result.
        if proba >= 0.30:
            risk_factors = [
                "Borderline case - monitor closely for escalation",
                "Reassess if caller reports additional casualties",
                "Upgrade to ALS if patient condition deteriorates"
            ]
        else:
            risk_factors = []
            if collision_type in ['Rear-end', 'Side impact']:
                risk_factors.append(
                    f"{collision_type} - typically lower severity impact"
                )
            if num_casualties == 0:
                risk_factors.append(
                    "No casualties reported - BLS response appropriate"
                )
            elif num_casualties <= 2 and not pedestrian_involved:
                risk_factors.append(
                    "Low casualty count - BLS response appropriate"
                )
            if vehicle_type == 'Car/Saloon':
                risk_factors.append(
                    "Passenger vehicle - standard safety profile"
                )
            if temporal['Is_rush_hour']:
                risk_factors.append(
                    "Rush hour - ensure clear route for response unit"
                )
            if current_weather == 'Raining':
                risk_factors.append(
                    "Raining - allow extra response travel time"
                )
            if not risk_factors:
                risk_factors.append(
                    "No elevated risk factors - standard BLS protocol"
                )

# Return top 3 most relevant factors only.
# Limiting to 3 keeps the display scannable for a dispatcher under time pressure during an emergency call.
    risk_factors = risk_factors[:3] if risk_factors else [
        "Standard risk profile - no elevated factors detected"
    ]

    return {
        'severity'    : severity,
        'confidence'  : confidence,
        'probability' : proba,
        'risk_factors': risk_factors,
        'weather'     : current_weather
    }