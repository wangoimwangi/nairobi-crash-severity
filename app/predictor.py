# ============================================================
# predictor.py
# Feature hydration and prediction logic
# Takes 7 dispatcher inputs + auto-fills remaining 21 features
# Passes complete 28-feature vector to trained RF model
# ============================================================

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import joblib
import json
import os

MODEL_PATH    = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_rf.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

THRESHOLD = metadata['optimal_thresholds']['Balanced Random Forest']

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

VEHICLE_MAPPING = {
    'Matatu/Minibus'      : 'Public (> 45 seats)',
    'Car/Saloon'          : 'Automobile',
    'Motorcycle/Boda Boda': 'Motorcycle',
    'Lorry/Truck'         : 'Lorry (41?100Q)',
    'Bus'                 : 'Public (> 45 seats)',
    'Pickup/SUV'          : 'Pick up upto 10Q',
    'Other'               : 'Other'
}

COLLISION_MAPPING = {
    'Head-on'        : 'Collision with roadside-parked vehicles',
    'Rear-end'       : 'Rear-end',
    'Rollover'       : 'Rollover',
    'Hit pedestrian' : 'Collision with pedestrians',
    'Side impact'    : 'Other',
    'Other'          : 'Other'
}

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


def get_weather():
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
        if precip > 0 or code in [51,53,55,61,63,65,80,81,82]:
            return 'Raining'
        elif code in [71,73,75,77]:
            return 'Cloudy'
        elif code in [45,48]:
            return 'Fog or mist'
        else:
            return 'Normal'
    except Exception:
        return 'Normal'


def get_temporal_features():
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


def predict(area_addis, vehicle_type, collision_type,
            num_vehicles, num_casualties,
            pedestrian_involved, cause_of_accident):
    df         = hydrate_features(area_addis, vehicle_type, collision_type,
                                   num_vehicles, num_casualties,
                                   pedestrian_involved, cause_of_accident)
    proba      = model.predict_proba(df)[0][1]
    severity   = 'HIGH' if proba >= THRESHOLD else 'LOW'
    confidence = round(proba * 100, 1)
    risk_factors = []
    if pedestrian_involved:
        risk_factors.append("Pedestrian involved — zero protection")
    if vehicle_type == 'Lorry/Truck':
        risk_factors.append("Heavy vehicle — high mass impact")
    if collision_type == 'Head-on':
        risk_factors.append("Head-on collision — maximum energy transfer")
    if collision_type == 'Rollover':
        risk_factors.append("Rollover — high injury risk")
    if num_casualties >= 3:
        risk_factors.append(f"{num_casualties} casualties — mass casualty event")
    if num_vehicles >= 3:
        risk_factors.append(f"{num_vehicles} vehicles — high energy crash")
    if cause_of_accident == 'Overspeeding':
        risk_factors.append("Overspeeding — high kinetic energy at impact")
    if cause_of_accident == 'Drunk/Impaired driving':
        risk_factors.append("Impaired driver — unpredictable behaviour")
    if cause_of_accident == 'Overtaking':
        risk_factors.append("Overtaking — high head-on collision risk")
    temporal = get_temporal_features()
    if temporal['Is_night']:
        risk_factors.append("Night time — reduced visibility")
    if temporal['Is_rush_hour']:
        risk_factors.append("Rush hour — high traffic density")
    current_weather = get_weather()
    if current_weather == 'Raining':
        risk_factors.append("Raining — reduced road grip and visibility")
    elif current_weather == 'Fog or mist':
        risk_factors.append("Fog or mist — severely reduced visibility")
    risk_factors = risk_factors[:3] if risk_factors else ["Standard risk profile"]
    return {
        'severity'    : severity,
        'confidence'  : confidence,
        'probability' : proba,
        'risk_factors': risk_factors,
        'weather'     : current_weather
    }
