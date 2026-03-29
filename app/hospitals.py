# ============================================================
# Nairobi area to Addis Ababa dataset mapping
# Hospital lookup by Nairobi area
# Geographic mapping layer — UI shows Nairobi context
# Model receives Addis Ababa equivalent values
# ============================================================

# ---- Nairobi area → Addis Ababa dataset mapping ----
# Maps Nairobi districts to equivalent Addis Ababa sub-cities
# based on traffic density and infrastructure similarity

NAIROBI_TO_ADDIS = {
    "CBD"               : "Arada",
    "Westlands"         : "Bole",
    "Upper Hill"        : "Kirkos",
    "Industrial Area"   : "Akaki Kaliti",
    "Langata/Ngong Road": "Nifas Silk-Lafto",
    "Kasarani/Thika Road": "Yeka",
    "Eastleigh"         : "Lideta",
    "Karen"             : "Bole",
    "Embakasi/JKIA"     : "Akaki Kaliti",
    "Mombasa Road"      : "Nifas Silk-Lafto",
    "Parklands"         : "Gulele",
    "South B/C"         : "Nifas Silk-Lafto"
}


# ---- Nairobi hospital lookup ----
# Returns nearest trauma center based on area selected
HOSPITAL_LOOKUP = {
    "CBD"               : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Nairobi Hospital"
    },
    "Westlands"         : {
        "primary"  : "Aga Khan University Hospital",
        "secondary": "MP Shah Hospital"
    },
    "Upper Hill"        : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Industrial Area"   : {
        "primary"  : "Mater Misericordiae Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Langata/Ngong Road": {
        "primary"  : "Nairobi Hospital",
        "secondary": "Karen Hospital"
    },
    "Kasarani/Thika Road": {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Guru Nanak Ramgarhia Hospital"
    },
    "Eastleigh"         : {
        "primary"  : "Mama Lucy Kibaki Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Karen"             : {
        "primary"  : "Karen Hospital",
        "secondary": "Nairobi Hospital"
    },
    "Embakasi/JKIA"     : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Mama Lucy Kibaki Hospital"
    },
    "Mombasa Road"      : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Mater Misericordiae Hospital"
    },
    "Parklands"         : {
        "primary"  : "MP Shah Hospital",
        "secondary": "Aga Khan University Hospital"
    },
    "South B/C"         : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Kenyatta National Hospital"
    }
}


def get_addis_area(nairobi_area):
    """Map Nairobi area to Addis Ababa dataset equivalent."""
    return NAIROBI_TO_ADDIS.get(nairobi_area, "Arada")


def get_hospitals(nairobi_area):
    """Return nearest trauma centers for a Nairobi area."""
    return HOSPITAL_LOOKUP.get(nairobi_area, {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Nairobi Hospital"
    })