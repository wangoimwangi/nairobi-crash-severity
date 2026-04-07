# ============================================================
# hospitals.py
# Nairobi area to Addis Ababa dataset mapping
# Hospital lookup by Nairobi area
# Geographic mapping layer — UI shows Nairobi context
# Model receives Addis Ababa equivalent values
# ============================================================

# ---- Nairobi area → Addis Ababa dataset mapping ----
# Maps Nairobi districts to equivalent Addis Ababa sub-cities
# based on traffic density and infrastructure similarity

NAIROBI_TO_ADDIS = {
    # Central
    "CBD"                                : "Arada",
    "Upper Hill"                         : "Kirkos",
    "Westlands"                          : "Bole",
    "Parklands"                          : "Gulele",

    # Major Corridors
    "Mombasa Road"                       : "Nifas Silk-Lafto",
    "Langata/Ngong Road/Southern Bypass" : "Nifas Silk-Lafto",
    "Thika Road/Kasarani"                : "Yeka",
    "Waiyaki Way"                        : "Bole",
    "Limuru Road"                        : "Yeka",
    "Outer Ring Road"                    : "Lideta",
    "Jogoo Road"                         : "Lideta",

    # Residential/Commercial
    "Eastleigh/Jogoo Road"               : "Lideta",
    "Karen"                              : "Bole",
    "Kilimani"                           : "Kirkos",
    "Lavington"                          : "Bole",
    "South B/C"                          : "Nifas Silk-Lafto",
    "Gigiri/Runda"                       : "Bole",

    # Industrial/Outer
    "Industrial Area"                    : "Akaki Kaliti",
    "Embakasi/JKIA"                      : "Akaki Kaliti",
    "Ruiru/Juja"                         : "Yeka",
    "Dagoretti"                          : "Nifas Silk-Lafto",
    "Kibera/Kawangware"                  : "Lideta",

    # Fallback
    "Other/Unknown"                      : "Arada"
}

# ---- Nairobi hospital lookup ----
# Returns nearest trauma center based on area selected
HOSPITAL_LOOKUP = {
    # Central
    "CBD"                                : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Nairobi Hospital"
    },
    "Upper Hill"                         : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Westlands"                          : {
        "primary"  : "Aga Khan University Hospital",
        "secondary": "MP Shah Hospital"
    },
    "Parklands"                          : {
        "primary"  : "MP Shah Hospital",
        "secondary": "Aga Khan University Hospital"
    },

    # Major Corridors
    "Mombasa Road"                       : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Mater Misericordiae Hospital"
    },
    "Langata/Ngong Road/Southern Bypass" : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Karen Hospital"
    },
    "Thika Road/Kasarani"                : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Guru Nanak Ramgarhia Sikh Hospital"
    },
    "Waiyaki Way"                        : {
        "primary"  : "Aga Khan University Hospital",
        "secondary": "MP Shah Hospital"
    },
    "Limuru Road"                        : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Guru Nanak Ramgarhia Sikh Hospital"
    },
    "Outer Ring Road"                    : {
        "primary"  : "Mama Lucy Kibaki Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Jogoo Road"                         : {
        "primary"  : "Mama Lucy Kibaki Hospital",
        "secondary": "Kenyatta National Hospital"
    },

    # Residential/Commercial
    "Eastleigh/Jogoo Road"               : {
        "primary"  : "Mama Lucy Kibaki Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Karen"                              : {
        "primary"  : "Karen Hospital",
        "secondary": "Nairobi Hospital"
    },
    "Kilimani"                           : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Aga Khan University Hospital"
    },
    "Lavington"                          : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Aga Khan University Hospital"
    },
    "South B/C"                          : {
        "primary"  : "Nairobi Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Gigiri/Runda"                       : {
        "primary"  : "Aga Khan University Hospital",
        "secondary": "MP Shah Hospital"
    },

    # Industrial/Outer
    "Industrial Area"                    : {
        "primary"  : "Mater Misericordiae Hospital",
        "secondary": "Kenyatta National Hospital"
    },
    "Embakasi/JKIA"                      : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Mama Lucy Kibaki Hospital"
    },
    "Ruiru/Juja"                         : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Guru Nanak Ramgarhia Sikh Hospital"
    },
    "Dagoretti"                          : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Nairobi Hospital"
    },
    "Kibera/Kawangware"                  : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Mater Misericordiae Hospital"
    },

    # Fallback
    "Other/Unknown"                      : {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Nairobi Hospital"
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