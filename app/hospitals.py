# ============================================================
# hospitals.py
# Accident Severity Classification System
# Nairobi Area Mapping & Hospital Lookup
# ============================================================

# MAPPING RATIONALE:
# The RTA training dataset uses generic land-use area categories, not geographic place names. 
# The four area categories are: "Office areas", "Residential areas", "Outside rural areas", "Industrial areas", "base/reference category"

# This mapping translates Nairobi-specific area names into these land-use categories, satisfying Functional Requirement iv
# The hospital lookup is a separate layer that uses the original Nairobi area name (not the mapped category) to return the geographically nearest trauma centres for dispatcher use.
# ============================================================


# ---- Nairobi area to model feature category mapping ----
NAIROBI_TO_ADDIS = {
    # Central business districts - commercial land use
    "CBD"                                : "Office areas",
    "Upper Hill"                         : "Office areas",
    "Westlands"                          : "Office areas",
    "Jogoo Road"                         : "Office areas",
    "Waiyaki Way"                        : "Office areas",

    # Major highways - outside rural / peri-urban corridors
    # These roads have highway characteristics despite being within or near the city boundary
    "Mombasa Road"                       : "Outside rural areas",
    "Langata/Ngong Road/Southern Bypass" : "Outside rural areas",
    "Thika Road/Kasarani"                : "Outside rural areas",
    "Limuru Road"                        : "Outside rural areas",
    "Outer Ring Road"                    : "Outside rural areas",
    "Ruiru/Juja"                         : "Outside rural areas",

    # Residential suburbs - low-to-medium density housing
    "Kilimani"                           : "Residential areas",
    "Parklands"                          : "Residential areas",
    "Eastleigh/Jogoo Road"               : "Residential areas",
    "Karen"                              : "Residential areas",
    "Lavington"                          : "Residential areas",
    "South B/C"                          : "Residential areas",
    "Gigiri/Runda"                       : "Residential areas",
    "Dagoretti"                          : "Residential areas",
    "Kibera/Kawangware"                  : "Residential areas",

    # Industrial and logistics zones
    "Industrial Area"                    : "Industrial areas",
    "Embakasi/JKIA"                      : "Industrial areas",

    # Fallback - base reference category
    "Other/Unknown"                      : "Other"
}


# ---- Nairobi hospital lookup ----
# Returns the two nearest trauma centres for a given Nairobi area. This lookup uses the original Nairobi area name  directly.

# Primary   = closest major trauma facility with full emergency and surgical capability
# Secondary = backup facility if primary is at capacity or unreachable due to traffic conditions

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
    """
    Translate a Nairobi area name into the land-use category used in the training dataset.
    This satisfies Functional Requirement iv  the internal mapping layer that ensures model feature space compatibility.
    Defaults to 'Other' (base reference category) if the area is not found in the mapping table.
    """
    return NAIROBI_TO_ADDIS.get(nairobi_area, "Other")


def get_hospitals(nairobi_area):
    """
    Return the nearest trauma centres for a given Nairobi area.
    Nearest trauma centre recommendation based on dispatcher-entered location.
    Returns a dict with 'primary' and 'secondary' hospital names.
    Defaults to KNH and Nairobi Hospital if area not found.
    """
    return HOSPITAL_LOOKUP.get(nairobi_area, {
        "primary"  : "Kenyatta National Hospital",
        "secondary": "Nairobi Hospital"
    })