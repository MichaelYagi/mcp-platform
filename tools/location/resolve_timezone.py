from tools.location.get_time_data import CITY_TIMEZONES, STATE_TIMEZONES, COUNTRY_TIMEZONES, DEFAULT_TZ

# Province/state abbreviation → full name (Canada, US, Australia, Germany, UK)
# Note: ambiguous codes (WA, NT, SA) default to US/Canada over Australia
_STATE_EXPAND = {
    # Canadian provinces & territories
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
    # US states
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
    # Australian states & territories
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "TAS": "Tasmania",
    "ACT": "Australian Capital Territory",
    # German states
    "BY": "Bavaria",
    "BE": "Berlin",
    "BB": "Brandenburg",
    "HB": "Bremen",
    "HH": "Hamburg",
    "HE": "Hesse",
    "MV": "Mecklenburg-Vorpommern",
    "NI": "Lower Saxony",
    "NW": "North Rhine-Westphalia",
    "RP": "Rhineland-Palatinate",
    "SL": "Saarland",
    "SN": "Saxony",
    "ST": "Saxony-Anhalt",
    "SH": "Schleswig-Holstein",
    "TH": "Thuringia",
    # UK
    "ENG": "England",
    "SCT": "Scotland",
    "WLS": "Wales",
    "NIR": "Northern Ireland",
}


def _title(s: str | None) -> str:
    """Normalize a location string to title case, or empty string if None."""
    return s.strip().title() if s else ""


def resolve_timezone(city: str, state: str, country: str) -> str:
    """
    Resolve timezone for a location using a cascading lookup strategy.

    Priority:
    1. City + Country exact match (case-insensitive via title-case normalization)
    2. State + Country exact match (tries abbreviation expansion)
    3. Country fallback
    4. UTC default
    """
    # Normalize to title case so e.g. "british columbia" matches "British Columbia"
    city_n    = _title(city)
    country_n = _title(country)
    state_n   = state.strip() if state else ""  # keep abbreviations as-is for _STATE_EXPAND

    # Try exact city + country match first
    city_key = (city_n, country_n)
    if city_key in CITY_TIMEZONES:
        return CITY_TIMEZONES[city_key]

    # Try state + country match — also try expanding abbreviations
    state_key = (state_n, country_n)
    if state_key in STATE_TIMEZONES:
        return STATE_TIMEZONES[state_key]

    # Try expanded state name (e.g. "BC" → "British Columbia")
    state_expanded = _STATE_EXPAND.get(state_n.upper(), _title(state_n))
    if state_expanded != state_n:
        expanded_key = (state_expanded, country_n)
        if expanded_key in STATE_TIMEZONES:
            return STATE_TIMEZONES[expanded_key]

    # Country-level fallback
    if country_n in COUNTRY_TIMEZONES:
        return COUNTRY_TIMEZONES[country_n]

    # Final fallback to UTC
    return DEFAULT_TZ