from tools.location.get_time_data import CITY_TIMEZONES, STATE_TIMEZONES, COUNTRY_TIMEZONES, DEFAULT_TZ

def resolve_timezone(city: str, state: str, country: str) -> str:
    """
    Resolve timezone for a location using a cascading lookup strategy.

    Priority:
    1. City + Country exact match
    2. State + Country exact match
    3. Country fallback
    4. UTC default
    """

    # Try exact city + country match first
    city_key = (city, country)
    if city_key in CITY_TIMEZONES:
        return CITY_TIMEZONES[city_key]

    # Try state + country match
    state_key = (state, country)
    if state_key in STATE_TIMEZONES:
        return STATE_TIMEZONES[state_key]

    # Country-level fallback
    if country in COUNTRY_TIMEZONES:
        return COUNTRY_TIMEZONES[country]

    # Final fallback to UTC
    return DEFAULT_TZ