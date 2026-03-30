from typing import Optional

from tools.location.detect_location import detect_default_location
from tools.location.get_time_data import STATE_TO_COUNTRY

def resolve_location(city: Optional[str], state: Optional[str], country: Optional[str]):
    """
    Normalizes location input. Only uses defaults if NO location info is provided.
    Infers country from state/province if country is not provided.
    """
    # Only use system defaults if EVERYTHING is missing
    if not city and not state and not country:
        return detect_default_location()

    # Clean up inputs
    city_clean = city.strip() if city else None
    state_clean = state.strip() if state else None
    country_clean = country.strip() if country else None

    # If country is missing but we have a state, try to infer it
    if not country_clean and state_clean:
        country_clean = STATE_TO_COUNTRY.get(state_clean)

    return {
        "city": city_clean,
        "state": state_clean,
        "country": country_clean
    }