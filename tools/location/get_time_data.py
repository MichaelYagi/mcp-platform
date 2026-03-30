# Map timezone → location (city, state/province/prefecture, country)
TZ_TO_LOCATION = {
    "America/Vancouver": {
        "city": "Surrey",
        "state": "British Columbia",
        "country": "Canada"
    },
    "America/Toronto": {
        "city": "Toronto",
        "state": "Ontario",
        "country": "Canada"
    },
    "America/New_York": {
        "city": "New York",
        "state": "New York",
        "country": "USA"
    },
    "Europe/London": {
        "city": "London",
        "state": "England",
        "country": "UK"
    },
    "Asia/Tokyo": {
        "city": "Tokyo",
        "state": "Tokyo Prefecture",
        "country": "Japan"
    },
}

DEFAULT_FALLBACK = {
    "city": "Surrey",
    "state": "British Columbia",
    "country": "Canada"
}

CITY_TIMEZONES = {
    # Canada
    ("Surrey", "Canada"): "America/Vancouver",
    ("Vancouver", "Canada"): "America/Vancouver",
    ("Burnaby", "Canada"): "America/Vancouver",
    ("Richmond", "Canada"): "America/Vancouver",
    ("Calgary", "Canada"): "America/Edmonton",
    ("Edmonton", "Canada"): "America/Edmonton",
    ("Winnipeg", "Canada"): "America/Winnipeg",
    ("Toronto", "Canada"): "America/Toronto",
    ("Ottawa", "Canada"): "America/Toronto",
    ("Montreal", "Canada"): "America/Toronto",
    ("Quebec City", "Canada"): "America/Toronto",

    # USA
    ("New York", "USA"): "America/New_York",
    ("Boston", "USA"): "America/New_York",
    ("Washington", "USA"): "America/New_York",
    ("Miami", "USA"): "America/New_York",
    ("Chicago", "USA"): "America/Chicago",
    ("Houston", "USA"): "America/Chicago",
    ("Dallas", "USA"): "America/Chicago",
    ("Denver", "USA"): "America/Denver",
    ("Phoenix", "USA"): "America/Phoenix",
    ("Los Angeles", "USA"): "America/Los_Angeles",
    ("San Francisco", "USA"): "America/Los_Angeles",
    ("Seattle", "USA"): "America/Los_Angeles",

    # UK
    ("London", "UK"): "Europe/London",
    ("Manchester", "UK"): "Europe/London",
    ("Birmingham", "UK"): "Europe/London",
    ("Edinburgh", "UK"): "Europe/London",

    # Europe
    ("Paris", "France"): "Europe/Paris",
    ("Marseille", "France"): "Europe/Paris",
    ("Berlin", "Germany"): "Europe/Berlin",
    ("Munich", "Germany"): "Europe/Berlin",
    ("Rome", "Italy"): "Europe/Rome",
    ("Milan", "Italy"): "Europe/Rome",
    ("Madrid", "Spain"): "Europe/Madrid",
    ("Barcelona", "Spain"): "Europe/Madrid",
    ("Amsterdam", "Netherlands"): "Europe/Amsterdam",
    ("Brussels", "Belgium"): "Europe/Brussels",
    ("Zurich", "Switzerland"): "Europe/Zurich",
    ("Vienna", "Austria"): "Europe/Vienna",
    ("Stockholm", "Sweden"): "Europe/Stockholm",
    ("Oslo", "Norway"): "Europe/Oslo",
    ("Copenhagen", "Denmark"): "Europe/Copenhagen",
    ("Warsaw", "Poland"): "Europe/Warsaw",
    ("Prague", "Czech Republic"): "Europe/Prague",
    ("Lisbon", "Portugal"): "Europe/Lisbon",
    ("Athens", "Greece"): "Europe/Athens",

    # Japan
    ("Tokyo", "Japan"): "Asia/Tokyo",
    ("Yokohama", "Japan"): "Asia/Tokyo",
    ("Osaka", "Japan"): "Asia/Tokyo",
    ("Nagoya", "Japan"): "Asia/Tokyo",
    ("Sapporo", "Japan"): "Asia/Tokyo",
    ("Fukuoka", "Japan"): "Asia/Tokyo",
    ("Chigasaki", "Japan"): "Asia/Tokyo",

    # East Asia
    ("Seoul", "South Korea"): "Asia/Seoul",
    ("Busan", "South Korea"): "Asia/Seoul",
    ("Shanghai", "China"): "Asia/Shanghai",
    ("Beijing", "China"): "Asia/Shanghai",
    ("Shenzhen", "China"): "Asia/Shanghai",
    ("Taipei", "Taiwan"): "Asia/Taipei",
    ("Hong Kong", "Hong Kong"): "Asia/Hong_Kong",

    # Southeast Asia
    ("Singapore", "Singapore"): "Asia/Singapore",
    ("Kuala Lumpur", "Malaysia"): "Asia/Kuala_Lumpur",
    ("Bangkok", "Thailand"): "Asia/Bangkok",
    ("Ho Chi Minh City", "Vietnam"): "Asia/Ho_Chi_Minh",
    ("Hanoi", "Vietnam"): "Asia/Bangkok",
    ("Manila", "Philippines"): "Asia/Manila",
    ("Jakarta", "Indonesia"): "Asia/Jakarta",

    # South Asia
    ("Delhi", "India"): "Asia/Kolkata",
    ("Mumbai", "India"): "Asia/Kolkata",
    ("Bangalore", "India"): "Asia/Kolkata",
    ("Karachi", "Pakistan"): "Asia/Karachi",
    ("Dhaka", "Bangladesh"): "Asia/Dhaka",

    # Middle East
    ("Dubai", "UAE"): "Asia/Dubai",
    ("Abu Dhabi", "UAE"): "Asia/Dubai",
    ("Riyadh", "Saudi Arabia"): "Asia/Riyadh",
    ("Doha", "Qatar"): "Asia/Qatar",
    ("Kuwait City", "Kuwait"): "Asia/Kuwait",
    ("Jerusalem", "Israel"): "Asia/Jerusalem",

    # Africa
    ("Johannesburg", "South Africa"): "Africa/Johannesburg",
    ("Cape Town", "South Africa"): "Africa/Johannesburg",
    ("Cairo", "Egypt"): "Africa/Cairo",
    ("Lagos", "Nigeria"): "Africa/Lagos",
    ("Nairobi", "Kenya"): "Africa/Nairobi",
    ("Casablanca", "Morocco"): "Africa/Casablanca",

    # Oceania
    ("Sydney", "Australia"): "Australia/Sydney",
    ("Melbourne", "Australia"): "Australia/Melbourne",
    ("Brisbane", "Australia"): "Australia/Brisbane",
    ("Perth", "Australia"): "Australia/Perth",
    ("Auckland", "New Zealand"): "Pacific/Auckland",
    ("Wellington", "New Zealand"): "Pacific/Auckland",
}

STATE_TIMEZONES = {
    # Canada (Provinces & Territories)
    ("British Columbia", "Canada"): "America/Vancouver",
    ("Alberta", "Canada"): "America/Edmonton",
    ("Saskatchewan", "Canada"): "America/Regina",
    ("Manitoba", "Canada"): "America/Winnipeg",
    ("Ontario", "Canada"): "America/Toronto",
    ("Quebec", "Canada"): "America/Toronto",
    ("New Brunswick", "Canada"): "America/Moncton",
    ("Nova Scotia", "Canada"): "America/Halifax",
    ("Prince Edward Island", "Canada"): "America/Halifax",
    ("Newfoundland and Labrador", "Canada"): "America/St_Johns",
    ("Yukon", "Canada"): "America/Whitehorse",
    ("Northwest Territories", "Canada"): "America/Yellowknife",
    ("Nunavut", "Canada"): "America/Iqaluit",

    # USA (States)
    ("New York", "USA"): "America/New_York",
    ("Massachusetts", "USA"): "America/New_York",
    ("Florida", "USA"): "America/New_York",
    ("District of Columbia", "USA"): "America/New_York",
    ("Illinois", "USA"): "America/Chicago",
    ("Texas", "USA"): "America/Chicago",
    ("Colorado", "USA"): "America/Denver",
    ("Arizona", "USA"): "America/Phoenix",
    ("California", "USA"): "America/Los_Angeles",
    ("Washington", "USA"): "America/Los_Angeles",
    ("Oregon", "USA"): "America/Los_Angeles",

    # UK (Countries)
    ("England", "UK"): "Europe/London",
    ("Scotland", "UK"): "Europe/London",
    ("Wales", "UK"): "Europe/London",
    ("Northern Ireland", "UK"): "Europe/London",

    # Australia (States & Territories)
    ("New South Wales", "Australia"): "Australia/Sydney",
    ("Victoria", "Australia"): "Australia/Melbourne",
    ("Queensland", "Australia"): "Australia/Brisbane",
    ("Western Australia", "Australia"): "Australia/Perth",
    ("South Australia", "Australia"): "Australia/Adelaide",
    ("Tasmania", "Australia"): "Australia/Hobart",
    ("Northern Territory", "Australia"): "Australia/Darwin",
    ("Australian Capital Territory", "Australia"): "Australia/Sydney",

    # India (States share one timezone)
    ("Maharashtra", "India"): "Asia/Kolkata",
    ("Karnataka", "India"): "Asia/Kolkata",
    ("Delhi", "India"): "Asia/Kolkata",
    ("Tamil Nadu", "India"): "Asia/Kolkata",
    ("West Bengal", "India"): "Asia/Kolkata",

    # Japan (Prefectures share one timezone)
    ("Tokyo", "Japan"): "Asia/Tokyo",
    ("Kanagawa", "Japan"): "Asia/Tokyo",
    ("Osaka", "Japan"): "Asia/Tokyo",
    ("Hokkaido", "Japan"): "Asia/Tokyo",
    ("Aichi", "Japan"): "Asia/Tokyo",
    ("Fukuoka", "Japan"): "Asia/Tokyo",

    # China (Provinces share one timezone)
    ("Guangdong", "China"): "Asia/Shanghai",
    ("Beijing", "China"): "Asia/Shanghai",
    ("Shanghai", "China"): "Asia/Shanghai",
    ("Zhejiang", "China"): "Asia/Shanghai",
    ("Jiangsu", "China"): "Asia/Shanghai",

    # Middle East (Regions share country timezone)
    ("Dubai", "UAE"): "Asia/Dubai",
    ("Abu Dhabi", "UAE"): "Asia/Dubai",
    ("Riyadh Province", "Saudi Arabia"): "Asia/Riyadh",
    ("Doha Municipality", "Qatar"): "Asia/Qatar",
    ("Kuwait", "Kuwait"): "Asia/Kuwait",
    ("Jerusalem District", "Israel"): "Asia/Jerusalem",

    # Africa (Regions share country timezone)
    ("Gauteng", "South Africa"): "Africa/Johannesburg",
    ("Western Cape", "South Africa"): "Africa/Johannesburg",
    ("Cairo Governorate", "Egypt"): "Africa/Cairo",
    ("Lagos State", "Nigeria"): "Africa/Lagos",
    ("Nairobi County", "Kenya"): "Africa/Nairobi",
    ("Casablanca-Settat", "Morocco"): "Africa/Casablanca",

    # Southeast Asia (Regions share country timezone)
    ("Bangkok", "Thailand"): "Asia/Bangkok",
    ("Ho Chi Minh", "Vietnam"): "Asia/Ho_Chi_Minh",
    ("Hanoi", "Vietnam"): "Asia/Bangkok",
    ("Metro Manila", "Philippines"): "Asia/Manila",
    ("Jakarta", "Indonesia"): "Asia/Jakarta",
    ("Kuala Lumpur", "Malaysia"): "Asia/Kuala_Lumpur",
}

COUNTRY_TIMEZONES = {
    # North America
    "Canada": "America/Toronto",          # default; west coast cities override via CITY_TIMEZONES
    "USA": "America/New_York",
    "Mexico": "America/Mexico_City",

    # South America
    "Brazil": "America/Sao_Paulo",
    "Argentina": "America/Argentina/Buenos_Aires",
    "Chile": "America/Santiago",
    "Colombia": "America/Bogota",
    "Peru": "America/Lima",

    # Europe
    "UK": "Europe/London",
    "Ireland": "Europe/Dublin",
    "France": "Europe/Paris",
    "Germany": "Europe/Berlin",
    "Spain": "Europe/Madrid",
    "Italy": "Europe/Rome",
    "Netherlands": "Europe/Amsterdam",
    "Belgium": "Europe/Brussels",
    "Sweden": "Europe/Stockholm",
    "Norway": "Europe/Oslo",
    "Finland": "Europe/Helsinki",
    "Denmark": "Europe/Copenhagen",
    "Poland": "Europe/Warsaw",
    "Czech Republic": "Europe/Prague",
    "Austria": "Europe/Vienna",
    "Switzerland": "Europe/Zurich",
    "Portugal": "Europe/Lisbon",
    "Greece": "Europe/Athens",
    "Turkey": "Europe/Istanbul",

    # Middle East
    "Israel": "Asia/Jerusalem",
    "Saudi Arabia": "Asia/Riyadh",
    "UAE": "Asia/Dubai",
    "Qatar": "Asia/Qatar",
    "Kuwait": "Asia/Kuwait",

    # Africa
    "South Africa": "Africa/Johannesburg",
    "Egypt": "Africa/Cairo",
    "Nigeria": "Africa/Lagos",
    "Kenya": "Africa/Nairobi",
    "Morocco": "Africa/Casablanca",

    # Asia
    "Japan": "Asia/Tokyo",
    "South Korea": "Asia/Seoul",
    "China": "Asia/Shanghai",
    "Taiwan": "Asia/Taipei",
    "Hong Kong": "Asia/Hong_Kong",
    "Singapore": "Asia/Singapore",
    "Malaysia": "Asia/Kuala_Lumpur",
    "Thailand": "Asia/Bangkok",
    "Vietnam": "Asia/Ho_Chi_Minh",
    "Philippines": "Asia/Manila",
    "India": "Asia/Kolkata",
    "Pakistan": "Asia/Karachi",
    "Bangladesh": "Asia/Dhaka",
    "Indonesia": "Asia/Jakarta",

    # Oceania
    "Australia": "Australia/Sydney",
    "New Zealand": "Pacific/Auckland",
}

# Map of state/province/prefecture abbreviations and names to countries
STATE_TO_COUNTRY = {
    # ==================== CANADA ====================
    # Canadian provinces (abbreviations)
    "BC": "Canada", "AB": "Canada", "SK": "Canada", "MB": "Canada",
    "ON": "Canada", "QC": "Canada", "NB": "Canada", "NS": "Canada",
    "PE": "Canada", "NL": "Canada", "YT": "Canada", "NT": "Canada", "NU": "Canada",

    # Canadian provinces (full names)
    "British Columbia": "Canada", "Alberta": "Canada", "Saskatchewan": "Canada",
    "Manitoba": "Canada", "Ontario": "Canada", "Quebec": "Canada",
    "New Brunswick": "Canada", "Nova Scotia": "Canada", "Prince Edward Island": "Canada",
    "Newfoundland and Labrador": "Canada", "Yukon": "Canada",
    "Northwest Territories": "Canada", "Nunavut": "Canada",

    # ==================== USA ====================
    # US states (abbreviations)
    "AL": "USA", "AK": "USA", "AZ": "USA", "AR": "USA", "CA": "USA",
    "CO": "USA", "CT": "USA", "DE": "USA", "FL": "USA", "GA": "USA",
    "HI": "USA", "ID": "USA", "IL": "USA", "IN": "USA", "IA": "USA",
    "KS": "USA", "KY": "USA", "LA": "USA", "ME": "USA", "MD": "USA",
    "MA": "USA", "MI": "USA", "MN": "USA", "MS": "USA", "MO": "USA",
    "MT": "USA", "NE": "USA", "NV": "USA", "NH": "USA", "NJ": "USA",
    "NM": "USA", "NY": "USA", "NC": "USA", "ND": "USA", "OH": "USA",
    "OK": "USA", "OR": "USA", "PA": "USA", "RI": "USA", "SC": "USA",
    "SD": "USA", "TN": "USA", "TX": "USA", "UT": "USA", "VT": "USA",
    "VA": "USA", "WA": "USA", "WV": "USA", "WI": "USA", "WY": "USA", "DC": "USA",

    # US states (full names - all 50)
    "Alabama": "USA", "Alaska": "USA", "Arizona": "USA", "Arkansas": "USA",
    "California": "USA", "Colorado": "USA", "Connecticut": "USA", "Delaware": "USA",
    "Florida": "USA", "Georgia": "USA", "Hawaii": "USA", "Idaho": "USA",
    "Illinois": "USA", "Indiana": "USA", "Iowa": "USA", "Kansas": "USA",
    "Kentucky": "USA", "Louisiana": "USA", "Maine": "USA", "Maryland": "USA",
    "Massachusetts": "USA", "Michigan": "USA", "Minnesota": "USA", "Mississippi": "USA",
    "Missouri": "USA", "Montana": "USA", "Nebraska": "USA", "Nevada": "USA",
    "New Hampshire": "USA", "New Jersey": "USA", "New Mexico": "USA", "New York": "USA",
    "North Carolina": "USA", "North Dakota": "USA", "Ohio": "USA", "Oklahoma": "USA",
    "Oregon": "USA", "Pennsylvania": "USA", "Rhode Island": "USA", "South Carolina": "USA",
    "South Dakota": "USA", "Tennessee": "USA", "Texas": "USA", "Utah": "USA",
    "Vermont": "USA", "Virginia": "USA", "Washington": "USA", "West Virginia": "USA",
    "Wisconsin": "USA", "Wyoming": "USA", "District of Columbia": "USA",

    # ==================== AUSTRALIA ====================
    "NSW": "Australia", "VIC": "Australia", "QLD": "Australia", "WA": "Australia",
    "SA": "Australia", "TAS": "Australia", "NT": "Australia", "ACT": "Australia",
    "New South Wales": "Australia", "Victoria": "Australia", "Queensland": "Australia",
    "Western Australia": "Australia", "South Australia": "Australia", "Tasmania": "Australia",
    "Northern Territory": "Australia", "Australian Capital Territory": "Australia",

    # ==================== JAPAN ====================
    # Major prefectures
    "Tokyo": "Japan", "Osaka": "Japan", "Kanagawa": "Japan", "Aichi": "Japan",
    "Hokkaido": "Japan", "Fukuoka": "Japan", "Saitama": "Japan", "Chiba": "Japan",
    "Hyogo": "Japan", "Kyoto": "Japan", "Shizuoka": "Japan", "Hiroshima": "Japan",
    "Ibaraki": "Japan", "Miyagi": "Japan", "Niigata": "Japan", "Nagano": "Japan",
    "Gifu": "Japan", "Tochigi": "Japan", "Gunma": "Japan", "Okayama": "Japan",
    "Mie": "Japan", "Kumamoto": "Japan", "Kagoshima": "Japan", "Okinawa": "Japan",
    "Shiga": "Japan", "Nara": "Japan", "Yamaguchi": "Japan", "Ehime": "Japan",

    # ==================== CHINA ====================
    # Major provinces and municipalities
    "Beijing": "China", "Shanghai": "China", "Guangdong": "China", "Zhejiang": "China",
    "Jiangsu": "China", "Sichuan": "China", "Henan": "China", "Hubei": "China",
    "Hunan": "China", "Shandong": "China", "Anhui": "China", "Hebei": "China",
    "Liaoning": "China", "Shaanxi": "China", "Jiangxi": "China", "Fujian": "China",
    "Yunnan": "China", "Shanxi": "China", "Heilongjiang": "China", "Guangxi": "China",
    "Tianjin": "China", "Chongqing": "China", "Xinjiang": "China", "Inner Mongolia": "China",

    # ==================== INDIA ====================
    # Major states
    "Maharashtra": "India", "Karnataka": "India", "Tamil Nadu": "India",
    "West Bengal": "India", "Uttar Pradesh": "India", "Gujarat": "India",
    "Rajasthan": "India", "Kerala": "India", "Andhra Pradesh": "India",
    "Telangana": "India", "Bihar": "India", "Madhya Pradesh": "India",
    "Delhi": "India", "Haryana": "India", "Punjab": "India", "Assam": "India",
    "Odisha": "India", "Jharkhand": "India", "Chhattisgarh": "India",

    # ==================== UK ====================
    "England": "UK", "Scotland": "UK", "Wales": "UK", "Northern Ireland": "UK",

    # ==================== GERMANY ====================
    # Major states (Länder)
    "Bavaria": "Germany", "Baden-Württemberg": "Germany", "North Rhine-Westphalia": "Germany",
    "Hesse": "Germany", "Saxony": "Germany", "Lower Saxony": "Germany",
    "Rhineland-Palatinate": "Germany", "Berlin": "Germany", "Hamburg": "Germany",
    "Schleswig-Holstein": "Germany", "Brandenburg": "Germany", "Saxony-Anhalt": "Germany",
    "Thuringia": "Germany", "Mecklenburg-Vorpommern": "Germany", "Bremen": "Germany",
    "Saarland": "Germany",

    # ==================== FRANCE ====================
    # Major regions
    "Île-de-France": "France", "Provence-Alpes-Côte d'Azur": "France",
    "Auvergne-Rhône-Alpes": "France", "Nouvelle-Aquitaine": "France",
    "Occitanie": "France", "Hauts-de-France": "France", "Brittany": "France",
    "Normandy": "France", "Grand Est": "France", "Pays de la Loire": "France",
    "Bourgogne-Franche-Comté": "France", "Centre-Val de Loire": "France",

    # ==================== ITALY ====================
    # Major regions
    "Lombardy": "Italy", "Lazio": "Italy", "Campania": "Italy", "Sicily": "Italy",
    "Veneto": "Italy", "Piedmont": "Italy", "Emilia-Romagna": "Italy",
    "Tuscany": "Italy", "Apulia": "Italy", "Calabria": "Italy",

    # ==================== SPAIN ====================
    # Autonomous communities
    "Andalusia": "Spain", "Catalonia": "Spain", "Madrid": "Spain",
    "Valencia": "Spain", "Galicia": "Spain", "Basque Country": "Spain",
    "Castile and León": "Spain", "Canary Islands": "Spain", "Aragon": "Spain",

    # ==================== BRAZIL ====================
    # Major states
    "São Paulo": "Brazil", "Rio de Janeiro": "Brazil", "Minas Gerais": "Brazil",
    "Bahia": "Brazil", "Rio Grande do Sul": "Brazil", "Paraná": "Brazil",
    "Pernambuco": "Brazil", "Ceará": "Brazil", "Santa Catarina": "Brazil",

    # ==================== MEXICO ====================
    # Major states
    "Mexico City": "Mexico", "Jalisco": "Mexico", "Nuevo León": "Mexico",
    "Veracruz": "Mexico", "Puebla": "Mexico", "Guanajuato": "Mexico",
    "Chiapas": "Mexico", "Estado de México": "Mexico",

    # ==================== SOUTH KOREA ====================
    "Seoul": "South Korea", "Busan": "South Korea", "Incheon": "South Korea",
    "Daegu": "South Korea", "Daejeon": "South Korea", "Gwangju": "South Korea",
    "Gyeonggi": "South Korea", "Gangwon": "South Korea", "Jeju": "South Korea",

    # ==================== SOUTH AFRICA ====================
    "Gauteng": "South Africa", "Western Cape": "South Africa", "KwaZulu-Natal": "South Africa",
    "Eastern Cape": "South Africa", "Limpopo": "South Africa", "Mpumalanga": "South Africa",

    # ==================== MIDDLE EAST ====================
    "Dubai": "UAE", "Abu Dhabi": "UAE", "Sharjah": "UAE",
    "Riyadh Province": "Saudi Arabia", "Makkah": "Saudi Arabia", "Eastern Province": "Saudi Arabia",
    "Jerusalem District": "Israel", "Tel Aviv": "Israel", "Haifa": "Israel",

    # ==================== SOUTHEAST ASIA ====================
    "Bangkok": "Thailand", "Chiang Mai": "Thailand",
    "Ho Chi Minh": "Vietnam", "Hanoi": "Vietnam",
    "Metro Manila": "Philippines", "Cebu": "Philippines",
    "Jakarta": "Indonesia", "West Java": "Indonesia", "East Java": "Indonesia",
    "Selangor": "Malaysia", "Johor": "Malaysia", "Penang": "Malaysia",
}

DEFAULT_TZ = "UTC"   # safe fallback