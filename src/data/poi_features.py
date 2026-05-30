"""
poi_features.py — OpenStreetMap POI feature extraction for site scoring.

Purpose:
    Given a candidate lat/lng, return structural features describing the
    location. These form the input vector for site selection scoring and
    station similarity computation.

Network note:
    The Overpass API (overpass-api.de and all tested mirrors) is blocked
    on this network. For the two known training site coordinates (Caltech,
    JPL), hardcoded feature vectors derived from manual Overpass Turbo
    queries on 2026-05-28 are returned directly. For any other coordinate,
    a live Overpass query is attempted — it will return zero counts if
    also blocked, with a printed warning.

    To restore live queries for all coordinates: remove the hardcoded
    fast-path block in get_poi_features() once network access is available.

Feature vector (14 features):
    OSM-derived (12):
        parking_count               parking lots/amenities within 1000m
        office_count                office buildings within 1000m
        restaurant_count            restaurants within 1000m
        amenity_count               all amenity nodes within 1000m
        fuel_station_count          fuel stations within 1000m
        building_count              all buildings within 1000m
        residential_building_count  residential buildings within 1000m
        transit_stop_count          bus/metro/train stops within 1000m
        nearest_charger_dist_m      distance to nearest EV charger
        highway_dist_m              distance to nearest primary highway node
        primary_road_count          primary roads within 1000m (traffic proxy)
        secondary_road_count        secondary roads within 1000m (traffic proxy)

    Caller-supplied (2):
        num_ports                   charging ports at candidate site
        location_type_encoded       workplace=0, public=1, retail=2
"""

import json
import math
import os
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 1000
REQUEST_TIMEOUT_S = 30
REQUEST_DELAY_S = 1.0
DEFAULT_CACHE_PATH = Path("data/cache/poi_cache.json")

LOCATION_TYPE_MAP = {
    "workplace": 0,
    "public": 1,
    "retail": 2,
}

# ---------------------------------------------------------------------------
# Hardcoded site feature vectors
# ---------------------------------------------------------------------------
# Derived from manual Overpass Turbo browser queries on 2026-05-28.
# Caltech counts from actual OSM data (Query 1 + Query 2).
# JPL counts from actual OSM data (Query 4 + Query 5).
# highway_dist_m = 9999 for both: primary road nodes not individually tagged
# in OSM for either area — roads exist as ways only.
# fuel_station_count = 0 for JPL: nearest fuel stations are >3km away,
# outside the 1000m counting radius.
# primary/secondary road counts = 0: no highway=primary/secondary ways
# appeared in either union query result.

HARDCODED_SITE_FEATURES = {
    'caltech': {
        'parking_count': 4,
        'office_count': 0,
        'restaurant_count': 29,
        'amenity_count': 95,
        'fuel_station_count': 2,
        'building_count': 5,
        'residential_building_count': 0,
        'transit_stop_count': 29,
        'nearest_charger_dist_m': 293.0,
        'highway_dist_m': 9999.0,
        'primary_road_count': 0,
        'secondary_road_count': 0,
    },
    'jpl': {
        'parking_count': 2,
        'office_count': 0,
        'restaurant_count': 17,
        'amenity_count': 110,
        'fuel_station_count': 0,
        'building_count': 1,
        'residential_building_count': 0,
        'transit_stop_count': 12,
        'nearest_charger_dist_m': 4510.0,
        'highway_dist_m': 9999.0,
        'primary_road_count': 0,
        'secondary_road_count': 0,
    },
}

# Cache key → site name mapping for hardcoded lookup
_HARDCODED_COORDS = {
    '34.1377_-118.1253': 'caltech',
    '34.2013_-118.1714': 'jpl',
}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Straight-line distance between two lat/lng points in meters."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(lat: float, lng: float) -> str:
    return f"{round(lat, 4)}_{round(lng, 4)}"


# ---------------------------------------------------------------------------
# Overpass query builders
# ---------------------------------------------------------------------------

def _build_overpass_query(lat: float, lng: float, radius: int = RADIUS_M) -> str:
    q = f"""
[out:json][timeout:25];
(
  node["amenity"="parking"](around:{radius},{lat},{lng});
  way["amenity"="parking"](around:{radius},{lat},{lng});
  node["building"="office"](around:{radius},{lat},{lng});
  way["building"="office"](around:{radius},{lat},{lng});
  node["amenity"="restaurant"](around:{radius},{lat},{lng});
  node["amenity"](around:{radius},{lat},{lng});
  node["amenity"="fuel"](around:{radius},{lat},{lng});
  node["building"](around:{radius},{lat},{lng});
  way["building"](around:{radius},{lat},{lng});
  node["building"="residential"](around:{radius},{lat},{lng});
  way["building"="residential"](around:{radius},{lat},{lng});
  node["highway"="bus_stop"](around:{radius},{lat},{lng});
  node["railway"="station"](around:{radius},{lat},{lng});
  node["railway"="halt"](around:{radius},{lat},{lng});
  way["highway"="primary"](around:{radius},{lat},{lng});
  way["highway"="secondary"](around:{radius},{lat},{lng});
);
out body;
"""
    return q.strip()


def _build_nearest_query(lat: float, lng: float, key: str, value: str, radius: int = 5000) -> str:
    return f"""
[out:json][timeout:25];
(
  node["{key}"="{value}"](around:{radius},{lat},{lng});
  way["{key}"="{value}"](around:{radius},{lat},{lng});
);
out body;
""".strip()


# ---------------------------------------------------------------------------
# Overpass HTTP caller
# ---------------------------------------------------------------------------

def _query_overpass(query: str) -> list:
    """
    Send a query to the Overpass API and return the list of elements.

    Sends as manually URL-encoded form data with explicit Content-Type.
    The original requests.post(data={"data": query}) caused 406 Not Acceptable
    because the Content-Type header was missing or mismatched.
    Fix: normalize whitespace, manually encode, set header explicitly.
    """
    normalized = " ".join(query.split())
    payload = "data=" + requests.utils.quote(normalized)

    try:
        response = requests.post(
            OVERPASS_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("elements", [])
    except requests.exceptions.Timeout:
        print(f"  [poi_features] Overpass timeout — returning empty result")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  [poi_features] Overpass request error: {e} — returning empty result")
        return []
    except json.JSONDecodeError:
        print(f"  [poi_features] Overpass returned non-JSON — returning empty result")
        return []


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _nearest_distance_m(lat: float, lng: float, elements: list) -> float:
    """Distance in meters to the nearest element. Returns 9999 if list empty."""
    min_dist = 9999.0
    for el in elements:
        el_lat = el.get("lat")
        el_lng = el.get("lon")
        if el_lat is not None and el_lng is not None:
            d = haversine_m(lat, lng, el_lat, el_lng)
            if d < min_dist:
                min_dist = d
    return min_dist


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def _extract_features_from_elements(
    lat: float,
    lng: float,
    elements: list,
    charger_elements: list,
    highway_elements: list,
) -> dict:
    parking_count = 0
    office_count = 0
    restaurant_count = 0
    amenity_count = 0
    fuel_station_count = 0
    building_count = 0
    residential_building_count = 0
    transit_stop_count = 0
    primary_road_count = 0
    secondary_road_count = 0

    for el in elements:
        tags = el.get("tags", {})
        amenity = tags.get("amenity", "")
        building = tags.get("building", "")
        highway = tags.get("highway", "")
        railway = tags.get("railway", "")

        if amenity in ("parking", "parking_space"):
            parking_count += 1
        if building == "office":
            office_count += 1
        if amenity == "restaurant":
            restaurant_count += 1
        if amenity:
            amenity_count += 1
        if amenity == "fuel":
            fuel_station_count += 1
        if building:
            building_count += 1
        if building == "residential":
            residential_building_count += 1
        if amenity == "bus_stop" or railway in ("station", "halt"):
            transit_stop_count += 1
        if highway == "primary":
            primary_road_count += 1
        if highway == "secondary":
            secondary_road_count += 1

    nearest_charger_dist_m = _nearest_distance_m(lat, lng, charger_elements)
    highway_dist_m = _nearest_distance_m(lat, lng, highway_elements)

    return {
        "parking_count": parking_count,
        "office_count": office_count,
        "restaurant_count": restaurant_count,
        "amenity_count": amenity_count,
        "fuel_station_count": fuel_station_count,
        "building_count": building_count,
        "residential_building_count": residential_building_count,
        "transit_stop_count": transit_stop_count,
        "nearest_charger_dist_m": nearest_charger_dist_m,
        "highway_dist_m": highway_dist_m,
        "primary_road_count": primary_road_count,
        "secondary_road_count": secondary_road_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_poi_features(
    lat: float,
    lng: float,
    num_ports: int,
    location_type: str = "workplace",
    cache_path: Path = DEFAULT_CACHE_PATH,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Get all 14 features for a candidate location.

    For Caltech and JPL coordinates, returns hardcoded features from
    manual Overpass Turbo queries (network access to Overpass is blocked).
    For any other coordinate, attempts a live Overpass query with cache.

    Args:
        lat, lng:       Decimal degree coordinates.
        num_ports:      Number of charging ports.
        location_type:  One of "workplace", "public", "retail".
        cache_path:     Path to JSON cache file.
        use_cache:      If False, bypass cache for non-hardcoded coordinates.
        verbose:        Print progress messages.

    Returns:
        Dict with all 14 features.
    """
    if location_type not in LOCATION_TYPE_MAP:
        raise ValueError(
            f"location_type must be one of {list(LOCATION_TYPE_MAP.keys())}, "
            f"got '{location_type}'"
        )

    key = _cache_key(lat, lng)

    # --- Hardcoded fast path for known training site coordinates ---
    if key in _HARDCODED_COORDS:
        site = _HARDCODED_COORDS[key]
        if verbose:
            print(f"  [poi_features] Using hardcoded features for {site} ({lat:.4f}, {lng:.4f})")
        osm_features = HARDCODED_SITE_FEATURES[site].copy()
        return {
            **osm_features,
            "num_ports": num_ports,
            "location_type_encoded": LOCATION_TYPE_MAP[location_type],
        }

    # --- Cache check for non-hardcoded coordinates ---
    if use_cache:
        cache = _load_cache(cache_path)
        if key in cache:
            if verbose:
                print(f"  [poi_features] Cache hit for ({lat:.4f}, {lng:.4f})")
            osm_features = cache[key]
            return {
                **osm_features,
                "num_ports": num_ports,
                "location_type_encoded": LOCATION_TYPE_MAP[location_type],
            }

    # --- Live Overpass query ---
    if verbose:
        print(f"  [poi_features] Querying Overpass for ({lat:.4f}, {lng:.4f})...")

    main_query = _build_overpass_query(lat, lng, RADIUS_M)
    main_elements = _query_overpass(main_query)
    time.sleep(REQUEST_DELAY_S)

    charger_query = _build_nearest_query(lat, lng, "amenity", "charging_station", radius=5000)
    charger_elements = _query_overpass(charger_query)
    time.sleep(REQUEST_DELAY_S)

    highway_query = _build_nearest_query(lat, lng, "highway", "primary", radius=5000)
    highway_elements = _query_overpass(highway_query)
    time.sleep(REQUEST_DELAY_S)

    osm_features = _extract_features_from_elements(
        lat, lng, main_elements, charger_elements, highway_elements
    )

    if verbose:
        print(f"  [poi_features] Extracted: {osm_features}")

    if use_cache:
        cache = _load_cache(cache_path)
        cache[key] = osm_features
        _save_cache(cache, cache_path)

    return {
        **osm_features,
        "num_ports": num_ports,
        "location_type_encoded": LOCATION_TYPE_MAP[location_type],
    }


def get_feature_names() -> list:
    """
    Canonical ordered list of feature names. Single source of truth.
    Both similarity scoring and profile building must use this function
    to construct feature vectors — never hardcode the order elsewhere.
    """
    return [
        "parking_count",
        "office_count",
        "restaurant_count",
        "amenity_count",
        "fuel_station_count",
        "building_count",
        "residential_building_count",
        "transit_stop_count",
        "nearest_charger_dist_m",
        "highway_dist_m",
        "primary_road_count",
        "secondary_road_count",
        "num_ports",
        "location_type_encoded",
    ]


def build_feature_vector(features: dict) -> list:
    """
    Convert a features dict to an ordered list for numpy/sklearn.
    Uses get_feature_names() as canonical ordering.

    Raises KeyError if any expected feature is missing.
    """
    names = get_feature_names()
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(
            f"Feature dict is missing expected keys: {missing}. "
            f"Make sure you're using get_poi_features() to build feature dicts."
        )
    return [float(features[n]) for n in names]