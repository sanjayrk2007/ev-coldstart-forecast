"""
poi_features.py — OpenStreetMap POI feature extraction for site scoring.

Purpose:
    Given a candidate lat/lng, query the OpenStreetMap Overpass API to extract
    structural features describing the location. These features form the input
    vector for site selection scoring and station similarity computation.

Why Overpass API:
    Free, no API key, global coverage, sufficient for this use case.
    Google Places would give higher quality data but adds cost and a dependency.

Feature vector (14 features total):
    OSM-derived (12):
        parking_count               — parking lots/amenities within 1000m
        office_count                — office buildings within 1000m
        restaurant_count            — restaurants within 1000m
        amenity_count               — all amenity nodes within 1000m
        fuel_station_count          — fuel stations within 1000m
        building_count              — all buildings within 1000m
        residential_building_count  — residential buildings within 1000m
        transit_stop_count          — bus/metro/train stops within 1000m
        nearest_charger_dist_m      — distance to nearest existing EV charger
        highway_dist_m              — distance to nearest highway
        primary_road_count          — primary roads within 1000m (traffic proxy)
        secondary_road_count        — secondary roads within 1000m (traffic proxy)

    Caller-supplied metadata (2):
        num_ports                   — number of charging ports at candidate site
        location_type_encoded       — workplace=0, public=1, retail=2

Caching:
    Results are cached to data/cache/poi_cache.json keyed by rounded lat/lng.
    Cache key precision: 4 decimal places (~11m). Queries within 11m of a
    cached point return the cached result without hitting Overpass.
    This matters because building station_profiles.json requires 107 queries —
    without caching, re-running the notebook re-queries all 107 stations.
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
# Be a polite API citizen — wait between requests to avoid hammering
# the public Overpass instance. 1 second is enough.
REQUEST_DELAY_S = 1.0

# Default cache path — relative to project root.
# Callers can override by passing cache_path to get_poi_features().
DEFAULT_CACHE_PATH = Path("data/cache/poi_cache.json")

# Location type encoding — must match whatever global_model.py expects.
# Document this mapping explicitly so it never silently drifts.
LOCATION_TYPE_MAP = {
    "workplace": 0,
    "public": 1,
    "retail": 2,
}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Straight-line distance between two lat/lng points in meters.

    Why not use a library: this is 6 lines of math and avoids a dependency.
    The haversine formula is accurate to within ~0.3% for distances under
    a few hundred km, which is more than sufficient for 1000m radius queries.

    Args:
        lat1, lng1: first point in decimal degrees
        lat2, lng2: second point in decimal degrees

    Returns:
        Distance in meters.
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict:
    """Load existing cache from disk. Returns empty dict if file doesn't exist."""
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Persist cache to disk. Creates parent directories if needed."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(lat: float, lng: float) -> str:
    """
    Cache key from lat/lng rounded to 4 decimal places.

    4 decimal places = ~11m precision. Two candidate locations within 11m
    of each other get the same cached result — correct behavior since
    their POI environments are essentially identical.
    """
    return f"{round(lat, 4)}_{round(lng, 4)}"


# ---------------------------------------------------------------------------
# Overpass query builder
# ---------------------------------------------------------------------------

def _build_overpass_query(lat: float, lng: float, radius: int = RADIUS_M) -> str:
    """
    Build the Overpass QL query for all POI features in a single HTTP request.

    Why a union block: sending one request with multiple node/way selectors
    is faster and more polite than sending 8 separate requests.

    The query structure:
        [out:json][timeout:25];  — response format and server-side timeout
        (                        — union block: collect results from all queries
          node[...](around:r,lat,lng);   — count nodes of type X within radius
          way[...](around:r,lat,lng);    — ways (roads, buildings) work the same
          ...
        );
        out body;                — return all collected elements

    For nearest-neighbor queries (charger distance, highway distance):
        We use a separate targeted query rather than counting —
        we want the distance to the nearest one, not a count.
    """
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
    """
    Build a query to find the nearest single OSM element of a given type.

    Uses a larger default radius (5000m) than the counting queries because
    'nearest charger' and 'nearest highway' are meaningful even if far away —
    the distance itself is the signal, not whether it exists within 1000m.

    Args:
        key, value: OSM tag to match, e.g. key="amenity", value="charging_station"
    """
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

    Sends as URL-encoded form data with explicit header.
    406 errors occur when Content-Type is missing or mismatched.
    """
    normalized = " ".join(query.split())
    payload = f"data={requests.utils.quote(normalized)}"

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
    """
    Given a list of Overpass elements, return the distance in meters to
    the nearest one. Returns 9999.0 if the list is empty (no match found).

    Why 9999 instead of None: downstream cosine similarity requires numeric
    values. 9999m is a meaningful sentinel — "very far away" — that the
    model can learn from rather than a missing value that breaks inference.

    For ways (roads, buildings), Overpass returns a center lat/lng in the
    'center' key when you use 'out center'. Since we're using 'out body',
    ways don't have a direct lat/lng. We skip ways for distance computation
    and rely on nodes only. For highway distance this is acceptable because
    highway nodes are dense along roads.
    """
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
    """
    Count and compute all OSM-derived features from Overpass response elements.

    This is separated from the HTTP calls so it can be unit-tested
    without hitting the network.

    Args:
        lat, lng: the query center point
        elements: results from the main union query
        charger_elements: results from nearest-charger query
        highway_elements: results from nearest-highway query

    Returns:
        Dict with all 12 OSM-derived features.
    """
    # Tag-based counts — iterate elements once and classify by tags.
    # An element can match multiple categories (e.g., a parking amenity
    # that's also tagged as a building). That's fine — each counter is
    # independent.
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

    # Distance features — nearest node from dedicated queries
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

    This is the main entry point. Call this for both candidate locations
    (site scoring) and existing stations (building station_profiles.json).
    The feature vector is identical in both cases — that's what makes
    cosine similarity valid.

    Args:
        lat, lng:         Decimal degree coordinates of the candidate location.
        num_ports:        Number of charging ports at the candidate site.
                          For existing stations, use the actual port count.
                          For candidates, use the planned port count.
        location_type:    One of "workplace", "public", "retail".
                          Encoded as 0, 1, 2 respectively.
                          Since ACN training data is all workplace, candidates
                          with other types will carry higher uncertainty.
        cache_path:       Path to the JSON cache file.
        use_cache:        If False, always re-query Overpass (useful for
                          testing or refreshing stale cached data).
        verbose:          Print progress messages (useful during the
                          107-station profile building loop).

    Returns:
        Dict with all 14 features. Keys match exactly what global_model.py
        and similarity scoring expect. Never returns None — missing OSM data
        falls back to 0 counts or 9999m distances.

    Example:
        >>> features = get_poi_features(
        ...     lat=34.1377,
        ...     lng=-118.1253,
        ...     num_ports=8,
        ...     location_type="workplace",
        ... )
        >>> features["office_count"]
        12
    """
    # Validate location_type early — a typo here would silently encode
    # as a wrong integer and corrupt the similarity computation.
    if location_type not in LOCATION_TYPE_MAP:
        raise ValueError(
            f"location_type must be one of {list(LOCATION_TYPE_MAP.keys())}, "
            f"got '{location_type}'"
        )

    key = _cache_key(lat, lng)

    # Cache check — return immediately if we have this location.
    # Note: cache stores only the OSM-derived features (12 features).
    # num_ports and location_type_encoded are appended after cache lookup
    # because they're caller-supplied, not OSM-derived. This means the
    # same lat/lng can be queried with different port counts without
    # invalidating the cached OSM data.
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

    if verbose:
        print(f"  [poi_features] Querying Overpass for ({lat:.4f}, {lng:.4f})...")

    # Main union query — counts all POI types in one request
    main_query = _build_overpass_query(lat, lng, RADIUS_M)
    main_elements = _query_overpass(main_query)
    time.sleep(REQUEST_DELAY_S)

    # Nearest charger query — separate because we need distance, not count,
    # and we use a larger search radius (5000m) so we always find something
    charger_query = _build_nearest_query(lat, lng, "amenity", "charging_station", radius=5000)
    charger_elements = _query_overpass(charger_query)
    time.sleep(REQUEST_DELAY_S)

    # Nearest highway query — highway nodes along primary/trunk/motorway roads
    # We query highway=primary nodes as a proxy for major road proximity.
    # This is an approximation: a dense node set along primary roads means
    # _nearest_distance_m will return a reasonable distance to the road.
    highway_query = _build_nearest_query(lat, lng, "highway", "primary", radius=5000)
    highway_elements = _query_overpass(highway_query)
    time.sleep(REQUEST_DELAY_S)

    # Extract features from raw elements
    osm_features = _extract_features_from_elements(
        lat, lng, main_elements, charger_elements, highway_elements
    )

    if verbose:
        print(f"  [poi_features] Extracted: {osm_features}")

    # Cache the OSM features (without caller-supplied metadata)
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
    Return the ordered list of feature names in the feature vector.

    This is the single source of truth for feature ordering. Both
    global_model.py (similarity scoring) and the notebook (profile building)
    must use this function to construct feature vectors — never hardcode
    the order elsewhere.

    Returns:
        List of 14 feature name strings in canonical order.
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
    Convert a features dict to an ordered list suitable for numpy/sklearn.

    Uses get_feature_names() as the canonical ordering so the vector
    is always consistent regardless of dict insertion order.

    Args:
        features: dict returned by get_poi_features()

    Returns:
        List of 14 floats in canonical feature order.

    Raises:
        KeyError if any expected feature is missing from the dict.
    """
    names = get_feature_names()
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(
            f"Feature dict is missing expected keys: {missing}. "
            f"Make sure you're using get_poi_features() to build feature dicts."
        )
    return [float(features[n]) for n in names]