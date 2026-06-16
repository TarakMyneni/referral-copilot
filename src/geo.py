import difflib
import math
import re

import pandas as pd

# Admin-level words that users append but centroid keys omit:
#   "agra division" → try "agra"; "jaipur district" → try "jaipur"
_ADMIN_SUFFIX = re.compile(
    r"\s+(?:division|district|tehsil|taluk|taluka|block|mandal|zone|"
    r"sector|ward|area|region|circle|sub.?district|sub.?division|"
    r"municipal corporation|nagar|nagar panchayat|gram panchayat)$",
    re.IGNORECASE,
)

# In-process cache so repeated searches for the same city don't hit Nominatim twice.
_nominatim_cache: dict = {}

# Optional callback set by app.py to persist new Nominatim results to Delta.
_geo_db_save_callback = None


def set_geo_save_callback(fn):
    """Register fn(query_lower, lat, lon) to persist new Nominatim hits to DB."""
    global _geo_db_save_callback
    _geo_db_save_callback = fn


def preload_nominatim_cache(entries):
    """Pre-populate in-process cache from DB: entries = [(query_lower, lat, lon), ...]"""
    for q, lat, lon in entries:
        _nominatim_cache[(q, False)] = (float(lat), float(lon))


def _nominatim_lookup(raw_query: str, is_pin: bool = False):
    """
    Geocode via OSM Nominatim (free, no API key).
    Returns (lat, lon) or None.
    Results are cached in-process to avoid repeated network calls.
    """
    cache_key = (raw_query.lower(), is_pin)
    if cache_key in _nominatim_cache:
        return _nominatim_cache[cache_key]

    result = None
    try:
        import requests
        if is_pin:
            params = {
                "postalcode": raw_query,
                "country":    "in",
                "format":     "json",
                "limit":      1,
            }
        else:
            params = {
                "q":            f"{raw_query}, India",
                "format":       "json",
                "limit":        1,
                "countrycodes": "in",
                "addressdetails": 0,
            }
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "Suvidha-Referral-Copilot/1.0"},
            timeout=5,
        )
        data = resp.json()
        if data:
            result = (float(data[0]["lat"]), float(data[0]["lon"]))
            print(f"[Geo] Nominatim: '{raw_query}' → {result}")
    except Exception as exc:
        print(f"[Geo] Nominatim failed for '{raw_query}': {exc}")

    _nominatim_cache[cache_key] = result
    if result and not is_pin and _geo_db_save_callback:
        try:
            _geo_db_save_callback(raw_query.lower(), result[0], result[1])
        except Exception:
            pass
    return result


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_postcode_centroids(df, postcode_col, lat_col, lon_col):
    """Build lat/lon centroids keyed by 6-digit PIN code from the facilities dataset."""
    if postcode_col not in df.columns:
        return {}
    work = df[[postcode_col, lat_col, lon_col]].copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col])
    work["_key"] = work[postcode_col].astype(str).str.strip()
    work = work[work["_key"].str.fullmatch(r"\d{6}")]
    if work.empty:
        return {}
    grouped = work.groupby("_key")[[lat_col, lon_col]].mean()
    result = {pin: {"lat": row[lat_col], "lon": row[lon_col]}
              for pin, row in grouped.iterrows()}
    print(f"[Geo] Built {len(result)} PIN code centroids")
    return result


def build_city_centroids(df, city_col, lat_col, lon_col):
    """
    Compute average lat/lon per city from the facilities dataset.
    Avoids external geocoding — every facility already has coordinates.
    """
    work = df[[city_col, lat_col, lon_col]].copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col])
    work["_key"] = work[city_col].astype(str).str.strip().str.lower()
    grouped = work.groupby("_key")[[lat_col, lon_col]].mean()
    return {
        city: {"lat": row[lat_col], "lon": row[lon_col]}
        for city, row in grouped.iterrows()
    }


def build_pincode_centroids(pincode_df):
    """
    Build location centroids from the India Post pincode directory at every
    available administrative level: district, division, region.
    """
    result = {}
    coord_cols = ["latitude", "longitude"]

    for level_col in ("region", "division", "district"):   # coarser first so finer overwrites
        if level_col not in pincode_df.columns:
            continue
        cols = [level_col] + coord_cols
        if not all(c in pincode_df.columns for c in cols):
            continue
        work = pincode_df[cols].copy()
        work["latitude"]  = pd.to_numeric(work["latitude"],  errors="coerce")
        work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
        work = work.dropna(subset=coord_cols)
        work["_key"] = work[level_col].astype(str).str.strip().str.lower()
        grouped = work.groupby("_key")[coord_cols].mean()
        for name, row in grouped.iterrows():
            if name:
                result[name] = {"lat": row["latitude"], "lon": row["longitude"]}

    return result


def _nearest_centroid(lat, lon, city_centroids, max_km=100):
    """
    Return the centroid key in city_centroids nearest to (lat, lon).
    Returns None if no centroid is within max_km.
    PIN-code-only keys (pure digits) are excluded.
    """
    best_key, best_dist = None, float("inf")
    for k, c in city_centroids.items():
        if k.isdigit():
            continue
        d = haversine_km(lat, lon, c["lat"], c["lon"])
        if d < best_dist:
            best_dist, best_key = d, k
    return best_key if best_dist <= max_km else None


def resolve_location(query_location, centroids):
    """
    Return (lat, lon, matched_name, match_type).
    match_type: "exact" | "fuzzy" | "not_found"

    Resolution order:
      0. 6-digit PIN code — exact hit in facility centroids, else Nominatim postalcode
      1. Nominatim geocoding — primary, handles old names / alternate spellings.
         After resolving coordinates, nearest-centroid lookup gives the canonical
         city name used in the dataset (e.g. "bombay" → Mumbai coords → "mumbai").
      2. Exact lowercase name match in centroids   ┐ fallback when
      3. Substring match (both directions, guarded) ┤ Nominatim is
      4. Admin-suffix stripped versions of 2-3      ┘ unavailable
      5. Edit-distance fuzzy (cutoff 0.82, typo correction only)
    """
    raw = query_location.strip()
    key = raw.lower()

    # --- 0. PIN code ---
    if re.fullmatch(r"\d{6}", key):
        if key in centroids:
            c = centroids[key]
            return c["lat"], c["lon"], key, "exact"
        coords = _nominatim_lookup(key, is_pin=True)
        if coords:
            return coords[0], coords[1], key, "exact"
        return None, None, None, "not_found"

    # City names only (exclude PIN-code keys which are pure digits)
    city_centroids = {k: v for k, v in centroids.items() if not k.isdigit()}

    # --- 1. Nominatim (primary) ---
    # Handles old names (Bombay→Mumbai, Calcutta→Kolkata), neighborhoods,
    # districts, and any Indian place not in the facility dataset.
    # Results are cached in-process and persisted to Delta between restarts.
    coords = _nominatim_lookup(raw)
    if coords:
        # Nearest centroid gives the canonical city name from our dataset.
        canonical = _nearest_centroid(coords[0], coords[1], city_centroids)
        matched_name = canonical if canonical else key
        print(f"[Geo] Nominatim: '{raw}' → {coords} → canonical='{matched_name}'")
        return coords[0], coords[1], matched_name, "fuzzy"

    # --- Fallback: centroid-based lookup (Nominatim unavailable) ---

    def _lookup(k):
        """Exact → guarded substring for a given lowercase key."""
        if not k:
            return None, None
        if k in city_centroids:
            return k, "exact"
        candidates = []
        for c in city_centroids:
            longer, shorter = (k, c) if len(k) >= len(c) else (c, k)
            if len(shorter) >= 4 and shorter in longer and len(shorter) >= len(longer) * 0.6:
                candidates.append(c)
        if candidates:
            return min(candidates, key=len), "fuzzy"
        return None, None

    # 2-3. Try full key
    matched, mtype = _lookup(key)
    if matched:
        c = city_centroids[matched]
        return c["lat"], c["lon"], matched, mtype

    # 4. Strip admin suffix and retry
    stripped = _ADMIN_SUFFIX.sub("", key).strip()
    if stripped and stripped != key:
        matched, mtype = _lookup(stripped)
        if matched:
            c = city_centroids[matched]
            print(f"[Geo] Suffix-stripped '{raw}' → '{matched}'")
            return c["lat"], c["lon"], matched, mtype

    # 5. Edit-distance fuzzy — typo correction only, high cutoff to avoid wrong cities
    close = difflib.get_close_matches(key, city_centroids.keys(), n=1, cutoff=0.82)
    if close:
        print(f"[Geo] Fuzzy matched '{raw}' → '{close[0]}'")
        c = city_centroids[close[0]]
        return c["lat"], c["lon"], close[0], "fuzzy"

    return None, None, None, "not_found"
