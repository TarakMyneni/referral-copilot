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


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_postcode_centroids(df, postcode_col, lat_col, lon_col):
    """Build lat/lon centroids keyed by 6-digit PIN code from the facilities dataset.
    Uses the dataset itself — no external postcode directory needed."""
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

    All three levels are merged into a single dict keyed by lowercase name.
    Priority when the same name appears at multiple levels: district wins over
    division wins over region (most precise wins).

    Facility city centroids (from build_city_centroids) should then override
    these with:  centroids = {**pincode_centroids, **facility_centroids}
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


def resolve_location(query_location, centroids):
    """
    Return (lat, lon, matched_name, match_type).
    match_type: "exact" | "fuzzy" | "not_found"

    Resolution order (each step also retried after stripping admin suffixes):
      0. 6-digit PIN code exact match
      1. Exact lowercase name match
      2. Substring match either direction — catches "new delhi" ↔ "delhi"
      3. Edit-distance fuzzy — catches typos like "kolekatta" → "kolkata"
    """
    raw = query_location.strip()
    key = raw.lower()

    # 0. PIN code — 6-digit number, keyed directly in centroids
    if re.fullmatch(r"\d{6}", key):
        if key in centroids:
            c = centroids[key]
            return c["lat"], c["lon"], key, "exact"
        return None, None, None, "not_found"

    def _lookup(k):
        """Try exact → substring → fuzzy for a given key string."""
        if not k:
            return None

        # exact
        if k in centroids:
            return k, "exact"

        # substring either direction
        candidates = [c for c in centroids if k in c or c in k]
        if candidates:
            return min(candidates, key=len), "fuzzy"

        # edit-distance fuzzy
        close = difflib.get_close_matches(k, centroids.keys(), n=1, cutoff=0.70)
        if close:
            print(f"[Geo] Fuzzy matched '{raw}' → '{close[0]}'")
            return close[0], "fuzzy"

        return None, None

    # 1-3. Try the full key as typed
    matched, mtype = _lookup(key)
    if matched:
        c = centroids[matched]
        return c["lat"], c["lon"], matched, mtype

    # 4. Strip admin suffix and retry ("agra division" → "agra")
    stripped = _ADMIN_SUFFIX.sub("", key).strip()
    if stripped and stripped != key:
        matched, mtype = _lookup(stripped)
        if matched:
            c = centroids[matched]
            print(f"[Geo] Suffix-stripped '{raw}' → '{matched}'")
            return c["lat"], c["lon"], matched, mtype

    return None, None, None, "not_found"
