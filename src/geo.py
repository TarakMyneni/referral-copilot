import math


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_city_centroids(df, city_col, lat_col, lon_col):
    """
    Compute average lat/lon per city from the facilities dataset.
    Avoids external geocoding — every facility already has coordinates.
    """
    work = df[[city_col, lat_col, lon_col]].dropna()
    work = work.copy()
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
        work = pincode_df[cols].dropna().copy()
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
    """
    key = query_location.strip().lower()

    if key in centroids:
        c = centroids[key]
        return c["lat"], c["lon"], key, "exact"

    # Fuzzy: substring match either direction
    candidates = [c for c in centroids if key in c or c in key]
    if candidates:
        best = min(candidates, key=len)
        c = centroids[best]
        return c["lat"], c["lon"], best, "fuzzy"

    return None, None, None, "not_found"
