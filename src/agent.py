"""
SupervisorAgent — orchestrates three tools then blends their signals:

  Tool 1 · location_tool  Deterministic haversine geo-filter.
                          Returns candidate facilities within radius_km.

  Tool 2 · rag_tool       Semantic vector search over unstructured text
                          (description, specialties, capability, procedure,
                          equipment). Returns similarity scores per facility.

  Tool 3 · feedback_tool  Historical save-counts from user_interactions Delta.
                          Returns boost scores per (care_need, facility_id).

Blended score = keyword_match + SEMANTIC_W * sem_score + FEEDBACK_W * fb_score

Evidence display stays fully rule-based (src/evidence.py) so every result
shows which field matched, what's missing, and what to flag — satisfying the
hackathon brief's "matching evidence" requirement without LLM opacity.
"""

import pandas as pd

from .config import COLUMNS
from .geo import haversine_km, resolve_location
from .ranking import normalize_care_need, match_score
from .evidence import evaluate_evidence
from . import rag as rag_tool
from . import feedback as feedback_tool

# Blending weights (tune these post-demo if needed)
SEMANTIC_W   = 0.40   # normalised semantic similarity contribution
FEEDBACK_W   = 0.50   # per-save boost (capped at FEEDBACK_CAP saves)
FEEDBACK_CAP = 5      # caps influence of very-frequently-saved facilities
SEM_FLOOR    = 0.10   # discard results with semantic score below this threshold
                      # (prevents very-low-relevance RAG hits from surfacing)

# Only these geo_source values carry reliable coordinates.
# CITY_AVG / STATE_AVG are statistical estimates and can be far off for small cities,
# so we treat them the same as UNKNOWN and use city-name string matching instead.
_RELIABLE_GEO = frozenset({"ORIGINAL", "POSTCODE"})

# ---------------------------------------------------------------------------
# Facility-type filtering
# Labs/diagnostics mention "prenatal tests", "gynecology panel", etc. in their
# capability/description, so keyword matching alone surfaces them for hospital
# queries. We detect them by name and org_type and exclude from non-diagnostic
# searches.
# ---------------------------------------------------------------------------

_LAB_NAME_WORDS = frozenset({
    "lab", "labs", "laboratory", "laboratories",
    "pathology", "diagnostics", "collection",
})

_LAB_ORG_SUBSTRINGS = (
    "laboratory", "laboratories", "pathology",
    "diagnostic center", "diagnostic centre",
    "collection center", "collection centre",
    "sample center", "sample centre",
)

# Care needs that legitimately expect diagnostic/imaging facilities
_DIAGNOSTIC_NEEDS = frozenset({
    "radiology", "pathology", "imaging", "diagnostic",
    "lab", "laboratory", "blood test", "scan",
})

# Org-type keywords that confirm a real clinical facility (override name check)
_CLINICAL_ORG_TYPES = (
    "hospital", "clinic", "nursing home", "medical center",
    "medical centre", "health center", "health centre", "dispensary",
)


def _is_lab_facility(name: str, org_type: str) -> bool:
    """Return True when a facility is primarily a diagnostic lab / collection centre."""
    org_lower = org_type.lower()
    # If org_type explicitly says hospital/clinic, trust that over the name
    if any(c in org_lower for c in _CLINICAL_ORG_TYPES):
        return False
    # Check name words
    name_words = set(
        name.lower().replace(",", " ").replace(".", " ").replace("-", " ").split()
    )
    if name_words & _LAB_NAME_WORDS:
        return True
    # Check org_type substrings
    return any(s in org_lower for s in _LAB_ORG_SUBSTRINGS)


def run(df: pd.DataFrame, centroids: dict,
        care_need_query: str, location_query: str,
        radius_km: float = 150, top_n: int = 10) -> tuple[list, dict]:
    """
    Entry point called by app.py.

    Returns
    -------
    results : list[dict]   ranked facilities, located first then unlocated
    meta    : dict         search metadata (or {"error": ...} on failure)
    """
    if not location_query.strip():
        return [], {"error": "Please enter a location."}

    # show_all mode: no specific care need — return all clinical facilities within radius
    show_all = not care_need_query.strip()

    # ── Parse care need ───────────────────────────────────────────────────────
    if show_all:
        need_key, keywords = "", []
    else:
        need_key, keywords = normalize_care_need(care_need_query)

    # ── Tool 1: location (deterministic) ─────────────────────────────────────
    lat, lon, matched_city, match_type = resolve_location(location_query, centroids)
    if lat is None:
        return [], {
            "error": (
                f"Could not resolve location '{location_query}'. "
                "Try a major city name present in the dataset."
            )
        }

    # ── Tool 2: RAG (semantic over unstructured text) ─────────────────────────
    if show_all:
        sem_scores       = {}
        semantic_active  = False
    else:
        rag_query = f"{care_need_query} {' '.join(keywords[:6])}"
        raw_sem_scores = rag_tool.semantic_search(rag_query, top_k=300)
        semantic_active = len(raw_sem_scores) > 0
        max_sem = max(raw_sem_scores.values(), default=1.0) or 1.0
        sem_scores = {fid: s / max_sem for fid, s in raw_sem_scores.items()}

    # ── Tool 3: feedback boost ────────────────────────────────────────────────
    boost_scores = feedback_tool.get_all_boosts(need_key)

    # ── Blend and filter ─────────────────────────────────────────────────────
    lat_col        = COLUMNS["latitude"]
    lon_col        = COLUMNS["longitude"]
    id_col         = COLUMNS["id"]
    geo_source_col = COLUMNS["geo_source"]
    city_col       = COLUMNS["city"]
    state_col      = COLUMNS["state"]
    # Use the resolved canonical city name (e.g. "mumbai" when user typed "bombay")
    # so that unlocated facilities whose city="mumbai" are matched correctly.
    loc_lower      = matched_city.lower() if matched_city else location_query.strip().lower()

    located   = []   # have usable coords, within radius
    unlocated = []   # geo_source=UNKNOWN but city/state matches

    for _, row in df.iterrows():
        facility_id = str(row.get(id_col, "") or "")

        # ── Relevance signals ─────────────────────────────────────────────
        fb_raw    = boost_scores.get(facility_id, 0)
        fb_score  = min(fb_raw, FEEDBACK_CAP) / FEEDBACK_CAP      # 0-1 float

        if show_all:
            # No specialty filter — accept any clinical facility; skip labs
            kw_score  = 0
            sem_score = 0.0
            facility_name = str(row.get(COLUMNS["name"], "") or "")
            facility_org  = str(row.get(COLUMNS.get("org_type", "organization_type"), "") or "")
            if _is_lab_facility(facility_name, facility_org):
                continue
        else:
            kw_score  = match_score(row, keywords)                 # 0-5 int
            sem_score = sem_scores.get(facility_id, 0.0)          # 0-1 float

            # Require at least one relevance signal; skip very low semantic hits
            if kw_score == 0 and sem_score < SEM_FLOOR:
                continue

            # Exclude diagnostic labs from clinical searches
            facility_name = str(row.get(COLUMNS["name"], "") or "")
            facility_org  = str(row.get(COLUMNS.get("org_type", "organization_type"), "") or "")
            if need_key not in _DIAGNOSTIC_NEEDS and _is_lab_facility(facility_name, facility_org):
                continue

        blended = kw_score + SEMANTIC_W * sem_score + FEEDBACK_W * fb_score

        # ── Geo ───────────────────────────────────────────────────────────
        geo_src  = str(row.get(geo_source_col, "UNKNOWN") or "UNKNOWN")
        flat_raw = row.get(lat_col)
        flon_raw = row.get(lon_col)
        try:
            flat, flon = float(flat_raw), float(flon_raw)
            has_coords = not (pd.isna(flat) or pd.isna(flon))
        except (ValueError, TypeError):
            has_coords = False

        ev = evaluate_evidence(row, keywords)
        base = {
            "id":            facility_id,
            "name":          str(row.get(COLUMNS["name"], "Unknown") or "Unknown"),
            "city":          str(row.get(city_col,  "") or ""),
            "state":         str(row.get(state_col, "") or ""),
            "blended_score": round(blended, 3),
            "kw_score":      kw_score,
            "sem_score":     round(sem_score, 3),
            "fb_saves":      fb_raw,
            "evidence":      ev,
            "geo_source":    geo_src,
            "phone":         str(row.get(COLUMNS.get("phone",    "phone"),    "") or ""),
            "website":       str(row.get(COLUMNS.get("website",  "website"),  "") or ""),
            "org_type":      str(row.get(COLUMNS.get("org_type", "organization_type"), "") or ""),
        }

        if has_coords and geo_src in _RELIABLE_GEO:
            dist = haversine_km(lat, lon, flat, flon)
            if dist > radius_km:
                continue
            located.append({**base, "distance_km": round(dist, 1),
                             "lat": flat, "lon": flon})
        else:
            # CITY_AVG / STATE_AVG coordinates are statistical estimates that can be
            # wrong for small cities (e.g. all Shimla facilities averaged = wrong spot).
            # UNKNOWN has no coords at all. In all cases, use city-name string match.
            fac_city  = str(row.get(city_col,  "") or "").lower()
            fac_state = str(row.get(state_col, "") or "").lower()
            if (fac_city  and (fac_city  in loc_lower or loc_lower in fac_city)) or \
               (fac_state and (fac_state in loc_lower or loc_lower in fac_state)):
                unlocated.append({**base, "distance_km": None,
                                   "lat": None, "lon": None})

    located.sort(key=lambda r: (-r["blended_score"], r["distance_km"]))
    unlocated.sort(key=lambda r: -r["blended_score"])
    combined = located + unlocated

    meta = {
        "resolved_location":   matched_city,
        "location_match_type": match_type,
        "care_need":           need_key,
        "keywords":            keywords,
        "total_matches":       len(combined),
        "located_count":       len(located),
        "unlocated_count":     len(unlocated),
        "max_distance_km":     radius_km,
        "search_lat":          lat,
        "search_lon":          lon,
        "semantic_active":     semantic_active,
    }
    return combined[:top_n], meta
