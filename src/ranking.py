import re

import pandas as pd

from .config import COLUMNS, CARE_NEED_SYNONYMS, EVIDENCE_TEXT_FIELDS
from .geo import haversine_km, resolve_location
from .evidence import evaluate_evidence


_SEPARATOR_WORDS = re.compile(r"\b(?:near|in|around|close to|nearby|at)\b", re.IGNORECASE)

_LLM_PROMPT = """\
Extract the medical care need and the Indian city or location from the user query.
Return ONLY a JSON object with two keys: "care_need" and "location".
If either is missing or unclear, return an empty string for that key.

Examples:
  "dialysis near Jaipur"           -> {{"care_need": "dialysis", "location": "Jaipur"}}
  "I need heart surgery in Mumbai" -> {{"care_need": "heart surgery", "location": "Mumbai"}}
  "hello wassup"                   -> {{"care_need": "", "location": ""}}

Query: {query}
"""


def _llm_parse(text):
    """Use Databricks Foundation Model API to extract care_need + location."""
    import json, os, requests
    try:
        host  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not host or not token:
            return text, ""

        resp = requests.post(
            f"{host}/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json={
                "messages": [{"role": "user",
                              "content": _LLM_PROMPT.format(query=text)}],
                "max_tokens": 60,
                "temperature": 0,
            },
            timeout=10,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            parsed    = json.loads(json_match.group())
            care_need = parsed.get("care_need", "").strip()
            location  = parsed.get("location",  "").strip()
            print(f"[LLM] '{text}' → care_need='{care_need}' location='{location}'")
            return care_need, location
    except Exception as e:
        print(f"[LLM] Parse failed: {e}")
    return text, ""


def parse_combined_query(text, centroids):
    """
    Parse a free-text query like "dialysis near Jaipur" into (care_need, location).

    Strategy:
      1. Look for "<need> near/in/around/close to/at <location>".
      2. Fall back to scanning for any city name the dataset actually knows about
         (handles phrasings without a separator word, e.g. "Jaipur dialysis").

    Returns ("", "") if nothing usable could be extracted.
    """
    text = text.strip()
    if not text:
        return "", ""

    m = re.search(r"^(.*?)\s+(?:near|in|around|close to|nearby|at)\s+(.+)$", text, re.IGNORECASE)
    if m:
        need, loc = m.group(1).strip(), m.group(2).strip()
        if need and loc:
            return need, loc

    text_lower = text.lower()
    for city in sorted(centroids.keys(), key=len, reverse=True):
        if city and city in text_lower:
            idx = text_lower.find(city)
            original_loc = text[idx:idx + len(city)]
            remaining = text[:idx] + text[idx + len(city):]
            need = _SEPARATOR_WORDS.sub("", remaining).strip()
            return (need or text), original_loc

    # LLM fallback — ask Databricks Foundation Model to extract intent
    return _llm_parse(text)


def normalize_care_need(text):
    """
    Map free-text care need to a canonical need name + list of search keywords.
    Falls back to treating the raw input as a single keyword if unrecognized.
    """
    text_l = text.strip().lower()
    if text_l in CARE_NEED_SYNONYMS:
        return text_l, CARE_NEED_SYNONYMS[text_l]

    for need, kws in CARE_NEED_SYNONYMS.items():
        if any(kw in text_l or text_l in kw for kw in kws):
            return need, kws

    return text_l, [text_l]


def match_score(row, keywords):
    """Count how many distinct evidence fields contain at least one keyword."""
    keywords_lower = [k.lower() for k in keywords]
    score = 0
    for field in EVIDENCE_TEXT_FIELDS:
        col = COLUMNS.get(field, field)
        text = str(row.get(col, "")).lower()
        if text and any(kw in text for kw in keywords_lower):
            score += 1
    return score


def search_facilities(df, location_query, care_need_query, centroids,
                      max_distance_km=150, top_n=10):
    """
    Returns (results, meta).
    results: list of dicts, sorted best-first.
      Records with resolvable coordinates are ranked by match_score desc, distance asc.
      Records with geo_source=UNKNOWN but matching city/state are appended at the end.
    meta: dict with resolved location / care-need info, or an "error" key.
    """
    if not location_query.strip() or not care_need_query.strip():
        return [], {"error": "Please enter both a location and a care need."}

    need_key, keywords = normalize_care_need(care_need_query)
    lat, lon, matched_city, match_type = resolve_location(location_query, centroids)

    if lat is None:
        return [], {
            "error": (
                f"Could not resolve location '{location_query}'. "
                f"Try a major city name present in the dataset."
            )
        }

    lat_col        = COLUMNS["latitude"]
    lon_col        = COLUMNS["longitude"]
    geo_source_col = COLUMNS["geo_source"]
    city_col       = COLUMNS["city"]
    state_col      = COLUMNS["state"]
    loc_lower      = location_query.strip().lower()

    located   = []   # records with usable coordinates, within radius
    unlocated = []   # geo_source=UNKNOWN, city/state matches the query location

    for _, row in df.iterrows():
        ms = match_score(row, keywords)
        if ms == 0:
            continue

        geo_src = str(row.get(geo_source_col, "UNKNOWN") or "UNKNOWN")

        flat_raw = row.get(lat_col)
        flon_raw = row.get(lon_col)
        try:
            flat, flon = float(flat_raw), float(flon_raw)
            has_coords = not (pd.isna(flat) or pd.isna(flon))
        except (ValueError, TypeError):
            has_coords = False

        ev = evaluate_evidence(row, keywords)
        base = {
            "name":        row.get(COLUMNS["name"], "Unknown"),
            "city":        row.get(city_col, ""),
            "state":       row.get(state_col, ""),
            "match_score": ms,
            "evidence":    ev,
            "geo_source":  geo_src,
            "phone":       str(row.get(COLUMNS.get("phone",    "phone"),    "") or ""),
            "website":     str(row.get(COLUMNS.get("website",  "website"),  "") or ""),
            "org_type":    str(row.get(COLUMNS.get("org_type", "organization_type"), "") or ""),
        }

        if has_coords:
            dist = haversine_km(lat, lon, flat, flon)
            if dist > max_distance_km:
                continue
            located.append({**base, "distance_km": round(dist, 1), "lat": flat, "lon": flon})
        elif geo_src == "UNKNOWN":
            # No coordinates could be resolved; include only if the facility's
            # city or state matches the typed location (name-based fallback).
            fac_city  = str(row.get(city_col,  "") or "").lower()
            fac_state = str(row.get(state_col, "") or "").lower()
            if (fac_city and (fac_city in loc_lower or loc_lower in fac_city)) or \
               (fac_state and (fac_state in loc_lower or loc_lower in fac_state)):
                unlocated.append({**base, "distance_km": None, "lat": None, "lon": None})

    located.sort(key=lambda r: (-r["match_score"], r["distance_km"]))
    unlocated.sort(key=lambda r: -r["match_score"])
    combined = located + unlocated

    meta = {
        "resolved_location":   matched_city,
        "location_match_type": match_type,
        "care_need":           need_key,
        "keywords":            keywords,
        "total_matches":       len(combined),
        "located_count":       len(located),
        "unlocated_count":     len(unlocated),
        "max_distance_km":     max_distance_km,
        "search_lat":          lat,
        "search_lon":          lon,
    }
    return combined[:top_n], meta
