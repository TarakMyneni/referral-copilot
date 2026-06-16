import re

import pandas as pd

from .config import COLUMNS, CARE_NEED_SYNONYMS, EVIDENCE_TEXT_FIELDS
from .geo import haversine_km, resolve_location
from .evidence import evaluate_evidence


_SEPARATOR_WORDS = re.compile(r"\b(?:near|in|around|close to|nearby|at)\b", re.IGNORECASE)

# Databricks Foundation Model endpoint — swap here to try a different model.
# Options (most powerful first):
#   databricks-claude-3-7-sonnet          ← best for structured extraction + multilingual
#   databricks-meta-llama-3-1-405b-instruct
#   databricks-meta-llama-3-3-70b-instruct  (previous default)
_LLM_MODEL = "databricks-claude-3-7-sonnet"

_LLM_PROMPT = """\
You are a medical query parser for an Indian healthcare referral app.
If the query is not in English, translate it to English first, then extract the fields.

Extract three things from the user query:
- care_need : the medical specialty or condition in English (see mapping below)
- location  : the Indian city, town, district, or 6-digit PIN code in English
- org_type  : "government" if they explicitly want a government/public hospital, "private" if they want private, "" otherwise

Symptom → specialty mapping (always use the specialty, not the raw symptom):
  fever, cold, cough, body ache, headache, weakness, vomiting, diarrhea, rash → "general medicine"
  diabetes, blood pressure, thyroid, routine checkup                           → "general medicine"
  leg injury, fracture, broken bone, sprain, joint pain, back pain             → "orthopedics"
  chest pain, heart problem, palpitations                                       → "cardiology"
  eye problem, eye pain, eye injury, vision                                     → "ophthalmology"
  pregnancy, delivery, antenatal, maternity                                     → "maternity"
  kidney, dialysis                                                               → "dialysis"
  seizure, brain, stroke, paralysis, numbness                                   → "neurology"
  child sick, infant, newborn                                                   → "pediatrics"
  stomach pain, acidity, jaundice, liver                                        → "gastroenterology"
  skin rash, eczema, skin infection                                             → "dermatology"
  breathing difficulty, asthma, lung                                            → "pulmonology"
  urine burning, kidney stone, urinary                                          → "urology"
  emergency, accident, unconscious, heavy bleeding                              → "emergency"

Rules:
- Strip filler words from care_need: remove "hospital", "clinic", "show me", "find", "i have", "i need", "government", "private" etc.
- Always return care_need and location in English regardless of input language.
- If a value is absent or unclear return an empty string for that key.
- Return ONLY a valid JSON object — no explanation, no markdown.

Examples:
{{"query":"dialysis near Jaipur"}} → {{"care_need":"dialysis","location":"Jaipur","org_type":""}}
{{"query":"I have fever in Nagpur"}} → {{"care_need":"general medicine","location":"Nagpur","org_type":""}}
{{"query":"i have leg injury within 5 km of Hyderabad"}} → {{"care_need":"orthopedics","location":"Hyderabad","org_type":""}}
{{"query":"government hospitals Hyderabad eye care"}} → {{"care_need":"ophthalmology","location":"Hyderabad","org_type":"government"}}
{{"query":"something in my eye near Chennai"}} → {{"care_need":"ophthalmology","location":"Chennai","org_type":""}}
{{"query":"need urgent help near Raipur chest problem"}} → {{"care_need":"cardiology","location":"Raipur","org_type":""}}
{{"query":"private hospital knee surgery Pune"}} → {{"care_need":"orthopedics","location":"Pune","org_type":"private"}}
{{"query":"headache and vomiting near Bhopal"}} → {{"care_need":"general medicine","location":"Bhopal","org_type":""}}
{{"query":"హైదరాబాద్ దగ్గర కంటి చికిత్స"}} → {{"care_need":"ophthalmology","location":"Hyderabad","org_type":""}}
{{"query":"जयपुर के पास डायलिसिस"}} → {{"care_need":"dialysis","location":"Jaipur","org_type":""}}
{{"query":"बुखार है मुंबई में"}} → {{"care_need":"general medicine","location":"Mumbai","org_type":""}}

Query: {query}
"""


def _llm_parse(text):
    """
    Use Databricks Foundation Model to extract care_need, location, org_type.
    Returns (care_need, location, org_type).
    """
    import json, os, requests
    try:
        host  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not host or not token:
            return "", "", ""

        resp = requests.post(
            f"{host}/serving-endpoints/{_LLM_MODEL}/invocations",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json={
                "messages": [{"role": "user",
                              "content": _LLM_PROMPT.format(query=text)}],
                "max_tokens": 80,
                "temperature": 0,
            },
            timeout=10,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            parsed   = json.loads(json_match.group())
            care     = parsed.get("care_need", "").strip()
            location = parsed.get("location",  "").strip()
            org      = parsed.get("org_type",  "").strip().lower()
            print(f"[LLM] '{text}' → care='{care}' loc='{location}' org='{org}'")
            return care, location, org
    except Exception as exc:
        print(f"[LLM] Parse failed: {exc}")
    return "", "", ""


def _regex_fallback(text, centroids):
    """
    Offline fallback when the LLM endpoint is unavailable.
    Handles only simple structured patterns.
    """
    # PIN code
    pin_m = re.search(r'\b(\d{6})\b', text)
    if pin_m:
        pin       = pin_m.group(1)
        remaining = (text[:pin_m.start()] + text[pin_m.end():]).strip()
        need      = _SEPARATOR_WORDS.sub("", remaining).strip()
        return need or text, pin, ""

    # "X near/in/around Y" with a known city name
    text_lower = text.lower()
    city_keys  = [k for k in centroids if not k.isdigit() and len(k) >= 3]
    for city in sorted(city_keys, key=len, reverse=True):
        if city in text_lower:
            idx          = text_lower.find(city)
            original_loc = text[idx:idx + len(city)]
            remaining    = (text[:idx] + text[idx + len(city):]).strip()
            need         = _SEPARATOR_WORDS.sub("", remaining).strip()
            return need or text, original_loc, ""

    # Plain separator pattern as last resort
    m = re.search(r"^(.*?)\s+(?:near|in|around|close to|nearby|at)\s+(.+)$",
                  text, re.IGNORECASE)
    if m:
        need, loc = m.group(1).strip(), m.group(2).strip()
        if need and loc:
            return need, loc, ""

    return text, "", ""


def parse_combined_query(text, centroids):
    """
    Parse a free-text query into (care_need, location, org_type).

    LLM is the primary parser — it handles any natural language, symptoms,
    and government/private intent.
    _regex_fallback is used only when the LLM endpoint is unreachable.
    """
    text = (text or "").strip()
    if not text:
        return "", "", ""

    care_need, location, org_type = _llm_parse(text)
    if care_need or location:
        return care_need, location, org_type

    # LLM unavailable — best-effort offline fallback
    print("[LLM] Falling back to regex parser")
    return _regex_fallback(text, centroids)


def normalize_care_need(text):
    """
    Map a care-need string to a canonical name + keyword list.
    The LLM already returns clean care-need text, so this is mostly a
    keyword-expansion step. Falls back to the raw text as a keyword if
    no synonym group matches.
    """
    text_l = text.strip().lower()
    if not text_l:
        return "", []

    # Direct key match
    if text_l in CARE_NEED_SYNONYMS:
        return text_l, CARE_NEED_SYNONYMS[text_l]

    # Score every care need by how many of its keywords appear in the text.
    # Best score wins — e.g. "broken leg" scores higher for orthopedics than emergency.
    best_need, best_kws, best_score = None, None, 0
    for need, kws in CARE_NEED_SYNONYMS.items():
        score = sum(1 for kw in kws if kw in text_l)
        if score > best_score:
            best_score = score
            best_need  = need
            best_kws   = kws

    if best_need:
        return best_need, best_kws

    return text_l, [text_l]


def match_score(row, keywords):
    """Count how many distinct evidence fields contain at least one keyword."""
    keywords_lower = [k.lower() for k in keywords]
    score = 0
    for field in EVIDENCE_TEXT_FIELDS:
        col  = COLUMNS.get(field, field)
        text = str(row.get(col, "")).lower()
        if text and any(kw in text for kw in keywords_lower):
            score += 1
    return score


def search_facilities(df, location_query, care_need_query, centroids,
                      max_distance_km=150, top_n=10):
    """
    Returns (results, meta).
    results: list of dicts, sorted best-first.
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
    loc_lower      = matched_city.lower() if matched_city else location_query.strip().lower()

    located   = []
    unlocated = []

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

        ev   = evaluate_evidence(row, keywords)
        base = {
            "name":        row.get(COLUMNS["name"], "Unknown"),
            "city":        row.get(city_col, ""),
            "state":       row.get(state_col, ""),
            "match_score": ms,
            "evidence":    ev,
            "geo_source":  geo_src,
            "phone":       str(row.get(COLUMNS.get("phone",       "phone"),       "") or ""),
            "website":     str(row.get(COLUMNS.get("website",     "website"),     "") or ""),
            "org_type":    str(row.get(COLUMNS.get("org_type",    "organization_type"), "") or ""),
            "description": str(row.get(COLUMNS.get("description", "description"), "") or ""),
            "num_doctors": str(row.get(COLUMNS.get("num_doctors", "num_doctors"), "") or ""),
            "capacity":    str(row.get(COLUMNS.get("capacity",    "capacity"),    "") or ""),
        }

        if has_coords:
            dist = haversine_km(lat, lon, flat, flon)
            if dist > max_distance_km:
                continue
            located.append({**base, "distance_km": round(dist, 1), "lat": flat, "lon": flon})
        elif geo_src == "UNKNOWN":
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
