import base64
import csv
import io
import os
import re
import threading
import uuid

import folium
import gradio as gr
import pandas as pd

from src.config import COLUMNS
from src.geo import resolve_location, build_postcode_centroids, preload_nominatim_cache, set_geo_save_callback
from src.ranking import parse_combined_query, _LLM_MODEL
from src.evidence import evaluate_evidence, trust_label
from src import agent as supervisor
from src import feedback as feedback_store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SILVER_TABLE    = "mediguide.referral_copilot.facilities_silver"
CENTROIDS_TABLE = "mediguide.referral_copilot.location_centroids"
GEO_CACHE_TABLE = "mediguide.referral_copilot.geo_cache"

_BASE    = os.path.dirname(os.path.abspath(__file__))
_NEEDED  = set(COLUMNS.values())
_SESSION = str(uuid.uuid4())

_DATA_DIR       = os.path.join(_BASE, "data")
_FACILITIES_CSV = os.path.join(_DATA_DIR, "facilities_silver.csv")
_CENTROIDS_CSV  = os.path.join(_DATA_DIR, "location_centroids.csv")

GRN_DK   = "#27500A"
GRN_MID  = "#3B6D11"
GRN_LT   = "#639922"
GRN_PALE = "#EAF3DE"
BG_PAGE  = "#F7FAF3"
BG_CARD  = "#FFFFFF"
BG_GOVT  = "#E6F1FB"
BORDER   = "#D3D1C7"
BORDER_G = "#C0DD97"
TXT_PRI  = "#2C2C2A"
TXT_SEC  = "#5F5E5A"
TXT_MUT  = "#888780"
AMBER    = "#854F0B"
RED_FLAG = "#993C1D"
GOVT_CLR = "#185FA5"

TRUST_CFG = {
    "✓ Strong evidence":    ("strong",  GRN_PALE,  "#97C459", GRN_DK,    "✓ Strong"),
    "◐ Partial evidence":   ("partial", "#FAEEDA",  "#FAC775", "#633806", "◐ Partial"),
    "⚠️ Needs verification": ("verify",  "#FAECE7",  "#F0997B", "#712B13", "⚠ Verify"),
}
_TRUST_DEFAULT = ("verify", "#FAECE7", "#F0997B", "#712B13", "⚠ Verify")

_TRUST_ORDER = {
    "✓ Strong evidence":    0,
    "◐ Partial evidence":   1,
    "⚠️ Needs verification": 2,
}

def _trust_tier(r):
    return _TRUST_ORDER.get(trust_label(r["evidence"]), 3)

_DEFAULT_SORT = "Evidence first"

FIELD_LABELS = {
    "specialties": "specialties", "description": "description",
    "capability": "capability", "procedure": "procedure",
    "equipment": "equipment", "num_doctors": "No. of doctors",
    "capacity": "capacity", "year_established": "Year established",
    "source_urls": "Source URL",
}

# ---------------------------------------------------------------------------
# Databricks SDK query
# ---------------------------------------------------------------------------

def _sdk_query(statement, wait=None):
    import time
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    w = WorkspaceClient()
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouse found.")
    wh_id = warehouses[0].id
    print(f"[SDK] Using warehouse: {warehouses[0].name} ({wh_id})")

    r = w.statement_execution.execute_statement(
        warehouse_id=wh_id, statement=statement, row_limit=20000,
    )
    terminal = {StatementState.SUCCEEDED, StatementState.FAILED,
                StatementState.CANCELED, StatementState.CLOSED}
    for _ in range(120):
        if r.status.state in terminal:
            break
        time.sleep(5)
        r = w.statement_execution.get_statement(r.statement_id)

    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Query failed ({r.status.state}): {r.status.error}")

    col_names = [c.name for c in r.manifest.schema.columns]
    rows = []
    chunk = r.result
    while chunk:
        if chunk.data_array:
            rows.extend(chunk.data_array)
        if chunk.next_chunk_index is None:
            break
        chunk = w.statement_execution.get_statement_result_chunk_n(
            statement_id=r.statement_id, chunk_index=chunk.next_chunk_index,
        )
    return col_names, rows

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_TEXT_COLS  = {"description", "capability", "procedure_text", "equipment", "source_urls"}
_TEXT_LIMIT = 300

def _load_silver():
    parts = []
    for c in sorted(_NEEDED):
        if not c:
            continue
        parts.append(f"SUBSTRING(`{c}`, 1, {_TEXT_LIMIT}) AS `{c}`"
                     if c in _TEXT_COLS else f"`{c}`")
    cols, rows = _sdk_query(f"SELECT {', '.join(parts)} FROM {SILVER_TABLE}")
    df = pd.DataFrame(rows, columns=cols)
    print(f"[App] Loaded {len(df):,} facilities from Delta")
    return df

def _load_centroids_delta():
    cols, rows = _sdk_query(f"SELECT level, name_key, lat, lon FROM {CENTROIDS_TABLE}")
    df_c = pd.DataFrame(rows, columns=cols)
    df_c["lat"] = pd.to_numeric(df_c["lat"], errors="coerce")
    df_c["lon"] = pd.to_numeric(df_c["lon"], errors="coerce")
    df_c = df_c.dropna(subset=["lat", "lon"])
    level_order = {"state": 0, "region": 1, "division": 2, "district": 3, "city": 4}
    df_c["_order"] = df_c["level"].map(level_order).fillna(0)
    df_c = df_c.sort_values("_order")
    return {row["name_key"]: {"lat": row["lat"], "lon": row["lon"]}
            for _, row in df_c.iterrows()}

def _load_facilities_csv():
    _df = pd.read_csv(_FACILITIES_CSV, dtype=str)
    for col in (COLUMNS["latitude"], COLUMNS["longitude"]):
        if col in _df.columns:
            _df[col] = pd.to_numeric(_df[col], errors="coerce")
    return _df

def _load_geo_cache():
    """Load Nominatim cache from Delta into memory so cold starts are fast."""
    try:
        _sdk_query(f"""
            CREATE TABLE IF NOT EXISTS {GEO_CACHE_TABLE} (
                query_lower STRING,
                lat         DOUBLE,
                lon         DOUBLE,
                city_name   STRING
            )
        """)
    except Exception:
        pass
    try:
        # Try loading with city_name; if column missing (old table), fall back
        try:
            cols, rows = _sdk_query(
                f"SELECT query_lower, lat, lon, city_name FROM {GEO_CACHE_TABLE}"
            )
            entries = [
                (r[0], r[1], r[2], r[3] or "")
                for r in rows if r[1] is not None and r[2] is not None
            ]
        except Exception:
            cols, rows = _sdk_query(
                f"SELECT query_lower, lat, lon FROM {GEO_CACHE_TABLE}"
            )
            entries = [
                (r[0], r[1], r[2], "")
                for r in rows if r[1] is not None and r[2] is not None
            ]
            try:
                _sdk_query(
                    f"ALTER TABLE {GEO_CACHE_TABLE} ADD COLUMN city_name STRING"
                )
            except Exception:
                pass
        preload_nominatim_cache(entries)
        print(f"[App] Geo cache loaded: {len(entries)} cities pre-cached")
    except Exception as e:
        print(f"[App] Geo cache skipped: {e}")


def _save_geo_cache_entry(query_lower, lat, lon, city_name=""):
    """Write a new Nominatim result to Delta in a background thread."""
    def _write():
        try:
            q = query_lower.replace("'", "''")
            c = (city_name or "").replace("'", "''")
            _sdk_query(f"""
                INSERT INTO {GEO_CACHE_TABLE} (query_lower, lat, lon, city_name)
                SELECT '{q}', {lat}, {lon}, '{c}'
                WHERE NOT EXISTS (
                    SELECT 1 FROM {GEO_CACHE_TABLE} WHERE query_lower = '{q}'
                )
            """)
        except Exception as e:
            print(f"[App] Geo cache write failed: {e}")
    threading.Thread(target=_write, daemon=True).start()


def _load_centroids_csv():
    df_c = pd.read_csv(_CENTROIDS_CSV, dtype=str)
    df_c["lat"] = pd.to_numeric(df_c["lat"], errors="coerce")
    df_c["lon"] = pd.to_numeric(df_c["lon"], errors="coerce")
    df_c = df_c.dropna(subset=["lat", "lon"])
    level_order = {"state": 0, "region": 1, "division": 2, "district": 3, "city": 4}
    df_c["_order"] = df_c["level"].map(level_order).fillna(0)
    df_c = df_c.sort_values("_order")
    return {row["name_key"]: {"lat": row["lat"], "lon": row["lon"]}
            for _, row in df_c.iterrows()}

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

df                = pd.DataFrame()
centroids         = {}
_STARTUP_ERROR    = None
_data_ready       = False


def _background_load():
    global df, centroids, _STARTUP_ERROR, _data_ready
    try:
        if os.path.exists(_FACILITIES_CSV):
            df = _load_facilities_csv()
            print(f"[App] Facilities from CSV ({len(df):,})")
        else:
            df = _load_silver()
        if os.path.exists(_CENTROIDS_CSV):
            centroids = _load_centroids_csv()
            print(f"[App] Centroids from CSV ({len(centroids):,})")
        else:
            centroids = _load_centroids_delta()
        try:
            feedback_store.load(_sdk_query)
        except Exception as fe:
            print(f"[App] Feedback skipped: {fe}")
        _load_geo_cache()
        set_geo_save_callback(_save_geo_cache_entry)
        # Enrich centroids with PIN codes derived from facilities coordinates.
        # Uses the dataset itself — no external postcode directory needed.
        try:
            pin_c = build_postcode_centroids(
                df,
                COLUMNS.get("postcode", "postcode"),
                COLUMNS["latitude"],
                COLUMNS["longitude"],
            )
            centroids = {**centroids, **pin_c}
        except Exception as _pe:
            print(f"[App] PIN centroids skipped: {_pe}")
        _data_ready = True
        print(f"[App] Ready — {len(df):,} facilities, {len(centroids):,} locations")
    except Exception as _e:
        import traceback
        _STARTUP_ERROR = f"{type(_e).__name__}: {_e}\n\n{traceback.format_exc()}"
        print(f"[App] STARTUP FAILED:\n{_STARTUP_ERROR}")

threading.Thread(target=_background_load, daemon=True).start()

# Load logo once at startup and embed as base64 to avoid static-file serving issues
_LOGO_B64 = ""
try:
    with open(os.path.join(_BASE, "logo.jpg"), "rb") as _f:
        _LOGO_B64 = base64.b64encode(_f.read()).decode("ascii")
except Exception:
    pass

# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _s(v):
    s = str(v or "")
    return "" if s in ("nan", "None") else s

_LIST_FIELDS = {"specialties", "capability", "procedure", "equipment"}

def _fmt_field(field, raw):
    """Format a raw DB field value for readable display in the chip popup.

    Silver expands camelCase → space-separated words, so we get strings like
    'gynecology And Obstetrics, neonatology Perinatal Medicine'.
    We normalise capitalisation and format list fields as readable bullet lines.
    """
    import re
    if not raw or not raw.strip():
        return "—"
    text = " ".join(raw.split())   # collapse whitespace

    if field == "description":
        # Paragraph — capitalise first letter, truncate, HTML-escape
        text = text[0].upper() + text[1:] if text else text
        out  = (text[:320] + "…") if len(text) > 320 else text
        return out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # List field: split on commas/semicolons, clean each item
    parts = [p.strip() for p in re.split(r"[,;]+", text) if p.strip()]
    seen, clean = set(), []
    for p in parts:
        item = p.lower().title()          # fix "And", "Of" → "And", "Of" (acceptable)
        key  = item.lower()
        if key not in seen:
            seen.add(key)
            clean.append(item)

    if not clean:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    shown = clean[:10]
    extra = len(clean) - 10
    lines = "".join(
        f'<span style="display:inline-block;background:#EAF3DE;border:0.5px solid #C0DD97;'
        f'border-radius:10px;padding:2px 8px;font-size:11px;color:#27500A;'
        f'margin:2px 3px 2px 0;">{item}</span>'
        for item in shown
    )
    if extra > 0:
        lines += (
            f'<span style="font-size:10px;color:#888780;margin-left:2px;">+{extra} more</span>'
        )
    return lines

def _is_govt(r):
    ot = _s(r.get("org_type", "")).lower()
    return any(w in ot for w in ("government", "govt", "public", "municipal", "district"))

_BUILDING_ICON = (
    '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" style="opacity:0.55">'
    '<rect x="3" y="8" width="18" height="13" rx="1" stroke="{c}" stroke-width="1.5"/>'
    '<rect x="9" y="8" width="6" height="6" stroke="{c}" stroke-width="1.2"/>'
    '<rect x="10" y="16" width="4" height="5" fill="{c}" opacity="0.4"/>'
    '<path d="M3 12h18" stroke="{c}" stroke-width="0.8" opacity="0.35"/>'
    '<path d="M12 3 L4 8 L20 8 Z" stroke="{c}" stroke-width="1.2" fill="none"/>'
    '</svg>'
)

# ---------------------------------------------------------------------------
# Inline JS helpers
# Svelte's {@html} does NOT execute <script> tags, so all bridge calls must
# be inline onclick attributes — those fire unconditionally via the browser's
# event system, no global function definitions required.
# ---------------------------------------------------------------------------

# Bridge element selectors
_QA = "#h-query textarea,#h-query input"
_RA = "#h-rad textarea,#h-rad input"
_FA = "#h-filter textarea,#h-filter input"
_SA = "#h-sort textarea,#h-sort input"
_BA = "#h-bm-id textarea,#h-bm-id input"
_RMA = "#h-rm-idx textarea,#h-rm-idx input"
_CLA = "#h-clear-tx textarea,#h-clear-tx input"
_EXA = "#h-export-tx textarea,#h-export-tx input"

def _jtap(val_sel, jsval, btn_id):
    """Set bridge textbox value then programmatically click the hidden trigger button.
    Using button.click() is more reliable than synthetic keydown in Gradio/Svelte."""
    bsel = f"#{btn_id} button,button#{btn_id}"
    return (
        f"(function(){{"
        f"var e=document.querySelector('{val_sel}');"
        f"if(!e)return;"
        f"e.value={jsval};"
        f"e.dispatchEvent(new Event('input',{{bubbles:true}}));"
        f"setTimeout(function(){{"
        f"var b=document.querySelector('{bsel}');"
        f"if(b)b.click();"
        f"}},80);"
        f"}})();"
    )

def _jclick(btn_id):
    """Inline JS: click a hidden trigger button (no value to pass)."""
    bsel = f"#{btn_id} button,button#{btn_id}"
    return f"(function(){{var b=document.querySelector('{bsel}');if(b)b.click();}})();"

# Confirm-and-reset JS — used on logo/app-name click
_RESET_JS = (
    "if(confirm('Clear all results and saved facilities and start fresh?')){"
    "var inp=document.getElementById('vi-query');if(inp)inp.value='';"
    "var b=document.querySelector('#h-reset-btn button,button#h-reset-btn');"
    "if(b)b.click();}"
)

# Inline search action — reads vi-query / vi-radius, sets bridge, clicks search.
_JS_SEARCH_INLINE = (
    "(function(){"
    "var inp=document.getElementById('vi-query');"
    "var rinp=document.getElementById('vi-radius');"
    "var q=inp?inp.value:'';"
    "var r=rinp?rinp.value:'50';"
    "if(inp)inp.value='';"  # clear immediately so user sees blank box right away
    "var ra=document.querySelector('#h-rad textarea,#h-rad input');"
    "var qa=document.querySelector('#h-query textarea,#h-query input');"
    "if(ra){ra.value=r;ra.dispatchEvent(new Event('input',{bubbles:true}));}"
    "if(qa){qa.value=q;qa.dispatchEvent(new Event('input',{bubbles:true}));}"
    "setTimeout(function(){"
    "var b=document.querySelector('#h-search-btn button,button#h-search-btn');"
    "if(b)b.click();"
    "},80);"
    "})()"
)

def _js_suggestion(q):
    """Inline JS for a suggestion pill: pre-fills vi-query and triggers search."""
    sq = q.replace("'", "\\'")
    return (
        f"(function(){{"
        f"var inp=document.getElementById('vi-query');if(inp)inp.value='{sq}';"
        f"var ra=document.querySelector('#h-rad textarea,#h-rad input');"
        f"var qa=document.querySelector('#h-query textarea,#h-query input');"
        f"if(ra){{ra.value='5';ra.dispatchEvent(new Event('input',{{bubbles:true}}));}}"
        f"if(qa){{qa.value='{sq}';"
        f"qa.dispatchEvent(new Event('input',{{bubbles:true}}));}}"
        f"setTimeout(function(){{"
        f"var b=document.querySelector('#h-search-btn button,button#h-search-btn');"
        f"if(b)b.click();"
        f"}},80);"
        f"}})();"
    )

# Bridge selectors for the new compare / checklist actions
_CMPA = "#h-cmp-id textarea,#h-cmp-id input"
_CHKA = "#h-chk-id textarea,#h-chk-id input"

# ---------------------------------------------------------------------------
# Care-need checklists
# ---------------------------------------------------------------------------

CARE_CHECKLISTS = {
    "maternity": [
        "Aadhaar card (patient + attendant)",
        "Health insurance / Ayushman Bharat card",
        "Mother & Child protection card",
        "Antenatal care records & ultrasound reports",
        "Blood group certificate",
        "Previous gynaecologist prescription",
        "Emergency contact number",
    ],
    "dialysis": [
        "Aadhaar card",
        "Health insurance card",
        "Nephrologist referral letter",
        "Recent lab reports (creatinine, BUN, electrolytes)",
        "Dialysis access records (fistula / catheter)",
        "Current medication list",
        "Previous dialysis session records",
    ],
    "cardiology": [
        "Aadhaar card",
        "Health insurance card",
        "ECG reports (latest)",
        "Echocardiogram report",
        "Angiography reports (if done previously)",
        "Current medication list",
        "Cardiologist referral letter",
    ],
    "emergency": [
        "Aadhaar card",
        "Health insurance card",
        "Any available medical history summary",
        "Current medications list",
        "Emergency contact details",
        "Blood group information",
    ],
    "oncology": [
        "Aadhaar card",
        "Health insurance card",
        "Biopsy / pathology reports",
        "Previous treatment records (chemo / radiation)",
        "Imaging reports (CT / MRI / PET scan)",
        "Oncologist referral letter",
        "Current medication list",
    ],
    "orthopedics": [
        "Aadhaar card",
        "Health insurance card",
        "X-ray / MRI reports of affected area",
        "Previous orthopaedic consultation notes",
        "Current medication list",
        "Physiotherapy records (if any)",
    ],
    "ophthalmology": [
        "Aadhaar card",
        "Health insurance card",
        "Previous eye prescription / glasses",
        "Fundus / retinal scan reports (if any)",
        "Current eye drops / medications",
        "History of eye surgeries (if any)",
    ],
    "neurology": [
        "Aadhaar card",
        "Health insurance card",
        "MRI / CT scan of brain / spine",
        "EEG reports (if epilepsy-related)",
        "Current medication list",
        "Neurologist referral letter",
    ],
    "pediatrics": [
        "Aadhaar card (child + parent / guardian)",
        "Health insurance card",
        "Vaccination / immunisation records",
        "Birth certificate / hospital discharge summary",
        "Growth and development records",
        "Current medication list",
    ],
    "icu": [
        "Aadhaar card",
        "Health insurance card",
        "All available medical records",
        "Current medications and dosage",
        "Blood group certificate",
        "Emergency contact details",
        "Power of attorney (if patient is incapacitated)",
    ],
    "general surgery": [
        "Aadhaar card",
        "Health insurance card",
        "Surgical referral letter",
        "Recent blood work and coagulation panel",
        "Imaging reports for the affected area",
        "Current medication list",
        "Fasting status (if elective procedure)",
    ],
    "radiology": [
        "Aadhaar card",
        "Doctor's referral / prescription for imaging",
        "Previous imaging reports (for comparison)",
        "Health insurance card",
        "Remove all metal accessories before arriving",
    ],
}

_GENERAL_CHECKLIST = [
    "Aadhaar card (patient + attendant)",
    "Health insurance / Ayushman Bharat card",
    "Any existing medical records or prescriptions",
    "Current medication list",
    "Blood group information",
    "Emergency contact details",
]


def _make_intake_qr_text(facility, care_need):
    """
    Build a mailto: URL as the QR payload.
    mailto:?subject=...&body=... is recognised as a link by ALL phone cameras
    (iOS and Android), opens the device email/Gmail app with pre-filled content.
    No server, no WhatsApp dependency — works offline once the QR is displayed.
    """
    import urllib.parse
    items = CARE_CHECKLISTS.get(care_need, _GENERAL_CHECKLIST)
    dept  = (care_need or "General").title()
    name  = facility.get("name", "Unknown")[:50]
    body_lines = [
        "SUVIDHA Incoming Referral",
        f"Facility: {name}",
        f"Department: {dept}",
        "",
        "Documents required:",
    ]
    for i, item in enumerate(items[:6], 1):
        body_lines.append(f"{i}. {item}")
    subject = f"SUVIDHA Referral – {dept}"
    body    = "\n".join(body_lines)
    return "mailto:?subject=" + urllib.parse.quote(subject) + "&body=" + urllib.parse.quote(body)


def _make_qr_svg(text, scale=3):
    """
    Generate a QR code as an inline SVG using segno.
    Falls back to api.qrserver.com if segno is not installed.
    The payload should be plain text (≤400 chars) for best scannability.
    """
    try:
        import segno, io
        qr  = segno.make(text, error="l", micro=False, boost_error=False)
        buf = io.StringIO()
        qr.save(buf, kind="svg", scale=scale, border=1,
                linecolor=GRN_DK, svgclass=None)
        svg = buf.getvalue()
        svg = svg.replace("<svg ", '<svg style="width:100%;max-width:200px;display:block;" ', 1)
        return svg
    except Exception as exc:
        print(f"[QR] segno failed ({exc}), using api.qrserver.com fallback")
        import urllib.parse
        encoded = urllib.parse.quote(text, safe="")
        return (
            f'<img src="https://api.qrserver.com/v1/create-qr-code/'
            f'?data={encoded}&size=200x200&margin=2&ecc=L&color=27500A" '
            f'width="200" height="200" '
            f'style="display:block;border-radius:6px;border:1px solid {BORDER_G};" '
            f'alt="QR code" />'
        )


# ---------------------------------------------------------------------------
# Conversational AI chat
# ---------------------------------------------------------------------------

_CHAT_SYSTEM = """\
You are Suvidha, a warm and caring Indian healthcare assistant helping people find the right hospital quickly.

CRITICAL — location and care-need priority rules:
1. If the CURRENT user message names a city or location, ALWAYS use that city — never use an older city from history.
2. If the CURRENT message has NO location, check the conversation history and use the most recent one.
3. Same rule for care need: current message wins; history fills the gap only when current message is silent.

━━━ DECISION LOGIC ━━━

1. SEARCH — when you know BOTH (a) an Indian location AND (b) what medical care is needed
   (from current or past messages), respond with ONLY this JSON (no other text):
   {"action":"search","care_need":"<specialty in English>","location":"<city>","org_type":"<government|private|>"}

2. FILTER — user wants to narrow current results to govt or private:
   {"action":"filter","org_type":"<government|private|all>"}

3. ASK — if location OR care need is still unclear, ask ONE warm, conversational question.
   Do NOT ask about something already answered in the history.

━━━ SYMPTOM → SPECIALTY MAPPING ━━━
Always map symptoms to the right specialty for care_need. Use these as a guide:

  Serious / urgent:
    chest pain, heart attack, palpitations     → "cardiology"
    can't breathe, severe breathlessness       → "pulmonology"
    unconscious, major accident, heavy bleeding → "emergency"
    seizure, sudden numbness, stroke signs     → "neurology"

  Injuries:
    leg injury, fracture, broken bone, sprain  → "orthopedics"
    arm injury, shoulder pain, joint pain      → "orthopedics"
    eye injury, foreign body in eye            → "ophthalmology"

  Specialist needs:
    kidney failure, dialysis                   → "dialysis"
    pregnancy, delivery, maternity             → "maternity"
    child / infant / kid / kids / baby sick    → "pediatrics"
    children's hospital / kid hospital / baby hospital → "pediatrics"
    ear infection / ear pain / earache / hearing problem → "ent"
    sore throat / throat pain / tonsil pain / sinusitis  → "ent"
    nose bleed / nasal problem / stuffy nose              → "ent"
    skin rash, eczema                          → "dermatology"
    stomach pain, acidity, jaundice            → "gastroenterology"
    urinary burning, kidney stone              → "urology"
    mental health, depression, anxiety         → "psychiatry"

  Mild / common (when symptoms are not clearly serious):
    fever, cold, cough, body ache, headache    → "general medicine"
    weakness, vomiting, diarrhea, rash         → "general medicine"
    routine checkup, blood pressure, diabetes  → "general medicine"
    any vague symptom without enough context   → ask ONE question to understand severity,
                                                 then route to the right specialty or "general medicine"

━━━ IMPORTANT RULES ━━━
- "hospitals near X" / "clinic near X" / "healthcare near X" — set care_need="" (empty) and search; do NOT ask for a condition.
- Only ask for a condition when the user's message has NO location either (truly nothing to go on).
- Neighborhoods like "Hebbal Bangalore" → extract main city (Bangalore).
- "within X km of Y" / "near Y" / "in Y" / "around Y" — Y is the location.
- org_type: "government" for govt/sarkari/public/sarkar, "private" for private, "" otherwise.
- Keep ASK responses warm, short, and conversational — like a caring friend, not a form.

━━━ EXAMPLES ━━━
  user: "maternity near Delhi"            → {"action":"search","care_need":"maternity","location":"Delhi","org_type":""}
  user: "dialysis" (prior loc=Delhi)      → {"action":"search","care_need":"dialysis","location":"Delhi","org_type":""}
  user: "show only govt ones"             → {"action":"filter","org_type":"government"}
  user: "eye problem near Chennai"        → {"action":"search","care_need":"ophthalmology","location":"Chennai","org_type":""}
  user: "leg injury within 5 km hyderabad" → {"action":"search","care_need":"orthopedics","location":"hyderabad","org_type":""}
  user: "i have fever near jaipur"        → {"action":"search","care_need":"general medicine","location":"jaipur","org_type":""}
  user: "i have fever"                    → ask "How long have you had the fever, and is it mild or quite high? Also, which city are you in?"
  user: "headache for 2 days, i am in pune" → {"action":"search","care_need":"general medicine","location":"pune","org_type":""}
  user: "severe headache suddenly near mumbai" → {"action":"search","care_need":"neurology","location":"mumbai","org_type":""}
  user: "govt hospitals for dialysis Jaipur" → {"action":"search","care_need":"dialysis","location":"Jaipur","org_type":"government"}
  user: "hospitals near Pune"             → {"action":"search","care_need":"","location":"Pune","org_type":""}
  user: "clinics in Chennai"             → {"action":"search","care_need":"","location":"Chennai","org_type":""}
  user: "I have chest pain near Nagpur"   → {"action":"search","care_need":"cardiology","location":"nagpur","org_type":""}
  [history has Delhi search] user: "now check Shimla for maternity" → {"action":"search","care_need":"maternity","location":"Shimla","org_type":""}
  [history has Mumbai search] user: "what about dialysis in Chennai?" → {"action":"search","care_need":"dialysis","location":"Chennai","org_type":""}
"""


def _llm_chat(user_msg, history):
    """
    Multi-turn LLM conversation. history is a list of (user, assistant) string pairs.
    Returns (response, is_action).
      is_action=True  → response is a parsed action dict
      is_action=False → response is a plain conversational string
      response=None   → LLM unavailable
    """
    import json as _json, os, requests as _req
    try:
        host  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not host or not token:
            return None, False

        msgs = [{"role": "system", "content": _CHAT_SYSTEM}]
        for u, a in history:
            msgs.append({"role": "user",      "content": str(u)})
            msgs.append({"role": "assistant", "content": str(a)})
        msgs.append({"role": "user", "content": user_msg})

        resp = _req.post(
            f"{host}/serving-endpoints/{_LLM_MODEL}/invocations",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"messages": msgs, "max_tokens": 150, "temperature": 0.3},
            timeout=12,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[Chat] '{user_msg[:60]}' → '{raw[:120]}'")

        hit = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if hit:
            try:
                action = _json.loads(hit.group())
                if "action" in action:
                    return action, True
            except Exception:
                pass
        return raw, False
    except Exception as exc:
        print(f"[Chat] LLM error: {exc}")
        return None, False


def _chat_thread_html(history):
    """Render chat history as a styled bubble thread. Returns '' when empty."""
    if not history:
        return ""

    bot_icon = (
        f'<div style="width:22px;height:22px;border-radius:50%;background:{GRN_MID};'
        f'display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:3px;">'
        f'<svg width="11" height="11" viewBox="0 0 40 40" fill="white">'
        f'<path d="M20 36s-14-10.5-14-20a14 14 0 0 1 28 0c0 9.5-14 20-14 20z"/>'
        f'<circle cx="20" cy="16" r="5" fill="white"/></svg></div>'
    )

    msgs = []
    for user_msg, ai_msg in history:
        u_esc = str(user_msg).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        msgs.append(
            f'<div style="display:flex;justify-content:flex-end;margin-bottom:7px;">'
            f'<div style="background:{GRN_MID};color:#fff;border-radius:18px 18px 4px 18px;'
            f'padding:8px 14px;max-width:78%;font-size:13px;line-height:1.45;">'
            f'{u_esc}</div></div>'
        )
        if ai_msg:
            # Inline bold style so it renders regardless of Gradio CSS resets
            ai_html = str(ai_msg).replace("<b>", '<b style="font-weight:700;color:inherit">')
            msgs.append(
                f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:7px;">'
                f'{bot_icon}'
                f'<div style="background:#F0F7FF;border:0.5px solid #B3D4F5;color:{TXT_PRI};'
                f'border-radius:4px 18px 18px 18px;padding:8px 14px;max-width:82%;'
                f'font-size:13px;line-height:1.5;">{ai_html}</div></div>'
            )

    return (
        f'<div id="sv-chat-thread" style="padding:10px 20px 6px;background:{BG_PAGE};'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
        + "".join(msgs) +
        f'</div>'
    )


def _checklist_modal_html(facility, care_need):
    """Render the full visit-checklist modal (position:fixed overlay)."""
    items  = CARE_CHECKLISTS.get(care_need, _GENERAL_CHECKLIST)
    dept   = (care_need or "General").title()

    qr_text = _make_intake_qr_text(facility, care_need)
    qr_svg  = _make_qr_svg(qr_text)

    items_html = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
        f'border-bottom:0.5px solid {BORDER};">'
        f'<span style="color:{GRN_MID};font-size:14px;flex-shrink:0;margin-top:1px;">&#9633;</span>'
        f'<span style="font-size:13px;color:{TXT_PRI};">{item}</span></div>'
        for item in items
    )

    close_js = _jclick("h-chk-close-btn")

    return (
        f'<div style="position:fixed;top:0;left:0;width:100%;height:100%;'
        f'background:rgba(0,0,0,0.55);z-index:9000;display:flex;'
        f'align-items:center;justify-content:center;'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
        f'<div style="background:#fff;border-radius:12px;width:min(640px,95vw);'
        f'max-height:88vh;overflow-y:auto;box-shadow:0 24px 80px rgba(0,0,0,.35);">'
        f'<div style="background:{GRN_MID};padding:16px 20px;border-radius:12px 12px 0 0;'
        f'display:flex;align-items:center;justify-content:space-between;">'
        f'<div>'
        f'<div style="font-size:15px;font-weight:700;color:#fff;">Visit Checklist</div>'
        f'<div style="font-size:12px;color:{GRN_PALE};margin-top:2px;">'
        f'{facility["name"]} &middot; {dept}</div>'
        f'</div>'
        f'<span onclick="{close_js}" style="color:#fff;font-size:22px;cursor:pointer;'
        f'line-height:1;padding:4px 8px;border-radius:4px;">&#x2715;</span>'
        f'</div>'
        f'<div style="padding:20px;display:flex;gap:24px;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:260px;">'
        f'<div style="font-size:11px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;'
        f'letter-spacing:.5px;margin-bottom:8px;">What to bring</div>'
        f'{items_html}'
        f'</div>'
        f'<div style="flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:8px;">'
        f'<div style="font-size:11px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;'
        f'letter-spacing:.5px;">Hospital Pass (QR)</div>'
        f'{qr_svg}'
        f'<div style="font-size:10px;color:{TXT_MUT};text-align:center;line-height:1.4;'
        f'max-width:170px;">Intake coordinator scans this QR to see routing &amp; requirements</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def _compare_panel_html(compare_list):
    """Render a side-by-side comparison table for 2-4 hospitals."""
    if len(compare_list) < 2:
        return ""

    clear_js = _jclick("h-cmp-clear-btn")

    def _row(label, vals, highlight=True):
        unique = {str(v).lower() for v in vals}
        cells = ""
        for v in vals:
            if highlight and len(unique) == 1:
                bg = f"background:{GRN_PALE};"
            elif highlight and len(unique) > 1:
                bg = "background:#FAEEDA;"
            else:
                bg = ""
            cells += (
                f'<td style="padding:8px 10px;font-size:12px;color:{TXT_PRI};'
                f'border-bottom:0.5px solid {BORDER};vertical-align:top;{bg}">{v or "&#8212;"}</td>'
            )
        return (
            f'<tr>'
            f'<td style="padding:8px 10px;font-size:11px;font-weight:600;color:{TXT_MUT};'
            f'background:#FAFCF7;border-bottom:0.5px solid {BORDER};white-space:nowrap;'
            f'border-right:0.5px solid {BORDER};">{label}</td>'
            f'{cells}</tr>'
        )

    def _spec_val(c):
        ms = [m for m in c["evidence"]["matching"]
              if m["field"] in ("specialties", "capability")]
        if not ms:
            return "&#8212;"
        raw   = ms[0].get("text", "")
        parts = [p.strip().lower().title()
                 for p in raw.replace(";", ",").split(",") if p.strip()][:4]
        return ", ".join(parts) or "&#8212;"

    headers = "".join(
        f'<th style="padding:10px 12px;font-size:13px;font-weight:600;color:{GRN_DK};'
        f'background:{GRN_PALE};border-bottom:2px solid {BORDER_G};text-align:left;">'
        f'{c["name"][:28]}{"&#8230;" if len(c["name"])>28 else ""}</th>'
        for c in compare_list
    )

    def _num_val(c, key):
        v = (c.get(key) or "").strip()
        try:
            return str(int(float(v))) if v else "&#8212;"
        except (ValueError, TypeError):
            return v or "&#8212;"

    rows = (
        _row("Type",        [("Government" if _is_govt(c) else "Private") for c in compare_list])
        + _row("Distance",  [f'{c.get("distance_km","&#8212;")} km' for c in compare_list])
        + _row("Trust",     [trust_label(c["evidence"]).split(" ", 1)[-1]
                             if c.get("evidence") else "&#8212;" for c in compare_list])
        + _row("Doctors",   [_num_val(c, "num_doctors") for c in compare_list])
        + _row("Capacity",  [_num_val(c, "capacity") for c in compare_list])
        + _row("Specialties", [_spec_val(c) for c in compare_list], highlight=False)
        + _row("Phone",     [(c.get("phone") or "&#8212;")[:22] for c in compare_list])
        + _row("City",      [f'{c.get("city","")}, {c.get("state","")}' for c in compare_list])
    )

    return (
        f'<div style="background:{BG_CARD};border:1px solid {BORDER_G};border-radius:10px;'
        f'margin:0 0 14px 0;overflow:hidden;'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
        f'<div style="background:{GRN_PALE};padding:10px 16px;display:flex;'
        f'align-items:center;justify-content:space-between;">'
        f'<span style="font-size:13px;font-weight:600;color:{GRN_DK};">'
        f'Comparing {len(compare_list)} hospitals &#x2014; '
        f'<span style="font-size:11px;font-weight:400;color:{GRN_MID};">'
        f'green = same &nbsp; amber = different</span></span>'
        f'<span onclick="{clear_js}" style="font-size:11px;color:{TXT_MUT};'
        f'cursor:pointer;text-decoration:underline;flex-shrink:0;">Clear</span></div>'
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>'
        f'<th style="padding:10px 12px;background:#F7FAF5;min-width:80px;'
        f'border-bottom:2px solid {BORDER_G};border-right:0.5px solid {BORDER};"></th>'
        f'{headers}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div></div>'
    )


_JS = """
<script>
(function(){
  if (window.__sv_bridge_loaded) return;
  window.__sv_bridge_loaded = true;

  /* Set a textbox value and dispatch input event so Svelte picks it up */
  function _set(sel, val) {
    var el = document.querySelector(sel);
    if (!el) return;
    el.value = val;
    el.dispatchEvent(new Event('input', {bubbles:true}));
  }

  /* Dispatch an Enter keydown on a textbox — triggers gr.Textbox .submit() handler */
  function _enter(sel) {
    var el = document.querySelector(sel);
    if (!el) return;
    el.dispatchEvent(new KeyboardEvent('keydown', {
      key:'Enter', code:'Enter', keyCode:13, which:13,
      bubbles:true, cancelable:true
    }));
  }

  window.sSearch = function() {
    var q = (document.getElementById('vi-query')  || {}).value || '';
    var r = (document.getElementById('vi-radius') || {}).value || '5';
    _set('#h-rad textarea, #h-rad input', r);
    _set('#h-query textarea, #h-query input', q);
    setTimeout(function(){ _enter('#h-query textarea, #h-query input'); }, 80);
  };
  window.sFilter = function(v) {
    _set('#h-filter textarea, #h-filter input', v);
    setTimeout(function(){ _enter('#h-filter textarea, #h-filter input'); }, 80);
  };
  window.sSort = function(v) {
    _set('#h-sort textarea, #h-sort input', v);
    setTimeout(function(){ _enter('#h-sort textarea, #h-sort input'); }, 80);
  };
  window.sBm = function(id) {
    _set('#h-bm-id textarea, #h-bm-id input', id);
    setTimeout(function(){ _enter('#h-bm-id textarea, #h-bm-id input'); }, 80);
  };
  window.sRm = function(idx) {
    _set('#h-rm-idx textarea, #h-rm-idx input', String(idx));
    setTimeout(function(){ _enter('#h-rm-idx textarea, #h-rm-idx input'); }, 80);
  };
  window.sClear = function() {
    _set('#h-clear-tx textarea, #h-clear-tx input', String(Date.now()));
    setTimeout(function(){ _enter('#h-clear-tx textarea, #h-clear-tx input'); }, 80);
  };
  window.sExport = function() {
    _set('#h-export-tx textarea, #h-export-tx input', String(Date.now()));
    setTimeout(function(){ _enter('#h-export-tx textarea, #h-export-tx input'); }, 80);
  };

  document.addEventListener('keydown', function(e){
    if (e.key === 'Enter' && (e.target.id === 'vi-query' || e.target.id === 'vi-radius'))
      window.sSearch();
  });
})();
</script>
"""


def _topbar_html(query, radius, n_saved, in_chat=False):
    if _LOGO_B64:
        # Crop to just the heart: image is square, heart occupies top ~65%, centered horizontally.
        # object-position 50% 28% focuses the crop window on the heart symbol.
        logo = (
            f'<img src="data:image/jpeg;base64,{_LOGO_B64}" '
            f'style="width:48px;height:48px;object-fit:cover;object-position:50% 28%;'
            f'border-radius:6px;flex-shrink:0;">'
        )
    else:
        logo = (
            '<svg viewBox="0 0 40 40" width="40" height="40" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="40" height="40" rx="8" fill="#3B6D11"/>'
            '<path d="M20 28C20 28 10 20 10 14a5 5 0 0 1 10-1 5 5 0 0 1 10 1c0 6-10 14-10 14z" fill="white"/>'
            '</svg>'
        )
    return f"""
<div style="background:#fff;border-bottom:1px solid {BORDER};padding:10px 20px;
            display:flex;align-items:center;gap:16px;flex-shrink:0;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div class="sv-topbar-logo" onclick="{_RESET_JS}"
       style="display:flex;align-items:center;gap:10px;flex-shrink:0;cursor:pointer;"
       title="Click to reset">
    {logo}
    <span style="font-size:22px;font-weight:700;color:{GRN_DK};letter-spacing:-0.3px;">SUVIDHA</span>
  </div>
  <div style="display:flex;align-items:center;border:1.5px solid #B4B2A9;border-radius:40px;
              background:#fff;flex:1;max-width:680px;height:48px;overflow:hidden;">
    <div style="flex:1;padding:6px 20px;height:100%;display:flex;flex-direction:column;
                justify-content:center;border-right:0.5px solid {BORDER};">
      <div style="font-size:9px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;
                  letter-spacing:0.5px;line-height:1;">ASK SUVIDHA</div>
      <input id="vi-query" value="" autofocus placeholder="{'Ask a follow-up…' if in_chat else 'e.g. I need dialysis near Jaipur…'}"
        onkeydown="if(event.key==='Enter'){{{_JS_SEARCH_INLINE}}}"
        style="border:none;outline:none;background:transparent;font-size:13px;
               color:{TXT_PRI};width:100%;padding:0;margin-top:2px;font-family:inherit;">
    </div>
    <div class="sv-pill-radius" style="flex:0 0 100px;padding:6px 16px;height:100%;
                display:flex;flex-direction:column;justify-content:center;
                border-right:0.5px solid {BORDER};">
      <div style="font-size:9px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;
                  letter-spacing:0.5px;line-height:1;">RADIUS</div>
      <div style="display:flex;align-items:center;gap:3px;margin-top:2px;">
        <input id="vi-radius" type="number" min="5" max="500" step="5" value="{int(radius or 5)}"
          onkeydown="if(event.key==='Enter'){{{_JS_SEARCH_INLINE}}}"
          style="border:none;outline:none;background:transparent;font-size:13px;
                 color:{TXT_PRI};width:38px;padding:0;font-family:inherit;">
        <span style="font-size:13px;color:{TXT_PRI};">km</span>
      </div>
    </div>
    <button onclick="{_JS_SEARCH_INLINE}"
      style="width:38px;height:38px;margin:5px;border-radius:50%;background:{GRN_MID};
             border:none;cursor:pointer;display:flex;align-items:center;
             justify-content:center;flex-shrink:0;padding:0;">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white"
           stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
    </button>
  </div>
  <div style="background:{GRN_PALE};border:0.5px solid {BORDER_G};border-radius:20px;
              padding:6px 14px;font-size:12px;color:{GRN_MID};white-space:nowrap;flex-shrink:0;">
    🔖 {n_saved} saved
  </div>
</div>"""


def _filterbar_html(filter_val, sort_val, n_results):
    def chip(label, val, count=None):
        active = filter_val == val
        txt = f"{label} ({count})" if count is not None else label
        if active:
            st = f"background:{GRN_MID};color:{GRN_PALE};border:1px solid {GRN_MID};"
        else:
            st = f"background:#fff;color:{GRN_MID};border:1px solid {BORDER_G};"
        js = _jtap(_FA, f"'{val}'", "h-filter-btn")
        return (f'<button onclick="{js}" style="{st}'
                f'border-radius:20px;padding:5px 14px;font-size:12px;cursor:pointer;'
                f'font-family:inherit;">{txt}</button>')

    _sort_cycle = {
        "Evidence first": "Nearest first",
        "Nearest first":  "Best match",
        "Best match":     "Evidence first",
    }
    new_sort = _sort_cycle.get(sort_val, "Nearest first")
    sort_js = _jtap(_SA, f"'{new_sort}'", "h-sort-btn")
    sort_btn = (
        f'<button onclick="{sort_js}" '
        f'style="background:#fff;border:0.5px solid {BORDER};border-radius:20px;'
        f'padding:5px 14px;font-size:12px;color:{TXT_SEC};cursor:pointer;'
        f'display:flex;align-items:center;gap:5px;font-family:inherit;">'
        f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="2" stroke-linecap="round"><path d="M3 6h18M7 12h10M11 18h2"/></svg>'
        f' Sort: {sort_val}</button>'
    )
    return f"""
<div style="background:#fff;border-bottom:1px solid {BORDER};padding:8px 20px;
            display:flex;align-items:center;gap:8px;flex-shrink:0;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  {chip("All", "All", n_results)}
  {chip("Government", "Government")}
  {chip("Private", "Private")}
  <div style="flex:1;"></div>
  {sort_btn}
</div>"""


def _card_html(rank, r, shortlist, compare=None, search_lat=None, search_lon=None):
    compare = compare or []
    ev    = r["evidence"]
    badge = trust_label(ev)
    _, tbg, tbdr, tclr, tlbl = TRUST_CFG.get(badge, _TRUST_DEFAULT)

    govt  = _is_govt(r)
    th_bg = BG_GOVT if govt else GRN_PALE
    ic    = GOVT_CLR if govt else GRN_MID
    ttype = "GOVERNMENT" if govt else "PRIVATE"
    icon  = _BUILDING_ICON.replace("{c}", ic)

    fid   = _s(r.get("id", r["name"]))
    saved = any(s.get("id") == fid for s in shortlist)
    dist  = r.get("distance_km")
    dist_s = f"{dist} km" if dist is not None else "—"

    # Evidence chips — click to expand and see the actual field text (formatted)
    id_slug = "".join(c for c in str(fid)[:16] if c.isalnum()) or "f"
    chips_html = ""
    for m in ev["matching"]:
        field   = m["field"]
        label   = FIELD_LABELS.get(field, field)
        raw     = m.get("text", "").strip()
        content = _fmt_field(field, raw)   # cleaned, formatted HTML
        cid     = f"ev-{id_slug}-{field}"
        toggle_js = (
            f"(function(){{"
            f"var d=document.getElementById('{cid}');"
            f"if(d)d.style.display=d.style.display==='block'?'none':'block';"
            f"}})();"
        )
        # Header label inside the expanded panel
        field_label_html = (
            f'<div style="font-size:9px;font-weight:600;color:{TXT_MUT};'
            f'text-transform:uppercase;letter-spacing:0.4px;margin-bottom:5px;">'
            f'{label}</div>'
        )
        chips_html += (
            f'<span onclick="{toggle_js}" title="Click to see {label} details"'
            f' style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
            f'border-radius:12px;padding:2px 10px;font-size:11px;color:{GRN_DK};'
            f'margin:2px 3px 2px 0;display:inline-block;cursor:pointer;'
            f'user-select:none;">{label} ▾</span>'
            f'<div id="{cid}" style="display:none;background:#F8FBF5;'
            f'border:0.5px solid {BORDER_G};border-radius:6px;color:{TXT_PRI};'
            f'padding:8px 10px;margin:3px 0 6px 0;line-height:1.6;">'
            f'{field_label_html}{content}</div>'
        )

    missing = ev.get("missing", [])
    miss_html = ""
    if missing:
        txt = " and ".join(FIELD_LABELS.get(m, m) for m in missing)
        miss_html = (
            f'<div style="font-size:11px;color:{AMBER};margin-top:5px;">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="{AMBER}" '
            f'stroke-width="2" stroke-linecap="round" style="vertical-align:-1px;margin-right:3px;">'
            f'<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>'
            f'{txt} not reported</div>'
        )

    flags_html = "".join(
        f'<div style="font-size:11px;color:{RED_FLAG};margin-top:4px;">'
        f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="{RED_FLAG}" '
        f'stroke-width="2" stroke-linecap="round" style="vertical-align:-1px;margin-right:3px;">'
        f'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        f'<path d="M12 9v4M12 17h.01"/></svg>{f}</div>'
        for f in ev.get("suspicious", [])
    )

    phone   = _s(r.get("phone", ""))
    website = _s(r.get("website", ""))
    _na = f'<span style="font-size:11px;color:{TXT_MUT};">Not available</span>'
    ph_html = (
        f'<a href="tel:{phone}" style="color:{GRN_MID};font-size:11px;'
        f'text-decoration:none;display:inline-flex;align-items:center;gap:3px;">'
        f'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="{GRN_MID}" '
        f'stroke-width="2" stroke-linecap="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.89 12a19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 3.81 1h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 8.91a16 16 0 0 0 6 6l.96-.96a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>'
        f'{phone[:20]}{"…" if len(phone)>20 else ""}</a>'
    ) if phone else _na
    dom = website.replace("https://","").replace("http://","").rstrip("/")
    wb_html = (
        f'<a href="{website if website.startswith("http") else "https://"+website}" '
        f'target="_blank" style="color:{GRN_MID};font-size:11px;'
        f'text-decoration:none;display:inline-flex;align-items:center;gap:3px;">'
        f'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="{GRN_MID}" '
        f'stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/>'
        f'<path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
        f'{dom[:22]}{"…" if len(dom)>22 else ""}</a>'
    ) if website else _na

    bm_bg  = GRN_PALE if saved else BG_CARD
    bm_bdr = GRN_MID  if saved else BORDER
    bm_ico = (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="{fc}" stroke="{fc}" stroke-width="2" '
        'stroke-linecap="round"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>'
    ).replace("{fc}", GRN_MID if saved else TXT_MUT)

    # Escape fid for inline JS
    safe_fid = fid.replace("'", "\\'")

    bm_onclick = _jtap(_BA, "'" + safe_fid + "'", "h-bm-btn")

    # Compare button
    in_compare  = any(c.get("id") == fid for c in compare)
    cmp_onclick = _jtap(_CMPA, "'" + safe_fid + "'", "h-cmp-btn")
    cmp_ico     = "&#x229F;" if in_compare else "&#x229E;"   # ⊟ / ⊞
    cmp_bg      = GRN_PALE   if in_compare else BG_CARD
    cmp_bdr     = GRN_MID    if in_compare else BORDER

    # Checklist button
    chk_onclick = _jtap(_CHKA, "'" + safe_fid + "'", "h-chk-btn")

    # Directions + nearby pharmacies — both use the hospital's own lat/lon
    lat_v, lon_v = r.get("lat"), r.get("lon")
    if lat_v and lon_v:
        if search_lat and search_lon:
            dir_url = (
                f"https://www.google.com/maps/dir/{search_lat},{search_lon}"
                f"/{lat_v},{lon_v}"
            )
        else:
            dir_url = f"https://www.google.com/maps/dir/?api=1&destination={lat_v},{lon_v}"
        dir_html = (
            f'<a href="{dir_url}" target="_blank" title="Get directions to this hospital"'
            f' style="width:30px;height:30px;border-radius:50%;border:0.5px solid {BORDER};'
            f'background:{BG_CARD};display:inline-flex;align-items:center;'
            f'justify-content:center;flex-shrink:0;text-decoration:none;">'
            f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{TXT_SEC}"'
            f' stroke-width="2" stroke-linecap="round">'
            f'<polygon points="3 11 22 2 13 21 11 13 3 11"/></svg></a>'
        )
        pharm_url  = f"https://www.google.com/maps/search/pharmacy/@{lat_v},{lon_v},15z"
        pharm_html = (
            f'<a href="{pharm_url}" target="_blank" title="Find pharmacies near this hospital"'
            f' style="width:30px;height:30px;border-radius:50%;border:0.5px solid {BORDER};'
            f'background:{BG_CARD};display:inline-flex;align-items:center;'
            f'justify-content:center;flex-shrink:0;text-decoration:none;">'
            f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{TXT_SEC}"'
            f' stroke-width="2" stroke-linecap="round">'
            f'<rect x="3" y="3" width="18" height="18" rx="2"/>'
            f'<path d="M9 12h6M12 9v6"/></svg></a>'
        )
    else:
        dir_html   = ""
        pharm_html = ""

    sem_pill = ""
    if r.get("sem_score", 0) > 0:
        sem_pill = (
            '<span style="font-size:10px;background:#E8F0FE;color:#1A56DB;'
            'border-radius:10px;padding:1px 6px;margin-left:5px;vertical-align:middle;">AI</span>'
        )

    evidence_section = ""
    if chips_html:
        evidence_section += (
            f'<div style="margin-top:8px;">'
            f'<div style="font-size:9px;font-weight:600;color:{TXT_SEC};'
            f'text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;">CONFIRMED IN</div>'
            f'<div>{chips_html}</div>'
            f'</div>'
        )
    if missing or ev.get("suspicious"):
        evidence_section += (
            f'<div style="border-top:0.5px solid {BORDER};margin-top:8px;padding-top:6px;">'
            f'{miss_html}{flags_html}</div>'
        )

    footer_links = " &nbsp; ".join(filter(None, [ph_html, wb_html]))

    return f"""
<div style="display:flex;border:0.5px solid {BORDER};border-radius:8px;background:{BG_CARD};
            margin-bottom:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
  <!-- Thumb -->
  <div style="width:64px;min-width:64px;background:{th_bg};display:flex;
              flex-direction:column;align-items:center;padding:14px 4px 0;
              position:relative;flex-shrink:0;">
    {icon}
    <div style="font-size:8px;font-weight:600;letter-spacing:0.5px;color:{ic};
                text-transform:uppercase;margin-top:6px;text-align:center;line-height:1.2;">
      {ttype}</div>
    <div style="position:absolute;bottom:0;left:0;right:0;background:{tbg};
                border-top:1px solid {tbdr};color:{tclr};font-size:10px;font-weight:500;
                text-align:center;padding:4px 2px;line-height:1;">{tlbl}</div>
  </div>
  <!-- Body -->
  <div style="flex:1;padding:12px 14px;display:flex;flex-direction:column;min-width:0;">
    <!-- Name + distance -->
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
      <div style="min-width:0;">
        <div style="font-size:14px;font-weight:600;color:{TXT_PRI};
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          {r['name']}{sem_pill}</div>
        <div style="font-size:11px;color:{TXT_MUT};margin-top:1px;">
          {_s(r.get('city'))}, {_s(r.get('state'))}</div>
        {(lambda d: f'<div style="font-size:11px;color:{TXT_PRI};margin-top:3px;line-height:1.4;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">{d[:160]}{"…" if len(d)>160 else ""}</div>' if d else "")(r.get("description","").strip())}
      </div>
      <div style="font-size:11px;color:{TXT_SEC};white-space:nowrap;flex-shrink:0;
                  display:flex;align-items:center;gap:3px;">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="{GRN_LT}" stroke="{GRN_LT}"
             stroke-width="0"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
        {dist_s}
      </div>
    </div>
    {evidence_section}
    <!-- Footer: contact links + action buttons -->
    <div style="border-top:0.5px solid {BORDER};margin-top:10px;padding-top:8px;
                display:flex;align-items:center;justify-content:space-between;gap:8px;">
      <div style="display:flex;gap:12px;flex-wrap:wrap;min-width:0;">{footer_links}</div>
      <div style="display:flex;align-items:center;gap:4px;flex-shrink:0;">
        {dir_html}
        {pharm_html}
        <button onclick="{cmp_onclick}" title="{'Remove from comparison' if in_compare else 'Add to comparison'}"
          style="width:30px;height:30px;border-radius:50%;border:0.5px solid {cmp_bdr};
                 background:{cmp_bg};cursor:pointer;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;padding:0;
                 font-size:16px;color:{GRN_MID};">{cmp_ico}</button>
        <button onclick="{chk_onclick}" title="Visit Checklist &amp; QR"
          style="width:30px;height:30px;border-radius:50%;border:0.5px solid {BORDER};
                 background:{BG_CARD};cursor:pointer;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;padding:0;">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{TXT_SEC}"
               stroke-width="2" stroke-linecap="round">
            <rect x="8" y="2" width="8" height="4" rx="1"/>
            <path d="M8 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2h-2"/>
            <line x1="12" y1="11" x2="16" y2="11"/><line x1="12" y1="15" x2="16" y2="15"/>
            <polyline points="8 11 9 12 11 10"/><polyline points="8 15 9 16 11 14"/>
          </svg>
        </button>
        <button onclick="{bm_onclick}" title="{'Remove from shortlist' if saved else 'Save to shortlist'}"
          style="width:30px;height:30px;border-radius:50%;border:0.5px solid {bm_bdr};
                 background:{bm_bg};cursor:pointer;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;padding:0;">
          {bm_ico}
        </button>
      </div>
    </div>
  </div>
</div>"""


def _map_html(results, lat, lon, radius_km, location_name):
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB Positron")
    folium.Circle(
        [lat, lon], radius=radius_km * 1000,
        color=GRN_MID, fill=True, fill_color=GRN_PALE, fill_opacity=0.07, weight=1.5,
    ).add_to(m)
    folium.CircleMarker(
        [lat, lon], radius=6, color=GRN_DK, fill=True, fill_color=GRN_DK,
        fill_opacity=1.0, tooltip=f"Search: {location_name}",
    ).add_to(m)
    color_map = {"strong": GRN_DK, "partial": "#BA7517", "verify": "#D85A30"}
    n = 0
    for r in results:
        rlat, rlon = r.get("lat"), r.get("lon")
        if not rlat or not rlon:
            continue
        badge = trust_label(r["evidence"])
        key, *_ = TRUST_CFG.get(badge, _TRUST_DEFAULT)
        c = color_map.get(key, GRN_LT)
        dist_s = f"{r['distance_km']} km" if r.get("distance_km") is not None else "—"
        folium.CircleMarker(
            [rlat, rlon], radius=7, color=c, fill=True, fill_color=c, fill_opacity=0.9,
            tooltip=f"{r['name']} · {dist_s}",
        ).add_to(m)
        n += 1
    b64 = base64.b64encode(m.get_root().render().encode("utf-8")).decode("ascii")
    label = (
        f'<div style="position:absolute;top:8px;left:8px;z-index:999;'
        f'background:rgba(255,255,255,0.93);padding:4px 9px;border-radius:6px;'
        f'font-size:11px;color:{TXT_SEC};border:0.5px solid {BORDER};font-family:inherit;">'
        f'{location_name} · {n} facilities · {radius_km} km radius</div>'
    )
    return (
        f'<div style="position:relative;width:100%;height:100%;min-height:260px;">'
        f'{label}'
        f'<iframe src="data:text/html;base64,{b64}" '
        f'style="width:100%;height:100%;border:none;display:block;"></iframe></div>'
    )


_INDIA_MAP_B64 = ""

def _build_india_map():
    global _INDIA_MAP_B64
    try:
        m = folium.Map(location=[22.5, 78.9], zoom_start=5, tiles="CartoDB Positron")
        _INDIA_MAP_B64 = base64.b64encode(
            m.get_root().render().encode("utf-8")
        ).decode("ascii")
        print("[App] Default India map ready")
    except Exception as e:
        print(f"[App] Default map failed: {e}")

threading.Thread(target=_build_india_map, daemon=True).start()


def _default_map_html():
    if _INDIA_MAP_B64:
        return (
            f'<div style="position:relative;width:100%;height:100%;min-height:260px;">'
            f'<iframe src="data:text/html;base64,{_INDIA_MAP_B64}" '
            f'style="width:100%;height:100%;border:none;display:block;"></iframe></div>'
        )
    return (
        f'<div style="width:100%;height:100%;min-height:260px;background:#EBF0E8;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:12px;color:{TXT_MUT};">Map loading…</div>'
    )


def _shortlist_panel_html(shortlist):
    n = len(shortlist)
    badge = (
        f'<span style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
        f'border-radius:10px;padding:1px 7px;font-size:10px;color:{GRN_DK};'
        f'margin-left:5px;">{n}</span>'
    )
    items = ""
    for i, s in enumerate(shortlist):
        _, tbg, tbdr, tclr, tlbl = TRUST_CFG.get(s.get("trust",""), _TRUST_DEFAULT)
        dist_s = f"{s['distance_km']} km" if s.get("distance_km") is not None else "—"
        rm_onclick = _jtap(_RMA, str(i), "h-rm-btn")
        items += f"""
<div style="background:{BG_PAGE};border:0.5px solid {BORDER_G};border-radius:7px;
            padding:7px 10px;margin-bottom:5px;display:flex;
            align-items:center;justify-content:space-between;gap:6px;">
  <div style="min-width:0;">
    <div style="font-size:12px;font-weight:500;color:{TXT_PRI};
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{s['name']}</div>
    <div style="font-size:10px;color:{TXT_SEC};margin-top:1px;">
      {dist_s} · {_s(s.get('city'))} ·
      <span style="background:{tbg};border:0.5px solid {tbdr};border-radius:8px;
                   padding:1px 5px;color:{tclr};">{tlbl}</span>
    </div>
  </div>
  <button onclick="{rm_onclick}"
    style="background:none;border:none;cursor:pointer;font-size:16px;
           color:{TXT_MUT};flex-shrink:0;padding:0;line-height:1;">×</button>
</div>"""

    clear_onclick  = _jclick("h-clear-btn")
    export_onclick = _jclick("h-export-btn-gr")

    export_btn = (
        f'<button onclick="{export_onclick}" style="width:100%;background:{GRN_MID};color:{GRN_PALE};'
        f'border:none;border-radius:8px;padding:10px;font-size:13px;font-weight:500;'
        f'cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;'
        f'margin-top:8px;font-family:inherit;">'
        f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{GRN_PALE}" '
        f'stroke-width="2" stroke-linecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        f'<polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
        f' Export shortlist</button>'
    )
    empty_msg = (
        f'<div style="font-size:12px;color:{TXT_MUT};font-style:italic;padding:4px 0 8px;">'
        f'No facilities saved yet.</div>'
    )
    return f"""
<div style="border-top:1px solid {BORDER};padding:12px 14px;background:{BG_CARD};
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <span style="font-size:13px;font-weight:500;color:{TXT_PRI};">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{GRN_MID}"
           stroke-width="2" stroke-linecap="round" style="vertical-align:-1px;margin-right:4px;">
        <line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>
        <line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/>
        <line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>
      </svg>Shortlist{badge}</span>
    <span onclick="{clear_onclick}"
      style="font-size:11px;color:{TXT_MUT};text-decoration:underline;cursor:pointer;">Clear</span>
  </div>
  {items if items else empty_msg}
  {export_btn}
</div>"""


_SUGGESTIONS = [
    "dialysis near Jaipur",
    "eye care near Hyderabad",
    "emergency surgery near Patna",
    "cardiac care near Mumbai",
    "maternity hospital near Delhi",
    "cancer treatment near Bangalore",
]

def _suggestions_bar_html(current_query):
    pills = ""
    for s in _SUGGESTIONS:
        active = s.lower() == (current_query or "").strip().lower()
        bg  = GRN_MID  if active else "#FFFFFF"
        clr = GRN_PALE if active else TXT_PRI
        bdr = GRN_MID  if active else GRN_MID
        pills += (
            f'<button onclick="{_js_suggestion(s)}" '
            f'style="background:{bg};color:{clr};border:1px solid {bdr};'
            f'border-radius:20px;padding:4px 12px;font-size:12px;cursor:pointer;'
            f'font-family:inherit;white-space:nowrap;flex-shrink:0;font-weight:500;">{s}</button>'
        )
    return (
        f'<div style="background:#FAFCF7;border-bottom:1px solid {BORDER};'
        f'padding:7px 20px;display:flex;align-items:center;gap:8px;'
        f'flex-wrap:nowrap;overflow-x:auto;flex-shrink:0;">'
        f'{pills}</div>'
    )


_FOLLOWUP_SPECIALTIES = [
    ("Eye Care",          "ophthalmology"),
    ("Heart / Cardiology","cardiology"),
    ("Orthopedics",       "orthopedics"),
    ("Emergency",         "emergency"),
    ("Dialysis",          "dialysis"),
    ("Cancer / Oncology", "oncology"),
    ("Maternity",         "maternity"),
    ("Neurology",         "neurology"),
    ("Pediatrics",        "pediatrics"),
    ("General Surgery",   "general surgery"),
]


def _followup_html(meta):
    """
    Contextual follow-up bar shown below search results.
    - No care_need: ask what specialty the user needs, offer quick-reply chips.
    - Has care_need + results: offer refinement chips (government/private/nearest).
    """
    loc       = meta.get("resolved_location", "") or ""
    care_need = meta.get("care_need", "") or ""
    n_results = meta.get("total_matches", 0)

    if not loc:
        return ""

    def _chip(label, query_text, emoji=""):
        js = _js_suggestion(query_text)
        txt = f"{emoji} {label}".strip()
        return (
            f'<button onclick="{js}" '
            f'style="background:#fff;color:{GRN_MID};border:1px solid {BORDER_G};'
            f'border-radius:20px;padding:4px 14px;font-size:12px;cursor:pointer;'
            f'font-family:inherit;white-space:nowrap;flex-shrink:0;font-weight:500;'
            f'transition:background 0.15s;"'
            f' onmouseover="this.style.background=\'{GRN_PALE}\'"'
            f' onmouseout="this.style.background=\'#fff\'">'
            f'{txt}</button>'
        )

    if not care_need:
        # Broad query — no specialty detected; ask what they need
        loc_title = loc.title()
        chips = "".join(
            _chip(label, f"{specialty} near {loc_title}")
            for label, specialty in _FOLLOWUP_SPECIALTIES
        )
        return (
            f'<div style="background:#F0F7FF;border:0.5px solid #B3D4F5;'
            f'border-radius:10px;padding:12px 16px;margin-bottom:14px;">'
            f'<div style="font-size:12px;font-weight:600;color:#0D47A1;margin-bottom:8px;">'
            f'&#x1F4AC; What kind of care are you looking for in {loc_title}?</div>'
            f'<div style="display:flex;gap:7px;flex-wrap:wrap;">{chips}</div>'
            f'</div>'
        )

    elif n_results > 0:
        # Has results — offer quick refinements
        loc_title = loc.title()
        care_title = care_need.title()
        refine_chips = "".join([
            _chip("Government only",   f"government {care_need} near {loc_title}", "🏛"),
            _chip("Private only",      f"private {care_need} near {loc_title}",    "🏥"),
            _chip("Nearest first",     f"{care_need} near {loc_title}",            "📍"),
            _chip("Wider search",      f"{care_need} near {loc_title}",            "🔍"),
        ])
        return (
            f'<div style="background:#F0F7FF;border:0.5px solid #B3D4F5;'
            f'border-radius:10px;padding:10px 16px;margin-bottom:14px;">'
            f'<div style="font-size:12px;font-weight:600;color:#0D47A1;margin-bottom:7px;">'
            f'&#x1F4AC; Found {n_results} {care_title} facilities near {loc_title}. Refine your search:</div>'
            f'<div style="display:flex;gap:7px;flex-wrap:wrap;">{refine_chips}</div>'
            f'</div>'
        )

    return ""


def _render_page(results, shortlist, filter_val, sort_val, query, radius, meta=None, compare=None):
    meta    = meta    or {}
    compare = compare or []
    n_saved = len(shortlist)

    # Apply filter + sort
    filtered = list(results)
    if filter_val == "Government":
        filtered = [r for r in filtered if _is_govt(r)]
    elif filter_val == "Private":
        filtered = [r for r in filtered if not _is_govt(r)]
    if sort_val == "Best match":
        filtered.sort(key=lambda r: -r.get("blended_score", 0))
    elif sort_val == "Nearest first":
        filtered.sort(key=lambda r: r.get("distance_km") or 9999)
    else:  # "Evidence first" (default)
        filtered.sort(key=lambda r: (_trust_tier(r), r.get("distance_km") or 9999))

    care_need = meta.get("care_need", "") if meta else ""
    loc       = meta.get("resolved_location", "") if meta else ""

    # Results body
    if not results and not meta:
        if _STARTUP_ERROR:
            results_body = (
                f'<div style="color:#c00;font-size:12px;padding:16px 0;">{_STARTUP_ERROR}</div>'
            )
        else:
            results_body = (
                f'<div style="color:{TXT_PRI};font-size:14px;font-weight:500;padding:20px 0;">'
                f'Select a suggestion above or type a query like '
                f'<b>dialysis near Jaipur</b> to find facilities.</div>'
            )
    elif "error" in meta:
        results_body = (
            f'<div style="background:#FFF8E1;border-left:3px solid #F9A825;'
            f'padding:12px 16px;border-radius:0 6px 6px 0;font-size:13px;color:{TXT_PRI};">'
            f'&#9888; {meta["error"]}</div>'
        )
    elif not results and not care_need and loc:
        # Location resolved but no specialty — show conversational follow-up only
        loc_title = loc.title()
        results_body = (
            f'<div style="font-size:13px;color:{TXT_PRI};padding-bottom:12px;">'
            f'Showing results near <b>{loc_title}</b>. What kind of care are you looking for?</div>'
        )
    elif not filtered:
        no_results_msg = (
            f'<div style="background:#FFF8E1;border-left:3px solid #F9A825;'
            f'padding:12px 16px;border-radius:0 6px 6px 0;font-size:13px;color:{TXT_PRI};">'
            f'No {filter_val.lower()} facilities found for '
            f'<b>{care_need}</b> within <b>{int(radius or 5)} km</b> of <b>{loc}</b>.'
            f'<br>Try switching to <b>All</b> filter, increasing the radius, or '
            f'checking the spelling.</div>'
        )
        results_body = no_results_msg
    else:
        n_loc   = meta.get("located_count", len(results))
        n_unloc = meta.get("unlocated_count", 0)
        unloc_s = f" + {n_unloc} unverified" if n_unloc else ""
        fuzzy   = (f' <span style="color:{TXT_MUT};font-size:11px;">(matched: {loc})</span>'
                   if meta.get("location_match_type") == "fuzzy" else "")
        expanded_from = meta.get("expanded_from")
        expand_banner = ""
        if expanded_from:
            expand_banner = (
                f'<div style="background:#E3F2FD;border-left:3px solid #1976D2;'
                f'padding:8px 14px;border-radius:0 6px 6px 0;font-size:12px;'
                f'color:#0D47A1;margin-bottom:10px;">'
                f'&#8505; No results within {int(expanded_from)} km — '
                f'automatically expanded to <b>{int(radius or 5)} km</b>.</div>'
            )
        summary = (
            expand_banner +
            f'<div style="font-size:12px;color:{TXT_SEC};margin-bottom:12px;">'
            f'<b style="color:{TXT_PRI};">{n_loc}</b> facilities for '
            f'<b style="color:{TXT_PRI};">{care_need}</b> within '
            f'<b style="color:{TXT_PRI};">{int(radius or 5)} km</b> of '
            f'<b style="color:{TXT_PRI};">{loc}</b>{fuzzy}{unloc_s}</div>'
        )
        slat  = meta.get("search_lat")
        slon  = meta.get("search_lon")
        cards = "".join(
            _card_html(i + 1, r, shortlist, compare, slat, slon)
            for i, r in enumerate(filtered)
        )
        compare_panel = _compare_panel_html(compare) if len(compare) >= 2 else ""
        results_body  = compare_panel + summary + cards

    # Map
    if meta.get("search_lat"):
        right_map = _map_html(
            results, meta["search_lat"], meta["search_lon"],
            int(radius or 5), meta.get("resolved_location", query or ""),
        )
    else:
        right_map = _default_map_html()

    shortlist_panel = _shortlist_panel_html(shortlist)
    chat_history = (meta or {}).get("chat_history", [])
    topbar       = _topbar_html(query, radius or 5, n_saved, in_chat=bool(chat_history))
    chat_thread  = _chat_thread_html(chat_history)
    # Hide suggestion pills once the user has started a conversation
    suggestions  = "" if chat_history else _suggestions_bar_html(query)
    filterbar    = _filterbar_html(filter_val, sort_val, len(results))

    responsive_css = f"""
<style>
b, strong {{ font-weight:700 !important; color:inherit !important; }}
.sv-body {{ display:flex; flex:1; min-height:600px; overflow:hidden; }}
.sv-results {{
  flex: 0 0 65%; padding:16px 20px 24px; overflow-y:auto;
  border-right:1px solid {BORDER}; box-sizing:border-box;
}}
.sv-right {{
  flex: 0 0 35%; display:flex; flex-direction:column;
  min-height:0; background:#fff;
}}
.sv-topbar-logo span {{ display:inline; }}
.sv-pill-radius {{ display:flex !important; flex-direction:column; }}

/* Medium screens: tighten pill */
@media (max-width:1100px) {{
  .sv-results {{ flex: 0 0 60%; }}
  .sv-right   {{ flex: 0 0 40%; }}
}}

/* Small screens: stack vertically */
@media (max-width:800px) {{
  .sv-body {{ flex-direction: column; min-height:unset; overflow:visible; }}
  .sv-results {{
    flex: none; width:100%; border-right:none;
    border-bottom:1px solid {BORDER};
  }}
  .sv-right {{ flex: none; width:100%; min-height:400px; }}
  .sv-topbar-logo span {{ display:none; }}
  .sv-pill-radius {{ display:none !important; }}
}}

/* Very small: tighten topbar pill gaps */
@media (max-width:600px) {{
  .sv-pill-section {{ padding: 4px 10px !important; }}
}}
</style>"""

    return f"""
{responsive_css}
<div style="display:flex;flex-direction:column;background:{BG_PAGE};
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    min-height:85vh;">
  {topbar}
  {suggestions}
  {chat_thread}
  {filterbar}
  <div class="sv-body">
    <div class="sv-results">{results_body}</div>
    <div class="sv-right">
      <div style="flex:1;min-height:260px;overflow:hidden;">{right_map}</div>
      {shortlist_panel}
    </div>
  </div>
  {_JS}
</div>"""

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_csv(shortlist):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Name", "City", "State", "Distance (km)", "Trust", "Phone", "Website"])
    for s in shortlist:
        w.writerow([s["name"], s.get("city",""), s.get("state",""),
                    s.get("distance_km") or "", s.get("trust",""),
                    s.get("phone",""), s.get("website","")])
    path = os.path.join(_DATA_DIR, "shortlist_export.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return path

# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

_AN = False  # api_name=False on all handlers suppresses schema crash

CSS = """
body, .gradio-container { background: #F7FAF3 !important; }
.gradio-container > .main { padding: 0 !important; max-width: 100% !important; }
footer { display: none !important; }
.gr-prose { padding: 0 !important; }
b, strong { font-weight: 700 !important; color: inherit !important; }

/* Hide bridge column and all bridge components.
   display:none keeps elements in DOM so querySelector still finds them and
   button.click() still fires — unlike visible=False which removes from DOM. */
#sv-bridge { display: none !important; }
.sv-hide   { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Suvidha — Healthcare Referrals") as demo:

    # Static JS — rendered once at startup; defines all window.sSearch / sFilter / etc.
    # Placed before page_html so functions are available before user interaction.
    gr.HTML(_JS)

    # State
    results_state   = gr.State([])
    shortlist_state = gr.State([])
    meta_state      = gr.State({})
    filter_state    = gr.State("All")
    sort_state      = gr.State(_DEFAULT_SORT)
    query_state     = gr.State("")
    radius_state    = gr.State(5)
    compare_state   = gr.State([])   # hospitals being compared

    # Main visible output
    page_html = gr.HTML(_render_page([], [], "All", _DEFAULT_SORT, "", 5))

    # Checklist modal — position:fixed overlay, empty = hidden
    checklist_html = gr.HTML("")

    # ── Bridge components ─────────────────────────────────────────────────
    # All wrapped in #sv-bridge (CSS display:none) and given class sv-hide.
    # CSS hides them visually but keeps them in DOM so button.click() fires.
    # Architecture: JS sets textbox value + dispatches input event (updates
    # Svelte's reactive binding), then clicks the hidden trigger button after
    # 80ms. Button.click() is more reliable than synthetic keydown events.
    _SHIDE = ["sv-hide"]
    with gr.Column(elem_id="sv-bridge"):
        h_query  = gr.Textbox(value="",              elem_id="h-query",  label="", elem_classes=_SHIDE)
        h_rad    = gr.Textbox(value="50",            elem_id="h-rad",    label="", elem_classes=_SHIDE)
        h_filter = gr.Textbox(value="All",           elem_id="h-filter", label="", elem_classes=_SHIDE)
        h_sort   = gr.Textbox(value=_DEFAULT_SORT,   elem_id="h-sort",   label="", elem_classes=_SHIDE)
        h_bm_id  = gr.Textbox(value="",             elem_id="h-bm-id",  label="", elem_classes=_SHIDE)
        h_rm_idx = gr.Textbox(value="",             elem_id="h-rm-idx", label="", elem_classes=_SHIDE)
        # One trigger button per action — JS calls element.click() on these
        h_cmp_id  = gr.Textbox(value="", elem_id="h-cmp-id",  label="", elem_classes=_SHIDE)
        h_chk_id  = gr.Textbox(value="", elem_id="h-chk-id",  label="", elem_classes=_SHIDE)
        h_search_btn     = gr.Button("", elem_id="h-search-btn",     elem_classes=_SHIDE)
        h_filter_btn     = gr.Button("", elem_id="h-filter-btn",     elem_classes=_SHIDE)
        h_sort_btn       = gr.Button("", elem_id="h-sort-btn",       elem_classes=_SHIDE)
        h_bm_btn         = gr.Button("", elem_id="h-bm-btn",         elem_classes=_SHIDE)
        h_rm_btn         = gr.Button("", elem_id="h-rm-btn",         elem_classes=_SHIDE)
        h_clear_btn      = gr.Button("", elem_id="h-clear-btn",      elem_classes=_SHIDE)
        h_export_btn_gr  = gr.Button("", elem_id="h-export-btn-gr",  elem_classes=_SHIDE)
        h_reset_btn      = gr.Button("", elem_id="h-reset-btn",      elem_classes=_SHIDE)
        h_cmp_btn        = gr.Button("", elem_id="h-cmp-btn",        elem_classes=_SHIDE)
        h_cmp_clear_btn  = gr.Button("", elem_id="h-cmp-clear-btn",  elem_classes=_SHIDE)
        h_chk_btn        = gr.Button("", elem_id="h-chk-btn",        elem_classes=_SHIDE)
        h_chk_close_btn  = gr.Button("", elem_id="h-chk-close-btn",  elem_classes=_SHIDE)

    export_file = gr.File(label="Download shortlist", visible=False)

    # ── Chat / Search ─────────────────────────────────────────────────────
    # Every user message goes through the LLM first. The LLM either:
    #   • asks a follow-up question  (conversational response)
    #   • emits {"action":"search",...}  → runs search, shows results
    #   • emits {"action":"filter",...}  → re-renders with new filter
    # chat_history is stored inside meta["chat_history"] so all downstream
    # handlers (filter, sort, bookmark) carry it through without changes.

    def _run_search(care_need, location, org_type_hint, radius, filter_val):
        """Run ONE supervisor call at max radius; filter client-side to avoid repeated RAG calls."""
        # Single RAG + vector search call (expensive) — always search wide
        all_results, meta = supervisor.run(
            df=df, centroids=centroids,
            care_need_query=care_need, location_query=location,
            radius_km=200, top_n=100,
        )

        def _within(r, km):
            d = r.get("distance_km")
            return d is None or d <= km

        # Filter to user's requested radius
        results = [r for r in all_results if _within(r, radius)]

        # Auto-expand client-side (no extra API calls)
        if not results and all_results:
            for bigger_r in [10, 25, 50, 100, 200]:
                if bigger_r <= radius:
                    continue
                results = [r for r in all_results if _within(r, bigger_r)]
                if results:
                    meta = dict(meta)
                    meta["expanded_from"] = radius
                    radius = bigger_r
                    break

        # Recount meta to reflect the filtered slice
        if results or all_results:
            meta = dict(meta)
            meta["total_matches"]   = len(results)
            meta["located_count"]   = sum(1 for r in results if r.get("distance_km") is not None)
            meta["unlocated_count"] = sum(1 for r in results if r.get("distance_km") is None)

        results = results[:10]

        # Always reset filter on a new search — ignore stale Private/Government state
        if org_type_hint in ("government", "govt", "public", "sarkari"):
            effective_filter = "Government"
        elif org_type_hint == "private":
            effective_filter = "Private"
        else:
            effective_filter = "All"
        return results, meta, effective_filter, radius

    def _do_chat(query, radius, shortlist, filter_val, sort_val, cur_results, cur_meta, cur_compare):
        try:
            radius = int(float(radius or "5"))
        except (ValueError, TypeError):
            radius = 5
        query = (query or "").strip()
        # Override radius if the user mentioned a distance in their message
        _rm = re.search(r'\b(\d+)\s*(?:km|kms|kilometer|kilometre)s?\b', query, re.IGNORECASE)
        if _rm:
            radius = max(5, min(500, int(_rm.group(1))))

        # Retrieve chat history stored in current meta
        chat_history = (cur_meta or {}).get("chat_history", [])

        def _ret(results, meta, filt, q, r, compare):
            """Attach chat_history to meta and render page. Always clears query box."""
            meta = dict(meta or {})
            meta["chat_history"] = chat_history
            html = _render_page(results, shortlist, filt, sort_val, q, r, meta, compare)
            return html, results, meta, "", r, compare, filt

        if not query:
            return _ret(cur_results or [], cur_meta or {}, filter_val,
                        (cur_meta or {}).get("query", ""), radius, cur_compare or [])

        if not _data_ready:
            ai_msg = _STARTUP_ERROR or "Still loading hospital data — please wait a moment and try again."
            chat_history = chat_history + [(query, ai_msg)]
            m = dict(cur_meta or {})
            m["chat_history"] = chat_history
            html = _render_page(cur_results or [], shortlist, filter_val, sort_val, query, radius, m, [])
            return html, cur_results or [], m, "", radius, [], filter_val

        # Terms too generic to use as care_need — would match most/all hospitals
        _GENERIC_CARE = {"hospital", "hospitals", "clinic", "clinics", "healthcare",
                         "medical", "doctor", "doctors", "care", "health", "centre", "center"}

        # ── Fast local action detection (no LLM needed) ───────────────────
        # Handles only filter commands — no LLM call here.
        def _local_action(q):
            ql = q.lower()
            govt_words = r"\b(govt|government|sarkari|public|sarkar|sarkari)\b"
            if re.search(govt_words, ql):
                return {"action": "filter", "org_type": "government"}
            if re.search(r"\bprivate\b", ql) and re.search(r"\b(show|only|filter|just)\b", ql):
                return {"action": "filter", "org_type": "private"}
            if re.search(r"\b(show all|remove filter|any type|both types)\b", ql):
                return {"action": "filter", "org_type": "all"}
            return None

        local = _local_action(query)
        if local:
            response, is_action = local, True
        else:
            response, is_action = _llm_chat(query, chat_history)

        # LLM unavailable — fall back to direct parse and run search
        if response is None:
            care_need, location, org_type_hint = parse_combined_query(query, centroids)
            # Fill gaps from previous search (no LLM context, so carry both)
            if not location:
                location = (cur_meta or {}).get("resolved_location", "")
            if not care_need:
                care_need = (cur_meta or {}).get("care_need", "")
            if care_need.lower().strip() in _GENERIC_CARE:
                care_need = ""
            if location:
                if not care_need and not _rm:
                    radius = 25  # default for "show all hospitals"
                results, meta, eff_filt, radius = _run_search(care_need, location, org_type_hint, radius, filter_val)
                n   = meta.get("total_matches", len(results))
                loc = meta.get("resolved_location", location)
                err = meta.get("error", "")
                if n:
                    fac_word = "facility" if n == 1 else "facilities"
                    if care_need:
                        ai_msg = (f"Found <b>{n}</b> {care_need.title()} {fac_word} near <b>{loc.title()}</b>! "
                                  f"Say \"show only government\" or \"show only private\" to filter.")
                    else:
                        ai_msg = (f"Found <b>{n}</b> {fac_word} within <b>{radius} km</b> of <b>{loc.title()}</b>. "
                                  f"Ask for a specific care (e.g. maternity, dialysis) to narrow down.")
                elif err and "resolve" in err.lower():
                    ai_msg = f"Hmm, I couldn't find <b>{location.title()}</b> on the map. Try a nearby bigger city?"
                else:
                    ai_msg = (f"I looked within <b>{radius} km</b> of <b>{loc.title()}</b> "
                              f"but didn't find any hospitals there. Want to try a nearby city?")
                chat_history = chat_history + [(query, ai_msg)]
                meta["chat_history"] = chat_history
                search_q = f"{care_need} near {location}" if care_need else f"hospitals near {location}"
                html = _render_page(results, shortlist, eff_filt, sort_val, search_q, radius, meta, [])
                return html, results, meta, "", radius, [], eff_filt
            if care_need:
                ai_msg = f"Got it — <b>{care_need}</b>. Which city or area are you near?"
            else:
                ai_msg = "Happy to help! Just tell me which city — e.g. <i>hospitals near Jaipur</i> or <i>dialysis near Chennai</i>."
            chat_history = chat_history + [(query, ai_msg)]
            m = dict(cur_meta or {})
            m["chat_history"] = chat_history
            html = _render_page(cur_results or [], shortlist, filter_val, sort_val, query, radius, m, [])
            return html, cur_results or [], m, "", radius, [], filter_val

        if is_action:
            action = response
            if action.get("action") == "search":
                care_need     = (action.get("care_need",  "") or "").strip()
                location      = (action.get("location",   "") or "").strip()
                org_type_hint = (action.get("org_type",   "") or "").strip().lower()
                # Reject generic care terms — treat them as "show all hospitals"
                if care_need.lower() in _GENERIC_CARE:
                    care_need = ""
                # Fill location gap only — trust LLM to carry care_need via history
                if not location:
                    location = (cur_meta or {}).get("resolved_location", "")
                if not location:
                    ai_msg = f"Which city are you near? I'll find hospitals there."
                    if care_need:
                        ai_msg = f"Which city are you near? I'll look for <b>{care_need}</b> facilities there."
                    chat_history = chat_history + [(query, ai_msg)]
                    return _ret(cur_results or [], cur_meta or {}, filter_val, query, radius, cur_compare or [])

                # Default to 25 km for "all hospitals" searches when user didn't specify a radius
                if not care_need and not _rm:
                    radius = 25

                results, meta, eff_filt, radius = _run_search(care_need, location, org_type_hint, radius, filter_val)
                n   = meta.get("total_matches", len(results))
                loc = meta.get("resolved_location", location)
                err = meta.get("error", "")
                if n:
                    fac_word = "facility" if n == 1 else "facilities"
                    if care_need:
                        ai_msg = (
                            f"Found <b>{n}</b> {care_need.title()} {fac_word} near <b>{loc.title()}</b>! "
                            f"You can say \"show only government\" or \"show only private\" to filter, or ask about a different care."
                        )
                    else:
                        ai_msg = (
                            f"Found <b>{n}</b> {fac_word} within <b>{radius} km</b> of <b>{loc.title()}</b>. "
                            f"You can ask for a specific care (e.g. maternity, dialysis) or say \"show only government\" to filter."
                        )
                elif err and "resolve" in err.lower():
                    ai_msg = (
                        f"Hmm, I couldn't find <b>{location.title()}</b> on the map. "
                        f"Could you try a nearby bigger city or district name?"
                    )
                else:
                    if care_need:
                        ai_msg = (
                            f"I looked up to <b>{radius} km</b> around <b>{loc.title()}</b> "
                            f"but didn't find {care_need} facilities there. "
                            f"Want to try a nearby bigger city, or a different type of care?"
                        )
                    else:
                        ai_msg = (
                            f"I looked within <b>{radius} km</b> of <b>{loc.title()}</b> "
                            f"but didn't find any hospitals there. "
                            f"Want to try a nearby bigger city?"
                        )
                chat_history = chat_history + [(query, ai_msg)]
                meta["chat_history"] = chat_history
                search_q = f"{care_need} near {location}" if care_need else f"hospitals near {location}"
                html = _render_page(results, shortlist, eff_filt, sort_val, search_q, radius, meta, [])
                return html, results, meta, "", radius, [], eff_filt

            elif action.get("action") == "filter":
                org = (action.get("org_type", "") or "").lower()
                new_filter = ("Government" if org in ("government", "govt", "public", "sarkari")
                              else "Private" if org == "private" else "All")
                type_label = new_filter.lower() if new_filter != "All" else "all types of"
                ai_msg = f"Sure! Filtering to <b>{type_label}</b> facilities now."
                chat_history = chat_history + [(query, ai_msg)]
                return _ret(cur_results or [], cur_meta or {}, new_filter,
                            (cur_meta or {}).get("query", query), radius, cur_compare or [])

        # Plain conversational response — show it, keep current results
        chat_history = chat_history + [(query, str(response))]
        return _ret(cur_results or [], cur_meta or {}, filter_val, query, radius, cur_compare or [])

    h_search_btn.click(
        _do_chat,
        [h_query, h_rad, shortlist_state, filter_state, sort_state,
         results_state, meta_state, compare_state],
        [page_html, results_state, meta_state, query_state, radius_state, compare_state, filter_state],
        api_name=_AN,
    )

    # ── Filter ────────────────────────────────────────────────────────────
    def _do_filter(fv, results, shortlist, sort_val, query, radius, meta, compare):
        return _render_page(results, shortlist, fv, sort_val, query, radius, meta, compare), fv

    h_filter_btn.click(
        _do_filter,
        [h_filter, results_state, shortlist_state, sort_state,
         query_state, radius_state, meta_state, compare_state],
        [page_html, filter_state], api_name=_AN,
    )

    # ── Sort ──────────────────────────────────────────────────────────────
    def _do_sort(sv, results, shortlist, filter_val, query, radius, meta, compare):
        return _render_page(results, shortlist, filter_val, sv, query, radius, meta, compare), sv

    h_sort_btn.click(
        _do_sort,
        [h_sort, results_state, shortlist_state, filter_state,
         query_state, radius_state, meta_state, compare_state],
        [page_html, sort_state], api_name=_AN,
    )

    # ── Bookmark ──────────────────────────────────────────────────────────
    def _do_bookmark(bm_id, results, shortlist, filter_val, sort_val, query, radius, meta, compare):
        if not bm_id or not results:
            return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, compare), shortlist
        candidate = next((r for r in results if _s(r.get("id", r["name"])) == bm_id), None)
        if candidate is None:
            return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, compare), shortlist
        fid = _s(candidate.get("id", candidate["name"]))
        if any(s.get("id") == fid for s in shortlist):
            shortlist = [s for s in shortlist if s.get("id") != fid]
        else:
            shortlist = shortlist + [{
                "id": fid, "name": candidate["name"],
                "city": candidate.get("city",""), "state": candidate.get("state",""),
                "distance_km": candidate.get("distance_km"),
                "trust": trust_label(candidate["evidence"]),
                "phone": candidate.get("phone",""), "website": candidate.get("website",""),
            }]
            try:
                feedback_store.record_save(
                    sdk_query_fn=_sdk_query, session_id=_SESSION,
                    care_need=meta.get("care_need","unknown"),
                    facility_id=fid, facility_name=candidate["name"],
                )
            except Exception:
                pass
        return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, compare), shortlist

    h_bm_btn.click(
        _do_bookmark,
        [h_bm_id, results_state, shortlist_state, filter_state, sort_state,
         query_state, radius_state, meta_state, compare_state],
        [page_html, shortlist_state], api_name=_AN,
    )

    # ── Remove from shortlist ─────────────────────────────────────────────
    def _do_remove(rm_idx, results, shortlist, filter_val, sort_val, query, radius, meta, compare):
        try:
            shortlist = [s for i, s in enumerate(shortlist) if i != int(rm_idx)]
        except (ValueError, TypeError):
            pass
        return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, compare), shortlist

    h_rm_btn.click(
        _do_remove,
        [h_rm_idx, results_state, shortlist_state, filter_state, sort_state,
         query_state, radius_state, meta_state, compare_state],
        [page_html, shortlist_state], api_name=_AN,
    )

    # ── Clear shortlist ───────────────────────────────────────────────────
    def _do_clear(results, shortlist, filter_val, sort_val, query, radius, meta, compare):
        return _render_page(results, [], filter_val, sort_val, query, radius, meta, compare), []

    h_clear_btn.click(
        _do_clear,
        [results_state, shortlist_state, filter_state, sort_state,
         query_state, radius_state, meta_state, compare_state],
        [page_html, shortlist_state], api_name=_AN,
    )

    # ── Export ────────────────────────────────────────────────────────────
    def _do_export(shortlist):
        if not shortlist:
            return gr.update(visible=False)
        path = _export_csv(shortlist)
        return gr.update(value=path, visible=True)

    h_export_btn_gr.click(_do_export, [shortlist_state], [export_file], api_name=_AN)

    # ── Reset (logo / app-name click) ─────────────────────────────────────
    def _do_reset():
        html = _render_page([], [], "All", _DEFAULT_SORT, "", 5, compare=[])
        return html, [], [], {}, "All", _DEFAULT_SORT, "", 5, []

    h_reset_btn.click(
        _do_reset,
        inputs=[],
        outputs=[page_html, results_state, shortlist_state, meta_state,
                 filter_state, sort_state, query_state, radius_state, compare_state],
        api_name=_AN,
    )

    # ── Compare toggle ────────────────────────────────────────────────────
    def _do_compare(cmp_id, results, shortlist, compare, filter_val, sort_val, query, radius, meta):
        compare = compare or []
        fac = next((r for r in results if _s(r.get("id", r["name"])) == cmp_id), None)
        if fac is None:
            return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, compare), compare
        fid = _s(fac.get("id", fac["name"]))
        if any(c.get("id") == fid for c in compare):
            compare = [c for c in compare if c.get("id") != fid]
        elif len(compare) < 4:
            compare = compare + [fac]
        return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, compare), compare

    h_cmp_btn.click(
        _do_compare,
        [h_cmp_id, results_state, shortlist_state, compare_state, filter_state,
         sort_state, query_state, radius_state, meta_state],
        [page_html, compare_state], api_name=_AN,
    )

    def _do_cmp_clear(results, shortlist, filter_val, sort_val, query, radius, meta):
        return _render_page(results, shortlist, filter_val, sort_val, query, radius, meta, []), []

    h_cmp_clear_btn.click(
        _do_cmp_clear,
        [results_state, shortlist_state, filter_state, sort_state,
         query_state, radius_state, meta_state],
        [page_html, compare_state], api_name=_AN,
    )

    # ── Checklist / QR modal ──────────────────────────────────────────────
    def _do_chk_open(chk_id, results, meta):
        fac = next((r for r in (results or []) if _s(r.get("id", r["name"])) == chk_id), None)
        if not fac:
            return ""
        care_need = (meta or {}).get("care_need", "")
        return _checklist_modal_html(fac, care_need)

    h_chk_btn.click(
        _do_chk_open,
        [h_chk_id, results_state, meta_state],
        [checklist_html], api_name=_AN,
    )

    def _do_chk_close():
        return ""

    h_chk_close_btn.click(
        _do_chk_close, [], [checklist_html], api_name=_AN,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("DATABRICKS_APP_PORT", 8080)),
    )
