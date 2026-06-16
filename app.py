import base64
import csv
import io
import os
import threading
import uuid

import folium
import gradio as gr
import pandas as pd

from src.config import COLUMNS
from src.geo import resolve_location
from src.ranking import parse_combined_query
from src.evidence import evaluate_evidence, trust_label
from src import agent as supervisor
from src import feedback as feedback_store

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SILVER_TABLE    = "mediguide.referral_copilot.facilities_silver"
CENTROIDS_TABLE = "mediguide.referral_copilot.location_centroids"

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
        _data_ready = True
        print(f"[App] Ready — {len(df):,} facilities, {len(centroids):,} locations")
    except Exception as _e:
        import traceback
        _STARTUP_ERROR = f"{type(_e).__name__}: {_e}\n\n{traceback.format_exc()}"
        print(f"[App] STARTUP FAILED:\n{_STARTUP_ERROR}")

threading.Thread(target=_background_load, daemon=True).start()

# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _s(v):
    s = str(v or "")
    return "" if s in ("nan", "None") else s

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

_JS = """
<script>
(function(){
  function _set(sel, val) {
    var el = document.querySelector(sel);
    if(el){ el.value = val; el.dispatchEvent(new Event('input', {bubbles:true})); }
  }
  function _click(sel) {
    var el = document.querySelector(sel);
    if(el) el.click();
  }
  window.sSearch = function() {
    var w = (document.getElementById('vi-where')||{}).value || '';
    var n = (document.getElementById('vi-need')||{}).value  || '';
    var r = (document.getElementById('vi-radius')||{}).value || '50';
    _set('#h-where textarea, #h-where input', w);
    _set('#h-need  textarea, #h-need  input', n);
    _set('#h-rad   input[type=number], #h-rad input', r);
    setTimeout(function(){ _click('#h-search button'); }, 120);
  };
  window.sFilter = function(v) {
    _set('#h-filter textarea, #h-filter input', v);
    setTimeout(function(){ _click('#h-filter-btn button'); }, 120);
  };
  window.sSort = function(v) {
    _set('#h-sort textarea, #h-sort input', v);
    setTimeout(function(){ _click('#h-sort-btn button'); }, 120);
  };
  window.sBm = function(id) {
    _set('#h-bm-id textarea, #h-bm-id input', id);
    setTimeout(function(){ _click('#h-bm-btn button'); }, 120);
  };
  window.sRm = function(idx) {
    _set('#h-rm-idx textarea, #h-rm-idx input', String(idx));
    setTimeout(function(){ _click('#h-rm-btn button'); }, 120);
  };
  window.sClear  = function(){ setTimeout(function(){ _click('#h-clear button');  }, 120); };
  window.sExport = function(){ setTimeout(function(){ _click('#h-export button'); }, 120); };
  document.addEventListener('keydown', function(e){
    if(e.key==='Enter' && ['vi-where','vi-need','vi-radius'].indexOf(e.target.id) >= 0)
      window.sSearch();
  });
})();
</script>
"""


def _topbar_html(where, need, radius, n_saved):
    logo = (
        '<svg viewBox="0 0 40 40" width="38" height="38" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="40" height="40" rx="9" fill="#3B6D11"/>'
        '<path d="M20 30 C20 30 9 21 9 14.5 A6.5 6.5 0 0 1 20 10 A6.5 6.5 0 0 1 31 14.5 C31 21 20 30 20 30Z" fill="white" opacity="0.9"/>'
        '<circle cx="20" cy="14" r="3" fill="#3B6D11" opacity="0.6"/>'
        '</svg>'
    )
    return f"""
<div style="background:#fff;border-bottom:1px solid {BORDER};padding:10px 20px;
            display:flex;align-items:center;gap:16px;flex-shrink:0;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div class="sv-topbar-logo" style="display:flex;align-items:center;gap:10px;flex-shrink:0;">
    {logo}
    <span style="font-size:22px;font-weight:700;color:{GRN_DK};letter-spacing:-0.3px;">SUVIDHA</span>
  </div>
  <div style="display:flex;align-items:center;border:1.5px solid #B4B2A9;border-radius:40px;
              background:#fff;flex:1;max-width:640px;height:48px;overflow:hidden;">
    <div class="sv-pill-section" style="flex:1;padding:4px 18px;border-right:0.5px solid {BORDER};height:100%;
                display:flex;flex-direction:column;justify-content:center;">
      <div style="font-size:9px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;
                  letter-spacing:0.5px;line-height:1;">WHERE</div>
      <input id="vi-where" value="{where}" placeholder="City or district"
        style="border:none;outline:none;background:transparent;font-size:13px;
               color:{TXT_PRI};width:100%;padding:0;margin-top:2px;font-family:inherit;">
    </div>
    <div class="sv-pill-section" style="flex:1;padding:4px 18px;border-right:0.5px solid {BORDER};height:100%;
                display:flex;flex-direction:column;justify-content:center;">
      <div style="font-size:9px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;
                  letter-spacing:0.5px;line-height:1;">CARE NEED</div>
      <input id="vi-need" value="{need}" placeholder="e.g. dialysis"
        style="border:none;outline:none;background:transparent;font-size:13px;
               color:{TXT_PRI};width:100%;padding:0;margin-top:2px;font-family:inherit;">
    </div>
    <div class="sv-pill-radius sv-pill-section" style="flex:0 0 110px;padding:4px 18px;border-right:0.5px solid {BORDER};height:100%;
                display:flex;flex-direction:column;justify-content:center;">
      <div style="font-size:9px;font-weight:600;color:{TXT_MUT};text-transform:uppercase;
                  letter-spacing:0.5px;line-height:1;">RADIUS</div>
      <div style="display:flex;align-items:center;gap:3px;margin-top:2px;">
        <input id="vi-radius" type="number" min="10" max="500" step="10" value="{int(radius or 50)}"
          style="border:none;outline:none;background:transparent;font-size:13px;
                 color:{TXT_PRI};width:38px;padding:0;font-family:inherit;">
        <span style="font-size:13px;color:{TXT_PRI};">km</span>
      </div>
    </div>
    <button onclick="sSearch()"
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
        return (f'<button onclick="sFilter(\'{val}\')" style="{st}'
                f'border-radius:20px;padding:5px 14px;font-size:12px;cursor:pointer;'
                f'font-family:inherit;">{txt}</button>')

    new_sort = "Best match" if sort_val == "Nearest first" else "Nearest first"
    sort_btn = (
        f'<button onclick="sSort(\'{new_sort}\')" '
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


def _card_html(rank, r, shortlist):
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

    # Evidence chips
    chips = "".join(
        f'<span style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
        f'border-radius:12px;padding:2px 10px;font-size:11px;color:{GRN_DK};'
        f'margin:2px 3px 2px 0;display:inline-block;">'
        f'{FIELD_LABELS.get(m["field"], m["field"])}</span>'
        for m in ev["matching"]
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
    ph_html = (
        f'<a href="tel:{phone}" style="color:{GRN_MID};font-size:11px;'
        f'text-decoration:none;display:inline-flex;align-items:center;gap:3px;">'
        f'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="{GRN_MID}" '
        f'stroke-width="2" stroke-linecap="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.89 12a19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 3.81 1h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 8.91a16 16 0 0 0 6 6l.96-.96a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>'
        f'{phone[:20]}{"…" if len(phone)>20 else ""}</a>'
    ) if phone else ""
    dom = website.replace("https://","").replace("http://","").rstrip("/")
    wb_html = (
        f'<a href="{website if website.startswith("http") else "https://"+website}" '
        f'target="_blank" style="color:{GRN_MID};font-size:11px;'
        f'text-decoration:none;display:inline-flex;align-items:center;gap:3px;">'
        f'<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="{GRN_MID}" '
        f'stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/>'
        f'<path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
        f'{dom[:22]}{"…" if len(dom)>22 else ""}</a>'
    ) if website else ""

    bm_bg  = GRN_PALE if saved else BG_CARD
    bm_bdr = GRN_MID  if saved else BORDER
    bm_ico = (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="{fc}" stroke="{fc}" stroke-width="2" '
        'stroke-linecap="round"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>'
    ).replace("{fc}", GRN_MID if saved else TXT_MUT)

    # Escape fid for inline JS (no quotes/backslashes expected in IDs)
    safe_fid = fid.replace("'", "\\'")

    sem_pill = ""
    if r.get("sem_score", 0) > 0:
        sem_pill = (
            '<span style="font-size:10px;background:#E8F0FE;color:#1A56DB;'
            'border-radius:10px;padding:1px 6px;margin-left:5px;vertical-align:middle;">AI</span>'
        )

    evidence_section = ""
    if chips:
        evidence_section += (
            f'<div style="margin-top:8px;">'
            f'<div style="font-size:9px;font-weight:600;color:{TXT_MUT};'
            f'text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;">CONFIRMED IN</div>'
            f'<div>{chips}</div>'
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
      </div>
      <div style="font-size:11px;color:{TXT_SEC};white-space:nowrap;flex-shrink:0;
                  display:flex;align-items:center;gap:3px;">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="{GRN_LT}" stroke="{GRN_LT}"
             stroke-width="0"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
        {dist_s}
      </div>
    </div>
    {evidence_section}
    <!-- Footer -->
    <div style="border-top:0.5px solid {BORDER};margin-top:10px;padding-top:8px;
                display:flex;align-items:center;justify-content:space-between;gap:8px;">
      <div style="display:flex;gap:14px;flex-wrap:wrap;min-width:0;">{footer_links}</div>
      <button onclick="sBm('{safe_fid}')"
        style="width:30px;height:30px;border-radius:50%;border:0.5px solid {bm_bdr};
               background:{bm_bg};cursor:pointer;display:flex;align-items:center;
               justify-content:center;flex-shrink:0;padding:0;">
        {bm_ico}
      </button>
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


def _empty_map_html():
    return (
        f'<div style="width:100%;height:100%;min-height:260px;background:#F2F4F0;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:12px;color:{TXT_MUT};">Map will appear after search</div>'
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
  <button onclick="sRm({i})"
    style="background:none;border:none;cursor:pointer;font-size:16px;
           color:{TXT_MUT};flex-shrink:0;padding:0;line-height:1;">×</button>
</div>"""

    export_btn = (
        f'<button onclick="sExport()" style="width:100%;background:{GRN_MID};color:{GRN_PALE};'
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
    <span onclick="sClear()"
      style="font-size:11px;color:{TXT_MUT};text-decoration:underline;cursor:pointer;">Clear</span>
  </div>
  {items if items else empty_msg}
  {export_btn}
</div>"""


def _render_page(results, shortlist, filter_val, sort_val, where, need, radius, meta=None):
    meta = meta or {}
    n_saved = len(shortlist)

    # Apply filter + sort
    filtered = list(results)
    if filter_val == "Government":
        filtered = [r for r in filtered if _is_govt(r)]
    elif filter_val == "Private":
        filtered = [r for r in filtered if not _is_govt(r)]
    if sort_val == "Best match":
        filtered.sort(key=lambda r: -r.get("blended_score", 0))
    else:
        filtered.sort(key=lambda r: r.get("distance_km") or 9999)

    # Results body
    if not results and not meta:
        results_body = (
            f'<div style="color:{TXT_MUT};font-size:13px;font-style:italic;padding:20px 0;">'
            f'Search above to see facilities.</div>'
        )
        if _STARTUP_ERROR:
            results_body = (
                f'<div style="color:#c00;font-size:12px;padding:16px 0;">{_STARTUP_ERROR}</div>'
            )
        elif not _data_ready:
            results_body = (
                f'<div style="font-size:13px;color:{TXT_MUT};padding:20px 0;">'
                f'⏳ Data loading — try searching in a moment.</div>'
            )
    elif "error" in meta:
        results_body = (
            f'<div style="background:#FFF8E1;border-left:3px solid #F9A825;'
            f'padding:12px 16px;border-radius:0 6px 6px 0;font-size:13px;color:{TXT_PRI};">'
            f'🤖 {meta["error"]}<br><span style="color:{TXT_MUT};font-size:12px;">'
            f'Try: <b>dialysis near Jaipur</b></span></div>'
        )
    elif not filtered:
        loc = meta.get("resolved_location", where)
        results_body = (
            f'<div style="color:{TXT_SEC};font-size:13px;padding:16px 0;">'
            f'No {filter_val.lower()} facilities found for '
            f'<b>{meta.get("care_need", need)}</b> within '
            f'<b>{int(radius or 50)} km</b> of <b>{loc}</b>. '
            f'Try "All" filter or a larger radius.</div>'
        )
    else:
        loc     = meta.get("resolved_location", where)
        n_loc   = meta.get("located_count", len(results))
        n_unloc = meta.get("unlocated_count", 0)
        unloc_s = f" + {n_unloc} unverified" if n_unloc else ""
        fuzzy   = (f' <span style="color:{TXT_MUT};font-size:11px;">'
                   f'(matched: {loc})</span>'
                   if meta.get("location_match_type") == "fuzzy" else "")
        summary = (
            f'<div style="font-size:12px;color:{TXT_SEC};margin-bottom:12px;">'
            f'<b style="color:{TXT_PRI};">{n_loc}</b> facilities for '
            f'<b style="color:{TXT_PRI};">{meta.get("care_need", need)}</b> within '
            f'<b style="color:{TXT_PRI};">{int(radius or 50)} km</b> of '
            f'<b style="color:{TXT_PRI};">{loc}</b>{fuzzy}{unloc_s}</div>'
        )
        cards = "".join(_card_html(i + 1, r, shortlist) for i, r in enumerate(filtered))
        results_body = summary + cards

    # Map
    if meta.get("search_lat"):
        right_map = _map_html(
            results, meta["search_lat"], meta["search_lon"],
            int(radius or 50), meta.get("resolved_location", where),
        )
    else:
        right_map = _empty_map_html()

    shortlist_panel = _shortlist_panel_html(shortlist)
    topbar  = _topbar_html(where, need, radius or 50, n_saved)
    filterbar = _filterbar_html(filter_val, sort_val, len(results))

    responsive_css = f"""
<style>
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
.sv-pill-radius {{ display:flex !important; }}

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
"""

with gr.Blocks(css=CSS, title="Suvidha — Healthcare Referrals") as demo:

    # State
    results_state  = gr.State([])
    shortlist_state = gr.State([])
    meta_state     = gr.State({})
    filter_state   = gr.State("All")
    sort_state     = gr.State("Nearest first")
    where_state    = gr.State("")
    need_state     = gr.State("")
    radius_state   = gr.State(50)

    # Main visible output
    page_html = gr.HTML(_render_page([], [], "All", "Nearest first", "", "", 50))

    # Hidden Gradio bridge components (JS writes to these, Python reads them)
    h_where     = gr.Textbox(value="",    visible=False, elem_id="h-where")
    h_need      = gr.Textbox(value="",    visible=False, elem_id="h-need")
    h_rad       = gr.Number( value=50,    visible=False, elem_id="h-rad")
    h_search    = gr.Button("s",          visible=False, elem_id="h-search")
    h_filter    = gr.Textbox(value="All", visible=False, elem_id="h-filter")
    h_filter_btn = gr.Button("f",         visible=False, elem_id="h-filter-btn")
    h_sort      = gr.Textbox(value="Nearest first", visible=False, elem_id="h-sort")
    h_sort_btn  = gr.Button("so",         visible=False, elem_id="h-sort-btn")
    h_bm_id     = gr.Textbox(value="",   visible=False, elem_id="h-bm-id")
    h_bm_btn    = gr.Button("bm",        visible=False, elem_id="h-bm-btn")
    h_rm_idx    = gr.Textbox(value="",   visible=False, elem_id="h-rm-idx")
    h_rm_btn    = gr.Button("rm",        visible=False, elem_id="h-rm-btn")
    h_clear     = gr.Button("cl",        visible=False, elem_id="h-clear")
    h_export    = gr.Button("ex",        visible=False, elem_id="h-export")

    export_file = gr.File(label="Download shortlist", visible=False)

    # ── Search ────────────────────────────────────────────────────────────
    def _do_search(where, need, radius, shortlist, filter_val, sort_val):
        radius = int(radius or 50)
        if not _data_ready:
            msg = _STARTUP_ERROR or "⏳ Data loading — try in a moment."
            m = {"error": msg}
            return (_render_page([], shortlist, filter_val, sort_val, where, need, radius, m),
                    [], m, where, need, radius)

        query = f"{need} near {where}" if where and need else (where or need or "").strip()
        if not query:
            m = {"error": "Enter a location and care need."}
            return (_render_page([], shortlist, filter_val, sort_val, where, need, radius, m),
                    [], m, where, need, radius)

        care_need, location = parse_combined_query(query, centroids)
        if not location:
            m = {"error": f"Couldn't find a location in '{query}'. Try 'dialysis near Jaipur'."}
            return (_render_page([], shortlist, filter_val, sort_val, where, need, radius, m),
                    [], m, where, need, radius)
        if not care_need:
            care_need = query

        results, meta = supervisor.run(
            df=df, centroids=centroids,
            care_need_query=care_need, location_query=location,
            radius_km=radius,
        )
        html = _render_page(results, shortlist, filter_val, sort_val, where, need, radius, meta)
        return html, results, meta, where, need, radius

    h_search.click(
        _do_search,
        [h_where, h_need, h_rad, shortlist_state, filter_state, sort_state],
        [page_html, results_state, meta_state, where_state, need_state, radius_state],
        api_name=_AN,
    )

    # ── Filter ────────────────────────────────────────────────────────────
    def _do_filter(fv, results, shortlist, sort_val, where, need, radius, meta):
        html = _render_page(results, shortlist, fv, sort_val, where, need, radius, meta)
        return html, fv

    h_filter_btn.click(
        _do_filter,
        [h_filter, results_state, shortlist_state, sort_state,
         where_state, need_state, radius_state, meta_state],
        [page_html, filter_state], api_name=_AN,
    )

    # ── Sort ──────────────────────────────────────────────────────────────
    def _do_sort(sv, results, shortlist, filter_val, where, need, radius, meta):
        html = _render_page(results, shortlist, filter_val, sv, where, need, radius, meta)
        return html, sv

    h_sort_btn.click(
        _do_sort,
        [h_sort, results_state, shortlist_state, filter_state,
         where_state, need_state, radius_state, meta_state],
        [page_html, sort_state], api_name=_AN,
    )

    # ── Bookmark ──────────────────────────────────────────────────────────
    def _do_bookmark(bm_id, results, shortlist, filter_val, sort_val,
                     where, need, radius, meta):
        if not bm_id or not results:
            return _render_page(results, shortlist, filter_val, sort_val, where, need, radius, meta), shortlist
        candidate = next((r for r in results if _s(r.get("id", r["name"])) == bm_id), None)
        if candidate is None:
            return _render_page(results, shortlist, filter_val, sort_val, where, need, radius, meta), shortlist
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
        html = _render_page(results, shortlist, filter_val, sort_val, where, need, radius, meta)
        return html, shortlist

    h_bm_btn.click(
        _do_bookmark,
        [h_bm_id, results_state, shortlist_state, filter_state, sort_state,
         where_state, need_state, radius_state, meta_state],
        [page_html, shortlist_state], api_name=_AN,
    )

    # ── Remove from shortlist ─────────────────────────────────────────────
    def _do_remove(rm_idx, results, shortlist, filter_val, sort_val,
                   where, need, radius, meta):
        try:
            idx = int(rm_idx)
            shortlist = [s for i, s in enumerate(shortlist) if i != idx]
        except (ValueError, TypeError):
            pass
        html = _render_page(results, shortlist, filter_val, sort_val, where, need, radius, meta)
        return html, shortlist

    h_rm_btn.click(
        _do_remove,
        [h_rm_idx, results_state, shortlist_state, filter_state, sort_state,
         where_state, need_state, radius_state, meta_state],
        [page_html, shortlist_state], api_name=_AN,
    )

    # ── Clear shortlist ───────────────────────────────────────────────────
    def _do_clear(results, shortlist, filter_val, sort_val, where, need, radius, meta):
        html = _render_page(results, [], filter_val, sort_val, where, need, radius, meta)
        return html, []

    h_clear.click(
        _do_clear,
        [results_state, shortlist_state, filter_state, sort_state,
         where_state, need_state, radius_state, meta_state],
        [page_html, shortlist_state], api_name=_AN,
    )

    # ── Export ────────────────────────────────────────────────────────────
    def _do_export(shortlist):
        if not shortlist:
            return gr.update(visible=False)
        path = _export_csv(shortlist)
        return gr.update(value=path, visible=True)

    h_export.click(_do_export, [shortlist_state], [export_file], api_name=_AN)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("DATABRICKS_APP_PORT", 8080)),
    )
