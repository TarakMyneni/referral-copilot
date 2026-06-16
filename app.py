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

SILVER_TABLE   = "mediguide.referral_copilot.facilities_silver"
CENTROIDS_TABLE = "mediguide.referral_copilot.location_centroids"

_BASE    = os.path.dirname(os.path.abspath(__file__))
_NEEDED  = set(COLUMNS.values())
_SESSION = str(uuid.uuid4())

_DATA_DIR       = os.path.join(_BASE, "data")
_FACILITIES_CSV = os.path.join(_DATA_DIR, "facilities_silver.csv")
_CENTROIDS_CSV  = os.path.join(_DATA_DIR, "location_centroids.csv")

# Spec colours
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
    "✓ Strong evidence":    ("strong",  GRN_PALE, "#97C459", GRN_DK,   "✓ Strong"),
    "◐ Partial evidence":   ("partial", "#FAEEDA", "#FAC775", "#633806", "◐ Partial"),
    "⚠️ Needs verification": ("verify",  "#FAECE7", "#F0997B", "#712B13", "⚠ Verify"),
}

FIELD_LABELS = {
    "specialties": "Specialties", "description": "Description",
    "capability": "Capability", "procedure": "Procedure",
    "equipment": "Equipment", "num_doctors": "No. of doctors",
    "capacity": "Capacity", "year_established": "Year established",
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
# Startup — background thread, CSV-first fallback
# ---------------------------------------------------------------------------

df                = pd.DataFrame()
centroids         = {}
_total_facilities = 0
_total_cities     = 0
_total_locations  = 0
_STARTUP_ERROR    = None
_data_ready       = False


def _background_load():
    global df, centroids, _total_facilities, _total_cities
    global _total_locations, _STARTUP_ERROR, _data_ready
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
        _total_facilities = len(df)
        _total_cities     = df[COLUMNS["city"]].dropna().nunique() if not df.empty else 0
        _total_locations  = len(centroids)
        _data_ready       = True
        print(f"[App] Ready — {_total_facilities:,} facilities, {_total_locations:,} locations")
    except Exception as _e:
        import traceback
        _STARTUP_ERROR = f"{type(_e).__name__}: {_e}\n\n{traceback.format_exc()}"
        print(f"[App] STARTUP FAILED:\n{_STARTUP_ERROR}")

threading.Thread(target=_background_load, daemon=True).start()

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _safe(v):
    s = str(v or "")
    return "" if s in ("nan", "None") else s


def _thumb(org_type):
    ot = (_safe(org_type)).lower()
    is_govt = any(w in ot for w in ("government", "govt", "public", "municipal", "district"))
    bg   = BG_GOVT if is_govt else GRN_PALE
    clr  = GOVT_CLR if is_govt else GRN_MID
    label = "GOVT" if is_govt else "PRIVATE"
    return bg, clr, label


def _build_card(rank, r, shortlist):
    ev         = r["evidence"]
    badge_text = trust_label(ev)
    _, tbg, tborder, tclr, tlabel = TRUST_CFG.get(
        badge_text, ("verify", "#FAECE7", "#F0997B", "#712B13", "⚠ Verify"))

    thumb_bg, thumb_clr, type_label = _thumb(r.get("org_type", ""))
    dist   = r.get("distance_km")
    dist_s = f"{dist} km" if dist is not None else "—"
    fid    = _safe(r.get("id", r["name"]))
    saved  = any(s.get("id") == fid for s in shortlist)

    # Evidence chips
    match_chips = "".join(
        f'<span style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
        f'border-radius:20px;padding:2px 8px;font-size:11px;color:{GRN_DK};'
        f'margin:2px 2px 2px 0;display:inline-block;">'
        f'{FIELD_LABELS.get(m["field"], m["field"])}</span>'
        for m in ev["matching"]
    )

    missing_items = ev.get("missing", [])
    missing_html = ""
    if missing_items:
        joined = " and ".join(FIELD_LABELS.get(m, m) for m in missing_items)
        missing_html = (
            f'<div style="font-size:11px;color:{AMBER};margin-top:4px;">'
            f'ℹ {joined} not reported</div>'
        )

    flags_html = "".join(
        f'<div style="font-size:11px;color:{RED_FLAG};margin-top:3px;">⚠ {f}</div>'
        for f in ev.get("suspicious", [])
    )

    phone   = _safe(r.get("phone", ""))
    website = _safe(r.get("website", ""))
    phone_html = (
        f'<a href="tel:{phone}" style="color:{GRN_MID};font-size:11px;text-decoration:none;">'
        f'📞 {phone[:18]}{"…" if len(phone)>18 else ""}</a>'
        if phone else ""
    )
    web_domain = website.replace("https://","").replace("http://","")
    web_html = (
        f'<a href="{website if website.startswith("http") else "https://"+website}" '
        f'target="_blank" style="color:{GRN_MID};font-size:11px;text-decoration:none;">'
        f'🌐 {web_domain[:24]}{"…" if len(web_domain)>24 else ""}</a>'
        if website else ""
    )

    bm_bg  = GRN_PALE if saved else BG_CARD
    bm_bdr = GRN_MID  if saved else BORDER
    bm_clr = GRN_MID  if saved else TXT_MUT
    bm_tip = "Saved" if saved else "Save to shortlist"

    # JS bookmark: sets hidden textbox value, dispatches input event, clicks hidden btn
    bm_js = (
        f"var b=document.querySelector('#bm-id-box textarea,#bm-id-box input');"
        f"if(b){{b.value='{fid}';b.dispatchEvent(new Event('input',{{bubbles:true}}));"
        f"var t=document.querySelector('#bm-trigger button');if(t)t.click();}}"
    )

    sem_pill = ""
    if r.get("sem_score", 0) > 0:
        sem_pill = (f'<span style="font-size:10px;background:#E8F0FE;color:#1A56DB;'
                    f'padding:1px 6px;border-radius:10px;margin-left:6px;">AI</span>')

    return f"""
<div style="display:flex;border:0.5px solid {BORDER};border-radius:10px;
            background:{BG_CARD};margin-bottom:10px;overflow:hidden;
            box-shadow:0 1px 2px rgba(0,0,0,0.04);">
  <!-- Thumb -->
  <div style="width:88px;min-width:88px;background:{thumb_bg};display:flex;
              flex-direction:column;align-items:center;justify-content:center;
              padding:12px 6px;position:relative;flex-shrink:0;">
    <div style="font-size:26px;opacity:0.45;color:{thumb_clr};">🏥</div>
    <div style="font-size:9px;font-weight:600;letter-spacing:0.4px;
                color:{thumb_clr};text-transform:uppercase;margin-top:4px;
                text-align:center;">{type_label}</div>
    <!-- Trust badge pinned to bottom -->
    <div style="position:absolute;bottom:0;left:0;right:0;
                background:{tbg};border-top:1px solid {tborder};
                color:{tclr};font-size:10px;font-weight:500;
                text-align:center;padding:5px 4px;">{tlabel}</div>
  </div>
  <!-- Body -->
  <div style="flex:1;padding:12px 14px;display:flex;flex-direction:column;gap:7px;
              min-width:0;">
    <!-- Name row -->
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">
      <div style="min-width:0;">
        <div style="font-size:14px;font-weight:500;color:{TXT_PRI};
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          <span style="font-size:11px;color:{TXT_MUT};font-weight:400;margin-right:4px;">
            #{rank}</span>{r['name']}{sem_pill}
        </div>
        <div style="font-size:11px;color:{TXT_MUT};margin-top:2px;">
          {_safe(r.get('city'))}, {_safe(r.get('state'))}
        </div>
      </div>
      <div style="font-size:11px;color:{TXT_SEC};white-space:nowrap;flex-shrink:0;">
        📍 {dist_s}
      </div>
    </div>
    <!-- Evidence -->
    {f'<div><div style="font-size:10px;color:{TXT_MUT};letter-spacing:0.3px;text-transform:uppercase;margin-bottom:4px;">Confirmed in</div><div>{match_chips}</div></div>' if match_chips else ''}
    {f'<div style="border-top:0.5px solid {BORDER};padding-top:6px;">{missing_html}{flags_html}</div>' if (missing_items or ev.get("suspicious")) else ''}
    <!-- Footer -->
    <div style="border-top:0.5px solid {BORDER};padding-top:8px;
                display:flex;align-items:center;justify-content:space-between;
                flex-wrap:wrap;gap:6px;margin-top:auto;">
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        {phone_html}
        {web_html}
      </div>
      <button onclick="{bm_js}" title="{bm_tip}"
        style="width:28px;height:28px;border-radius:50%;border:0.5px solid {bm_bdr};
               background:{bm_bg};cursor:pointer;font-size:14px;color:{bm_clr};
               display:flex;align-items:center;justify-content:center;flex-shrink:0;
               line-height:1;">{"🔖" if saved else "🏷"}</button>
    </div>
  </div>
</div>
"""


def _build_map(results, search_lat, search_lon, radius_km, location_name):
    import base64
    m = folium.Map(location=[search_lat, search_lon], zoom_start=8,
                   tiles="CartoDB Positron")
    folium.Circle(
        [search_lat, search_lon], radius=radius_km * 1000,
        color=GRN_MID, fill=True, fill_color=GRN_PALE,
        fill_opacity=0.08, weight=1.5,
    ).add_to(m)
    folium.CircleMarker(
        [search_lat, search_lon], radius=7, color=GRN_DK,
        fill=True, fill_color=GRN_DK, fill_opacity=0.9,
        tooltip=f"Search: {location_name}",
    ).add_to(m)
    color_map = {"strong": GRN_DK, "partial": "#BA7517", "verify": "#D85A30"}
    for r in results:
        lat, lon = r.get("lat"), r.get("lon")
        if not lat or not lon:
            continue
        badge_text = trust_label(r["evidence"])
        key, *_ = TRUST_CFG.get(badge_text, ("verify",))
        c = color_map.get(key, GRN_LT)
        dist_s = f"{r['distance_km']} km" if r.get("distance_km") is not None else "—"
        folium.CircleMarker(
            [lat, lon], radius=7, color=c,
            fill=True, fill_color=c, fill_opacity=0.85,
            tooltip=f"{r['name']} · {dist_s}",
        ).add_to(m)
    n = sum(1 for r in results if r.get("lat"))

    map_bytes = m.get_root().render().encode("utf-8")
    b64 = base64.b64encode(map_bytes).decode("ascii")
    label = (
        f'<div style="position:absolute;top:8px;left:8px;z-index:999;'
        f'background:rgba(255,255,255,0.9);padding:4px 8px;border-radius:6px;'
        f'font-size:11px;color:{TXT_SEC};border:0.5px solid {BORDER};">'
        f'{location_name} · {n} facilities · {radius_km} km radius</div>'
    )
    return (
        f'<div style="position:relative;width:100%;height:260px;border-radius:8px;'
        f'overflow:hidden;border:0.5px solid {BORDER};">'
        f'{label}'
        f'<iframe src="data:text/html;base64,{b64}" '
        f'style="width:100%;height:100%;border:none;"></iframe></div>'
    )


def _build_shortlist_panel(shortlist):
    count = len(shortlist)
    badge = (
        f'<span style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
        f'border-radius:10px;padding:1px 7px;font-size:10px;color:{GRN_DK};">'
        f'{count}</span>'
    )
    items_html = ""
    for i, s in enumerate(shortlist):
        dist_s  = f"{s['distance_km']} km" if s.get("distance_km") is not None else "—"
        _, tbg, tbdr, tclr, tlbl = TRUST_CFG.get(
            s.get("trust", ""), ("verify", "#FAECE7", "#F0997B", "#712B13", "⚠ Verify"))
        rm_js = (
            f"var b=document.querySelector('#rm-idx-box textarea,#rm-idx-box input');"
            f"if(b){{b.value='{i}';b.dispatchEvent(new Event('input',{{bubbles:true}}));"
            f"var t=document.querySelector('#rm-trigger button');if(t)t.click();}}"
        )
        items_html += f"""
<div style="background:{BG_PAGE};border:0.5px solid {BORDER_G};border-radius:8px;
            padding:7px 8px;margin-bottom:5px;display:flex;
            align-items:center;justify-content:space-between;gap:6px;">
  <div style="min-width:0;">
    <div style="font-size:12px;font-weight:500;color:{TXT_PRI};
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{s['name']}</div>
    <div style="font-size:10px;color:{TXT_SEC};">{dist_s} · {s['city']}
      <span style="background:{tbg};border:0.5px solid {tbdr};border-radius:8px;
                   padding:1px 5px;color:{tclr};margin-left:4px;">{tlbl}</span>
    </div>
  </div>
  <button onclick="{rm_js}"
    style="background:none;border:none;cursor:pointer;font-size:14px;
           color:{TXT_MUT};flex-shrink:0;line-height:1;">×</button>
</div>"""

    return f"""
<div style="background:{BG_CARD};border-top:1px solid {BORDER};padding:12px 14px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <span style="font-size:13px;font-weight:500;color:{TXT_PRI};">Shortlist {badge}</span>
    <span id="clear-shortlist-link" onclick="
      var b=document.querySelector('#clear-trigger button');if(b)b.click();"
      style="font-size:11px;color:{TXT_MUT};text-decoration:underline;cursor:pointer;">
      Clear</span>
  </div>
  {items_html if items_html else f'<div style="font-size:12px;color:{TXT_MUT};font-style:italic;">No facilities saved yet.</div>'}
</div>"""


def _export_csv(shortlist):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Name", "City", "State", "Distance (km)", "Trust", "Phone", "Website"])
    for s in shortlist:
        w.writerow([s["name"], s["city"], s["state"],
                    s.get("distance_km") or "", s.get("trust") or "",
                    s.get("phone") or "", s.get("website") or ""])
    path = os.path.join(_BASE, "data", "shortlist_export.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return path

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _render_results(results, shortlist, meta, location, radius_km):
    if not results:
        return (
            f'<div style="color:{TXT_SEC};padding:16px 0;font-size:13px;">'
            f'No facilities found for <b>{meta.get("care_need","")}</b> within '
            f'<b>{radius_km} km</b> of <b>{location}</b>. Try a larger radius.</div>'
        ), ""

    n_loc   = meta.get("located_count", len(results))
    n_unloc = meta.get("unlocated_count", 0)
    need    = meta.get("care_need", "")
    sem_tag = (f'<span style="background:#E8F0FE;color:#1A56DB;border-radius:10px;'
               f'padding:1px 7px;font-size:10px;margin-left:6px;">AI-enhanced</span>'
               if meta.get("semantic_active") else "")
    fuzzy   = (f' <span style="color:{TXT_MUT};font-size:11px;">'
               f'(matched: {meta["resolved_location"]})</span>'
               if meta.get("location_match_type") == "fuzzy" else "")
    unloc   = (f' + {n_unloc} unverified' if n_unloc else "")

    summary = (
        f'<div style="font-size:12px;color:{TXT_SEC};margin-bottom:10px;">'
        f'<b style="color:{TXT_PRI};">{n_loc}</b> facilities for '
        f'<b style="color:{TXT_PRI};">{need}</b>{sem_tag} within '
        f'<b style="color:{TXT_PRI};">{radius_km} km</b> of '
        f'<b style="color:{TXT_PRI};">{location}</b>{fuzzy}{unloc}'
        f'</div>'
    )
    cards = "".join(_build_card(i + 1, r, shortlist) for i, r in enumerate(results))
    map_html = _build_map(results, meta["search_lat"], meta["search_lon"],
                          radius_km, meta.get("resolved_location", location))
    return summary + cards, map_html


def do_search(query_text, radius_km, shortlist):
    if not _data_ready:
        msg = (_STARTUP_ERROR
               if _STARTUP_ERROR
               else "⏳ Data loading — please wait a moment and try again.")
        return (f'<div style="color:{TXT_MUT};padding:16px;font-size:13px;">{msg}</div>',
                [], "", "", shortlist, _build_shortlist_panel(shortlist))

    if not query_text.strip():
        return (
            f'<div style="color:{TXT_MUT};padding:16px;font-size:13px;font-style:italic;">'
            f'Enter something like <b>"dialysis near Jaipur"</b></div>',
            [], "", "", shortlist, _build_shortlist_panel(shortlist),
        )

    care_need, location = parse_combined_query(query_text, centroids)
    if not location:
        return (
            f'<div style="background:#FFF8E1;border-left:3px solid #F9A825;'
            f'padding:12px 14px;border-radius:0 6px 6px 0;font-size:13px;">'
            f'🤖 Couldn\'t find a location in that query.<br>'
            f'Try: <b>"dialysis near Jaipur"</b> or <b>"heart surgery near Mumbai"</b></div>',
            [], "", "", shortlist, _build_shortlist_panel(shortlist),
        )
    if not care_need:
        care_need = query_text

    results, meta = supervisor.run(
        df=df, centroids=centroids,
        care_need_query=care_need, location_query=location,
        radius_km=radius_km,
    )

    if "error" in meta:
        return (
            f'<div style="color:#c00;padding:12px;font-size:13px;">{meta["error"]}</div>',
            [], "", "", shortlist, _build_shortlist_panel(shortlist),
        )

    results_html, map_html = _render_results(results, shortlist, meta, location, radius_km)
    return (results_html, results, map_html,
            meta.get("care_need", care_need),
            shortlist, _build_shortlist_panel(shortlist))


def do_filter_sort(results, shortlist, filter_val, sort_val):
    if not results:
        return ""
    filtered = results
    if filter_val == "Government":
        filtered = [r for r in results
                    if any(w in (_safe(r.get("org_type",""))).lower()
                           for w in ("government","govt","public","municipal","district"))]
    elif filter_val == "Private":
        filtered = [r for r in results
                    if not any(w in (_safe(r.get("org_type",""))).lower()
                               for w in ("government","govt","public","municipal","district"))]
    if sort_val == "Best match":
        filtered = sorted(filtered, key=lambda r: -r.get("blended_score", 0))
    else:
        filtered = sorted(filtered, key=lambda r: (r.get("distance_km") or 9999))

    cards = "".join(_build_card(i + 1, r, shortlist) for i, r in enumerate(filtered))
    return cards


def do_bookmark(bm_id, results, shortlist, care_need):
    if not bm_id or not results:
        return shortlist, _build_shortlist_panel(shortlist), ""
    candidate = next((r for r in results
                      if _safe(r.get("id", r["name"])) == bm_id), None)
    if candidate is None:
        return shortlist, _build_shortlist_panel(shortlist), ""
    fid = _safe(candidate.get("id", candidate["name"]))
    already = next((i for i, s in enumerate(shortlist) if s.get("id") == fid), None)
    if already is not None:
        shortlist = [s for s in shortlist if s.get("id") != fid]
    else:
        shortlist = shortlist + [{
            "id": fid, "name": candidate["name"],
            "city": candidate["city"], "state": candidate["state"],
            "distance_km": candidate.get("distance_km"),
            "trust": trust_label(candidate["evidence"]),
            "phone": candidate.get("phone", ""),
            "website": candidate.get("website", ""),
        }]
        try:
            feedback_store.record_save(
                sdk_query_fn=_sdk_query, session_id=_SESSION,
                care_need=care_need or "unknown",
                facility_id=fid, facility_name=candidate["name"],
            )
        except Exception:
            pass
    return shortlist, _build_shortlist_panel(shortlist), ""


def do_remove(rm_idx, shortlist):
    try:
        idx = int(rm_idx)
        shortlist = [s for i, s in enumerate(shortlist) if i != idx]
    except (ValueError, TypeError):
        pass
    return shortlist, _build_shortlist_panel(shortlist), ""


def do_clear(shortlist):
    return [], _build_shortlist_panel([])


def do_export(shortlist):
    if not shortlist:
        return None
    return _export_csv(shortlist)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = f"""
* {{ box-sizing: border-box; }}
body, .gradio-container {{
  background: {BG_PAGE} !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  margin: 0 !important; padding: 0 !important;
}}
.gradio-container > .main {{ padding: 0 !important; }}
footer {{ display: none !important; }}
/* Topbar */
#suvidha-topbar {{
  background: {BG_CARD}; border-bottom: 1px solid {BORDER};
  padding: 10px 20px; display: flex; align-items: center; gap: 14px;
}}
/* Search inputs — minimal */
#where-box textarea, #need-box textarea,
#where-box input, #need-box input {{
  border: none !important; background: transparent !important;
  box-shadow: none !important; padding: 2px 4px !important;
  font-size: 13px !important; color: {TXT_PRI} !important;
  resize: none !important; min-height: 0 !important;
}}
#where-box label, #need-box label {{
  font-size: 9px !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: 0.5px;
  color: {TXT_MUT} !important; margin-bottom: 0 !important;
}}
/* Search pill wrapper */
.search-pill {{
  display: flex; align-items: center; flex: 1;
  border: 1.5px solid #B4B2A9; border-radius: 40px;
  background: {BG_CARD}; overflow: hidden; max-width: 640px;
}}
.pill-section {{
  flex: 1; padding: 6px 14px; min-width: 0;
  border-right: 0.5px solid {BORDER};
}}
.pill-section:last-of-type {{ border-right: none; }}
/* Search button */
#search-btn button {{
  width: 36px !important; height: 36px !important;
  border-radius: 50% !important; background: {GRN_MID} !important;
  color: #fff !important; border: none !important;
  font-size: 16px !important; padding: 0 !important;
  min-width: 0 !important;
}}
/* Filter chips */
.filter-chip button {{
  border-radius: 20px !important; padding: 4px 14px !important;
  font-size: 12px !important; border: 1px solid {BORDER_G} !important;
  background: {BG_CARD} !important; color: {GRN_MID} !important;
  min-width: 0 !important;
}}
.filter-chip.active button {{
  background: {GRN_MID} !important; color: {GRN_PALE} !important;
  border-color: {GRN_MID} !important;
}}
/* Sort */
#sort-radio .wrap {{ flex-direction: row !important; gap: 8px !important; }}
#sort-radio label span {{ font-size: 12px !important; color: {TXT_SEC} !important; }}
/* Results column */
#results-col {{ background: {BG_PAGE}; padding: 14px 20px; overflow-y: auto; }}
/* Right column */
#right-col {{ width: 320px; min-width: 320px; flex-shrink: 0; }}
/* Radius slider */
#radius-slider .wrap {{ gap: 6px !important; }}
/* Filter bar */
#filter-bar {{
  background: {BG_CARD}; border-bottom: 1px solid {BORDER};
  padding: 8px 20px;
}}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Suvidha — Healthcare Referrals", css=CSS) as demo:

    results_state   = gr.State([])
    shortlist_state = gr.State([])
    care_need_state = gr.State("")

    # ── Topbar ────────────────────────────────────────────────────────────
    with gr.Row(elem_id="suvidha-topbar"):
        # Logo
        gr.HTML(f"""
<div style="display:flex;align-items:center;gap:10px;flex-shrink:0;">
  <div style="width:32px;height:32px;background:{GRN_MID};border-radius:8px;
              display:flex;align-items:center;justify-content:center;font-size:16px;">
    📍</div>
  <div>
    <div style="font-size:16px;font-weight:500;color:{GRN_DK};line-height:1.2;">Suvidha</div>
    <div style="font-size:11px;color:{GRN_LT};line-height:1.2;">सुविधा · Healthcare Referrals</div>
  </div>
</div>""")

        # Search pill — three sections
        with gr.Group(elem_classes=["search-pill"]):
            with gr.Column(elem_classes=["pill-section"], min_width=0):
                where_box = gr.Textbox(label="WHERE", placeholder="City or district",
                                       elem_id="where-box", lines=1,
                                       container=False)
            with gr.Column(elem_classes=["pill-section"], min_width=0):
                need_box  = gr.Textbox(label="CARE NEED", placeholder="e.g. dialysis",
                                       elem_id="need-box", lines=1,
                                       container=False)
            with gr.Column(elem_classes=["pill-section"], min_width=0, scale=0):
                radius_slider = gr.Slider(minimum=10, maximum=500, value=150, step=10,
                                          label="RADIUS (km)", elem_id="radius-slider",
                                          container=False)

        search_btn = gr.Button("🔍", elem_id="search-btn", scale=0, min_width=36)

        # Saved count
        saved_count_html = gr.HTML(
            f'<div style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
            f'border-radius:20px;padding:5px 12px;font-size:12px;color:{GRN_MID};'
            f'white-space:nowrap;flex-shrink:0;">🔖 0 saved</div>',
            elem_id="saved-count",
        )

    # Also support free-text combined search (smart parse)
    with gr.Row(elem_id="smart-row",
                visible=True):
        smart_box = gr.Textbox(
            placeholder='"dialysis near Jaipur"  or  "heart surgery near Mumbai"',
            label="Or search naturally", scale=5, container=True,
        )
        smart_btn = gr.Button("Search", variant="primary", scale=1, min_width=80)

    gr.Examples(
        examples=["dialysis near Jaipur", "emergency surgery near Patna",
                  "cardiology near Gurgaon", "oncology near Delhi",
                  "maternity near Chennai", "ophthalmology near Hyderabad"],
        inputs=[smart_box], label="Quick examples",
    )

    # ── Filter + Sort bar ─────────────────────────────────────────────────
    with gr.Row(elem_id="filter-bar"):
        filter_all  = gr.Button("All",        elem_classes=["filter-chip", "active"], scale=0)
        filter_govt = gr.Button("Government", elem_classes=["filter-chip"],           scale=0)
        filter_priv = gr.Button("Private",    elem_classes=["filter-chip"],           scale=0)
        gr.HTML('<div style="flex:1;"></div>')
        sort_radio  = gr.Radio(["Nearest first", "Best match"], value="Nearest first",
                               label="Sort", elem_id="sort-radio", container=False,
                               scale=0)

    filter_state = gr.State("All")
    sort_state   = gr.State("Nearest first")

    # ── Main content: two-column ──────────────────────────────────────────
    with gr.Row(equal_height=False):
        # Left — results
        with gr.Column(scale=3, elem_id="results-col"):
            results_html = gr.HTML(
                f'<div style="color:{TXT_MUT};font-size:13px;font-style:italic;'
                f'padding:20px 0;">Search above to see facilities.</div>'
            )

        # Right — map + shortlist
        with gr.Column(scale=1, min_width=320, elem_id="right-col"):
            map_html      = gr.HTML()
            shortlist_html = gr.HTML(_build_shortlist_panel([]))
            export_btn    = gr.Button("⬇ Export shortlist", variant="secondary")
            export_file   = gr.File(label="Download", visible=False)

    # Hidden elements for JS bookmark / remove
    bm_id_box  = gr.Textbox(value="", visible=False, elem_id="bm-id-box")
    bm_trigger = gr.Button("bm",      visible=False, elem_id="bm-trigger")
    rm_idx_box = gr.Textbox(value="", visible=False, elem_id="rm-idx-box")
    rm_trigger = gr.Button("rm",      visible=False, elem_id="rm-trigger")
    clear_trigger = gr.Button("clr",  visible=False, elem_id="clear-trigger")

    # ── Event wiring ──────────────────────────────────────────────────────

    def _combined_search(where, need, radius, shortlist):
        query = f"{need} near {where}" if where and need else (where or need or "")
        return do_search(query, radius, shortlist)

    def _smart_search(query, radius, shortlist):
        return do_search(query, radius, shortlist)

    _SEARCH_OUTS = [results_html, results_state, map_html,
                    care_need_state, shortlist_state, shortlist_html]

    _AN = False  # api_name=False on every handler suppresses the schema crash

    search_btn.click(
        _combined_search,
        [where_box, need_box, radius_slider, shortlist_state],
        _SEARCH_OUTS, api_name=_AN,
    )
    where_box.submit(
        _combined_search,
        [where_box, need_box, radius_slider, shortlist_state],
        _SEARCH_OUTS, api_name=_AN,
    )
    need_box.submit(
        _combined_search,
        [where_box, need_box, radius_slider, shortlist_state],
        _SEARCH_OUTS, api_name=_AN,
    )
    smart_btn.click(
        _smart_search,
        [smart_box, radius_slider, shortlist_state],
        _SEARCH_OUTS, api_name=_AN,
    )
    smart_box.submit(
        _smart_search,
        [smart_box, radius_slider, shortlist_state],
        _SEARCH_OUTS, api_name=_AN,
    )

    # Filter/sort
    def _set_filter(val, results, shortlist, sort_val):
        cards = do_filter_sort(results, shortlist, val, sort_val)
        return val, cards

    filter_all.click(
        lambda r, sl, sv: _set_filter("All", r, sl, sv),
        [results_state, shortlist_state, sort_state],
        [filter_state, results_html], api_name=_AN,
    )
    filter_govt.click(
        lambda r, sl, sv: _set_filter("Government", r, sl, sv),
        [results_state, shortlist_state, sort_state],
        [filter_state, results_html], api_name=_AN,
    )
    filter_priv.click(
        lambda r, sl, sv: _set_filter("Private", r, sl, sv),
        [results_state, shortlist_state, sort_state],
        [filter_state, results_html], api_name=_AN,
    )
    sort_radio.change(
        lambda sv, r, sl, fv: (sv, do_filter_sort(r, sl, fv, sv)),
        [sort_radio, results_state, shortlist_state, filter_state],
        [sort_state, results_html], api_name=_AN,
    )

    # Bookmark
    def _bookmark(bm_id, results, shortlist, care_need):
        sl, sl_html, _ = do_bookmark(bm_id, results, shortlist, care_need)
        n = len(sl)
        saved_html = (
            f'<div style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
            f'border-radius:20px;padding:5px 12px;font-size:12px;color:{GRN_MID};'
            f'white-space:nowrap;flex-shrink:0;">🔖 {n} saved</div>'
        )
        cards = "".join(_build_card(i + 1, r, sl)
                        for i, r in enumerate(results))
        return sl, sl_html, "", saved_html, cards

    bm_trigger.click(
        _bookmark,
        [bm_id_box, results_state, shortlist_state, care_need_state],
        [shortlist_state, shortlist_html, bm_id_box, saved_count_html, results_html],
        api_name=_AN,
    )

    # Remove from shortlist
    def _remove(rm_idx, shortlist):
        sl, sl_html, _ = do_remove(rm_idx, shortlist)
        n = len(sl)
        saved_html = (
            f'<div style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
            f'border-radius:20px;padding:5px 12px;font-size:12px;color:{GRN_MID};'
            f'white-space:nowrap;flex-shrink:0;">🔖 {n} saved</div>'
        )
        return sl, sl_html, "", saved_html

    rm_trigger.click(
        _remove,
        [rm_idx_box, shortlist_state],
        [shortlist_state, shortlist_html, rm_idx_box, saved_count_html],
        api_name=_AN,
    )

    clear_trigger.click(
        lambda sl: ([],
                    _build_shortlist_panel([]),
                    f'<div style="background:{GRN_PALE};border:0.5px solid {BORDER_G};'
                    f'border-radius:20px;padding:5px 12px;font-size:12px;color:{GRN_MID};'
                    f'white-space:nowrap;flex-shrink:0;">🔖 0 saved</div>'),
        [shortlist_state],
        [shortlist_state, shortlist_html, saved_count_html],
        api_name=_AN,
    )

    export_btn.click(do_export, [shortlist_state], [export_file], api_name=_AN)
    export_btn.click(lambda: gr.update(visible=True), outputs=[export_file], api_name=_AN)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("DATABRICKS_APP_PORT", 8080)),
    )
