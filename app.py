import os
import uuid

import folium
import gradio as gr
import pandas as pd

from src.config import COLUMNS
from src.geo import build_city_centroids, build_pincode_centroids
from src.ranking import parse_combined_query
from src.evidence import trust_label
from src import agent as supervisor
from src import feedback as feedback_store


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SILVER_TABLE   = "mediguide.referral_copilot.facilities_silver"
PINCODE_SILVER = "mediguide.referral_copilot.pincode_silver"

_BASE      = os.path.dirname(os.path.abspath(__file__))
_NEEDED    = set(COLUMNS.values())
_SESSION   = str(uuid.uuid4())   # per-deployment session ID for interaction logs


# ---------------------------------------------------------------------------
# Databricks SDK query helper
# ---------------------------------------------------------------------------

def _sdk_query(statement, wait="120s"):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    w = WorkspaceClient()
    warehouses = list(w.warehouses.list())
    print(f"[SDK] Found {len(warehouses)} warehouse(s): {[(wh.name, wh.id, wh.state) for wh in warehouses]}")
    if not warehouses:
        raise RuntimeError("No SQL warehouse found — check warehouse permissions for the app service principal.")
    wh_id = warehouses[0].id
    print(f"[SDK] Using warehouse: {warehouses[0].name} ({wh_id})")

    r = w.statement_execution.execute_statement(
        warehouse_id=wh_id,
        statement=statement,
        wait_timeout=wait,
        row_limit=20000,
    )
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Query failed: {r.status.error}")

    col_names = [c.name for c in r.manifest.schema.columns]
    rows = []
    chunk = r.result
    while chunk:
        if chunk.data_array:
            rows.extend(chunk.data_array)
        if chunk.next_chunk_index is None:
            break
        chunk = w.statement_execution.get_statement_result_chunk_n(
            statement_id=r.statement_id,
            chunk_index=chunk.next_chunk_index,
        )
    return col_names, rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_silver():
    needed_sql = ", ".join(f"`{c}`" for c in _NEEDED if c)
    cols, rows = _sdk_query(f"SELECT {needed_sql} FROM {SILVER_TABLE}")
    df = pd.DataFrame(rows, columns=cols)
    print(f"[App] Loaded {len(df):,} facilities from {SILVER_TABLE}")
    return df


def _load_pincode():
    try:
        cols, rows = _sdk_query(
            f"SELECT district, division, region, state, latitude, longitude "
            f"FROM {PINCODE_SILVER}",
            wait="30s",
        )
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        print(f"[App] Pincode Silver unavailable: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Startup: load data, build indexes, wire feedback
# ---------------------------------------------------------------------------

_STARTUP_ERROR = None

try:
    df          = _load_silver()
    _pincode_df = _load_pincode()

    _facility_centroids = build_city_centroids(df, COLUMNS["city"], COLUMNS["latitude"], COLUMNS["longitude"])
    _pincode_centroids  = build_pincode_centroids(_pincode_df)
    centroids = {**_pincode_centroids, **_facility_centroids}

    feedback_store.load(_sdk_query)

    _total_facilities = len(df)
    _total_cities     = df[COLUMNS["city"]].dropna().nunique()
    _total_locations  = len(centroids)

except Exception as _e:
    import traceback
    _STARTUP_ERROR = f"{type(_e).__name__}: {_e}\n\n{traceback.format_exc()}"
    print(f"[App] STARTUP FAILED:\n{_STARTUP_ERROR}")
    df                  = pd.DataFrame()
    centroids           = {}
    _total_facilities   = 0
    _total_cities       = 0
    _total_locations    = 0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "specialties":      "specialties",
    "description":      "description",
    "capability":       "capability",
    "procedure":        "procedure",
    "equipment":        "equipment",
    "num_doctors":      "number of doctors",
    "capacity":         "capacity",
    "year_established": "year established",
    "source_urls":      "source URL",
}

_TRUST_STYLES = {
    "✓ Strong evidence":    ("background:#1a7f4b;color:#fff;", "✓ Strong"),
    "◐ Partial evidence":   ("background:#b8621a;color:#fff;", "◐ Partial"),
    "⚠️ Needs verification": ("background:#c0392b;color:#fff;", "⚠ Verify"),
}

_GEO_NOTE = {
    "POSTCODE": "<span style='font-size:10px;color:#888;'>(postcode centroid)</span>",
    "DISTRICT": "<span style='font-size:10px;color:#888;'>(district centroid)</span>",
    "DIVISION": "<span style='font-size:10px;color:#999;'>(division centroid)</span>",
    "REGION":   "<span style='font-size:10px;color:#f90;'>(region centroid)</span>",
    "STATE":    "<span style='font-size:10px;color:#f90;'>(state centroid)</span>",
    "UNKNOWN":  "<span style='font-size:10px;color:#c00;'>(location unverified)</span>",
}


def _phone_link(phone):
    if not phone or str(phone) in ("nan", "None", ""):
        return ""
    p = str(phone)
    display = p[:20] + "…" if len(p) > 20 else p
    return f'📞 <a href="tel:{p}" style="color:#0B2026;">{display}</a>'


def _website_link(site):
    if not site or str(site) in ("nan", "None", ""):
        return ""
    s = str(site)
    url = s if s.startswith("http") else f"https://{s}"
    display = s.replace("https://", "").replace("http://", "")
    if len(display) > 35:
        display = display[:35] + "…"
    return f'🌐 <a href="{url}" target="_blank" style="color:#0B2026;">{display}</a>'


def format_result_card(rank, r):
    ev         = r["evidence"]
    badge_text = trust_label(ev)
    badge_style, badge_label = _TRUST_STYLES.get(badge_text, ("background:#666;color:#fff;", badge_text))
    badge_html = (
        f'<span style="padding:4px 12px;border-radius:20px;font-size:11px;'
        f'font-weight:600;white-space:nowrap;{badge_style}">{badge_label}</span>'
    )

    # Distance + geo precision note
    dist      = r.get("distance_km")
    dist_str  = f"{dist} km away" if dist is not None else "distance unknown"
    geo_note  = _GEO_NOTE.get(r.get("geo_source", ""), "")
    geo_line  = f"{dist_str} {geo_note}".strip()

    # Org type
    org = r.get("org_type", "")
    org_tag = f" · {org.title()}" if org and org not in ("nan", "None") else ""

    meta_line = f"📍 {r['city']}, {r['state']} &nbsp;·&nbsp; {geo_line}{org_tag}"

    # Score pills (only shown when semantic or feedback is active)
    score_pills = ""
    if r.get("sem_score", 0) > 0:
        score_pills += (
            f'<span style="font-size:10px;background:#e8f4fd;color:#1565c0;'
            f'padding:2px 7px;border-radius:10px;margin-right:4px;">🔍 sem {r["sem_score"]:.2f}</span>'
        )
    if r.get("fb_saves", 0) > 0:
        score_pills += (
            f'<span style="font-size:10px;background:#fdf6e3;color:#7d5a00;'
            f'padding:2px 7px;border-radius:10px;">⭐ {r["fb_saves"]} save(s)</span>'
        )

    def _col(title, color, items):
        if not items:
            return ""
        rows = "".join(f'<div style="font-size:12px;padding:1px 0;">{i}</div>' for i in items)
        return (
            f'<div><div style="font-size:12px;font-weight:600;color:{color};margin-bottom:3px;">'
            f'{title}</div>{rows}</div>'
        )

    matching_items = [f'• {FIELD_LABELS.get(m["field"], m["field"])}' for m in ev["matching"]]
    missing_items  = [f'• {FIELD_LABELS.get(m, m)} not reported'      for m in ev["missing"]]

    evidence_html = (
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));'
        f'gap:10px;margin-top:10px;">'
        + _col("Matching evidence", "#1a7f4b", matching_items)
        + _col("Missing data",      "#b8621a", missing_items)
        + _col("⚠ Flags",           "#c0392b", ev["suspicious"])
        + "</div>"
    )

    contact_parts = list(filter(None, [_phone_link(r.get("phone", "")),
                                       _website_link(r.get("website", ""))]))
    contact_html = ""
    if contact_parts:
        contact_html = (
            f'<div style="margin-top:10px;font-size:12px;color:#555;'
            f'border-top:1px solid #f0f0f0;padding-top:8px;">'
            + " &nbsp;·&nbsp; ".join(contact_parts) + "</div>"
        )

    return f"""
<div style="border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin:8px 0;
            background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:11px;color:#888;font-weight:600;letter-spacing:0.5px;
                  text-transform:uppercase;margin-bottom:4px;">#{rank}</div>
      <div style="font-size:17px;font-weight:700;color:#0B2026;">{r['name']}</div>
      <div style="color:#555;font-size:13px;margin-top:3px;">{meta_line}</div>
      {'<div style="margin-top:5px;">' + score_pills + '</div>' if score_pills else ''}
    </div>
    {badge_html}
  </div>
  {evidence_html}
  {contact_html}
</div>
"""


def build_map_html(results, search_lat, search_lon):
    f = folium.Figure(width="100%", height="380px")
    m = folium.Map(location=[search_lat, search_lon], zoom_start=8, tiles="CartoDB positron")
    f.add_child(m)

    folium.CircleMarker(
        location=[search_lat, search_lon],
        radius=9, color="#FF3621", fill=True,
        fill_color="#FF3621", fill_opacity=0.85,
        tooltip="Search centre",
    ).add_to(m)

    color_map = {
        "✓ Strong evidence":    "green",
        "◐ Partial evidence":   "orange",
        "⚠️ Needs verification": "red",
    }
    for i, r in enumerate(results):
        lat, lon = r.get("lat"), r.get("lon")
        if not lat or not lon:
            continue
        trust = trust_label(r["evidence"])
        dist_label = f"{r['distance_km']} km" if r.get("distance_km") is not None else "dist unknown"
        popup_html = (
            f"<div style='min-width:180px'><b>{r['name']}</b><br>"
            f"{r['city']}, {r['state']}<br>"
            f"{dist_label} · {trust}</div>"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{i+1}. {r['name']}",
            icon=folium.Icon(color=color_map.get(trust, "blue"),
                             icon="plus-sign", prefix="glyphicon"),
        ).add_to(m)

    return f._repr_html_()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_search(location, care_need, radius_km):
    if not location.strip() or not care_need.strip():
        return "Please enter both a location and a care need.", [], "", ""

    results, meta = supervisor.run(
        df=df,
        centroids=centroids,
        care_need_query=care_need,
        location_query=location,
        radius_km=radius_km,
    )

    if "error" in meta:
        return f'<p style="color:#c00;padding:12px 0;">{meta["error"]}</p>', [], "", ""

    if not results:
        return (
            f'<p style="color:#666;padding:16px 0;">No facilities found matching '
            f'<b>{care_need}</b> within {radius_km} km of <b>{meta["resolved_location"]}</b>. '
            f'Try a larger radius.</p>'
        ), [], "", meta.get("care_need", care_need)

    match_note = ""
    if meta["location_match_type"] == "fuzzy":
        match_note = f" (closest match: <em>{meta['resolved_location']}</em>)"

    unloc_note = (
        f" + {meta['unlocated_count']} with unverified location"
        if meta.get("unlocated_count") else ""
    )
    sem_badge = (
        ' <span style="font-size:11px;background:#e8f4fd;color:#1565c0;'
        'padding:2px 8px;border-radius:10px;">🔍 semantic</span>'
        if meta.get("semantic_active") else ""
    )

    summary = (
        f'<div style="background:#f0f7f4;border-left:4px solid #1a7f4b;padding:10px 14px;'
        f'border-radius:0 6px 6px 0;margin-bottom:4px;font-size:14px;">'
        f'<b>{meta["located_count"]} match(es)</b> for <b>{meta["care_need"]}</b>{sem_badge} '
        f'within <b>{radius_km} km</b> of <b>{location}</b>{match_note}{unloc_note}. '
        f'Showing top {len(results)}.</div>'
    )

    cards    = "".join(format_result_card(i + 1, r) for i, r in enumerate(results))
    map_html = build_map_html(results, meta["search_lat"], meta["search_lon"])
    return summary + cards, results, map_html, meta.get("care_need", care_need)


def smart_search(query_text, radius_km):
    if not query_text.strip():
        return (
            '<p style="color:#aaa;padding:20px 0;">Enter something like '
            '<b>"dialysis near Jaipur"</b> or <b>"emergency surgery near Patna"</b>.</p>'
        ), [], "", ""

    care_need, location = parse_combined_query(query_text, centroids)
    if not location:
        return (
            f'<p style="color:#888;padding:16px 0;">Could not find a location in '
            f'<em>"{query_text}"</em>. Try <b>"&lt;care need&gt; near &lt;city&gt;"</b>.</p>'
        ), [], "", ""

    return run_search(location, care_need, radius_km)


# ---------------------------------------------------------------------------
# Shortlist
# ---------------------------------------------------------------------------

def update_picker_choices(results):
    choices = [f"{i + 1}. {r['name']}" for i, r in enumerate(results)]
    return gr.update(choices=choices, value=None)


def add_to_shortlist(selected_label, results, shortlist, care_need):
    if not selected_label or not results:
        return shortlist, format_shortlist(shortlist)
    idx = int(selected_label.split(".")[0]) - 1
    if idx < 0 or idx >= len(results):
        return shortlist, format_shortlist(shortlist)

    candidate = results[idx]
    if not any(s["name"] == candidate["name"] for s in shortlist):
        shortlist = shortlist + [{
            "name":        candidate["name"],
            "city":        candidate["city"],
            "state":       candidate["state"],
            "distance_km": candidate["distance_km"],
            "trust":       trust_label(candidate["evidence"]),
            "phone":       candidate.get("phone", ""),
            "website":     candidate.get("website", ""),
        }]
        # Persist interaction to Delta (fire-and-forget; non-blocking)
        feedback_store.record_save(
            sdk_query_fn  = _sdk_query,
            session_id    = _SESSION,
            care_need     = care_need or "unknown",
            facility_id   = candidate.get("id", ""),
            facility_name = candidate["name"],
        )

    return shortlist, format_shortlist(shortlist)


def format_shortlist(shortlist):
    if not shortlist:
        return '<p style="color:#888;font-style:italic;">No facilities saved yet.</p>'
    cards = []
    for s in shortlist:
        trust_text = s.get("trust", "")
        style, label = _TRUST_STYLES.get(trust_text, ("background:#666;color:#fff;", trust_text))
        dist_str = f" · {s['distance_km']} km" if s.get("distance_km") is not None else ""
        contact = " &nbsp;·&nbsp; ".join(
            filter(None, [_phone_link(s.get("phone", "")),
                          _website_link(s.get("website", ""))])
        )
        cards.append(
            f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px;'
            f'margin:6px 0;background:#fff;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'flex-wrap:wrap;gap:6px;">'
            f'<div><div style="font-weight:600;color:#0B2026;">{s["name"]}</div>'
            f'<div style="font-size:12px;color:#666;">📍 {s["city"]}, {s["state"]}{dist_str}</div>'
            f'</div>'
            f'<span style="padding:3px 10px;border-radius:16px;font-size:11px;'
            f'font-weight:600;{style}">{label}</span></div>'
            + (f'<div style="margin-top:6px;font-size:12px;">{contact}</div>' if contact else "")
            + "</div>"
        )
    return "".join(cards)


def clear_shortlist():
    return [], format_shortlist([])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
body, .gradio-container {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
"""

with gr.Blocks(title="Referral Copilot", css=CSS) as demo:

    gr.HTML(
        f'<div style="background:linear-gradient(135deg,#0B2026 0%,#1a3a45 100%);'
        f'border-radius:10px;padding:20px 24px;margin-bottom:8px;">'
        f'<div style="font-size:26px;font-weight:700;color:#F9F7F4;margin-bottom:6px;">🏥 Referral Copilot</div>'
        f'<div style="color:#EEEDE9;font-size:13px;line-height:1.5;opacity:0.85;">'
        f'Enter a care need and location to get a ranked shortlist of Indian healthcare facilities — '
        f'with the evidence behind each match, what\'s missing, and what to verify before referring a patient.'
        f'</div>'
        f'<div style="font-size:11px;color:#EEEDE9;opacity:0.5;margin-top:8px;">'
        f'{_total_facilities:,} facilities &nbsp;·&nbsp; {_total_cities:,} cities &nbsp;·&nbsp; '
        f'{_total_locations:,} searchable locations &nbsp;·&nbsp; DAIS for Good Hackathon 2026'
        f'</div></div>'
        + (f'<div style="background:#fff0f0;border:1px solid #c00;border-radius:8px;'
           f'padding:12px 16px;margin-top:8px;font-family:monospace;font-size:11px;'
           f'color:#c00;white-space:pre-wrap;overflow-x:auto;max-height:300px;overflow-y:auto;">'
           f'&#9888; Startup error:\n{_STARTUP_ERROR}</div>' if _STARTUP_ERROR else "")
    )

    with gr.Row():
        search_box = gr.Textbox(
            label="What do you need, and where?",
            placeholder='"dialysis near Jaipur"  or  "emergency surgery near Patna"',
            scale=5, container=False,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1, min_width=90)

    radius_input = gr.Slider(
        label="Search radius (km)", minimum=10, maximum=500, value=150, step=10
    )

    gr.Examples(
        examples=[
            "dialysis near Jaipur",
            "emergency surgery near Patna",
            "cardiology near Gurgaon",
            "ophthalmology near Hyderabad",
            "oncology near Delhi",
            "maternity near Chennai",
        ],
        inputs=[search_box],
        label="Example searches",
    )

    results_output  = gr.HTML(
        '<p style="color:#aaa;padding:20px 0;font-style:italic;">Results will appear here.</p>'
    )
    results_state   = gr.State([])
    care_need_state = gr.State("")    # tracks care_need across search → shortlist save
    map_output      = gr.HTML()

    gr.Markdown("---\n## Shortlist")
    with gr.Row():
        candidate_picker = gr.Dropdown(label="Select a result to save", choices=[], scale=4)
        save_btn  = gr.Button("Save to shortlist", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    shortlist_output = gr.HTML('<p style="color:#888;font-style:italic;">No facilities saved yet.</p>')
    shortlist_state  = gr.State([])

    def _search(q, r):
        return smart_search(q, r)

    search_btn.click(
        _search,
        [search_box, radius_input],
        [results_output, results_state, map_output, care_need_state],
    )
    search_box.submit(
        _search,
        [search_box, radius_input],
        [results_output, results_state, map_output, care_need_state],
    )
    results_state.change(update_picker_choices, [results_state], [candidate_picker])
    save_btn.click(
        add_to_shortlist,
        [candidate_picker, results_state, shortlist_state, care_need_state],
        [shortlist_state, shortlist_output],
    )
    clear_btn.click(clear_shortlist, outputs=[shortlist_state, shortlist_output])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("DATABRICKS_APP_PORT", 8080)),
    )
