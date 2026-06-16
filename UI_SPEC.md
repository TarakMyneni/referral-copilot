# Suvidha UI Spec — For Claude Code

## Overview
Rebuild `app.py` as a Gradio app matching the Suvidha mockup exactly.
The app is fully light-themed (no dark mode). All colors are hardcoded below.
Do NOT use Gradio's default theme — use `gr.Blocks(css=...)` with custom CSS.

---

## Color Palette (hardcoded, no CSS variables)

```
Background page:      #F7FAF3
Background card:      #FFFFFF
Background govt thumb:#E6F1FB
Background priv thumb:#EAF3DE
Border default:       #D3D1C7
Border light:         #E8E6DF
Border green:         #C0DD97

Green dark:           #27500A   (logo, strong badge text, saved bookmark)
Green mid:            #3B6D11   (logo icon bg, search btn, active chip, export btn, links)
Green light:          #639922   (map pin icons, distance icons)
Green pale:           #EAF3DE   (chip bg, shortlist item bg, private thumb bg, ev chip bg)

Text primary:         #2C2C2A
Text secondary:       #5F5E5A
Text muted:           #888780

Strong badge bg:      #EAF3DE  border: #97C459  text: #27500A
Partial badge bg:     #FAEEDA  border: #FAC775  text: #633806
Verify badge bg:      #FAECE7  border: #F0997B  text: #712B13

Amber (missing):      #854F0B
Red (flags):          #993C1D

Govt thumb bg:        #E6F1FB   text/icon: #185FA5
```

---

## App Layout — Two-column, full height

```
┌─────────────────────────────────────────────────────┐
│  TOPBAR (logo left | search pill center | saved right)│
├─────────────────────────────────────────────────────┤
│  FILTER BAR (chips left | sort right)               │
├──────────────────────────────┬──────────────────────┤
│  RESULTS COLUMN              │  RIGHT COLUMN        │
│  (scrollable cards)          │  MAP (top, flex:1)   │
│                              │  SHORTLIST (bottom)  │
└──────────────────────────────┴──────────────────────┘
```

Use `gr.Blocks` with a single HTML component updated via Python callbacks.
Render the entire UI as one `gr.HTML` output — do NOT use Gradio's built-in
Card/Row/Column layout for the results and map — they won't match the design.

---

## 1. Topbar

- Height: ~56px, white bg, bottom border #D3D1C7
- LEFT: Logo
  - 32×32px dark green (#3B6D11) rounded square (border-radius 8px) with white map-pin icon
  - Next to it: "Suvidha" in 16px/500 #27500A, below it "सुविधा · Healthcare Referrals" in 11px #639922
- CENTER: Search pill
  - Border: 1.5px solid #B4B2A9, border-radius 40px, white bg
  - Three sections separated by 0.5px #D3D1C7 dividers:
    - "WHERE" label (9px uppercase #888780) + value (13px #2C2C2A)
    - "CARE NEED" label + value
    - "RADIUS" label + value (e.g. "50 km")
  - Right end: 34×34px circle button, bg #3B6D11, white search icon
  - Clicking this circle or pressing Enter triggers the search
- RIGHT: Saved count pill
  - bg #EAF3DE, border 0.5px #C0DD97, border-radius 20px, padding 5px 10px
  - bookmark icon + "N saved" in 12px #3B6D11
  - Updates live as items are added to shortlist

---

## 2. Filter Bar

- White bg, bottom border #D3D1C7, padding 10px 20px
- LEFT: Three filter chips — "All (N)", "Government", "Private"
  - Active chip: bg #3B6D11, color #EAF3DE, border #3B6D11
  - Inactive chip: bg #FFFFFF, color #3B6D11, border #C0DD97
  - border-radius 20px, padding 5px 14px, font-size 12px
  - Clicking filters the results list in place (no full re-search)
- RIGHT: Sort pill — "Sort: Nearest first" / "Sort: Best match"
  - bg #fff, border 0.5px #D3D1C7, border-radius 20px, padding 5px 12px
  - Sort icon + text, 12px #5F5E5A
  - Toggling changes sort order of current results

---

## 3. Results Column

- Background #F7FAF3, padding 14px 20px, flex column, gap 10px
- Scrollable, takes remaining height
- Result count line: "N facilities for <need> within X km of <location>"
  - 12px #5F5E5A, "N" in bold #2C2C2A

### Facility Card

```
┌──────────────┬──────────────────────────────────────┐
│  THUMB       │  CARD BODY                           │
│  96px wide   │                                      │
│              │  Name + distance (top row)           │
│  Icon        │  Subtitle (city, state)              │
│  Type label  │                                      │
│              │  "Confirmed in" label                │
│  ─────────── │  Green chips row                     │
│  Trust badge │  ── hairline divider ──              │
│  (bottom)    │  Missing / flag rows                 │
│              │                                      │
│              │  ── footer divider ──                │
│              │  Phone · Website      [bookmark btn] │
└──────────────┴──────────────────────────────────────┘
```

#### Thumb panel (96px wide)
- Private facility: bg #EAF3DE, icon color #3B6D11 (opacity 0.5), type label #3B6D11
- Government facility: bg #E6F1FB, icon color #185FA5 (opacity 0.5), type label #185FA5
- Icon: ti-building-hospital at 24px
- Type label: 9px uppercase, letter-spacing 0.4px
- Trust badge: absolutely pinned to bottom of thumb, full width
  - Strong:  bg #EAF3DE, text #27500A, top border #97C459  → "✓ Strong"
  - Partial: bg #FAEEDA, text #633806, top border #FAC775  → "◐ Partial"
  - Verify:  bg #FAECE7, text #712B13, top border #F0997B  → "⚠ Verify"
  - 10px/500, centered, padding 5px 4px

#### Card body
- Padding: 12px 14px, flex column, gap 9px

**Top row:**
- Name: 14px/500 #2C2C2A
- Subtitle: 11px #888780 (city, state)
- Distance (top-right, flex-shrink:0): map-pin icon #639922 + "X.X km" in 11px #5F5E5A

**Evidence section:**
- "CONFIRMED IN" label: 10px uppercase #888780, letter-spacing 0.3px
- Green chips: bg #EAF3DE, border 0.5px #C0DD97, border-radius 20px,
  padding 2px 8px, font-size 11px, color #27500A
  - One chip per matching field (specialties, description, capability, procedure, equipment)
- Hairline divider: border-top 0.5px #E8E6DF
- Missing row (if any): info-circle icon + text in 11px #854F0B
  - e.g. "Capacity not reported" or "No. of doctors and capacity not reported"
- Flag rows (if suspicious, shown instead of/in addition to missing):
  - alert-triangle icon + text in 11px #993C1D
  - e.g. "Specialty not corroborated in description, capability or equipment"
  - e.g. "No source URL — claims cannot be verified"

**Footer:**
- Border-top 0.5px #E8E6DF, padding-top 8px
- Left: phone icon + number (11px #3B6D11), globe icon + domain (11px #3B6D11)
- Right: 28×28px circle bookmark button
  - Unsaved: border 0.5px #D3D1C7, bg #fff, bookmark icon #888780
  - Saved: border #3B6D11, bg #EAF3DE, bookmark icon #3B6D11
  - Clicking toggles save/unsave and updates shortlist

---

## 4. Right Column (320px fixed width)

### Map area (top, flex:1, min-height 270px)

Use **Folium** to render a real interactive map:
```python
import folium
m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
               tiles='CartoDB Positron')
# Add circle for search radius
folium.Circle([center_lat, center_lon], radius=radius_km*1000,
              color='#3B6D11', fill=True, fill_opacity=0.05).add_to(m)
# Add markers per result
for r in results:
    color = {'strong':'#27500A','partial':'#BA7517','verify':'#D85A30'}[r['trust']]
    folium.CircleMarker(
        [r['lat'], r['lon']], radius=7, color=color, fill=True,
        fill_color=color, fill_opacity=0.9,
        tooltip=f"{r['name']} · {r['distance_km']} km"
    ).add_to(m)
```
Render as HTML iframe inside the Gradio HTML component.
Map has a small label in top-left showing city + "N facilities · X km radius".

### Shortlist panel (bottom, fixed)

- White bg, top border #D3D1C7, padding 12px 14px
- Header row: "Shortlist" (13px/500 #2C2C2A) + saved count badge + "Clear" link (11px underline #888780)
- Saved count badge: bg #EAF3DE, border #C0DD97, 10px #27500A
- Each shortlist item:
  - bg #F7FAF3, border 0.5px #C0DD97, border-radius 8px, padding 7px 8px
  - Left: facility name (12px/500 #2C2C2A) + meta "X km · City · ✓ Trust" (10px #5F5E5A)
  - Right: × button (12px #888780) — removes from shortlist
- Export button: full width, bg #3B6D11, color #EAF3DE, border-radius 8px,
  padding 9px, 13px, download icon + "Export shortlist"
  - On click: generates a plain-text or CSV summary of saved facilities and
    triggers a `gr.File` download

---

## 5. Interactions & State

```python
# State
results_state   = gr.State([])   # current search results (list of dicts)
shortlist_state = gr.State([])   # saved facilities (list of dicts)
filter_state    = gr.State("all")  # "all" | "government" | "private"
sort_state      = gr.State("distance")  # "distance" | "match"

# Main search trigger: search pill button OR Enter key in any pill input
# → calls smart_search(where, care_need, radius_km)
# → returns updated HTML for entire results+map+topbar area

# Filter chips
# → client-side filter on results_state, no re-search needed
# → re-renders results list only

# Sort toggle
# → re-sorts results_state in place
# → re-renders results list only

# Bookmark button on card
# → add/remove from shortlist_state
# → re-renders shortlist panel + topbar saved count + bookmark icon state

# × button in shortlist
# → remove from shortlist_state
# → re-renders shortlist panel + topbar saved count

# Export button
# → generate CSV string from shortlist_state
# → gr.File download
```

---

## 6. Data wiring (replaces sample CSV)

```python
import os
from databricks import sql   # or spark.table if running in a notebook

TABLE = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.<actual_table_name>"

# On app startup (not per-request):
df = spark.table(TABLE).toPandas()
centroids = build_city_centroids(df, ...)
```

If `spark` is not available (local dev), fall back to:
```python
df = pd.read_csv("data/facilities_sample.csv")
```

---

## 7. File delivery

```
app.py              Main Gradio app (all CSS inline in gr.Blocks)
app.yaml            command: ["python", "app.py"]
requirements.txt    gradio, pandas, folium
src/config.py       COLUMNS + CARE_NEED_SYNONYMS (update once real schema confirmed)
src/geo.py          haversine_km, build_city_centroids, resolve_location
src/ranking.py      search_facilities, parse_combined_query, normalize_care_need
src/evidence.py     evaluate_evidence, trust_label
data/facilities_sample.csv   fallback for local dev
```

---

## 8. Databricks Apps deployment note

Check the official "Gradio Hello world" template's `app.yaml` and `launch()` call
and align `app.py`'s `__main__` block with it.
Current best-effort default in the existing code:
```python
demo.launch(server_name="0.0.0.0",
            server_port=int(os.environ.get("DATABRICKS_APP_PORT", 8080)))
```
Verify this against the live template before final deploy.
