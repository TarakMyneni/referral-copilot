# Referral Copilot

**Track 3 — Referral Copilot**: enter a location and a care need (e.g. "dialysis near Jaipur")
and get a ranked, evidence-attached shortlist of candidate facilities.

## Status

Working MVP, tested locally with a 15-row synthetic dataset that mirrors the structure of the
official hackathon dataset (10,000 Indian healthcare facilities, 51 columns). **Not yet wired
to the real dataset** — see "Switching to the real dataset" below, that's the next step.

## How it works

1. **Location resolution** — every facility in the dataset already has lat/lon, so we compute
   each city's centroid (average lat/lon of its facilities) and use that to resolve "near
   <city>" queries. No external geocoding API calls (works fine under Free Edition's
   restricted egress).
2. **Care-need matching** — free-text care need is normalized against a synonym dictionary
   (`src/config.py: CARE_NEED_SYNONYMS`), then we search for those keywords across
   `specialties`, `description`, `capability`, `procedure`, `equipment`.
3. **Ranking** — candidates within the search radius are sorted by (a) how many evidence
   fields matched, then (b) distance.
4. **Evidence engine** (`src/evidence.py`) — for each candidate:
   - **Matching evidence**: which fields contained a supporting keyword
   - **Missing evidence**: low-coverage fields (numberDoctors, capacity, yearEstablished,
     source_urls) that are blank for this facility
   - **Suspicious evidence**: internal inconsistencies — e.g. a specialty is claimed but not
     corroborated anywhere else, or the facility matches but reports 0 doctors, or there's no
     source URL at all
   - A simple **trust badge** (Strong evidence / Partial evidence / Needs verification)
     summarizes the above
5. **Shortlist** — pick any result from the dropdown and save it; shortlist persists for the
   current session (in-memory `gr.State`, not yet persisted to a table — see Stretch ideas).

## Project structure

```
app.py                  Gradio UI (entrypoint)
app.yaml                Databricks Apps run command
requirements.txt        gradio, pandas
data/facilities_sample.csv   Synthetic placeholder dataset (15 rows)
src/config.py           Column-name mapping + care-need synonym dictionary
src/geo.py              Haversine distance + city-centroid location resolution
src/ranking.py          Search/filter/rank pipeline
src/evidence.py          Evidence engine + trust badge
```

## Switching to the real dataset

1. Drop the real CSV into `data/` (e.g. `data/facilities.csv`) and update `DATA_PATH` in
   `app.py`.
2. Open `src/config.py` and update the `COLUMNS` dict so each logical name points at the
   real column header (e.g. if the real column is `facility_name` instead of `name`, change
   `"name": "name"` to `"name": "facility_name"`). **Nothing else needs to change.**
3. Check the real `specialties` values and expand `CARE_NEED_SYNONYMS` in `src/config.py` to
   match the actual controlled vocabulary used in the dataset.
4. Re-run the test below to sanity-check before opening the UI.

```bash
python3 -c "
import pandas as pd
from src.config import COLUMNS
from src.geo import build_city_centroids
from src.ranking import search_facilities

df = pd.read_csv('data/facilities.csv')
centroids = build_city_centroids(df, COLUMNS['city'], COLUMNS['latitude'], COLUMNS['longitude'])
results, meta = search_facilities(df, 'Jaipur', 'dialysis', centroids)
print(meta)
for r in results:
    print(r['name'], r['distance_km'], r['match_score'])
"
```

## Running locally

```bash
pip install -r requirements.txt
python3 app.py
```

Open the printed local URL.

## Deploying to Databricks Apps

1. In the workspace: **Compute → Apps → Create app → "Gradio Hello world" template**.
2. Check the template's generated `app.py` for its exact `launch()` call (port/host) and
   align our `app.py`'s `__main__` block with it if it differs — the launch line in this repo
   is a reasonable default but hasn't been verified against the live Apps runtime.
3. Sync these files into the app's source directory (via `databricks sync` or the workspace
   file editor) and deploy.

## Stretch ideas (if time allows, in priority order)

1. **Map view** of ranked candidates (facility lat/lon already available) — high
   visual-impact, low effort (`pydeck` or `folium`).
2. **Shortlist persistence** to a Delta table (Unity Catalog) instead of in-session state.
3. **LLM narrative summary** of each candidate's evidence (Foundation Model API) — keep
   ranking/flagging rule-based and deterministic; use the LLM only to phrase the summary, so a
   flaky API call can't break the core demo.
4. **Broader location gazetteer**: fall back to the India Post PIN code directory for
   locations not covered by the facility dataset's city list.
5. **District-level health context** (NFHS-5) for the social-good narrative — requires a
   spatial join (lat/lon → district polygon); treat as a pitch-deck addition unless a
   teammate can build it independently.

## Known limitations / assumptions

- Column names in `src/config.py` are **placeholders** based on the hackathon's dataset
  description, not yet verified against the real file.
- Shortlist is per-session only (resets on page reload).
- Location resolution requires the query city to appear in the dataset (or be a substring
  match of one that does); arbitrary towns not in the dataset will return "not found".
- `app.py`'s `launch()` port configuration for Databricks Apps is a best-effort default —
  verify against the official Gradio template.
