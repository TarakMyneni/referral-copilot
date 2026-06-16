# Referral Copilot — Project Brief

## 1. Original objective (Track 3: Referral Copilot)

> Build an app where a user enters a location and a care need, such as "dialysis near Jaipur"
> or "emergency surgery near Patna," and receives an evidence-attached shortlist of candidate
> facilities.
>
> Minimum workflow: location and need in; ranked candidates out; each candidate shows distance,
> matching evidence, missing or suspicious evidence, and can be saved to a shortlist.

Event: **Databricks Apps & Agents for Good Hackathon 2026** (DAIS 2026), via Devpost.
Deadline: **Jun 16, 2026 @ 2:30pm PDT**.
Judging criteria: Business Applicability, Data Relevance, Creativity, Thoroughness, Well-Architected.

## 2. Deliverable constraints (confirmed with user)

- **Final output must be a Databricks App** (Databricks Apps platform), built with **Gradio**
  (chosen over Streamlit/Dash/Flask for build speed — user has no framework preference).
- **Databricks access: Free Edition** — notebooks, SQL warehouses, Unity Catalog, and Apps
  (1 app per account quota; there have been intermittent "Compute error" reports on app
  creation — test the deploy step early, don't leave it to the end).
- **Submission via git.** *Confidence ~80%*: this most likely means a GitHub repo link in the
  Devpost submission form (standard pattern), NOT that the Databricks App itself must be
  deployed from a git-synced source — but the official rules PDF was inaccessible (Google
  Drive sign-in wall) so this isn't 100% confirmed. If you can access the rules PDF, verify.

## 3. Dataset

Official dataset lives in Unity Catalog:
`databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset` (table name(s) TBD —
user confirmed `SHOW TABLES` returns non-empty results and `spark.table(...)` works).

**Described on the hackathon Resources page** (not yet verified against real columns):
~10,000 Indian healthcare facility records, 51 columns, including: name, state, city, postcode,
latitude, longitude, specialties (controlled vocabulary), description, capability, procedure,
equipment, numberDoctors, capacity, yearEstablished, source_urls.

Field coverage (as documented): description 100%, capability 99.7%, procedure 92.5%,
equipment 77.0%, numberDoctors 36.4%, capacity 25.2%, yearEstablished 47.8%.

**Critical framing from the hackathon docs**: these fields are "claims to verify, not ground
truth." This directly motivated the evidence-engine design below — don't lose this framing if
you redesign anything.

### IMMEDIATE FIRST STEP

Query the real table to get actual column names + sample rows, e.g.:

```python
from databricks import sql  # or use spark.table(...) if running in a Databricks notebook
# ... connect, then:
df = spark.table("databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.<table_name>")
pdf = df.toPandas()
print(pdf.shape)
print(pdf.columns.tolist())
print(pdf.head(3).to_dict(orient="records"))
```

Then update `src/config.py`'s `COLUMNS` dict (currently placeholder values) to match the real
column names — that's the only file that should need changing for the schema swap. Also expand
`CARE_NEED_SYNONYMS` once you see the real `specialties` controlled-vocabulary values.

### Supplemental datasets (phase-2, not yet integrated)

A separate doc describes two public datasets the user may also have access to:
- **India Post PIN code directory** (165,627 rows: pincode, district, state, lat/lon — ~12,600
  rows missing lat/lon). Potential use: broader location gazetteer fallback for "near <town>"
  queries where the town isn't in the facility dataset.
- **NFHS-5 district health indicators** (706 districts, 109 columns of health/demographic
  data). Potential use: "for Good" framing — show district-level health burden context
  alongside referral results. Requires a spatial join (facility lat/lon → district polygon via
  geoBoundaries/DataMeet shapefiles + GeoPandas) since facility data has city/state, not
  district, and string-matching district names across sources is unreliable.

Treat both as optional — don't let them block the core deliverable.

## 4. Current architecture (working, tested end-to-end with synthetic data)

```
app.py                  Gradio UI (entrypoint)
app.yaml                Databricks Apps run command: ["python", "app.py"]
requirements.txt        gradio, pandas
data/facilities_sample.csv   15-row SYNTHETIC placeholder dataset (real data not yet wired in)
src/config.py           COLUMNS mapping (placeholder) + CARE_NEED_SYNONYMS dict
src/geo.py              Haversine distance + city-centroid location resolution (no external geocoding API)
src/ranking.py          search_facilities() + parse_combined_query() (smart-search parser) + normalize_care_need()
src/evidence.py         evaluate_evidence() (matching/missing/suspicious) + trust_label()
README.md               Full setup/deploy docs
```

Git repo already initialized, 2 commits so far.

### Key design decisions & rationale

- **Location resolution without external APIs**: every facility already has lat/lon, so we
  compute each city's centroid from the dataset itself and resolve "near <city>" against that.
  This avoids external geocoding calls entirely, which matters given Free Edition's restricted
  outbound network access.
- **Rules-based evidence engine, not LLM**: deterministic and explainable (good for
  Thoroughness/Well-Architected judging), and avoids LLM latency/flakiness in the critical
  path. Evidence has three buckets:
  - *Matching*: care-need keyword found in specialties/description/capability/procedure/equipment
  - *Missing*: low-coverage fields (numberDoctors, capacity, yearEstablished, source_urls) blank
    for this facility
  - *Suspicious*: need-level inconsistency checks — e.g. specialty claimed but not corroborated
    in any other field, matches the need but reports 0 doctors, no source URL at all
  - Collapsed into a single **trust badge** (Strong evidence / Partial evidence / Needs
    verification) for scannability
- **Single smart-search box** (not separate location/care-need fields): matches the brief's own
  example phrasing ("dialysis near Jaipur"). `parse_combined_query()` handles "X near Y", "X Y",
  and "Y X" orderings via regex + city-name fallback matching against known cities. Enter-to-
  search and one-click `gr.Examples` (with `run_on_click=True`, confirmed working on gradio
  6.18.0) minimize interaction steps.
- **Shortlist**: in-session `gr.State`, not yet persisted (see stretch ideas).

## 5. Outstanding work, in priority order

1. **Get real dataset schema** (see step above) and update `src/config.py`. Re-run the smoke
   test in README.md ("Switching to the real dataset" section) against real data.
2. **Verify Databricks Apps deployment** end-to-end with the real `app.py` — use the official
   "Gradio Hello world" template as a reference for the correct `launch()`/port config (current
   `app.py` uses `DATABRICKS_APP_PORT` env var as a best-effort default, *confidence ~40%*
   unverified against the live Apps runtime). Given the 1-app-per-account quota, get this right
   early.
3. **Push to GitHub** for submission (repo is already git-initialized locally).
4. Stretch, in priority order:
   - Map view of ranked candidates (lat/lon already available — `pydeck`/`folium`)
   - Per-result inline "Save to shortlist" button (currently dropdown + button = 2 actions)
   - Shortlist persistence to a Delta table
   - LLM narrative summary of evidence per candidate (Foundation Model API) — keep
     ranking/flagging rule-based; LLM only phrases the summary so a flaky call can't break the
     core demo
   - PIN code gazetteer fallback for locations outside the facility dataset
   - NFHS-5 district health context for the social-good narrative

## 6. Known unknowns / things to double-check

- Real `specialties` field format (comma/semicolon-separated string vs list vs coded values?) —
  `CARE_NEED_SYNONYMS` and the matching logic assume substring search on a text field.
- Exact Databricks Apps `launch()` port configuration for Gradio (see #2 above).
- Exact git submission requirement (see section 2).
