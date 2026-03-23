# Block Changelog

Turn Philadelphia open data into an approachable changelog for every city block.

## One-line pitch

Enter any Philadelphia address and see what changed on that block: 311 activity, crime incidents, new licenses, demolitions, lane closures, cleanup work, and other city events.

## Product framing

Use a plain-English framing throughout the product:

- headline: **What changed in Philly?**
- drill-down question: **What changed on this block?**

The project should feel approachable to anyone, not just technical audiences.

---

# MVP goal

Build the **smallest believable version** of the idea:

1. user enters an address or block
2. we identify the census block
3. we show a reverse-chronological feed of recent events for that block
4. we show a few summary counts for the last 30 / 90 / 365 days

That is enough for a compelling demo.

## MVP non-goals

Do **not** build these in v1:

- citywide rankings
- fancy scoring systems
- AI summaries
- too many datasets
- perfect event taxonomy
- production ETL complexity
- dbt / orchestration / warehouse infrastructure

Keep it small, fast, and real.

---

# Recommended stack

## Data / pipeline

- `philly` for source discovery and loading
- `Python` for orchestration and normalization
- `DuckDB` for local analytical transforms
- `Parquet` for intermediate and final data
- `GeoPandas` / `Shapely` for spatial joins to census blocks

## Frontend

- simple static HTML/JS app
- address search
- feed view
- optional simple map highlight

## Why this stack

This is the lightest stack that still gives us a real derived dataset pipeline.

Avoid overkill for now:

- no dbt yet
- no Airflow / Dagster
- no backend server required

If the project grows, `dbt-duckdb` is a natural later upgrade.

---

# Directory plan

```text
explorations/block-changelog/
  README.md
  data/
    raw/
    silver/
    gold/
  scripts/
    build_pipeline.py
    fetch_raw.py
    assign_blocks.py
    normalize_311.py
    normalize_crime.py
    normalize_business_licenses.py
    normalize_demolitions.py
    normalize_lane_closures.py
    normalize_cleaning_tasks.py
    build_gold.py
  sql/
    union_events.sql
    block_summary.sql
  app/
    index.html
    main.js
    styles.css
```

Current implementation status:

- `scripts/build_pipeline.py` now builds the first working MVP pipeline for **Census Blocks + 311 + Crime**
- it also derives **Philadelphia Neighborhoods** summaries from the block/event data for a simpler top-level browsing view
- it writes raw parquet extracts, normalized silver parquet tables, and gold outputs including `block_log.parquet`, `block_summary.parquet`, and `neighborhood_summary.parquet`
- later we can split that script into smaller source-specific files if needed

For the first pass, even fewer files is fine.

---

# Real MVP scope

## Datasets for v1

Use only a small, high-value set.

### Core event datasets

1. `311 Service and Information Requests`
2. `Crime Incidents`
3. `Licenses and Inspections Business Licenses`
4. `Building Demolitions`
5. `Street Lane Closures`
6. `Citywide Cleaning Program Tasks`

### Geography

7. `Census Blocks`

## Why these

They give a nice mix of:

- things breaking
- things opening
- things being maintained
- things changing physically
- things residents will immediately understand

That is enough to make a block feel alive.

---

# Data model

We are creating a **derived dataset**.

## Concept

Many unrelated source datasets become one normalized event log.

## Bronze / raw

Raw source extracts as parquet.

Examples:

- `data/raw/311_2024.parquet`
- `data/raw/crime_2024.parquet`
- `data/raw/business_licenses_recent.parquet`
- `data/raw/demolitions_recent.parquet`
- `data/raw/lane_closures_current.parquet`
- `data/raw/cleaning_tasks_recent.parquet`
- `data/raw/census_blocks.geojson` or parquet

## Silver / normalized

One normalized table per source.

Examples:

- `data/silver/events_311.parquet`
- `data/silver/events_crime.parquet`
- `data/silver/events_business_licenses.parquet`
- `data/silver/events_demolitions.parquet`
- `data/silver/events_lane_closures.parquet`
- `data/silver/events_cleaning_tasks.parquet`

## Gold / product-ready

- `data/gold/block_log.parquet`
- `data/gold/block_summary.parquet`
- optional compact JSON exports for frontend use

---

# Normalized event schema

Each source should be transformed into the same event schema.

## Required fields

- `event_id`
- `source_dataset`
- `source_record_id`
- `event_type`
- `event_subtype`
- `event_date`
- `event_end_date` nullable
- `title`
- `description`
- `address` nullable
- `lat` nullable
- `lng` nullable
- `block_id`
- `tract_id` nullable
- `status` nullable
- `severity` nullable
- `tags` nullable

## Useful optional fields

- `source_url`
- `raw_payload_json`
- `icon`
- `color`

## Event type examples

Keep these simple.

- `service_request`
- `crime_incident`
- `business_license`
- `demolition`
- `lane_closure`
- `cleanup`

---

# Source-specific first-pass mapping

## 1. 311 Service and Information Requests

### Use for MVP

- request creation events
- optionally closed events later

### Suggested mapping

- `event_type`: `service_request`
- `event_subtype`: `service_name`
- `event_date`: `requested_datetime`
- `event_end_date`: `closed_datetime`
- `title`: `service_name`
- `description`: `status` + `agency_responsible`
- `address`: `address`
- `lat/lng`: `lat`, `lon`

### Good first filter

Use recent data only, e.g. last 12-18 months.

## 2. Crime Incidents

### Suggested mapping

- `event_type`: `crime_incident`
- `event_subtype`: `text_general_code`
- `event_date`: `dispatch_date_time`
- `title`: `text_general_code`
- `description`: `location_block`
- `lat/lng`: `lat`, `lng`

## 3. Business Licenses

### Suggested mapping

- `event_type`: `business_license`
- `event_subtype`: `licensetype`
- `event_date`: `mostrecentissuedate`
- `event_end_date`: `expirationdate`
- `title`: `business_name` or `licensetype`
- `description`: `licensestatus`
- `address`: `address`
- `lat/lng`: `lat`, `lng`

### Note

For v1, only include rows with usable coordinates and meaningful issuance dates.

## 4. Building Demolitions

### Suggested mapping

- `event_type`: `demolition`
- `event_subtype`: `typeofwork`
- `event_date`: `start_date`
- `event_end_date`: `completed_date`
- `title`: `record_type + typeofwork`
- `description`: `status`
- `address`: `address`
- `lat/lng`: `lat`, `lng`

## 5. Street Lane Closures

### Suggested mapping

- `event_type`: `lane_closure`
- `event_subtype`: `occupancytype`
- `event_date`: `effectivedate`
- `event_end_date`: `expirationdate`
- `title`: `purpose`
- `description`: `status`
- `address`: `address`

### Note

If exact block assignment is hard for line-like street closures in v1, approximate via representative point or defer if needed.

## 6. Citywide Cleaning Program Tasks

### Suggested mapping

- `event_type`: `cleanup`
- `event_subtype`: `task`
- `event_date`: `created_date`
- `title`: `task`
- `description`: `last_edited_date`

---

# Geography assignment

This is the key step.

## Goal

Every event should be assigned to a census block.

## Source of truth

- `Census Blocks`
- use `GEOID10` as `block_id`

## Approach

### For point datasets

Use GeoPandas spatial join:

- convert rows with coordinates into points
- spatial join to census block polygons
- assign `block_id`

### For line-like or address-range datasets

For MVP, use the simplest workable method:

- if there are coordinates, use them
- if not, skip those records in v1
- if lane closures are too awkward, include only closures with a usable point representation or leave closures out of first shipped demo

## Important rule

Prefer a smaller correct event log over a broad messy one.

---

# MVP outputs

## 1. Block log

A table of normalized events.

### Example columns

- `block_id`
- `event_date`
- `event_type`
- `event_subtype`
- `title`
- `description`
- `source_dataset`
- `address`

## 2. Block summary

Aggregated by block.

### Example fields

- `block_id`
- `events_30d`
- `events_90d`
- `events_365d`
- `service_requests_365d`
- `crime_incidents_365d`
- `business_licenses_365d`
- `demolitions_365d`
- `cleanups_365d`
- `last_event_date`

## 3. Frontend-friendly export

Could be:

- one compact JSON keyed by `block_id`
- or a pair of files:
  - summaries
  - event feed records

---

# UI / UX plan

## MVP UI

Very simple.

### Entry points

- search by address
- optionally click block on map later

### Main panel

- block ID / address
- recent activity summary
- toggle: `30d | 90d | 365d`
- reverse chronological event feed

### Event feed card example

- date
- type icon / color
- title
- short description
- source dataset label

## Optional copy style

- headline: **What changed on this block?**
- subhead: **A changelog built from Philadelphia open data**

Keep the interface quiet and legible.

---

# Recommended implementation order

## Phase 1: prove the data model

1. fetch census blocks
2. fetch 311
3. fetch crime
4. normalize both
5. assign blocks
6. union into one event log
7. export block log + block summary

If this works, the concept is already validated.

## Phase 2: add more event types

8. add business licenses
9. add demolitions
10. add cleanup tasks
11. add lane closures if easy enough

## Phase 3: frontend demo

12. build a simple search + feed UI
13. wire address to block lookup
14. render summaries and event feed

---

# What to cut if time gets tight

Cut in this order:

1. lane closures
2. cleanup tasks
3. demolitions
4. business licenses

Do **not** cut:

- census blocks
- 311
- crime
- normalized event log
- block summaries
- address/block lookup

A demo built from only 311 + crime + block geography would still be real and compelling.

---

# Data quality notes

## Known realities

- some source datasets have poor CSV formatting
- some ArcGIS resources are cleaner to sample than to count
- some Carto datasets have computed lat/lng columns in the original query
- some datasets lack reliable coordinates
- some event semantics are messy or ambiguous

## Strategy

- use only records with valid coordinates in v1
- keep normalization simple
- do not try to perfectly model the city
- make all assumptions explicit in the README and UI copy

---

# Success criteria for MVP

This project is successful if it can do these 5 things:

1. given an address, identify a block
2. show at least 3 meaningful event types on that block
3. render a convincing chronological feed
4. compute simple recent-activity summaries
5. make a user say: **"oh wow, this is actually a changelog for a real place"**

---

# Future extensions after MVP

Only after the MVP works:

- citywide “most changed blocks” ranking
- compare one year vs another
- block stability score / change velocity score
- tree / heat / vacancy / food / Wi-Fi enrichment
- neighborhood compare
- clearer event summaries
- downloadable derived dataset

---

# Frontend architecture

## Recommendation

Use a **static frontend** with:

- `MapLibre GL JS`
- `Vite`
- vanilla JS modules
- plain CSS

This keeps the project:

- static-hosting friendly
- backend-free
- lightweight
- visually modern
- easy to iterate on

## Why not React for v1

React would still work as a static site, but it is not necessary for the MVP.

The app only needs:

- one full-screen map
- one search box
- one selected-block panel
- a few filters / toggles
- one event feed

That is simple enough to build cleanly in vanilla JS.

## Proposed frontend structure

```text
explorations/block-changelog/app/
  index.html
  main.js
  styles.css
  data/
    blocks.geojson
    neighborhoods.geojson
    neighborhood_index.json
    address_index.json
    manifest.json
    tracts/
      42101000100.json
      ...
```

If we want a slightly nicer dev experience, use Vite with:

```text
explorations/block-changelog/web/
  index.html
  src/
    main.js
    styles.css
    lib/
      map.js
      ui.js
      data.js
```

## Frontend responsibilities

### `main.js`

- initialize the map
- load the derived data files
- wire address search to block selection
- update the panel and feed
- handle filters and time windows

### `map.js`

- create MapLibre map instance
- add block layer
- highlight selected block
- fly to selected block
- attach click / hover handlers

### `data.js`

- load summary and log files
- index events by block
- filter events by date range and type

### `ui.js`

- render summary metrics
- render feed rows
- render empty states
- manage filter UI state

---

# Visual direction

## Design goal

The experience should feel:

- modern
- premium
- quiet
- map-first
- legible to non-technical users

The inspiration can be similar to `mapcn`, but the implementation should remain static and framework-light.

## Core visual language

### 1. Full-screen map

- edge-to-edge map
- minimal chrome
- no GIS-style clutter

### 2. Floating glass panels

Use:

- dark translucent panels
- subtle backdrop blur
- thin soft borders
- low-contrast shadows
- generous rounded corners

### 3. Strong typography hierarchy

- large plain-English heading
- tiny uppercase labels
- muted secondary text
- short, scannable feed rows

### 4. Clean color system

Use restrained semantic accents:

- blue / cyan for service requests
- red / orange for incidents or demolition
- green for business / improvement / cleanup
- neutral gray for secondary metadata

### 5. Smooth but subtle motion

Use small transitions for:

- panel opening
- feed refresh
- block highlight
- map fly-to

No flashy animation.

---

# Visual spec

## Main layout

### Top center

Search bar:

- rounded pill or rounded rectangle
- dark translucent background
- address input + optional search icon
- placeholder: `Enter a Philadelphia address`

### Left panel or right panel

Selected block panel:

- title: `What changed on this block?`
- secondary line: plain-language location context
- recent activity counts
- date-range toggle
- event-type filters
- event feed

### Map content

- block polygons lightly visible only on interaction
- selected block strongly outlined
- optional hover state for nearby blocks

## Summary row

Show 3 small stat pills/cards:

- `30d`
- `90d`
- `365d`

Each shows:

- event count
- maybe percentile later, but not required in v1

## Feed item design

Each row should include:

- event date
- small type badge or icon
- title
- one-line description
- source label

Example:

- `Mar 12` · `cleanup`
- `Sanitation Block Cleaning (after)`
- `Citywide cleaning task recorded`
- `Citywide Cleaning Program Tasks`

## Empty state

If a block has no recent events:

- do not make it feel broken
- show something like:
  - `Quiet block.`
  - `No recent tracked events in the selected time range.`

---

# Map / basemap recommendations

## Map engine

Use **MapLibre GL JS**.

## Basemap

Use a clean, premium-feeling style:

- CARTO light / dark basemap
- or another MapLibre-compatible neutral style

For MVP, prefer a basemap that keeps the data panel readable and does not fight for attention.

## Initial camera

Start centered on Philadelphia with a zoom level that makes block interaction feel immediate.

## Interaction rules

- click block → select block and open panel
- search address → geocode / resolve block and fly to it
- keep interaction simple and predictable

---

# Address search options

## MVP choice

Use a simple client-side geocoding approach if possible.

Possible options:

- a public geocoder
- precomputed address-to-block lookup if we derive one later
- lightweight geocoding API call from the client if terms/limits allow

## Better long-term option

Generate our own derived lookup table:

- address or parcel centroids
- mapped to `block_id`

That would keep the project more self-contained and aligned with the “derived civic dataset” idea.

---

# Immediate next build tasks

## Data tasks

- [x] fetch `Census Blocks`
- [x] fetch recent `311 Service and Information Requests`
- [x] fetch recent `Crime Incidents`
- [x] spatially assign both to blocks
- [x] define normalized event schema
- [x] export first `block_log.parquet`
- [x] export first `block_summary.parquet`
- [x] export `census_blocks.parquet` for frontend/map work
- [x] export frontend-ready `blocks.geojson` into `app/data/`
- [x] export frontend-ready `neighborhoods.geojson` into `app/data/`
- [x] export a `neighborhood_index.json` drilldown lookup into `app/data/`
- [x] export lazy-loaded tract event JSON files into `app/data/tracts/`
- [x] export an `address_index.json` lookup into `app/data/`

## Product tasks

- [ ] decide final name shown in UI
- [ ] choose address lookup approach
- [ ] design simple feed interface
- [ ] pick 30 / 90 / 365-day summary format

## Decision

If we need a smaller first step than the full MVP, do this:

> build a working block changelog using **only 311 + crime + census blocks**

That is the true minimal version worth shipping internally first.
