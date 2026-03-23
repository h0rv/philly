# Philly Transit 3D — Project Specification

> A real-time, 3D animated transit visualizer for Philadelphia built on SEPTA and regional open data. Inspired by Mini Tokyo 3D, designed for the US context, and powered by an existing Python/CLI open data library.

---

## Overview

Philly Transit 3D is a browser-based interactive map that renders all active SEPTA vehicles (buses, trolleys, subway/el, metro, and regional rail) — plus Amtrak and NJ Transit where data is available — as animated 3D models moving in real time across a tilted, isometric-style 3D map of Philadelphia. Users can also rewind and replay the last 24 hours of transit activity at variable speed.

The project is generic by design: the data layer is abstracted so any GTFS-compatible transit agency can be plugged in with minimal configuration.

---

## Goals

- Render all active transit vehicles as low-poly 3D models moving in real time
- Support a history/replay mode (last 24 hours, scrubable, variable playback speed)
- Be city-agnostic at the data layer — Philly/SEPTA is the first instance, not a hard dependency
- Leverage the existing Python/CLI open data library as the primary data source
- Be fully open source and deployable as a static site (no backend required for the front end in read-only mode)

---

## Supported Vehicle Types (Phase 1)

| Type          | Agency     | Notes                                                                                                         |
| ------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| Bus           | SEPTA      | All active routes                                                                                             |
| Trolley       | SEPTA      | Routes 10, 11, 13, 15, 34, 36                                                                                 |
| Subway/El     | SEPTA      | Market-Frankford Line (`L1`), Broad Street Line local/express/spur (`B1`/`B2`/`B3`) via SEPTA Metro trips API |
| High Speed    | SEPTA      | Norristown High Speed Line (`M1`) via SEPTA Metro trips API                                                   |
| Regional Rail | SEPTA      | All 13 lines                                                                                                  |
| Regional Rail | Amtrak     | Where GTFS-RT is available                                                                                    |
| Regional Rail | NJ Transit | Where GTFS-RT is available                                                                                    |

---

## Tech Stack

### Front End

| Layer             | Choice                    | Rationale                                                                     |
| ----------------- | ------------------------- | ----------------------------------------------------------------------------- |
| Framework         | React 19 + TypeScript     | Component model, ecosystem, strong types                                      |
| Build tool        | Vite                      | Fast dev server, ES module native                                             |
| 3D map engine     | MapLibre GL JS v4         | WebGL2, open source, no vendor lock-in, 3D buildings + terrain out of the box |
| Vehicle rendering | deck.gl `ScenegraphLayer` | GPU-instanced glTF models, syncs perfectly with MapLibre camera               |
| State management  | Zustand                   | Lightweight, no boilerplate                                                   |
| UI components     | shadcn/ui + Tailwind v4   | Accessible, headless, easily styled                                           |
| Map tiles         | Protomaps (PMTiles)       | Single static file, self-hostable, no tile server needed                      |
| 3D buildings      | MapTiler free tier        | `fill-extrusion` layer for Philly building footprints                         |

### Data / Backend

| Layer            | Choice                                                         | Rationale                                                                          |
| ---------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Open data access | Existing Python/CLI library                                    | Wraps OpenDataPhilly + SEPTA APIs; already handles auth, pagination, normalization |
| Real-time feed   | SEPTA TransitView API + GTFS-RT + SEPTA Metro `/api/v2/trips/` | Vehicle positions every ~15s; Metro route IDs include `L1`, `B1`, `B2`, `B3`, `M1` |
| History storage  | SQLite (via the CLI tool)                                      | Append vehicle positions with timestamps; lightweight, no infra needed             |
| WebSocket server | Node.js / Bun                                                  | Decodes GTFS-RT protobuf, interpolates positions, pushes deltas to browser         |
| Deployment       | Vercel or Cloudflare Pages                                     | Static front end; WebSocket server as an edge function or small standalone service |

### 3D Vehicle Models

Models are glTF/GLB format, loaded by deck.gl's `ScenegraphLayer`. Models should be low-poly (stylized, slightly chunky — think Monopoly piece energy, not photorealistic) for performance and aesthetic consistency.

| Vehicle               | Source                                       |
| --------------------- | -------------------------------------------- |
| SEPTA bus             | Sketchfab free asset or custom Blender model |
| Articulated bus       | Separate model                               |
| Trolley               | Custom Blender model                         |
| Subway/El car         | Custom Blender model                         |
| Regional Rail cab car | Custom Blender model                         |
| Amtrak locomotive     | Sketchfab or custom                          |

Models are oriented along the direction of travel. Heading is derived from the bearing between consecutive GPS positions.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser (React)                    │
│  MapLibre GL JS  ──── deck.gl ScenegraphLayer        │
│  (basemap + 3D buildings)   (animated vehicle models)│
│                  ↑                                   │
│           WebSocket client                           │
└──────────────────┬──────────────────────────────────┘
                   │ WebSocket (position deltas, ~1s)
┌──────────────────┴──────────────────────────────────┐
│              Real-time pipeline (Bun/Node)           │
│  - Polls SEPTA TransitView / GTFS-RT / Metro trips  │
│    API every 15s                                    │
│  - Dead-reckoning interpolator (fills gaps)         │
│  - Writes timestamped positions to SQLite           │
│  - Pushes interpolated frames at ~1fps to clients   │
└──────────────────┬──────────────────────────────────┘
                   │ Python subprocess / REST / file
┌──────────────────┴──────────────────────────────────┐
│         Python/CLI open data library                 │
│  - Wraps OpenDataPhilly, SEPTA APIs, GTFS feeds     │
│  - Agency-agnostic GTFS-RT decoder                  │
│  - History export / replay data prep                │
└─────────────────────────────────────────────────────┘
```

---

## Position Interpolation (Dead Reckoning)

SEPTA's vehicle feeds update every 15–30 seconds. To achieve smooth 60fps animation, the pipeline must interpolate between known positions.

Algorithm:

1. On each feed update, store `{vehicleId, lat, lng, bearing, timestamp, speed}`.
2. Between updates, compute estimated position using last known bearing and speed.
3. Apply cubic easing when a new position arrives to smoothly correct any drift.
4. Clamp interpolated positions to the known route geometry (snap-to-route) to prevent vehicles from jumping off roads/rails.

Snap-to-route is optional for buses (freeform routing) but important for rail (fixed tracks).

---

## History / Replay Mode

All vehicle positions are timestamped and written to SQLite by the pipeline. The front end can query any time window and replay it.

### Storage

- Schema: `(vehicle_id, route_id, agency, lat, lng, bearing, speed, recorded_at)`
- Retention: rolling 24 hours (configurable)
- Write rate: ~15s per vehicle; at peak ~700 active SEPTA vehicles = ~2,800 rows/minute = ~4M rows/day — manageable in SQLite, can migrate to DuckDB if needed

### Playback UI

- Timeline scrubber (full 24h range)
- Playback speed selector: 1×, 5×, 15×, 60×, 120×
- Play / pause / jump-to-now controls
- Clock display showing the "current" replay time
- Visual indicator distinguishing live mode from replay mode

### Replay Data Flow

In replay mode, the WebSocket server streams pre-recorded positions from SQLite at the requested playback speed instead of live feed data. The front end does not need to know the difference — it consumes the same position delta format.

---

## Map Configuration

- Default camera pitch: 45–55° (isometric feel)
- Default bearing: ~15° off north (slight angle, like a Monopoly board)
- Default zoom: ~13 (neighborhood level, most of central Philly visible)
- 3D buildings: enabled, MapTiler building layer
- Terrain: disabled by default (Philly is flat; optional toggle)
- Day/night lighting: sun position calculated from real time, ambient light adjusts accordingly
- Underground mode: Market-Frankford El runs partially underground — toggle to see subway tunnels

---

## Vehicle Rendering Details

Each vehicle is rendered as a glTF model via `ScenegraphLayer` with:

- **Position**: interpolated lat/lng (see dead reckoning above)
- **Orientation**: bearing derived from position delta, smoothed with exponential moving average to avoid jitter
- **Scale**: uniform per vehicle type; slightly exaggerated for visibility at city zoom levels
- **Color**: matched to SEPTA route color (e.g. MFL = blue, BSL = orange, Regional Rail lines use their published colors)

Click on any vehicle to show a popup with: route name, vehicle ID, next stop, scheduled vs actual time, current delay (if available).

---

## Generic / Multi-Agency Design

The data layer is abstracted behind an `Agency` interface. Adding a new agency requires:

1. A GTFS-RT or polling endpoint for vehicle positions
2. A GTFS static feed for route/stop metadata
3. An entry in `agencies.config.ts` with the endpoint, refresh interval, and vehicle type mappings

The Python library already handles GTFS normalization — the pipeline consumes its output format regardless of source agency.

---

## Phase 1 Scope

- [ ] Real-time positions for all SEPTA vehicle types
- [ ] 3D model rendering for bus, trolley, subway, regional rail
- [ ] Dead-reckoning interpolation
- [ ] Click-to-inspect vehicle popup
- [ ] 24-hour history recording
- [ ] Replay mode with scrubber + speed control
- [ ] Protomaps basemap + MapTiler 3D buildings
- [ ] Deploy to Vercel/Cloudflare Pages

## Phase 2 (Future)

- Amtrak + NJ Transit overlays
- Route filtering / show only selected lines
- Delay heatmap overlay (aggregate view)
- Station-level popups with live arrivals
- Mobile-optimized layout
- Eco mode (reduced frame rate for battery saving)
- Embed API (drop the map into any page as a `<script>` tag)

---

## Development Workflow

The Python/CLI open data library is the primary tool for data exploration and iteration. Claude Code can use it directly to:

- Fetch and inspect live SEPTA vehicle positions, including Metro route IDs `L1`, `B1`, `B2`, `B3`, and `M1`
- Query historical data from SQLite
- Test GTFS-RT decoding and normalization
- Generate sample data fixtures for front-end development

The front end can run against either live WebSocket data or a fixture file for offline development.

---

## Key Gotchas

**GTFS-RT protobuf decoding**: The raw SEPTA GTFS-RT feed is binary protobuf. The pipeline must decode it using the standard `gtfs-realtime.proto` schema before normalizing to the internal position format.

**Bearing jitter**: GPS positions from transit vehicles are noisy. Compute bearing from a rolling window of the last 3–5 positions rather than just the last two, and apply EMA smoothing. Without this, model orientation flickers constantly.

**SEPTA Metro `NO GPS` responses**: SEPTA's Metro trips endpoint exists for `L1`, `B1`, `B2`, `B3`, and `M1`, but Market-Frankford and Broad Street trips may return active trains with `status = "NO GPS"` and null coordinates. The renderer should treat these as real active trips but only animate trains with coordinates; if desired, a later fallback can infer station-to-station motion from schedule progress.

**ScenegraphLayer + MapLibre interleaving**: To have 3D vehicle models correctly occlude behind buildings, use `MapboxOverlay` with `interleaved: true`. This requires WebGL2 (`maplibre-gl >= 3`). Without interleaving, vehicles will render on top of buildings regardless of depth.

**glTF model up-axis**: Many glTF models exported from Blender are Y-up. deck.gl's `ScenegraphLayer` expects Z-up. Set `getOrientation` to compensate: `getOrientation: d => [0, 0, d.bearing]` usually needs an additional 90° X-axis rotation for Blender exports.

**SQLite write contention**: The pipeline writes positions at high frequency. Use WAL mode (`PRAGMA journal_mode=WAL`) to allow concurrent reads (replay queries) without blocking writes.

---

## Inspiration / Prior Art

| Project             | City         | Notes                                             |
| ------------------- | ------------ | ------------------------------------------------- |
| Mini Tokyo 3D       | Tokyo        | Gold standard; open source; Japan-specific APIs   |
| Live Tube Map       | London       | 3D, real-time, beautiful aesthetic; closed source |
| AP Transit          | New York     | Native mobile app, 3D subway map                  |
| TransitFlow         | Philadelphia | 2017, pre-rendered video only, not interactive    |
| SEPTA Real-time Map | Philadelphia | Official, 2D dot map                              |

**No one has built this for a US city as an open-source, interactive, 3D, history-capable web app. This is the gap.**
