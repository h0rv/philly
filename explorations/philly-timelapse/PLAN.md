# Philadelphia Through Time: 1860–2023

### A full-city aerial timelapse viewer

---

## The Idea

Philadelphia has publicly available aerial/map imagery for 23 distinct years spanning 1860 to 2023 — all served as live ArcGIS tile map services. Every single one follows the same tile format. You can point a map viewer at any of them and pan around the entire city.

This is a browser-based interactive map with a time scrubber. You drag the slider, the city changes beneath you. 163 years of Philadelphia, explorable at any zoom level, any neighborhood, any block.

No downloads. No preprocessing. Tiles load on demand from the city's own servers, cached aggressively at the edge.

---

## Tile Inventory

All 23 snapshots, in chronological order:

| Year | Type                            | Resolution | Tile URL Base                                                                                                    |
| ---- | ------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| 1860 | Historic atlas (Hexamer Locher) | —          | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/HistoricHexamerLocherAtlas_1860/MapServer` |
| 1875 | Historic atlas (GM Hopkins)     | —          | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/HistoricGMHopkinsAtlas_1875/MapServer`     |
| 1895 | Historic atlas (Bromley)        | —          | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/HistoricBromleyAtlas_1895/MapServer`       |
| 1910 | Historic atlas (Bromley)        | —          | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/HistoricBromleyAtlas_1910/MapServer`       |
| 1942 | Historic land use (WPA)         | —          | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/HistoricLandUse_1942/MapServer`            |
| 1962 | Historic land use (WPA)         | —          | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/HistoricLandUse_1962/MapServer`            |
| 1996 | Aerial photo                    | 6 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_1996_6in/MapServer`            |
| 2000 | Aerial photo                    | 18 in      | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2000_18in/MapServer`           |
| 2004 | Aerial photo                    | 6 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2004_6in/MapServer`            |
| 2005 | Aerial photo                    | 16 in      | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2005_16in/MapServer`           |
| 2008 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2008_3in/MapServer`            |
| 2009 | Aerial photo                    | 6 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2009_6in/MapServer`            |
| 2010 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2010_3in/MapServer`            |
| 2011 | Aerial photo (leaf-on)          | 12 in      | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2011_12in_LEAFON/MapServer`    |
| 2012 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2012_3in/MapServer`            |
| 2014 | Aerial photo                    | 6 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2014_6in/MapServer`            |
| 2015 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2015_3in/MapServer`            |
| 2016 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2016_3in/MapServer`            |
| 2017 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2017_3in/MapServer`            |
| 2018 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2018_3in/MapServer`            |
| 2019 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2019_3in/MapServer`            |
| 2020 | Aerial photo                    | 3 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2020_3in/MapServer`            |
| 2022 | Aerial photo                    | 2 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2022_2in/MapServer`            |
| 2023 | Aerial photo                    | 2 in       | `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2023/MapServer`                |

**ArcGIS tile URL format:** `{base}/tile/{z}/{y}/{x}` _(note: y before x — differs from standard slippy map)_

---

## Architecture

### Frontend: MapLibre GL JS

MapLibre (open-source fork of Mapbox GL) handles this perfectly:

- Raster tile layers with opacity transitions — smooth crossfades between years
- All tile fetching, caching, and rendering handled natively
- Supports custom tile URL templates

Each year is a raster layer. On slider change, the active layer fades in, previous fades out. At any moment only 2 layers are loaded (current + next/previous for prefetch).

**No backend required for v1.** All tiles are fetched directly from ArcGIS's CDN.

### Tile URL Adapter

ArcGIS uses `{z}/{y}/{x}` order. MapLibre expects `{z}/{x}/{y}`. A simple proxy edge function flips the parameters:

```
/api/tiles/{year}/{z}/{x}/{y} → tiles.arcgis.com/.../tile/{z}/{y}/{x}
```

This proxy also:

- Sets aggressive `Cache-Control` headers (these tiles will never change — historical data)
- Enables CORS for any origin
- Can be deployed as a Cloudflare Worker, Vercel Edge Function, etc.

### Caching Strategy

- **Immutable tiles**: Historical imagery never changes. Set `Cache-Control: public, max-age=31536000, immutable`
- **CDN layer**: Cloudflare in front of the proxy caches the tiles at the edge globally
- **Browser cache**: MapLibre respects cache headers, so repeat visits are instant
- **Prefetching**: When the slider is at year N, preload year N+1 and N-1 tiles for the current viewport

---

## UI/UX Design

### Layout

- Full-bleed map, no chrome
- Single timeline scrubber pinned to the bottom center
- Large year display (e.g., `1942`) floating over the map, top-left
- Subtle auto-play button (▶) — steps through years automatically with crossfades
- On hover over the scrubber: thumbnail previews of each year

### Timeline

```
1860  1875  1895  1910    1942  1962    1996 2000 2004 ··· 2023
  ●────●────●────●─────────●────●────────●────●────●─────────●
                                                         ▲
                                                      [cursor]
```

The gap between 1910 and 1942 is visual — years are not evenly spaced. The scrubber positions are proportional to real time, making the density of modern data visually obvious.

### Year character

Each era has a visual distinction that happens naturally from the source material:

- **1860–1910**: Hand-drawn atlas maps, sepia/ink style
- **1942–1962**: Early aerial, black & white
- **1996–2005**: Color aerial, lower resolution, washed-out
- **2008–2023**: High-res color, crisp — you can see individual cars

---

## Build Phases

### Phase 1 — Proof of concept (single HTML file)

- MapLibre GL JS loaded from CDN
- Hardcode the 24 tile sources
- Basic range input slider, swaps tile layer on change
- Instant crossfade via opacity

**Success criteria**: Can pan around the city and scrub through all years in the browser. ~100 lines of HTML/JS.

### Phase 2 — Edge tile proxy

- Cloudflare Worker (or Vercel Edge Function) that:
  - Proxies ArcGIS tiles
  - Fixes the y/x ordering
  - Sets immutable cache headers
- Route all tile requests through the proxy

**Success criteria**: Tiles cached at edge, tile URLs no longer hit ArcGIS directly for repeat requests.

### Phase 3 — Polish

- Smooth crossfade transitions (500ms opacity ease)
- Auto-play mode with configurable speed
- Year label with era context (e.g., "1942 — WWII-era Philadelphia")
- Thumbnail strip on scrubber hover
- Mobile touch support (pinch zoom, swipe scrubber)
- Deep-linkable URLs: `?year=1962&lat=39.95&lng=-75.16&z=15`

### Phase 4 — Overlay data layer (optional, post-launch)

- Toggle overlay: building demolitions 2006–2024 (dots appear/disappear over time)
- Toggle overlay: vacant property indicators
- These make the physical changes legible — you can literally watch buildings disappear

---

## Open Questions

1. **CORS on ArcGIS tiles**: Test whether `tiles.arcgis.com` serves CORS headers for browser requests. If yes, Phase 2 proxy is optional for v1. If no, proxy is required from day 1.

2. **Zoom level support per year**: Older imagery (1996, 2000) may not support deep zoom levels. Need to test max zoom per year and clamp the slider accordingly so the map doesn't show blank tiles.

3. **Projection for historic atlases (1860–1910)**: The Bromley/Hopkins atlases may be served in a different projection or at different tile matrix sets than the aerial photos. Test these first — they might need special handling.

4. **Performance on mobile**: At high zoom levels, switching years means loading a large number of tiles. Prefetching strategy needs to be conservative on mobile.

---

## First Steps

1. Open browser devtools, manually test a tile URL:
   `https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/CityImagery_2023/MapServer/tile/17/49223/38225`
   (That's approximately Center City at zoom 17)

2. Write the single-file HTML prototype (Phase 1). This is ~80 lines and takes 30 minutes. Validates the whole concept before investing in infrastructure.

3. If CORS works: ship Phase 1 as-is and iterate. If not: stand up the Cloudflare Worker first.
