# Agent Guide

Guidance for AI coding agents working in this repository.

## What this is

`philly` is a Python library + CLI (`phl`) for querying OpenDataPhilly's ~478 public datasets with server-side filtering, caching, and streaming. The repo also hosts data-journalism "explorations" and an Astro site that publishes them to GitHub Pages.

Two installable packages live under `src/`:
- `philly` — the library (`from philly import Philly`)
- `phl` — CLI output/formatting helpers; the `phl` entrypoint resolves to `philly.__main__:main`

## Commands

Use `uv` for everything. Tasks are defined as Poe tasks in `pyproject.toml`.

```bash
uv run poe test                      # run all tests (pytest)
uv run pytest tests/test_search.py   # single test file
uv run pytest tests/test_search.py::TestName::test_case  # single test
uv run poe fmt                       # ruff format
uv run poe lint                      # ruff check --fix
uv run poe type-check                # ty check src
uv run poe all                       # fmt + lint + type-check (NOT test)

uv run phl datasets                  # exercise the CLI
```

Website (separate Bun/Astro project under `website/`, not part of the Python package):

```bash
uv run poe site-install              # cd website && bun install
uv run poe site-dev                  # local Astro dev server
uv run poe site-build                # build for GitHub Pages
uv run poe site-sync                 # copy ready explorations into website/public
```

Refresh the packaged dataset catalog from upstream OpenDataPhilly:

```bash
uv run python scripts/update_datasets.py   # regenerates src/philly/datasets/*.yaml
```

## Architecture

### Catalog is local, data is remote
Each dataset is a checked-in YAML file in `src/philly/datasets/` (one per dataset, packaged via `package-data`). `Philly.__init__` loads all of them into memory at construction and keys them by `title`. Dataset *metadata* ships with the package; the actual *rows* are always fetched live over HTTP from the resource URLs. `scripts/update_datasets.py` regenerates these YAMLs by pulling from the `opendataphilly/opendataphilly-jkan` GitHub repo (`_datasets` dir).

A dataset has multiple **resources**, each with a `format` (csv, geojson, json, shp, api, html) and a `url`. Resource selection is format-driven: when no resource is named, the code walks `config.defaults.format_preference` (default `[csv, geojson, json]`) and picks the first match via `find_resource_by_format`.

### Backend detection drives every capability
`filtering.py::detect_backend(url)` classifies each resource URL into `CARTO`, `ARCGIS`, `STATIC`, or `UNKNOWN`. This single decision determines whether server-side filtering, counting, and streaming are possible:
- **Carto** (`phl.carto.com/api/v2/sql`): `build_carto_query` rewrites the SQL `q=` param — extracts the table name, then injects `SELECT cols ... WHERE ... LIMIT ... OFFSET`. WHERE clauses pass through `validate_where_clause` which blocks `DROP/DELETE/UPDATE/INSERT/TRUNCATE`.
- **ArcGIS** (`arcgis.com` or `FeatureServer`): `build_arcgis_query` sets `where`, `outFields`, `resultRecordCount`, `resultOffset` query params.
- **Static/Unknown**: server-side filtering is impossible; `load` logs a warning and downloads the whole file. Filtering then has to happen client-side.

When adding features that filter/limit/count/stream, branch on `BackendType` — never assume a single transport.

### The `Philly` facade
`philly/philly.py` is the public API; it's a thin orchestrator that delegates to single-purpose modules: `loaders` (format-aware download → DataFrame/GeoDataFrame/dict), `streaming` (paginated Carto/ArcGIS + static-file streamers), `sample` (header-only previews without full download), `metadata`, `search` (lazy-built index, fuzzy optional), `filters` (schema/column/example discovery from a sample), `cache`, `updates`, `format_selection`. Most public methods are `async` (network I/O); the CLI wraps them with `asyncio.run` via local `_load()/_stream()` closures.

### Caching
`FileCache` (`cache.py`) is a TTL + optional LRU file cache keyed by a SHA-256 of `{dataset, resource, format, params}` (filter params included, so filtered queries cache independently). Enabled by default at `~/.cache/philly`. `updates.py` compares cached entries' stored metadata against remote `Last-Modified`/size to flag staleness.

### Sampling without downloading
`sample.py` reads only enough bytes to get N rows (partial CSV/GeoJSON/JSON reads). `get_columns`, `get_schema`, `get_filter_schema`, and `get_filter_examples` are all built on top of `sample()` — they infer structure from a small sample, never a full load. Formats `html/shp/api/xml` can't be sampled and are skipped.

### Config
`philly.example.yml` / `phl init` scaffolds `~/.config/philly/config.yml` (or `./philly.yml`). Controls cache (enabled/ttl/directory/max_size_mb) and `defaults.format_preference`. Passing `config_file` explicitly to `Philly()` makes its cache settings override constructor args.

## Explorations + website pipeline
`explorations/` holds standalone data-journalism projects (Python fetch scripts + static `index.html`); `explorations/IDEAS.md` is the running idea backlog and a useful map of dataset clusters and join keys. Publish metadata lives in `website/config/explorations.mjs`; `website/scripts/build-explorations.mjs` copies ready artifacts into `website/public/explorations/`. An exploration missing required generated assets stays "build-pending" rather than shipping a broken link. CI: `.github/workflows/pages.yaml` + `publish.yaml`.

## Conventions
- Python ≥3.11; models are Pydantic (`philly/models/`). `Dataset.from_file` parses a YAML into a `Dataset`; `ResourceFormat` is a YAML-backed enum.
- Async tests use `@pytest.mark.asyncio`.
- `poe all` does not run tests — run `poe test` separately before claiming green.
