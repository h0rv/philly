# Philly

<img src="./assets/logo.png" width="256" alt="logo">

Python library and CLI for working with [OpenDataPhilly](https://opendataphilly.org/) datasets.

## Installation

```bash
pip install philly
```

Or with uv:

```bash
uv pip install philly
```

## Quick Start

```python
from philly import Philly

phl = Philly()
data = await phl.load("Crime Incidents", format="csv", limit=1000)
```

## CLI

```bash
# List datasets
phl datasets

# Search
phl search "crime" --fuzzy

# Load data
phl load "Crime Incidents" --format csv --limit 1000

# Sample preview
phl sample "Crime Incidents" --n 10

# Dataset info
phl info "Crime Incidents"
phl columns "Crime Incidents"
phl schema "Crime Incidents"

# Cache
phl cache-info
phl cache-clear

# Config
phl config show
phl config init
```

Output formats: `--format json|csv|table`

## Features

- **Formats**: CSV, JSON, GeoJSON, Shapefile, GeoPackage, GTFS, and more
- **Caching**: Automatic with configurable TTL and LRU eviction
- **Filtering**: Server-side WHERE, LIMIT, OFFSET for Carto/ArcGIS APIs
- **Search**: Fuzzy search across 400+ datasets
- **Streaming**: Memory-efficient iteration over large datasets
- **Preview**: Sample data without downloading everything

## Configuration

Create `philly.yml` in your project or `~/.config/philly/config.yml`:

```yaml
cache:
  enabled: true
  ttl: 3600
  directory: ~/.cache/philly

defaults:
  format_preference: [csv, geojson, json]
```

See `philly.example.yml` for all options.

## License

MIT
