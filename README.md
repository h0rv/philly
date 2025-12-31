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

# Stream data (memory-efficient, ideal for large datasets)
phl stream "Crime Incidents" --output-format csv
phl load "Crime Incidents" --stream  # equivalent

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

Output formats: `--format json|csv|table` (or `--output-format` for specific commands)

### Streaming for Unix Pipelines

Stream large datasets line-by-line without loading everything into memory:

```bash
# Stream and pipe to Unix tools
phl stream "Crime Incidents" --output-format csv | awk -F',' '$13 == "300"'
phl stream "Crime Incidents" --output-format jsonl | jq '.text_general_code' | sort | uniq -c

# Filter server-side before streaming
phl stream "Crime Incidents" --where "hour = '14'" --output-format csv | head -100
```

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
