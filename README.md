# Philly

<img src="./assets/logo.png" width="256" alt="logo">

Query Philadelphia's 400+ public datasets with server-side filtering, smart caching, and streaming.

## Installation

```bash
uv add philly
```

## Quick Start

```python
from philly import Philly

phl = Philly()

# Load with server-side filtering (only matching rows are downloaded)
data = await phl.load("Crime Incidents", where="dispatch_date >= '2024-01-01'", limit=1000)

# Stream large datasets without loading into memory
async for chunk in phl.stream("Crime Incidents"):
    process(chunk)
```

## CLI

```bash
# Discovery
phl datasets                           # List all 400+ datasets
phl search "crime" --fuzzy             # Fuzzy search
phl info "Crime Incidents"             # Dataset metadata

# Load data
phl load "Crime Incidents" --limit 100
phl load "Crime Incidents" --where "hour = '14'" --format csv

# Stream to Unix pipes
phl stream "Crime Incidents" --output-format csv | head -1000
phl stream "Crime Incidents" --output-format jsonl | jq '.text_general_code'

# Preview
phl sample "Crime Incidents" --limit 10
phl columns "Crime Incidents"
phl schema "Crime Incidents"
phl count "Crime Incidents"

# Cache management
phl cache-info
phl cache-clear
```

## Why Philly?

|                       | requests + pandas   | philly                          |
| --------------------- | ------------------- | ------------------------------- |
| Server-side filtering | Manual URL building | `--where "year = 2024"`         |
| Format handling       | Per-format code     | Auto-detects from 40+ formats   |
| Caching               | DIY                 | Built-in with TTL + LRU         |
| Dataset discovery     | Browse website      | `phl search "permits"`          |
| Streaming             | Manual chunking     | `phl stream` / async generators |

## Configuration

Create `~/.config/philly/config.yml`:

```yaml
cache:
  enabled: true
  ttl: 3600
  directory: ~/.cache/philly

defaults:
  format_preference: [csv, geojson, json]
```

## Website

This repo now includes a small Astro site for GitHub Pages, kept separate from the Python package source.
We use Poe tasks from the repo root.

```bash
uv run poe site-install
uv run poe site-dev
uv run poe site-build
```

The site build does two things:

1. Builds the Astro landing page and explorations index
2. Copies any ready exploration artifacts into `website/public/explorations/`

Exploration publish metadata lives in `website/config/explorations.mjs`.
If an exploration is missing required generated assets, it stays listed as build-pending instead of shipping a broken link.

## License

MIT
