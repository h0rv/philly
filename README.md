# Philly

<img src="https://raw.githubusercontent.com/h0rv/philly/main/assets/philly.svg" width="320" alt="Philly">

Query Philadelphia public data from Python, your terminal, or an MCP client.

Catalog metadata comes from [OpenDataPhilly](https://opendataphilly.org/),
maintained in its [upstream repository](https://github.com/opendataphilly/opendataphilly-jkan).
Data is supplied by individual publishers, including City departments, nonprofits
and researchers, under each dataset's terms; see [OpenDataPhilly's terms](https://opendataphilly.org/about/#terms).
Philly is an independent project. Source URLs and recorded licenses accompany MCP results;
missing license metadata is not a grant of permission. The code's MIT license does
not relicense datasets. Upstream notices are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

## Installation

```bash
uv add philly
```

## MCP and website

The [MCP service](mcp/README.md) adds bounded, read-only discovery, schema,
preview, structured filtering and aggregation using the official TypeScript SDK
2.3.0. It runs locally over stdio or Streamable HTTP and is prepared for
Cloudflare Workers with static assets. **No public MCP endpoint is deployed yet.**
See [deployment readiness and approval gates](mcp/DEPLOYMENT.md).

The website now starts with install/connect instructions and a five-row SVG
wordmark adapted from [sprts](https://github.com/h0rv/sprts) with
[MIT attribution](assets/NOTICE). City Atlas is preserved at
`/explorations/city-atlas/`, alongside the existing exploration gallery. The MCP catalog is generated from the packaged YAML with
`cd mcp && npm run catalog`.

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
phl datasets                           # List available datasets
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

This repo includes a static Astro site, kept separate from the Python package source. The existing GitHub Pages workflow remains in place; the MCP project adds an undeployed Cloudflare configuration.
We use Poe tasks from the repo root.

```bash
uv run poe site-install
uv run poe site-dev
uv run poe site-build
```

The site build does two things:

1. Builds the install/connect landing page, machine-readable guide and explorations routes
2. Copies any ready exploration artifacts into `website/public/explorations/`

Exploration publish metadata lives in `website/config/explorations.mjs`.
If an exploration is missing required generated assets, it stays listed as build-pending instead of shipping a broken link.

## License

MIT

## Refresh catalog metadata

```sh
uv run python scripts/update_datasets.py
npm --prefix mcp run catalog
npm --prefix mcp run catalog:check
```

The updater pins the upstream Git revision, validates records before writing,
retains existing IDs, and records source paths, hashes and rejected records in
[src/philly/catalog-source.json](src/philly/catalog-source.json). Use
`--source /path/to/upstream-checkout` for an offline refresh. Invalid or duplicate
records stop the write; after reviewing the report, `--allow-partial` updates valid
records and retains legacy records without silently dropping them.

## Or ask an agent

Install and sign in to your agent, then replace the URL with your deployed MCP endpoint.
The [website](https://philly.horv.co/) offers copyable Claude Code, Codex and Pi tabs.

- **Claude Code:** pass an HTTP entry through `--mcp-config` and `--strict-mcp-config` for one run. [Setup docs](https://code.claude.com/docs/en/mcp).
- **Codex:** pass `-c 'mcp_servers.philly.url="https://YOUR-PHILLY-MCP-HOST/mcp"'` for one run. [Setup docs](https://developers.openai.com/codex/mcp/).
- **Pi 1+:** `pi mcp add philly --url https://YOUR-PHILLY-MCP-HOST/mcp` saves or replaces the user connection; Codemode activates automatically. [Setup docs](https://pi.dev/docs/latest/mcp).

Ask: "Use the philly MCP server to find building-permit datasets and show five sample records with source links."

These use standard MCP; no separate Philly plugin or skill is required. Keep the
agent's normal permission prompts. The remote Worker requires separate deployment.
CLI syntax/configuration and local MCP plumbing were checked without an authenticated
model call. See [MCP workflow evidence and limits](mcp/EFFECTIVENESS.md).
