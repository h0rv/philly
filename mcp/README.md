# Philly MCP

<img src="../assets/philly.svg" width="240" alt="Philly">


Read-only Philadelphia open data, using the official TypeScript MCP SDK **2.3.0** and protocol **2026-07-28**. `createMcpHandler` creates a fresh server per request, with default legacy stateless Streamable HTTP compatibility. Local stdio serves both eras too. The Python `phl` CLI and library are installed separately.

Connect using Streamable HTTP at **https://philly.horv.co/mcp**. The public, read-only service requires no API key. Cloudflare serves the website and MCP together; see [deployment details and verification status](DEPLOYMENT.md).

## Run locally

Stdio requires Node 24+ and npm. Local HTTP also requires Bun 1.3.11 to build the website assets used by the Worker configuration.

```sh
cd mcp
npm ci
npm run stdio
```

A desktop client's stdio configuration (replace the absolute path):

```json
{
  "mcpServers": {
    "philly": {
      "command": "node",
      "args": ["--experimental-strip-types", "/absolute/path/to/philly/mcp/src/stdio.ts"]
    }
  }
}
```

Claude Code:

```sh
claude mcp add philly -- node --experimental-strip-types /absolute/path/to/philly/mcp/src/stdio.ts
```

For VS Code, use `.vscode/mcp.json` with the same command/args beneath `"servers": { "philly": { "type": "stdio", ... } }`. [Claude Code reference](https://code.claude.com/docs/en/mcp); [VS Code reference](https://code.visualstudio.com/docs/agent-customization/mcp-servers).

For local HTTP, run this complete setup from the repository root:

```sh
cd website
bun install --frozen-lockfile
bun run build
cd ../mcp
npm ci
npm run dev
```

Use **http://127.0.0.1:8787/mcp** in a local Streamable HTTP client. `GET /mcp/health` reports health/version, catalog count and limits without calling upstream. Cloud-hosted assistants cannot reach localhost. The website provides copyable setup examples at `/connect/` and machine-readable guidance at `/llms.txt`.

## Tools

| Tool | Inputs | Result |
| --- | --- | --- |
| `search_datasets` | `query`, `offset`, `limit` | Word search, stable dataset IDs, licenses and queryability |
| `describe_dataset` | `datasetId`, resource-page `offset`, `limit` | Source metadata, resource IDs, explicit capabilities |
| `get_schema` | `datasetId`, optional `resourceId` | CARTO `LIMIT 0` or ArcGIS layer field metadata, not a sampled schema |
| `preview_dataset` | Same as query | Bounded preview; default 20 rows, no geometry in default columns |
| `query_dataset` | IDs, `columns`, `filters`, `orderBy`, `limit`, `offset` | Live rows, applied filters, attribution and pagination |
| `aggregate_dataset` | IDs, `operation`, `column`, `groupBy`, `filters`, `limit` | Full-source count or numeric sum/avg/min/max; bounded groups |

`philly://usage` describes operational limits. Call `tools/list` for exact input and output schemas. Tool results include identical JSON in `structuredContent` and a text block so clients that only read text receive the data too. On failure `isError` is true and both representations contain `error: {code, message}`; upstream bodies are not echoed. Output schemas include success and error branches for SDK v1 compatibility.

Example tool arguments, after discovering the dataset and resource IDs:

```json
{
  "datasetId": "Crime_Incidents",
  "columns": ["cartodb_id", "text_general_code"],
  "filters": [{"column": "hour", "op": "eq", "value": 14}],
  "limit": 5
}
```

Filters are AND-only: `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `is_null`, `not_null`. Comparisons require a non-null value; null operators need no value. Column identifiers must exist in backend schema metadata. Strings are quoted, never interpreted as SQL. Numeric aggregates require a backend numeric column; `count` defaults to `COUNT(*)`, or counts non-null values with `column`. Grouped results expose `group_value` and `value`. A full group page may be truncated; it has no continuation cursor.

Resource selection preserves the first approved CARTO resource **in catalog order** when present, otherwise selects the first approved ArcGIS layer. This can be one year's resource, not the entire dataset. Pass `resourceId` explicitly when scope matters. Original query predicates are retained inside a subquery, including crime-year boundaries. Count scope is the full matching **resource**, never a preview sample. There is no raw SQL, arbitrary URL, write tool, join, regex predicate or automatic fanout.

## Catalog and backend policy

`npm run catalog` deterministically generates the Worker catalog from all checked-in Python YAML files. No network is used. Dataset IDs are YAML filenames without `.yaml`; resource IDs use a preserved explicit ID when present, otherwise a hash of URL + name + format. Resource reordering does not change IDs. Metadata is a snapshot, not a freshness guarantee. CI fails if generated content drifts.

Only HTTPS `phl.carto.com/api/v2/sql` resources with a narrow, generator-validated SELECT grammar are enabled. The grammar permits a single table, optional known lat/lng projections, and simple date comparisons. It rejects arbitrary expressions, joins and other catalog SQL. Reviewed source queries are compiled into the service; clients cannot supply source SQL. Use discovery to inspect current resource capabilities. Descriptions paginate resources; default 10, maximum 20.

Approved ArcGIS resources use only HTTPS `services.arcgis.com`, the Philadelphia tenant `fLeGjb7u4uXqeF9q`, and numbered FeatureServer layers. Catalog URLs with scoped predicates or unknown parameters are rejected. The GeotechnicalTest service root has an explicit reviewed layer-0 mapping that preserves its existing resource ID. Schema comes from layer metadata; query requires pagination and ordering capabilities and adds the object ID as a tie breaker. Geometry is excluded. ArcGIS aggregation is unsupported.

Static files, unrecognized CARTO queries and other hosts remain **discovery-only**. These return source links and explicit unsupported errors for remote operations. They are not passed through the Python static loaders, which can download entire resources. Use the local Python library for broader format support. Add a new adapter only with backend capability checks, limits and fixtures.

Every successful data response identifies the dataset/resource, original source URL, source license (null when absent), retrieval timestamp and operation scope. Query and aggregate responses include filters and truncation. Schema comes from authoritative backend metadata, not row inference. Responses add `sourceId: "opendataphilly"` while preserving dataset/resource IDs. `completeness` and `completenessReasons` distinguish row/group caps, omitted default columns, offsets, upstream transfer caps, unknown completeness, and continuation limits. `truncated` remains the compatible row truncation flag. Data and descriptions are untrusted content, not instructions. Cite sources when presenting results; code's MIT license does not replace dataset licenses.

## Limits and failure behavior

- HTTP body / stdio buffer: 16 KiB. HTTP streamed bytes bounded before SDK JSON parsing; encoded requests rejected.
- Upstream: 128 KiB decoded bytes, checked on the stream before JSON parsing, regardless of Content-Length. Oversized/non-JSON/error/redirect responses fail; no retries.
- Tool structured result: 60 KiB, with room reserved for the matching JSON text fallback. HTTP output: 192 KiB including protocol envelope. Legacy SSE replies are buffered with the same cap; modern exchanges use JSON. No subscriptions, progress streams, sessions or HTTP batches. Send one MCP operation per request.
- Query: 100 rows + one lookahead, 20 selected columns, 8 filters, offset ≤10,000. Default columns omit geometry and indicate omissions.
- Operations: 8 seconds across schema + data calls, at most two sequential subrequests, no parallel fanout. Four admitted HTTP requests and four data operations per isolate; excess work gets HTTP 429 or an MCP busy error. Slow request reading/output has a 9-second deadline. Cancellation propagates to upstream fetch/stream reads.
- Only fixed approved origins/paths are fetched; redirects use `redirect: error`. Query and columns are built from validated structured arguments and backend fields.
- Exact configured `PUBLIC_ORIGIN`, URL host and Host header required. An absent Origin is valid for native MCP clients; a present Origin must exactly match. Browser CORS is same-origin only. GET/DELETE session endpoints are not supported.
- Pagination defaults to `cartodb_id` ordering where available; otherwise supply `orderBy` for offsets. Prefer a unique key; live changes/ties can cause skips or duplicates. A cap may prevent a next page even when `truncated` is true.

The concurrency gate is isolate-local, not a distributed rate limit. Public abuse can exhaust the account's free daily request allowance. No persistent rate-limit database, accounts or paid services are provisioned.

## Verification

```sh
npm run catalog
npm run check
npm test
npm run build          # dry-run packaging only; build website first
npm run test:workerd   # offline current + legacy clients against bundled workerd
npm run benchmark     # local Node microbenchmark, not production CPU proof
npm run test:browser  # install Chromium first; or set CHROMIUM_PATH
```

From the root also run `uv run pytest tests -n 4` and `uv run poe all` (which does **not** run tests). Shared YAML/JSON contract tests ensure the Worker retains Python dataset/resource identities and licenses. Fixtures cover preserved resource predicates, schema, structured filters, count/aggregate, output caps, unsupported operations, malformed upstream data, cancellation and concurrency. Tests include real SDK v2 and v1 Streamable HTTP clients plus both stdio eras. Browser tests cover mobile/desktop, keyboard navigation, light/dark layout and copy fallback.

SDK references: [2.3.0 release](https://github.com/modelcontextprotocol/typescript-sdk/releases/tag/v2.3.0), [v2 documentation](https://ts.sdk.modelcontextprotocol.io/v2/), [current protocol](https://modelcontextprotocol.io/specification/2026-07-28).

See [workflow evidence](EFFECTIVENESS.md) for tested questions and limitations. Row queries accept `orderDirection`; grouped aggregates accept `orderBy: "value"` and `orderDirection: "desc"` for ranked totals. Defaults remain ascending.

## Client code composition

The six bounded tools remain composable from client code. Discover and describe IDs, read schema, then query selected columns and reduce rows in the client. Return only the final projection to the model; preserve attribution and completeness metadata. No server-side eval or extra execution tool is introduced. [Cloudflare Code Mode patterns](https://developers.cloudflare.com/agents/model-context-protocol/codemode/) describe server-side wrappers for clients needing their own execution host; clients already providing code composition can call these typed tools directly.

For example, query `PWD_Geotechnical_Tests` with `columns: ["OBJECTID"]` after confirming its live schema. Field names must be discovered, not assumed. Live ArcGIS checks on 2026-10-17 were blocked by the execution environment proxy (CONNECT HTTP 403); fixture tests do not establish current upstream availability or data correctness.

## Scoped provider catalog (draft, requires this server version)

Additional sources are defined separately from the generated OpenDataPhilly snapshot in `src/provider-catalog.ts`. They use distinct source/dataset IDs and the same six tools. They are not installed into the Python local static-file loaders.

| Dataset ID | Source scope | Paging |
| --- | --- | --- |
| `loc:philadelphia_engineering` | LoC historic buildings/landscapes/engineering collection search for Philadelphia engineering | Native search pages; keep `limit` fixed and `offset` divisible by it |
| `usgs:schuylkill_latest` | USGS v1 latest-continuous, monitoring location `USGS-01474500` | Bounded native pages; source units, timestamps, parameter and approval codes retained |
| `musicbrainz:philadelphia_places` | MusicBrainz places whose associated area matches Philadelphia | Native relevance pages; includes studios and closed places |
| `noaa:philadelphia_tides_today` | Station `8545240`, today's UTC hourly predictions | Bounded 24-hour window; heights in meters relative to MLLW |

These providers support schema, selected columns, preview/query and native paging. Filters, custom ordering and aggregation fail explicitly rather than applying approximate semantics. Provider schemas describe versioned normalized projections, identified by `schemaSource`; they are not live upstream metadata. `complete` means the entire specified resource scope was returned, not all Philadelphia records across all time. NOAA predictions are not observations; USGS latest values are not historical series or flood forecasts. Null licenses mean no universal license is asserted; check item/source rights. MusicBrainz only returns core-data fields, not supplementary tags/ratings.

All four endpoints and scopes are fixed in the reviewed catalog. Provider response links are never followed. The MusicBrainz admission cooldown allows one request per second per service instance and returns `BUSY` otherwise; this is not a cross-isolate/IP-wide limiter. Upstream throttling fails without retries. All providers share operation/byte/row/output limits. Live probes on 2026-10-17 were blocked by the environment CONNECT proxy (HTTP 403), so these adapters remain fixture-verified draft work pending live checks from an authorized environment.

Official API references: [LoC JSON requests](https://www.loc.gov/apis/json-and-yaml/requests/), [USGS OGC v1](https://api.waterdata.usgs.gov/docs/ogcapi/), [MusicBrainz place search](https://musicbrainz.org/doc/MusicBrainz_API/Search#Place), [NOAA CO-OPS](https://api.tidesandcurrents.noaa.gov/api/prod/).

`npm run test:python` exercises the Python SDK and actual `phl remote` CLI through a local Worker MCP transport into a provider fixture. Run `uv sync --locked --group test` first. CI runs this cross-language check; `PHILLY_PYTHON` can select another interpreter. It does not query live providers.

MusicBrainz places additionally expose nullable WGS84 coordinates, original partial founded/closed dates, the reported `ended` flag, and a canonical record URL. See [field provenance and an address/location join example](MUSICBRAINZ.md). These support candidate connections to local records without adding recordings lookups or city-boundary/coverage claims.
