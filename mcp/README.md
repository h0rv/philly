# Philly MCP

Read-only Philadelphia open data, using the official TypeScript MCP SDK **2.3.0** and protocol **2026-07-28**. `createMcpHandler` creates a fresh server per request, with default legacy stateless Streamable HTTP compatibility. Local stdio serves both eras too. The Python `phl` CLI and library remain independent and unchanged.

**Status: prepared and locally tested, not deployed.** No public MCP URL exists yet. The existing GitHub Pages production site is not changed by this branch. See [deployment readiness](DEPLOYMENT.md).

## Run locally

Requires Node 24+ and npm; Bun 1.3.11 builds the optional website.

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

For local HTTP, first build static assets from the repository root:

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
| `get_schema` | `datasetId`, optional `resourceId` | CARTO query field metadata using `LIMIT 0`, not a sampled schema |
| `preview_dataset` | Same as query | Bounded preview; default 20 rows, no geometry in default columns |
| `query_dataset` | IDs, `columns`, `filters`, `orderBy`, `limit`, `offset` | Live rows, applied filters, attribution and pagination |
| `aggregate_dataset` | IDs, `operation`, `column`, `groupBy`, `filters`, `limit` | Full-source count or numeric sum/avg/min/max; bounded groups |

`philly://usage` describes operational limits. Call `tools/list` for exact input schemas. Tool results include identical JSON in `structuredContent` and a text block so clients that only read text receive the data too. On failure `isError` is true; upstream bodies are not echoed.

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

Resource selection defaults to the first approved resource **in catalog order**. This can be one year's resource, not the entire dataset. Pass `resourceId` explicitly when scope matters. Original query predicates are retained inside a subquery, including crime-year boundaries. Count scope is the full matching **resource**, never a preview sample. There is no raw SQL, arbitrary URL, write tool, join, regex predicate or automatic fanout.

## Catalog and backend policy

`npm run catalog` deterministically generates the Worker catalog and website counts from all checked-in Python YAML files. No network is used. Dataset IDs are YAML filenames without `.yaml`; resource IDs hash URL + name + format, so resource reordering does not change IDs. Metadata is a snapshot, not a freshness guarantee. CI fails if generated content drifts.

Only HTTPS `phl.carto.com/api/v2/sql` resources with a narrow, generator-validated SELECT grammar are enabled. The grammar permits a single table, optional known lat/lng projections, and simple date comparisons. It rejects arbitrary expressions, joins and other catalog SQL. Reviewed source queries are compiled into the service; clients cannot supply source SQL. The generated catalog currently enables 67 datasets out of 491. Descriptions paginate resources; default 10, maximum 20.

ArcGIS, static files, unrecognized CARTO queries and other hosts remain **discovery-only**. These return source links and explicit unsupported errors for remote operations. They are not passed through the Python static loaders, which can download entire resources. Use the local Python library for broader format support. Add a new adapter only with backend capability checks, limits and fixtures.

Every successful data response identifies the dataset/resource, original source URL, source license (null when absent), retrieval timestamp and operation scope. Query and aggregate responses include filters and truncation. Schema comes from CARTO field metadata, not row inference. Data and descriptions are untrusted content, not instructions. Cite sources when presenting results; code's MIT license does not replace dataset licenses.

## Limits and failure behavior

- HTTP body / stdio buffer: 16 KiB. HTTP streamed bytes bounded before SDK JSON parsing; encoded requests rejected.
- Upstream: 128 KiB decoded bytes, checked on the stream before JSON parsing, regardless of Content-Length. Oversized/non-JSON/error/redirect responses fail; no retries.
- Tool structured result: 60 KiB, with room reserved for the matching JSON text fallback. HTTP output: 192 KiB including protocol envelope. Legacy SSE replies are buffered with the same cap; modern exchanges use JSON. No subscriptions, progress streams, sessions or HTTP batches. Send one MCP operation per request.
- Query: 100 rows + one lookahead, 20 selected columns, 8 filters, offset ≤10,000. Default columns omit geometry and indicate omissions.
- Operations: 8 seconds across schema + data calls, at most two sequential subrequests, no parallel fanout. Four admitted HTTP requests and four data operations per isolate; excess work gets HTTP 429 or an MCP busy error. Slow request reading/output has a 9-second deadline. Cancellation propagates to upstream fetch/stream reads.
- Only the fixed approved origin/path is fetched; redirects use `redirect: error`. Query and columns are built from validated structured arguments and backend fields.
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
