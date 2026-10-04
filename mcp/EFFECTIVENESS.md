# MCP workflow evidence

Offline audit and fixture verification, 2026-10-04. The generated catalog is checked against every packaged YAML record by the catalog contract test. This is not a live CARTO availability, accuracy, or model-answer evaluation.

| Question or step | Supported workflow | Evidence |
| --- | --- | --- |
| Find crime or 311 data for 2025 | Search includes resource names as well as dataset text. Describe resources and select the matching year's resource ID. | `year discovery, schema, latest preview and ranked counts preserve scope` finds both topics with 2025. |
| Identify supported sources and years | Paginated describe returns resource names, IDs, source URLs, backend capabilities and next offset. | Catalog pagination test; Crime Incidents has more resources than one page. |
| Inspect fields | `get_schema` fetches backend field metadata with LIMIT 0. | Schema test requires zero rows and rejects malformed metadata. |
| Retrieve recent matching records | Query/preview use structured filters, selected columns and `orderDirection: "desc"`; choose the appropriate timestamp/key as `orderBy`. | Workflow fixture uses descending unique IDs, filters hour >= 14, returns IDs 3 and 2 and next offset 2. |
| Count or rank categories | Aggregate uses full selected resource and filters; `orderBy: "value", orderDirection: "desc"` ranks totals, with ascending group key as tie breaker. | Workflow fixture asserts generated SQL and returns Theft=2, Burglary=1, unlike an ordinary row fixture. |
| Attribute the answer | Schema/query/aggregate retain selected resource name and ID, URL, license, retrieval timestamp, filters and truncation. | Workflow assertions compare all three responses to the explicitly selected 2025 resource. |
| Reject unsafe or unsupported requests | Identifiers are schema checked; ordering choices are enums. No raw SQL, arbitrary URL, or unbounded adapter. | Injection/error tests and ordering rejection assertions. |

Run `npm run check && npm test` from `mcp/`. The new workflow test uses fixed, internally consistent backend contract responses and inspects emitted SQL; it does not execute PostgreSQL. Existing protocol tests separately exercise tool discovery, invocation, structured/text responses and errors over current and legacy HTTP, plus stdio discovery. Preview and query share the same bounded service implementation.

## Limits

- The bundled catalog is a metadata snapshot. Search across resource names establishes a match, not a verified coverage guarantee. Describe and explicitly select the required resource, particularly for year-partitioned datasets. An omitted resource ID still chooses the first approved resource; `resourceName` now makes that scope visible.
- Defaults remain ascending row order and ascending group order. Request value/descending ordering explicitly for top categories.
- Pagination is offset based, capped at 10,000, and is not a snapshot. Nonunique sort keys and changing data can repeat or skip rows. Aggregate groups are bounded and may be truncated; aggregate pagination is not provided.
- Date/time min/max, date bucketing, spatial predicates, joins, OR filters and cross-resource aggregation are not implemented. Numeric aggregation remains intentionally restricted. Multi-year comparisons require separate explicit resources; do not sum potentially overlapping resources blindly.
- Static and ArcGIS resources are discovery only. Unsupported operations fail rather than download complete resources.
- No live CARTO request or model call was used in this verification. Real backend schema, current availability, answer accuracy and deployed performance still require approved bounded acceptance checks.

## Pi connection verification

Pi 1.0.2 (`@earendil-works/pi-coding-agent`) was checked against its CLI help and
[official MCP documentation](https://pi.dev/docs/latest/mcp). `pi mcp list --json`
connected to the built Worker over loopback Streamable HTTP using an isolated
`PI_CODING_AGENT_DIR` fixture. It reported `connected`, default `codemode` exposure,
all Philly tools and no errors. No real user configuration or credentials changed.
The homepage's `pi mcp add` command intentionally saves the user's connection; its
URL must be replaced after deployment. No authenticated model or public endpoint
run was performed.
