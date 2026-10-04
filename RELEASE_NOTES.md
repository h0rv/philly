# 1.1.0

- Refresh catalog metadata from OpenDataPhilly commit `72c443beb6d83d7f9d3cd8b9d4cd0a1d59f33d12`: update Litter Index and add Dumpster, Unsafe Property Complaints by Hex Bins, and PIT Zones metadata. Record rejected/ambiguous upstream entries and retain legacy records; no live resource availability claim.
- Validate refreshes before writes, preserve dataset/resource identifiers, package source provenance, and check generated catalog consistency in CI.
- Add bounded read-only MCP discovery, schema, queries and aggregation with explicit provenance. Search resource years, sort recent rows and rank aggregate groups. Current and legacy MCP clients plus stdio are fixture-tested; live CARTO and Cloudflare deployment remain unverified.
- Simplify the install/connect site, share one SVG logo, remove drifting catalog counts, and add concise CLI and accessible Claude Code, Codex and Pi MCP setup tabs.
- Credit OpenDataPhilly as the catalog source, distinguish individual data publishers and terms, and include upstream notices in release distributions.
- Report the installed Python package version from distribution metadata instead of a separate CLI constant.

The Python library and `phl` CLI are the PyPI distribution. The TypeScript MCP server remains a source-checkout installation. GitHub Pages hosts the static website; the remote MCP URL is a placeholder for a separately deployed service.
