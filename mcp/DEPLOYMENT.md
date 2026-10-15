# Deployment

Website: **https://philly.horv.co/** · Remote MCP: **https://philly.horv.co/mcp**

Cloudflare deployment was confirmed on 2026-10-15 for main commit `a939489531095f08f79098d44a4b7086724ace39`: build `ae2fc263-d135-4ab5-8651-c587f584ddf9`, Worker version `b7bfc786-2610-4a53-9f58-6e208ad71cd4` at 100%. The custom domain is attached, with an active wildcard certificate; the dedicated certificate was pending at confirmation. Automatic builds from main are enabled.

## Configuration

One stateless `philly-mcp` Worker serves the existing Astro site and `/mcp`. Static assets are served first; `/mcp` and `/mcp/*` run the Worker first. `PUBLIC_ORIGIN` is `https://philly.horv.co`. The production environment selects the existing Cloudflare account. Local development retains its loopback origin.

The deployment uses Workers Free, 23 static assets and the existing authorized build credential. No new credentials, paid resources, Durable Objects, D1, KV, R2, scheduled jobs or session storage were added. Observability storage, workers.dev and preview URLs remain disabled. The custom domain is managed separately from Wrangler. The existing GitHub Pages workflow remains in place; the unrelated `isophilly` Worker is unchanged.

## Build and deploy

Use Node 24+ and Bun 1.3.11. From the repository root:

```sh
cd website
bun install --frozen-lockfile
bun run build
cd ../mcp
npm ci
npm run build -- --env production
```

The last command only packages a dry run. With the existing authorized Cloudflare build credential, deploy from `mcp`:

```sh
npx --no-install wrangler deploy --env production
```

The Worker entrypoint is `src/worker.ts`; the `ASSETS` binding uses `../website/dist`. The reviewed bundle was 407 KiB gzip. Do not enable paid services or change credential permissions for this deployment.

## Verification status

Local validation passed: 509 Python tests, 25 MCP tests, 10 browser tests, lint/type/catalog checks, Astro build, production dry-run packaging, and current/legacy MCP clients against workerd. PR and main CI passed for the deployment commit.

Deployment confirmation is not a live protocol test. This environment's proxy denied the direct HTTPS check with CONNECT 403; no bypass was attempted. Public TLS, current/legacy MCP discovery, origin rejection, and bounded live CARTO schema/preview/filter/count checks remain unverified here. Production CPU, memory and traffic headroom have not been measured. No authenticated agent model call is claimed.

The public service has isolate-local concurrency limits, not a distributed rate limit. Abuse can exhaust the account's Free daily request allowance. Keep the account on Free; do not upgrade to address quota exhaustion.

Rollback uses a previously verified Worker version through the existing deployment connection. Check domain and asset behavior after rollback; the Python package is independent.

References: [Workers limits](https://developers.cloudflare.com/workers/platform/limits/), [static asset billing](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/), [asset routing](https://developers.cloudflare.com/workers/static-assets/routing/advanced/).
