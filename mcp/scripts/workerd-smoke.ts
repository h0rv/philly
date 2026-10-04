import assert from "node:assert/strict";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";
import {
  Client,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
const mf = new Miniflare(
  convertV4MiniflareOptions({
    host: "127.0.0.1",
    port: 8788,
    modules: true,
    scriptPath: "dist/worker.js",
    compatibilityDate: "2026-10-03",
    bindings: { PUBLIC_ORIGIN: "http://127.0.0.1:8788" },
  }),
);
try {
  await mf.ready;
  for (const mode of ["legacy", { pin: "2026-07-28" }] as const) {
    const client = new Client(
      { name: "workerd-test", version: "1" },
      { versionNegotiation: { mode } },
    );
    await client.connect(
      new StreamableHTTPClientTransport(new URL("http://127.0.0.1:8788/mcp")),
    );
    assert.equal((await client.listTools()).tools.length, 6);
    const result = await client.callTool({
      name: "search_datasets",
      arguments: { query: "crime" },
    });
    assert.ok(JSON.stringify(result).includes("Crime_Incidents"));
    await client.close();
  }
  assert.equal((await fetch("http://127.0.0.1:8788/mcp/health")).status, 200);
  console.log(
    "workerd: current + legacy MCP clients, search, schema listing and health passed (offline).",
  );
} finally {
  await mf.dispose();
}
