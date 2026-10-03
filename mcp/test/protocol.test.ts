import { test } from "node:test";
import assert from "node:assert/strict";
import {
  Client,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
import { Client as LegacyClient } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport as LegacyTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createWorker } from "../src/worker.ts";
import { LIMITS } from "../src/bounds.ts";
import { DataService } from "../src/data.ts";
const origin = "http://127.0.0.1:8787";
const env = { PUBLIC_ORIGIN: origin };
function setup() {
  let network = 0;
  const worker = createWorker(
    new DataService(async () => {
      network++;
      throw new Error("No network allowed in protocol fixtures");
    }),
  );
  const fetcher: typeof fetch = async (url, init) =>
    worker.fetch(new Request(url, init), env);
  return { worker, fetcher, network: () => network };
}
for (const modern of [true, false]) {
  test(`${modern ? "SDK v2 pinned 2026-07-28" : "SDK v1 legacy StreamableHTTP"} client discovery, tools and errors`, async () => {
    const { fetcher, network } = setup();
    const client = modern
      ? new Client(
          { name: "fixture", version: "1" },
          { versionNegotiation: { mode: { pin: "2026-07-28" } } },
        )
      : new LegacyClient({ name: "fixture", version: "1" });
    const transport = modern
      ? new StreamableHTTPClientTransport(new URL(`${origin}/mcp`), {
          fetch: fetcher,
        })
      : new LegacyTransport(new URL(`${origin}/mcp`), { fetch: fetcher });
    try {
      await client.connect(transport);
      const listed = await client.listTools();
      assert.equal(listed.tools.length, 6);
      const result = await client.callTool({
        name: "search_datasets",
        arguments: { query: "crime" },
      });
      assert.ok(JSON.stringify(result).includes("Crime_Incidents"));
      const content = result.content;
      assert.ok(Array.isArray(content));
      const text = content.find((item) => item.type === "text");
      assert.ok(text && text.type === "text");
      assert.deepEqual(JSON.parse(text.text), result.structuredContent);
      const bad = await client.callTool({
        name: "query_dataset",
        arguments: { datasetId: "Crime_Incidents", limit: 999999 },
      });
      assert.equal(bad.isError, true);
      const unsupported = await client.callTool({
        name: "query_dataset",
        arguments: { datasetId: "does-not-exist" },
      });
      assert.equal(unsupported.isError, true);
      assert.equal(network(), 0);
      assert.equal((await client.listResources()).resources.length, 1);
    } finally {
      await client.close();
    }
  });
}
test("HTTP perimeter allows originless clients, rejects spoofed origins/hosts and encoded/oversized bodies", async () => {
  const { worker } = setup();
  const req = (path: string, init?: RequestInit) =>
    worker.fetch(new Request(origin + path, init), env);
  assert.equal((await req("/mcp/health")).status, 200);
  for (const invalidOrigin of [
    "",
    "null",
    origin + "/",
    "https://evil.test",
    origin + ", https://evil.test",
  ]) {
    assert.equal(
      (await req("/mcp/health", { headers: { Origin: invalidOrigin } })).status,
      403,
    );
  }
  assert.equal(
    (await req("/mcp/health", { headers: { Origin: origin } })).status,
    200,
  );
  assert.equal(
    (await req("/mcp/health", { headers: { Origin: "https://evil.test" } }))
      .status,
    403,
  );
  assert.equal(
    (await req("/mcp/health", { headers: { Host: "evil.test" } })).status,
    403,
  );
  assert.equal(
    (await worker.fetch(new Request("https://evil.test/mcp"), env)).status,
    403,
  );
  assert.equal((await req("/mcp")).status, 405);
  assert.equal(
    (
      await req("/mcp", {
        method: "POST",
        headers: { "content-encoding": "gzip" },
        body: "{}",
      })
    ).status,
    415,
  );
  assert.equal(
    (
      await req("/mcp", {
        method: "POST",
        body: "x".repeat(LIMITS.requestBytes + 1),
      })
    ).status,
    413,
  );
  assert.equal(
    (
      await req("/mcp", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json, text/event-stream",
        },
        body: "{",
      })
    ).status,
    400,
  );
});
test("concurrent HTTP body reads have a bounded admission gate and release after cancellation", async () => {
  const { worker } = setup();
  const controllers = Array.from(
    { length: LIMITS.concurrency },
    () => new AbortController(),
  );
  const held = controllers.map((controller) =>
    worker.fetch(
      new Request(origin + "/mcp", {
        method: "POST",
        body: new ReadableStream({}),
        duplex: "half",
        signal: controller.signal,
      } as RequestInit),
      env,
    ),
  );
  const busy = await worker.fetch(
    new Request(origin + "/mcp", { method: "POST", body: "{}" }),
    env,
  );
  assert.equal(busy.status, 429);
  controllers.forEach((c) => c.abort());
  await Promise.all(held);
  assert.notEqual(
    (
      await worker.fetch(
        new Request(origin + "/mcp", { method: "POST", body: "{}" }),
        env,
      )
    ).status,
    429,
  );
});
test("legacy batches cannot amplify work behind one HTTP admission slot", async () => {
  const { worker, network } = setup();
  const request = {
    jsonrpc: "2.0",
    id: 1,
    method: "tools/call",
    params: {
      name: "query_dataset",
      arguments: { datasetId: "Crime_Incidents" },
    },
  };
  const response = await worker.fetch(
    new Request(origin + "/mcp", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json,text/event-stream",
      },
      body: " \n" + JSON.stringify([request, { ...request, id: 2 }]),
    }),
    env,
  );
  assert.equal(response.status, 400);
  assert.equal((await response.json()).error.code, -32600);
  assert.equal(network(), 0);
});
