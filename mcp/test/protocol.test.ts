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
import { outputSchemas } from "../src/contracts.ts";
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
      for (const tool of listed.tools) assert.ok(tool.outputSchema, tool.name);
      const result = await client.callTool({
        name: "search_datasets",
        arguments: { query: "crime" },
      });
      assert.ok(JSON.stringify(result).includes("Crime_Incidents"));
      outputSchemas.search_datasets.parse(result.structuredContent);
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
      assert.equal(
        (unsupported.structuredContent as { error: { code: string } }).error
          .code,
        "DATASET_NOT_FOUND",
      );
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

test("MCP output contracts cover discovery, CARTO and ArcGIS calls and preview alias", async () => {
  const arcMeta = {
    fields: [{ name: "OBJECTID", type: "esriFieldTypeOID" }],
    objectIdField: "OBJECTID",
    maxRecordCount: 1000,
    advancedQueryCapabilities: {
      supportsPagination: true,
      supportsOrderBy: true,
    },
  };
  const service = new DataService(async (url) => {
    const u = new URL(url);
    if (u.hostname === "services.arcgis.com")
      return Response.json(
        u.pathname.endsWith("/query")
          ? { features: [{ attributes: { OBJECTID: 1 } }] }
          : arcMeta,
      );
    return Response.json({
      fields: { cartodb_id: { type: "number" } },
      rows: u.searchParams.get("q")!.endsWith("LIMIT 0")
        ? []
        : [{ cartodb_id: 1, value: 1 }],
    });
  });
  const worker = createWorker(service);
  const client = new Client({ name: "contracts", version: "1" });
  await client.connect(
    new StreamableHTTPClientTransport(new URL(`${origin}/mcp`), {
      fetch: async (url, init) => worker.fetch(new Request(url, init), env),
    }),
  );
  try {
    for (const datasetId of ["Crime_Incidents", "PWD_Geotechnical_Tests"]) {
      for (const name of [
        "describe_dataset",
        "get_schema",
        "query_dataset",
        "preview_dataset",
        "aggregate_dataset",
      ] as const) {
        const result = await client.callTool({
          name,
          arguments: { datasetId },
        });
        if (
          name === "aggregate_dataset" &&
          datasetId === "PWD_Geotechnical_Tests"
        ) {
          assert.equal(result.isError, true);
          assert.equal(
            (result.structuredContent as { error: { code: string } }).error
              .code,
            "UNSUPPORTED_OPERATION",
          );
        } else {
          assert.ok(!result.isError, JSON.stringify(result));
          outputSchemas[
            name === "preview_dataset" ? "query_dataset" : name
          ].parse(result.structuredContent);
        }
        const content = result.content as { type: string; text: string }[];
        assert.deepEqual(JSON.parse(content[0].text), result.structuredContent);
      }
    }
  } finally {
    await client.close();
  }
});
