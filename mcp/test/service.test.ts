import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { catalog, describe, search } from "../src/catalog.ts";
import {
  DataService,
  querySchema,
  aggregateSchema,
  filterSchema,
  where,
} from "../src/data.ts";
import { LIMITS, readBounded } from "../src/bounds.ts";
const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/carto.json", import.meta.url), "utf8"),
);
const datasetId = "Crime_Incidents";
const input = querySchema.parse({ datasetId, limit: 2 });
function fake() {
  const calls: { url: URL; init?: RequestInit }[] = [];
  const service = new DataService(async (url, init) => {
    const u = new URL(url);
    calls.push({ url: u, init });
    return Response.json({
      ...fixture,
      rows: u.searchParams.get("q")?.endsWith("LIMIT 0") ? [] : fixture.rows,
    });
  });
  return { service, calls };
}
test("catalog IDs/count, backend capabilities, and pagination are coherent", () => {
  assert.equal(new Set(catalog.map((d) => d.id)).size, catalog.length);
  assert.equal(search("", 0, 20).total, catalog.length);
  assert.equal(
    search("crime", 0, 20).datasets.some((d) => d.id === datasetId),
    true,
  );
  const d = describe(datasetId, 0, 3);
  assert.equal(d.nextOffset, 3);
  assert.equal(d.resources[0].backend, "discovery-only");
  assert.equal(d.resources[2].backend, "carto");
  assert.ok(!("query" in d.resources[2]));
});
test("preview preserves year-specific source predicates, bounds and attribution", async () => {
  const { service, calls } = fake();
  const result = await service.query(input);
  assert.equal(calls.length, 2);
  for (const call of calls) {
    assert.equal(call.url.origin, "https://phl.carto.com");
    assert.equal(call.url.pathname, "/api/v2/sql");
    assert.equal(call.init?.redirect, "error");
  }
  const sql = calls[1].url.searchParams.get("q")!;
  assert.match(sql, /2026-01-01/);
  assert.match(sql, /2027-01-01/);
  assert.match(sql, /LIMIT 3 OFFSET 0$/);
  assert.match(sql, /ORDER BY "cartodb_id"/);
  assert.equal(result.rows.length, 2);
  assert.equal(result.nextOffset, 2);
  assert.equal(result.truncated, true);
  assert.equal(result.license, "City of Philadelphia License");
  assert.ok(result.sourceUrl);
  assert.ok(result.resourceId);
  assert.ok(result.retrievedAt);
  assert.equal(result.columns.includes("the_geom"), false);
});
test("schema uses metadata LIMIT 0, never sample inference", async () => {
  const { service, calls } = fake();
  const result = await service.schema({ datasetId });
  assert.deepEqual(result.fields, fixture.fields);
  assert.match(calls[0].url.searchParams.get("q")!, /LIMIT 0$/);
  assert.match(result.schemaSource, /not sample/);
});
test("structured filters quote values and reject SQL/URL/column injection", () => {
  assert.equal(
    where(
      [
        filterSchema.parse({
          column: "text_general_code",
          op: "eq",
          value: "x' OR 1=1--",
        }),
      ],
      fixture.fields,
    ),
    " WHERE \"text_general_code\" = 'x'' OR 1=1--'",
  );
  for (const bad of [
    { where: "1=1" },
    { url: "http://localhost" },
    { limit: 101 },
    { offset: 10001 },
    { columns: ["x;drop"] },
    { filters: [{ column: "hour", op: "eq", value: "\\" }] },
  ])
    assert.equal(querySchema.safeParse({ datasetId, ...bad }).success, false);
  assert.throws(
    () =>
      where([filterSchema.parse({ column: "hour", op: "eq" })], fixture.fields),
    /require a value/,
  );
});
test("unsupported resources and unknown columns never fetch data rows", async () => {
  const { service, calls } = fake();
  await assert.rejects(
    service.query({
      ...input,
      resourceId: catalog.find((d) => d.id === datasetId)!.resources[0].id,
    }),
    /Unsupported/,
  );
  assert.equal(calls.length, 0);
  await assert.rejects(
    service.query({ ...input, columns: ["nonexistent"] }),
    /Unknown column/,
  );
  assert.equal(calls.length, 1);
});
test("aggregate/count use full filtered source, with numeric validation and bounded groups", async () => {
  const { service, calls } = fake();
  const result = await service.aggregate(
    aggregateSchema.parse({
      datasetId,
      operation: "sum",
      column: "hour",
      groupBy: "text_general_code",
      filters: [{ column: "hour", op: "gte", value: 14 }],
      limit: 2,
    }),
  );
  assert.match(calls[1].url.searchParams.get("q")!, /SUM\("hour"\) AS value/);
  assert.match(
    calls[1].url.searchParams.get("q")!,
    /WHERE "hour" >= 14 GROUP BY/,
  );
  assert.equal(result.truncated, true);
  assert.match(result.scope, /not a preview/);
  await assert.rejects(
    service.aggregate(
      aggregateSchema.parse({
        datasetId,
        operation: "sum",
        column: "text_general_code",
      }),
    ),
    /numeric/,
  );
  await service.aggregate(aggregateSchema.parse({ datasetId }));
  assert.match(calls.at(-1)!.url.searchParams.get("q")!, /COUNT\(\*\)/);
});
test("decoded streaming bytes capped before parse even with missing or false Content-Length", async () => {
  let canceled = false;
  const body = new ReadableStream({
    pull(c) {
      c.enqueue(new Uint8Array(65536));
    },
    cancel() {
      canceled = true;
    },
  });
  await assert.rejects(readBounded(body, LIMITS.upstreamBytes), /Byte limit/);
  assert.equal(canceled, true);
  const service = new DataService(
    async () =>
      new Response("x".repeat(LIMITS.upstreamBytes + 1), {
        headers: { "content-type": "application/json", "content-length": "1" },
      }),
  );
  await assert.rejects(service.schema({ datasetId }), /Byte limit/);
});
test("rejects redirects, invalid JSON, upstream errors, bad schema and excessive rows", async () => {
  for (const response of [
    new Response(null, {
      status: 302,
      headers: { location: "http://localhost" },
    }),
    new Response("<html>"),
    Response.json({ error: ["bad"] }),
    Response.json({ rows: [], fields: { bad: null } }),
    Response.json({ ...fixture, rows: Array(102).fill({}) }),
  ]) {
    const service = new DataService(async () => response);
    await assert.rejects(service.schema({ datasetId }));
  }
});
test("read cancellation and service concurrency admission are bounded", async () => {
  const controller = new AbortController();
  const pending = readBounded(new ReadableStream({}), 100, controller.signal);
  controller.abort();
  await assert.rejects(pending);
  let release!: () => void;
  const held = new Promise<void>((r) => {
    release = r;
  });
  const service = new DataService(async () => {
    await held;
    return Response.json({ ...fixture, rows: [] });
  });
  const requests = Array.from({ length: LIMITS.concurrency }, () =>
    service.schema({ datasetId }),
  );
  await assert.rejects(service.schema({ datasetId }), /Busy/);
  release();
  await Promise.all(requests);
  await service.schema({ datasetId });
});
test("tool result bytes are capped and parse errors do not echo upstream bodies", async () => {
  const { boundedResult } = await import("../src/bounds.ts");
  assert.throws(
    () => boundedResult({ text: "x".repeat(LIMITS.resultBytes) }),
    /Result limit/,
  );
  const service = new DataService(
    async () =>
      new Response("private upstream snippet", {
        headers: { "content-type": "application/json" },
      }),
  );
  await assert.rejects(service.schema({ datasetId }), {
    message: "Upstream returned malformed JSON.",
  });
});
test("operation cancellation aborts the upstream request", async () => {
  const controller = new AbortController();
  let observed = false;
  const service = new DataService(
    async (_, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => {
            observed = true;
            reject(new Error("aborted"));
          },
          { once: true },
        );
      }),
  );
  const pending = service.schema({ datasetId }, controller.signal);
  controller.abort();
  await assert.rejects(pending, /aborted/);
  assert.equal(observed, true);
});
