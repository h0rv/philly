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
test("schema rejects malformed containers and servers that ignore LIMIT 0", async () => {
  for (const data of [
    { rows: [], fields: [] },
    { rows: [], fields: {} },
    { rows: [], fields: "" },
    { rows: [{}], fields: fixture.fields },
  ]) {
    await assert.rejects(
      new DataService(async () => Response.json(data)).schema({ datasetId }),
      /schema unavailable/,
    );
  }
});
test("aggregate group ordering cannot collide with the aggregate's value alias", async () => {
  const calls: string[] = [];
  const service = new DataService(async (url) => {
    const sql = new URL(url).searchParams.get("q")!;
    calls.push(sql);
    return Response.json({
      fields: { value: { type: "number" } },
      rows: sql.endsWith("LIMIT 0") ? [] : [{ group_value: 2, value: 1 }],
    });
  });
  await service.aggregate(
    aggregateSchema.parse({ datasetId, groupBy: "value" }),
  );
  assert.match(calls[1], /GROUP BY "value" ORDER BY group_value ASC LIMIT/);
});
test("JSON text fallback is identical and leaves room for the full response even with escaping", async () => {
  const { boundedResult } = await import("../src/bounds.ts");
  const value = { text: '"\\\n'.repeat(9000) };
  const result = boundedResult(value);
  assert.deepEqual(
    JSON.parse(result.content[0].text),
    result.structuredContent,
  );
  assert.ok(
    new TextEncoder().encode(JSON.stringify({ jsonrpc: "2.0", id: 1, result }))
      .byteLength < LIMITS.responseBytes,
  );
});
test("upstream JSON media type is exact and case-insensitive", async () => {
  const body = JSON.stringify({ ...fixture, rows: [] });
  await new DataService(
    async () =>
      new Response(body, {
        headers: { "content-type": "APPLICATION/JSON; charset=utf-8" },
      }),
  ).schema({ datasetId });
  await assert.rejects(
    new DataService(
      async () =>
        new Response(body, {
          headers: { "content-type": "text/html; fake=application/json" },
        }),
    ).schema({ datasetId }),
    /non-JSON/,
  );
});

test("year discovery, schema, latest preview and ranked counts preserve scope", async () => {
  assert.ok(
    search("crime 2025", 0, 20).datasets.some((d) => d.id === datasetId),
  );
  assert.ok(
    search("311 2025", 0, 20).datasets.some(
      (d) => d.id === "311_Service_and_Information_Requests",
    ),
  );
  const resource = describe(datasetId, 0, 20).resources.find(
    (r) => r.name.includes("2025") && r.backend === "carto",
  )!;
  assert.ok(resource);
  const calls: string[] = [];
  const service = new DataService(async (url) => {
    const sql = new URL(url).searchParams.get("q")!;
    calls.push(sql);
    if (sql.endsWith("LIMIT 0"))
      return Response.json({ fields: fixture.fields, rows: [] });
    // Fixed backend contract responses, not a SQL interpreter.
    if (sql.includes("COUNT(*)")) {
      assert.match(
        sql,
        /GROUP BY "text_general_code" ORDER BY value DESC, group_value ASC LIMIT 3$/,
      );
      return Response.json({
        rows: [
          { group_value: "Theft", value: 2 },
          { group_value: "Burglary", value: 1 },
        ],
      });
    }
    assert.match(
      sql,
      /WHERE "hour" >= 14 ORDER BY "cartodb_id" DESC LIMIT 3 OFFSET 0$/,
    );
    return Response.json({ rows: [...fixture.rows].reverse() });
  });
  const selection = { datasetId, resourceId: resource.id };
  const schema = await service.schema(selection);
  assert.equal(schema.fields.hour.type, "number");
  const filters = [{ column: "hour", op: "gte", value: 14 }];
  const page = await service.query(
    querySchema.parse({
      ...selection,
      filters,
      orderDirection: "desc",
      limit: 2,
    }),
  );
  assert.deepEqual(
    page.rows.map((r) => r.cartodb_id),
    [3, 2],
  );
  assert.equal(page.nextOffset, 2);
  assert.equal(page.orderDirection, "desc");
  const totals = await service.aggregate(
    aggregateSchema.parse({
      ...selection,
      filters,
      groupBy: "text_general_code",
      orderBy: "value",
      orderDirection: "desc",
      limit: 2,
    }),
  );
  assert.deepEqual(totals.rows, [
    { group_value: "Theft", value: 2 },
    { group_value: "Burglary", value: 1 },
  ]);
  assert.equal(totals.truncated, false);
  assert.equal(totals.orderBy, "value");
  for (const result of [schema, page, totals]) {
    assert.equal(result.resourceId, resource.id);
    assert.equal(result.resourceName, resource.name);
    assert.equal(result.sourceUrl, resource.url);
    assert.equal(result.license, "City of Philadelphia License");
    assert.ok(!Number.isNaN(Date.parse(result.retrievedAt)));
  }
  for (const sql of calls) {
    assert.match(sql, /2025-01-01/);
    assert.match(sql, /2026-01-01/);
  }
  assert.deepEqual(totals.appliedFilters, page.appliedFilters);
  assert.equal(
    querySchema.safeParse({
      ...selection,
      orderDirection: "desc; DROP TABLE x",
    }).success,
    false,
  );
  assert.equal(
    aggregateSchema.safeParse({ ...selection, orderBy: "COUNT(*)" }).success,
    false,
  );
});
