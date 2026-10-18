// Offline Python SDK/CLI -> real Worker MCP -> provider fixture contract.
// Run uv sync --locked --group test from the root before this check.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createWorker } from "../src/worker.ts";
import { DataService } from "../src/data.ts";
const root = fileURLToPath(new URL("../../", import.meta.url));
const python = process.env.PHILLY_PYTHON ?? root + ".venv/bin/python";
const worker = createWorker(
  new DataService(async (url) => {
    const upstream = new URL(url);
    if (upstream.origin === "https://musicbrainz.org") {
      assert.equal(upstream.searchParams.get("query"), 'area:"Philadelphia"');
      return Response.json(
        JSON.parse(
          readFileSync(
            new URL(
              "../test/fixtures/musicbrainz-place-search.json",
              import.meta.url,
            ),
            "utf8",
          ),
        ),
      );
    }
    assert.equal(upstream.origin, "https://api.tidesandcurrents.noaa.gov");
    return Response.json({
      predictions: [{ t: "2026-10-17 00:00", v: "0.4" }],
    });
  }),
);
const server = createServer(async (req, res) => {
  try {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const origin = `http://127.0.0.1:${server.address().port}`;
    const response = await worker.fetch(
      new Request(origin + req.url, {
        method: req.method,
        headers: req.headers,
        body: Buffer.concat(chunks),
      }),
      { PUBLIC_ORIGIN: origin },
    );
    res.writeHead(response.status, Object.fromEntries(response.headers));
    res.end(Buffer.from(await response.arrayBuffer()));
  } catch {
    res.statusCode = 500;
    res.end("Fixture transport failed.");
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const endpoint = `http://127.0.0.1:${server.address().port}/mcp`;
async function run(args, source = "noaa") {
  const child = spawn(python, args, {
    cwd: root,
    stdio: ["ignore", "pipe", "inherit"],
  });
  let output = "";
  child.stdout.on("data", (chunk) => {
    output += chunk;
  });
  const code = await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", resolve);
  });
  assert.equal(code, 0);
  const result = JSON.parse(output);
  assert.equal(result.sourceId, source);
  if (source === "musicbrainz") {
    assert.equal(result.rows[0].latitude, 51.9414);
    assert.equal(result.rows[0].longitude, -1.548);
    assert.equal(result.rows[0].begin_date, "1971");
    assert.equal(result.rows[0].end_date, "1999-10");
    assert.equal(result.rows[0].ended, true);
    assert.equal(
      result.rows[0].place_url,
      "https://musicbrainz.org/place/d1ab65f8-d082-492a-bd70-ce375548dabf",
    );
  } else {
    assert.equal(result.rows[0].height_meters, "0.4");
    assert.equal(result.rows[0].datum, "MLLW");
  }
  assert.equal(result.completeness, "complete");
}
try {
  await run([
    "-c",
    `import asyncio,json; from philly import RemotePhilly; print(json.dumps(asyncio.run(RemotePhilly(${JSON.stringify(endpoint)}).query('noaa:philadelphia_tides_today'))))`,
  ]);
  await run([
    "-m",
    "philly",
    "--cache=False",
    "remote",
    "query",
    "noaa:philadelphia_tides_today",
    "--endpoint",
    endpoint,
    "--limit",
    "1",
  ]);
  await run(
    [
      "-c",
      `import asyncio,json; from philly import RemotePhilly; print(json.dumps(asyncio.run(RemotePhilly(${JSON.stringify(endpoint)}).query('musicbrainz:philadelphia_places'))))`,
    ],
    "musicbrainz",
  );
  // Respect the shared service's existing MusicBrainz admission cooldown.
  await new Promise((resolve) => setTimeout(resolve, 1100));
  await run(
    [
      "-m",
      "philly",
      "--cache=False",
      "remote",
      "query",
      "musicbrainz:philadelphia_places",
      "--endpoint",
      endpoint,
      "--limit",
      "1",
    ],
    "musicbrainz",
  );
  console.log(
    "Python SDK and CLI -> real Worker MCP -> provider fixture: passed (offline).",
  );
} finally {
  await new Promise((resolve) => server.close(resolve));
}
