// Offline Python SDK/CLI -> real Worker MCP -> provider fixture contract.
// Run uv sync --locked --group test from the root before this check.
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
    assert.equal(new URL(url).origin, "https://api.tidesandcurrents.noaa.gov");
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
async function run(args) {
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
  assert.equal(result.sourceId, "noaa");
  assert.equal(result.rows[0].height_meters, "0.4");
  assert.equal(result.rows[0].datum, "MLLW");
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
  console.log(
    "Python SDK and CLI -> real Worker MCP -> provider fixture: passed (offline).",
  );
} finally {
  await new Promise((resolve) => server.close(resolve));
}
