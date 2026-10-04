import { search } from "../src/catalog.ts";
import { readBounded, LIMITS } from "../src/bounds.ts";
import { createWorker } from "../src/worker.ts";
const samples: Record<string, unknown> = {};
async function bench(
  name: string,
  operation: () => unknown | Promise<unknown>,
  n = 500,
) {
  for (let i = 0; i < 20; i++) await operation();
  const times: number[] = [];
  const cpu = process.cpuUsage();
  for (let i = 0; i < n; i++) {
    const start = performance.now();
    await operation();
    times.push(performance.now() - start);
  }
  const used = process.cpuUsage(cpu);
  times.sort((a, b) => a - b);
  samples[name] = {
    iterations: n,
    p50Ms: times[Math.floor(n * 0.5)],
    p95Ms: times[Math.floor(n * 0.95)],
    meanCpuMs: (used.user + used.system) / 1000 / n,
  };
}
await bench("catalog search", () => search("crime", 0, 20));
const payload = JSON.stringify({
  rows: Array.from({ length: 100 }, (_, i) => ({
    id: i,
    text: "x".repeat(1200),
  })),
});
await bench("bounded decode + JSON parse (120 KiB)", async () =>
  JSON.parse(
    await readBounded(new Response(payload).body, LIMITS.upstreamBytes),
  ),
);
const worker = createWorker();
await bench(
  "legacy HTTP search including SDK + result encoding",
  () =>
    worker.fetch(
      new Request("http://127.0.0.1:8787/mcp", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json, text/event-stream",
        },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "tools/call",
          params: { name: "search_datasets", arguments: { query: "crime" } },
        }),
      }),
      { PUBLIC_ORIGIN: "http://127.0.0.1:8787" },
    ),
  200,
);
console.log(
  JSON.stringify(
    {
      runtime: process.version,
      notes:
        "Local Node CPU microbenchmark, not Cloudflare billing/production evidence.",
      samples,
    },
    null,
    2,
  ),
);
