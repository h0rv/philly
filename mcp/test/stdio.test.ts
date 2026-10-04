import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
for (const mode of ["legacy", { pin: "2026-07-28" }] as const) {
  test(`stdio discovers tools (${JSON.stringify(mode)})`, async () => {
    const client = new Client(
      { name: "stdio-fixture", version: "1" },
      { versionNegotiation: { mode } },
    );
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [
        "--experimental-strip-types",
        fileURLToPath(new URL("../src/stdio.ts", import.meta.url)),
      ],
      stderr: "pipe",
    });
    try {
      await client.connect(transport);
      assert.equal((await client.listTools()).tools.length, 6);
      assert.ok(
        JSON.stringify(
          await client.callTool({
            name: "search_datasets",
            arguments: { query: "crime" },
          }),
        ).includes("Crime_Incidents"),
      );
    } finally {
      await client.close();
    }
  });
}

test("agent example configuration starts Philly and discovers year resources", async () => {
  const { readFileSync } = await import("node:fs");
  const config = JSON.parse(
    readFileSync(new URL("../claude.json", import.meta.url), "utf8"),
  );
  const client = new Client({ name: "agent-example", version: "1" });
  const transport = new StdioClientTransport({
    ...config.mcpServers.philly,
    cwd: fileURLToPath(new URL("../../", import.meta.url)),
    stderr: "pipe",
  });
  try {
    await client.connect(transport);
    const found = await client.callTool({
      name: "search_datasets",
      arguments: { query: "crime 2025" },
    });
    assert.ok(JSON.stringify(found).includes("Crime_Incidents"));
    const described = await client.callTool({
      name: "describe_dataset",
      arguments: { datasetId: "Crime_Incidents", limit: 20 },
    });
    assert.ok(JSON.stringify(described).includes("2025"));
    assert.ok(JSON.stringify(described).includes("source"));
  } finally {
    await client.close();
  }
});
