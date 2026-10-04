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
