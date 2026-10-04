import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { catalog } from "../src/catalog.ts";

test("generated catalog matches every packaged YAML record", () => {
  execFileSync(process.execPath, [
    new URL("../scripts/catalog.mjs", import.meta.url).pathname,
    "--check",
  ]);
  const names = readdirSync(
    new URL("../../src/philly/datasets/", import.meta.url),
  )
    .filter((name) => name.endsWith(".yaml"))
    .map((name) => name.slice(0, -5))
    .sort();
  assert.deepEqual(catalog.map((dataset) => dataset.id).sort(), names);
});
