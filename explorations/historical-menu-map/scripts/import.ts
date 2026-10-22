import { readFile, writeFile } from "node:fs/promises";
import { parseArchive } from "../src/model.ts";
import type { MenuArchive } from "../src/model.ts";
import { importNypl } from "./nypl.ts";

// A reviewed export can be imported without network access. Its per-record source
// IDs, original catalog fields and snapshot hashes are validated, never invented.
const input = process.argv[2];
if (input === undefined)
  throw new Error("Usage: npm run import -- <reviewed-export.json> [four-CSV-directory]");
const archive = parseArchive(JSON.parse(await readFile(input, "utf8")));
const nyplDirectory = process.argv[3];
let result: MenuArchive = archive;
if (nyplDirectory !== undefined) {
  const snapshot = archive.snapshots.find((s) => s.id === "nypl-2021-08-01");
  if (snapshot === undefined)
    throw new Error("Export must preserve the NYPL archive snapshot SHA-256 and retrieval time");
  result = {
    ...archive,
    menus: [
      ...archive.menus.filter((m) => m.provider !== "NYPL"),
      ...(await importNypl(nyplDirectory, snapshot.id)),
    ],
  };
}
const validated = parseArchive(result);
await writeFile("public/menus.json", `${JSON.stringify(validated)}\n`);
process.stdout.write(`Imported ${String(validated.menus.length)} source records\n`);
