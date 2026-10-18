import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { z } from "zod";
import { nyplArchiveUrl, ciaApiUrl, ciaQueryUrl } from "./selection.ts";

const querySchema = z.object({
  records: z.array(
    z.object({
      pointer: z.union([z.string().regex(/^\d+$/u), z.int().nonnegative()]).transform(String),
    }),
  ),
});

async function download(url: string, destination: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(
      `Source download denied or failed: HTTP ${String(response.status)} ${url}. No alternate host or credential workaround attempted.`,
    );
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(destination, bytes);
  const snapshot = {
    url,
    retrievedAt: new Date().toISOString(),
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
  await writeFile(`${destination}.provenance.json`, `${JSON.stringify(snapshot, null, 2)}\n`);
  return destination.endsWith(".json") ? JSON.parse(new TextDecoder().decode(bytes)) : null;
}

await mkdir("raw/cia", { recursive: true });
if (process.argv.includes("--nypl")) await download(nyplArchiveUrl, "raw/nypl.tgz");
if (process.argv.includes("--cia")) {
  const query = querySchema.parse(await download(ciaQueryUrl, "raw/cia-query.json"));
  if (query.records.length !== 30)
    throw new Error(`CIA parent count changed: ${String(query.records.length)}`);
  // Sequential requests respect the library service rather than bursting all 30.
  await query.records.reduce(async (previous, record) => {
    await previous;
    await download(ciaApiUrl(record.pointer), `raw/cia/${record.pointer}.json`);
  }, Promise.resolve());
}
if (!process.argv.includes("--nypl") && !process.argv.includes("--cia"))
  throw new Error("Choose --nypl and/or --cia");
