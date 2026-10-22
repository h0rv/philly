import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { z } from "zod";
import { parseArchive } from "../src/model.ts";
import type { Menu } from "../src/model.ts";

// A small truthful fallback while the approved source downloads are blocked.
// This preserves received metadata only; no dish, pin or scan-page ID is made up.
const raw = await readFile("snapshots/parent-audit.json", "utf8");
const audit = z
  .object({
    receivedAt: z.iso.datetime(),
    description: z.string(),
    records: z.array(
      z.object({
        provider: z.literal("CIA"),
        collection: z.string(),
        recordId: z.string(),
        name: z.string(),
        year: z.int().nullable(),
        address: z.string(),
        url: z.url(),
      }),
    ),
  })
  .parse(JSON.parse(raw));
const menus: Menu[] = audit.records.map((record) => ({
  id: `cia-${record.recordId}`,
  name: record.name,
  provider: record.provider,
  collection: "The Culinary Institute of America · Menu Collection (p16940coll1)",
  recordId: record.recordId,
  url: record.url,
  rights:
    "Item-specific rights have not been retrieved. Open the holding library record before reusing any scan.",
  snapshotId: "parent-audit-notes",
  date:
    record.year === null
      ? {
          label: "Undated",
          year: null,
          value: null,
          precision: "unknown",
          basis: "Undated in the parent source audit",
          catalogValue: "",
        }
      : {
          label: String(record.year),
          year: record.year,
          value: String(record.year),
          precision: "year",
          basis: "Year supplied in the parent source audit; original catalog payload is pending",
          catalogValue: "",
        },
  location: null,
  locationNote: `The source audit records ${record.address}. A verified coordinate has not been supplied; no pin assigned.`,
  cuisines: [],
  cuisineBasis: "Not supplied in the source audit",
  notes: [
    "Metadata-only starter record from parent research notes. Original JSON, page transcripts and scan-page IDs are pending the blocked source download.",
  ],
  currency: "",
  pages: [],
  dishes: [],
  catalog: { audit_address: record.address },
}));
const archive = parseArchive({
  schemaVersion: 1,
  title: "Philadelphia historical menus · metadata starter",
  snapshots: [
    {
      id: "parent-audit-notes",
      url: "https://cdm16940.contentdm.oclc.org/digital/collection/p16940coll1",
      retrievedAt: audit.receivedAt,
      sha256: createHash("sha256").update(raw).digest("hex"),
      description: audit.description,
    },
  ],
  menus,
});
await writeFile("public/menus.json", `${JSON.stringify(archive)}\n`);
