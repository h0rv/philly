import { readdir, readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { parse } from "yaml";
const root = new URL("../../", import.meta.url);
const hash = (s) => createHash("sha256").update(s).digest("hex").slice(0, 16);
const catalog = [];
// This intentionally recognizes a small grammar, not arbitrary catalog SQL.
const sourcePattern =
  /^select\s+\*(?:\s*,\s*ST_Y\(the_geom\)\s+AS\s+lat\s*,\s*ST_X\(the_geom\)\s+AS\s+lng)?\s+from\s+([a-z_][a-z0-9_]*)(?:\s+where\s+([a-z_][a-z0-9_]*\s*(?:>=|<=|=|<|>)\s*'\d{4}-\d{2}-\d{2}'(?:\s+and\s+[a-z_][a-z0-9_]*\s*(?:>=|<=|=|<|>)\s*'\d{4}-\d{2}-\d{2}')*))?\s*;?\s*$/i;
for (const file of (await readdir(new URL("src/philly/datasets/", root)))
  .filter((f) => f.endsWith(".yaml"))
  .sort()) {
  const d = parse(
    await readFile(new URL(`src/philly/datasets/${file}`, root), "utf8"),
  );
  const resources = (d.resources ?? []).map((r) => {
    let query = null;
    try {
      const u = new URL(r.url);
      const match = (u.searchParams.get("q") ?? "").match(sourcePattern);
      if (
        u.origin === "https://phl.carto.com" &&
        u.pathname === "/api/v2/sql" &&
        !u.username &&
        !u.password &&
        match
      ) {
        query = (u.searchParams.get("q") ?? "").trim().replace(/;$/, "");
      }
    } catch {
      /* Non-web metadata links are discovery-only. */
    }
    if (r.id !== undefined && !/^[a-f0-9]{16}$/.test(r.id)) {
      throw new Error(`Invalid resource ID in ${file}`);
    }
    return {
      id:
        r.id ??
        hash(
          String(r.url ?? "") +
            "\n" +
            String(r.name ?? "") +
            "\n" +
            String(r.format ?? "unknown"),
        ),
      name: String(r.name ?? "").slice(0, 300),
      format: r.format ?? "unknown",
      url: r.url ?? "",
      query,
    };
  });
  catalog.push({
    id: file.slice(0, -5),
    title: d.title,
    description: String(d.notes ?? "").slice(0, 1500),
    organization: d.organization ?? null,
    license: d.license ?? null,
    categories: d.category ?? [],
    resources,
  });
}
const content = JSON.stringify(catalog) + "\n";
const output = new URL("mcp/src/generated/catalog.json", root);
if (process.argv.includes("--check")) {
  if ((await readFile(output, "utf8")) !== content) {
    throw new Error(
      "Catalog is stale. Run npm run catalog and commit the result.",
    );
  }
} else {
  await mkdir(new URL("mcp/src/generated/", root), { recursive: true });
  await writeFile(output, content);
}
console.log({ datasets: catalog.length, revision: hash(content) });
