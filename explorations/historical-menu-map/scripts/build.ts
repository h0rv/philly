import { context, build } from "esbuild";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { parseArchive } from "../src/model.ts";

// Validate checked-in data before publishing any output. No network or runtime API.
parseArchive(JSON.parse(await readFile(new URL("../public/menus.json", import.meta.url), "utf8")));
await mkdir("dist", { recursive: true });
await cp("public", "dist", { recursive: true });
await cp("index.html", "dist/index.html");
await cp("node_modules/leaflet/dist/images", "dist/images", { recursive: true });
const options = {
  entryPoints: ["src/main.ts", "src/styles.css"],
  bundle: true,
  minify: true,
  outdir: "dist",
  format: "esm",
  target: "es2022",
  legalComments: "eof",
  loader: { ".png": "file" },
  assetNames: "images/[name]-[hash]",
} satisfies Parameters<typeof build>[0];
if (process.argv.includes("--serve")) {
  const dev = await context(options);
  await dev.watch();
  const server = await dev.serve({ servedir: "dist", host: "127.0.0.1", port: 4173 });
  process.stdout.write(
    `Local preview: http://${server.hosts[0] ?? "127.0.0.1"}:${String(server.port)}\n`,
  );
} else {
  const result = await build({ ...options, metafile: true });
  await writeFile("dist/build-meta.json", `${JSON.stringify(result.metafile, null, 2)}\n`);
}
