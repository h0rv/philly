# Philly, à la carte

An isolated, static Philadelphia historical-menu exploration. It does not change the landing page, register an exploration for publication, change the MCP Worker, or deploy anything. Branch starts from `b1f2d57` (main with PR #9); PRs #7/#8 are not included.

**Current preview is data-blocked:** the executor's approved NYPL S3 archive and CIA CONTENTdm requests returned HTTP 403, including permission-approved retries. The working preview contains **two metadata-only CIA records from supplied parent audit notes**, with original names and known/unknown years. It contains **zero guessed coordinates, zero item transcriptions, and zero prices**. The checked-in banner says so. This is a UI/import contribution awaiting the reviewed source export, not a completed 102-record atlas. Do not infer that this subset represents city coverage.

## Run locally

Node 24.19+ and npm 11.9 are used here. Dependencies come from the official npm registry and are pinned in `package.json` and `package-lock.json`. No credentials are needed.

```sh
cd explorations/historical-menu-map
npm ci --ignore-scripts
npm run check
npm run dev
# http://127.0.0.1:4173
```

`npm run build` emits portable HTML/CSS/JS/JSON and original SVGs in `dist/`. Any static HTTP server can serve it, including from a nested path. There is no backend, database, runtime source API or external font. No production source content is inserted as HTML.

```sh
CHROMIUM_PATH=/usr/bin/chromium npm run test:browser
# Omit CHROMIUM_PATH to use an already installed Playwright browser.
```

The default map is an **original, approximate locator drawing**, with modern streets/neighborhood labels. It is not a geographic dataset or a historical street reconstruction. Leaflet supplies zoom/pan/keyboard controls. Optional OpenStreetMap street tiles require network access, carry OSM/ODbL attribution, and fail back to the locator. No tile scraping, prefetching or image rehosting is performed. Source records remain fully accessible through the adjacent list, including when their location/date is unresolved.

## Data architecture

The user's requested [modern Python skill](https://github.com/h0rv/agent-skills/blob/abb546e49c2068b55d4b0b925066dd73a0ab88ac/modern-python/SKILL.md) was read with its data-modeling, parse-don't-validate, type-narrowing, immutability, exhaustiveness and testing references. Its principles are applied in TypeScript:

- `src/model.ts` defines Zod-first runtime schemas and infers domain types with `z.infer`. Raw JSON is `unknown` until parsed once at the source boundary.
- Readonly values and a discriminated date union preserve the distinction between unknown, year, month and day precision. Original catalog dates and corrections coexist.
- Source boundaries reject invalid date/year relationships, duplicate IDs, orphan dish/page links, missing snapshots, negative prices and out-of-bounds pins. Domain code consumes typed values without casts or lint suppressions.
- Tests target filter behavior, price semantics, source-boundary relationships, keyboard focus and load/empty states.

Every menu preserves provider, collection, record ID/URL, rights, snapshot identity, dating basis/precision, original catalog fields and page-level URLs/IDs when available. Item rows remain **occurrences**; dish identity is separate. Source records are not claimed to be unique restaurants. Cuisine is only assigned when a source or explicit reviewed annotation supports it. Missing cuisine stays unspecified.

`public/menus.json` is the normalized browser payload. `snapshots/parent-audit.json` is a truthful snapshot of **received research notes**, not a downloaded library API response. Its SHA-256 hashes that exact local file; the UI and snapshot description explain that boundary. Neither original CIA catalog date nor original page IDs are invented for these notes.

## Finish the real-data import

The parent has audited 72 dated NYPL menus and 30 CIA parent records (16 dated, 14 undated). The selected NYPL IDs are pinned in `scripts/selection.ts`. The expected join cardinalities are 72 menus, 432 source pages, 3,469 item occurrences, 2,534 distinct dishes and 1,488 priced rows. These are **expected source counts**, not the current preview's contents.

1. Supply the reviewed normalized export following `archiveSchema` in `src/model.ts`, preserving snapshot hashes, retrieval timestamps, source records and verified location evidence.
2. Run `npm run import -- /path/to/reviewed-export.json`.
3. Alternatively, add the four exact NYPL CSVs with `npm run import -- /path/to/reviewed-export.json /path/to/csv-directory`. The export must include the `nypl-2021-08-01` snapshot and its real archive checksum/retrieval time. The streaming importer joins `Menu → MenuPage → MenuItem → Dish`, validates expected counts and fails if they change. It never geocodes catalog locations.
4. Re-run checks/browser QA, inspect pin placement against evidence, then request a new data review. A current preview cannot substantiate the full import until this happens.

`node scripts/fetch.ts --nypl --cia` records exact downloaded bytes and SHA-256 sidecars under ignored `raw/` when the original providers are accessible. It stops on HTTP failures. CIA requests are sequential. Raw four-CSV archive extraction should be reviewed locally; this script does not automatically extract arbitrary archive paths. CIA normalization awaits an actual reviewed raw response/export; the fetcher does not pretend its raw payloads are normalized menus.

The NYPL adapter conservatively exposes catalog **year** precision until a printed date is reviewed. Specific audit corrections are explicit: 33270 uses printed `1915-01-02` while retaining catalog `1915-01-01`; 33662 remains catalog-only. The known 29916 page-4 Rosoff advertisement is excluded from display. The source-page count remains 432 before display exclusion. **The exact NYPL page_number convention for that exclusion remains to be verified against the blocked snapshot.** Horn & Hardart 30411 remains unlocated rather than being pinned to an arbitrary outlet. All other NYPL coordinates require review; the source's x/y item positions are points, not image bounding boxes.

## Tooling verification (2026-10-18)

The official [TypeScript 7 announcement](https://devblogs.microsoft.com/typescript/announcing-typescript-7-0/) says the Go-based compiler is released for production under `typescript`; [npm's stable package](https://registry.npmjs.org/typescript/latest) resolves to **7.0.2** here. This project uses its native `tsc`, not `@typescript/native-preview`. The installed executable reports `Version 7.0.2`.

Oxfmt **0.73.0**, Oxlint **1.88.0** and oxlint-tsgolint **7.0.2004** were resolved from npm's stable tags and pinned. [Oxlint's official docs](https://oxc.rs/docs/guide/usage/linter/type-aware.html) document Go-backed type-aware rules plus type-check diagnostics. The configuration enables correctness/suspicious/performance rules and explicit strict type-aware rules (unsafe operations/assertions, nullable conditions, promises, exhaustiveness and unnecessary conditions). All warnings fail. Independent `tsc --noEmit` also runs, with `strict`, unchecked index access, exact optional properties, no unused code, no implicit returns, no fallthrough and **no skipped library checks**. No blanket disables, `any`, unsafe casts or suppression comments are used.

esbuild **0.28.2** is only a build tool; no application framework is included. Leaflet **1.9.4** is the [official stable release](https://leafletjs.com/download.html), rather than its 2.0 alpha. [Simple.css](https://simplecss.org/) **2.3.7** is the lightweight classless base, themed with original typography, cream paper, mustard/ink and rowhouse/storefront linework. Zod **4.6.7** defines the data boundary, using its documented [Mini entry point](https://zod.dev/packages/mini) in the browser for tree shaking. The current browser bundle is about 196 KB (59 KB gzip), including Leaflet. Playwright is a dev-only dependency.

## Sources and image rights

- [NYPL bulk snapshot](https://s3.amazonaws.com/menusdata.nypl.org/gzips/2021_08_01_07_01_17_data.tgz) and [four-CSV dataset documentation](https://clickhouse.com/docs/get-started/sample-datasets/menus).
- CIA [Philadelphia parent query](https://cdm16940.contentdm.oclc.org/digital/bl/dmwebservices/index.php?q=dmQuery/p16940coll1/locati%5EPhiladelphia%5Eexact%5Eand/title!date!descri!decade!locati/title/1024/1/0/0/json), [8063](https://cdm16940.contentdm.oclc.org/digital/api/singleitem/collection/p16940coll1/id/8063) and [17865](https://cdm16940.contentdm.oclc.org/digital/collection/p16940coll1/id/17865). CIA item rights are not presumed open.
- The two Bookbinder venues are distinct: Old Original Bookbinder's at 125 Walnut is not Bookbinders at 215 South 15th. Middle East Restaurant retains its original name.
- [OldPhilly](https://github.com/h0rv/oldphilly/tree/fe86106dd14f6cff153c948ffdca43ee92cea354) was inspected, including its [metadata model](https://github.com/h0rv/oldphilly/blob/fe86106dd14f6cff153c948ffdca43ee92cea354/oldphilly/models.py), export and site display code. The checkout's data/release directories are empty placeholders. Its [Hugging Face metadata dataset](https://huggingface.co/datasets/h0rv/phillyhistory-metadata) may support later small build-time photo links. Photos need rights, attribution, photo-date precision and a verified venue match; proximity alone is only neighborhood context. No photos were downloaded, republished, or claimed as matches in this contribution.
- Sandwich context links to the [Inquirer](https://www.inquirer.com/philly-tips/philly-cheesesteak-guide-20180705.html) and [Visit Philadelphia](https://www.visitphilly.com/media-center/press-releases/phillys-signature-sandwiches-cheesesteaks-hoagies-roast-pork/). No vintage sandwich menu or price was fabricated. No Michelin awards or current restaurant claims are included.

The code's MIT license does not relicense source material. Scans are linked at their library and not rehosted. `storefront.svg` and `locator.svg` are original drawings for this contribution. Dependencies retain their package licenses; bundled Leaflet legal comments are preserved.

## Review evidence

Independent [Luna review](review/LUNA.md), [desktop](review/desktop.png), [mobile](review/mobile.png), [desktop detail](review/desktop-detail.png) and [mobile detail](review/mobile-detail.png) are checked in. Strict checks/build and 7 boundary/behavior tests pass; browser QA covers desktop/mobile filters, unknown states, keyboard focus, source deep links, load failures, tile failures, overflow, plus a clearly synthetic intercepted record for pin/transcript/price/markup behavior. These UI checks do not verify the blocked full source import.
