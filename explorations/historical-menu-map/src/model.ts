import * as z from "zod/mini";

const sourceUrl = z.url({ protocol: /^https$/u });
const nonempty = z.string().check(z.minLength(1));
const year = z.int().check(z.minimum(1600), z.maximum(2100));
const dateCommon = { label: nonempty, basis: nonempty, catalogValue: z.string() };
export const menuDateSchema = z.readonly(
  z
    .discriminatedUnion("precision", [
      z.strictObject({
        ...dateCommon,
        precision: z.literal("unknown"),
        year: z.null(),
        value: z.null(),
      }),
      z.strictObject({
        ...dateCommon,
        precision: z.literal("year"),
        year,
        value: z.string().check(z.regex(/^\d{4}$/u)),
      }),
      z.strictObject({
        ...dateCommon,
        precision: z.literal("month"),
        year,
        value: z.string().check(z.regex(/^\d{4}-(0[1-9]|1[0-2])$/u)),
      }),
      z.strictObject({ ...dateCommon, precision: z.literal("day"), year, value: z.iso.date() }),
    ])
    .check(
      z.refine(
        (d) => d.value === null || Number(d.value.slice(0, 4)) === d.year,
        "Year and date disagree",
      ),
    ),
);

export const locationSchema = z.readonly(
  z.strictObject({
    lat: z.number().check(z.minimum(39.86), z.maximum(40.14)),
    lng: z.number().check(z.minimum(-75.3), z.maximum(-74.95)),
    area: nonempty,
    address: nonempty,
    basis: nonempty,
    evidence: sourceUrl,
  }),
);
export const menuPageSchema = z.readonly(
  z.strictObject({ id: nonempty, label: nonempty, url: sourceUrl, transcript: z.string() }),
);
export const dishSchema = z.readonly(
  z.strictObject({
    id: nonempty,
    dishId: nonempty,
    pageId: nonempty,
    name: nonempty,
    price: z.nullable(z.number().check(z.nonnegative())),
    priceHigh: z.nullable(z.number().check(z.nonnegative())),
  }),
);

export const menuSchema = z.readonly(
  z
    .strictObject({
      id: nonempty,
      name: nonempty,
      provider: z.enum(["NYPL", "CIA"]),
      collection: nonempty,
      recordId: nonempty,
      url: sourceUrl,
      rights: nonempty,
      snapshotId: nonempty,
      date: menuDateSchema,
      location: z.nullable(locationSchema),
      locationNote: nonempty,
      cuisines: z.readonly(z.array(nonempty)),
      cuisineBasis: nonempty,
      notes: z.readonly(z.array(z.string())),
      currency: z.string(),
      pages: z.readonly(z.array(menuPageSchema)),
      dishes: z.readonly(z.array(dishSchema)),
      catalog: z.readonly(z.record(z.string(), z.string())),
    })
    .check(
      z.refine(
        (menu) => new Set(menu.pages.map((p) => p.id)).size === menu.pages.length,
        "Duplicate page ID",
      ),
      z.refine(
        (menu) => new Set(menu.dishes.map((d) => d.id)).size === menu.dishes.length,
        "Duplicate item ID",
      ),
      z.refine((menu) => {
        const pages = new Set(menu.pages.map((p) => p.id));
        return menu.dishes.every((d) => pages.has(d.pageId));
      }, "Dish references a missing page"),
    ),
);

export const snapshotSchema = z.readonly(
  z.strictObject({
    id: nonempty,
    url: sourceUrl,
    retrievedAt: z.iso.datetime({ offset: true }),
    sha256: z.string().check(z.regex(/^[a-f0-9]{64}$/u)),
    description: nonempty,
  }),
);
export const archiveSchema = z.readonly(
  z
    .strictObject({
      schemaVersion: z.literal(1),
      title: nonempty,
      snapshots: z.readonly(z.array(snapshotSchema)),
      menus: z.readonly(z.array(menuSchema)),
    })
    .check(
      z.refine(
        (archive) => new Set(archive.snapshots.map((s) => s.id)).size === archive.snapshots.length,
        "Duplicate snapshot ID",
      ),
      z.refine(
        (archive) => new Set(archive.menus.map((m) => m.id)).size === archive.menus.length,
        "Duplicate menu ID",
      ),
      z.refine((archive) => {
        const ids = new Set(archive.snapshots.map((s) => s.id));
        return archive.menus.every((m) => ids.has(m.snapshotId));
      }, "Menu references a missing snapshot"),
    ),
);

export type MenuDate = z.infer<typeof menuDateSchema>;
export type MenuLocation = z.infer<typeof locationSchema>;
export type MenuPage = z.infer<typeof menuPageSchema>;
export type Dish = z.infer<typeof dishSchema>;
export type Menu = z.infer<typeof menuSchema>;
export type Snapshot = z.infer<typeof snapshotSchema>;
export type MenuArchive = z.infer<typeof archiveSchema>;

export function parseArchive(value: unknown): MenuArchive {
  return archiveSchema.parse(value);
}
