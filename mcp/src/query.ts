import { z } from "zod";
import { LIMITS } from "./bounds.ts";
export const identifier = z
  .string()
  .max(63)
  .regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/);
export const selection = {
  datasetId: z.string().min(1).max(200),
  resourceId: z
    .string()
    .regex(/^[a-f0-9]{16}$/)
    .optional(),
};
export const filterSchema = z.strictObject({
  column: identifier,
  op: z.enum(["eq", "ne", "lt", "lte", "gt", "gte", "is_null", "not_null"]),
  value: z
    .union([
      z
        .string()
        .max(200)
        .regex(/^[^\x00-\x1f\\]*$/),
      z.number().finite(),
      z.boolean(),
      z.null(),
    ])
    .optional(),
});
export const filtersSchema = z
  .array(filterSchema)
  .max(LIMITS.filters)
  .default([]);
export const querySchema = z.strictObject({
  ...selection,
  filters: filtersSchema,
  columns: z.array(identifier).min(1).max(LIMITS.columns).optional(),
  limit: z.number().int().min(1).max(LIMITS.rows).default(20),
  offset: z.number().int().min(0).max(LIMITS.offset).default(0),
  orderBy: identifier.optional(),
  orderDirection: z.enum(["asc", "desc"]).default("asc"),
});
export const aggregateSchema = z.strictObject({
  ...selection,
  filters: filtersSchema,
  operation: z.enum(["count", "sum", "avg", "min", "max"]).default("count"),
  column: identifier.optional(),
  groupBy: identifier.optional(),
  orderBy: z.enum(["group", "value"]).default("group"),
  orderDirection: z.enum(["asc", "desc"]).default("asc"),
  limit: z.number().int().min(1).max(50).default(20),
});
export type Fields = Record<string, { type: string }>;
const quote = (s: string) => `"${identifier.parse(s)}"`;
export function where(
  filters: z.infer<typeof filtersSchema>,
  fields: Fields,
  quoteField = quote,
) {
  return filters.length
    ? " WHERE " +
        filters
          .map((f) => {
            checkColumn(f.column, fields);
            if (f.op === "is_null" || f.op === "not_null")
              return `${quoteField(f.column)} IS ${f.op === "not_null" ? "NOT " : ""}NULL`;
            if (f.value === undefined || f.value === null)
              throw new Error(
                "Comparisons require a value; use is_null / not_null for null.",
              );
            const literal =
              typeof f.value === "string"
                ? `'${f.value.replaceAll("'", "''")}'`
                : String(f.value);
            const op = {
              eq: "=",
              ne: "<>",
              lt: "<",
              lte: "<=",
              gt: ">",
              gte: ">=",
            }[f.op];
            return `${quoteField(f.column)} ${op} ${literal}`;
          })
          .join(" AND ")
    : "";
}
export function checkColumn(name: string, fields: Fields) {
  identifier.parse(name);
  if (!Object.hasOwn(fields, name))
    throw new Error(`Unknown column: ${name}. Use get_schema.`);
}
