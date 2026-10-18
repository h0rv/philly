import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { parseArchive } from "../src/model.ts";

const archive = parseArchive(
  JSON.parse(await readFile(new URL("../public/menus.json", import.meta.url), "utf8")),
);

test("real archive loads, map controls work, and the viewport has no horizontal overflow", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => {
    errors.push(error.message);
  });
  await page.goto("/");
  await expect(page.locator(".menu-row")).toHaveCount(archive.menus.length);
  await expect(page.getByRole("heading", { name: "The city has good taste." })).toBeVisible();
  await page.getByRole("button", { name: "Zoom in" }).click();
  await page.getByRole("button", { name: "Recenter" }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(errors).toEqual([]);
  await page.screenshot({ path: `review/${info.project.name}.png`, fullPage: true });
});

test("keyboard opens a menu and Back returns focus to its list row", async ({ page }, info) => {
  await page.goto("/");
  const first = page.locator(".menu-row").first();
  const id = await first.getAttribute("data-menu-id");
  await first.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#detail-view h2")).toBeFocused();
  await expect(page.locator("#detail-view")).toBeVisible();
  await expect(
    page.locator("#detail-view").getByText("Rights / reuse", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: `review/${info.project.name}-detail.png`, fullPage: true });
  await page.getByRole("button", { name: "Back to the menu file" }).click();
  await expect(page.locator(`button[data-menu-id="${id ?? ""}"]`)).toBeFocused();
});

test("filters compose, include unknowns, reset and show useful empty states", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("combobox", { name: "Decade", exact: true }).selectOption("undated");
  await expect(page.locator(".menu-row")).toHaveCount(
    archive.menus.filter((m) => m.date.year === null).length,
  );
  await page
    .getByRole("combobox", { name: "Neighborhood / area", exact: true })
    .selectOption("unlocated");
  await expect(page.locator(".menu-row")).toHaveCount(
    archive.menus.filter((m) => m.date.year === null && m.location === null).length,
  );
  await page.getByLabel("On the menu", { exact: true }).fill("__no_such_item__");
  await expect(page.getByRole("heading", { name: "Nothing on this page." })).toBeVisible();
  await page.locator("#menu-list").getByRole("button", { name: "Reset all filters" }).click();
  await expect(page.locator(".menu-row")).toHaveCount(archive.menus.length);
  await page.getByText("Dates & unknowns", { exact: true }).click();
  await page.getByLabel("From year", { exact: true }).fill("2000");
  await page.getByLabel("To year", { exact: true }).fill("1800");
  await expect(page.getByText(/start year comes after/u)).toBeVisible();
});

test("undated Middle East retains original name and deep link", async ({ page }) => {
  await page.goto("/#menu=cia-17865");
  await expect(page.locator("#detail-view h2")).toHaveText("Middle East Restaurant");
  await expect(page.locator(".detail-cover").getByText("Undated", { exact: true })).toBeVisible();
  await expect(page.locator("#detail-view a").first()).toHaveAttribute(
    "href",
    /p16940coll1\/id\/17865/u,
  );
});

test("failed data and rejected data render accessible load states", async ({ page }) => {
  await page.route("**/menus.json", (route) => route.fulfill({ status: 503, body: "unavailable" }));
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("archive couldn't open");
  await page.unroute("**/menus.json");
  await page.route("**/menus.json", (route) => route.fulfill({ json: { schemaVersion: 999 } }));
  await page.reload();
  await expect(page.getByRole("alert")).toContainText("archive couldn't open");
});

test("external tiles can fail without losing the locator or list", async ({ page }) => {
  await page.route("https://tile.openstreetmap.org/**", (route) => route.abort());
  await page.goto("/");
  await expect(page.locator(".menu-row")).toHaveCount(archive.menus.length);
  await page.getByRole("button", { name: "Load street map" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Street tiles are unavailable" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Use original locator" }).click();
  await expect(page.locator(".leaflet-image-layer")).toBeVisible();
  await expect(page.locator(".menu-row")).toHaveCount(archive.menus.length);
});

// Synthetic records exist only in this intercepted test response. They do not
// ship in public/menus.json or the recorded preview screenshots.
test("pin selection and transcription rendering preserve zero/missing prices and escape markup", async ({
  page,
}) => {
  const base = archive.menus[0];
  if (base === undefined) throw new Error("The reviewed archive needs a source record");
  const fixture = {
    ...base,
    id: "fixture-only",
    name: "Synthetic browser fixture",
    currency: "USD",
    location: {
      lat: 39.95,
      lng: -75.16,
      area: "Test-only area",
      address: "Synthetic address",
      basis: "Browser test fixture, not historical data",
      evidence: base.url,
    },
    pages: [
      {
        id: "fixture-page",
        label: "Fixture page",
        url: base.url,
        transcript: "Existing transcript test <script>not markup</script>",
      },
    ],
    dishes: [
      {
        id: "fixture-zero",
        dishId: "fixture-dish",
        pageId: "fixture-page",
        name: "Fixture item <img src=x>",
        price: 0,
        priceHigh: null,
      },
      {
        id: "fixture-missing",
        dishId: "fixture-dish-2",
        pageId: "fixture-page",
        name: "Missing-price fixture",
        price: null,
        priceHigh: null,
      },
    ],
  };
  await page.route("**/menus.json", (route) =>
    route.fulfill({ json: { ...archive, menus: [fixture] } }),
  );
  await page.goto("/");
  await expect(page.locator("#pin-count")).toHaveText("1 point · 1 record");
  await page
    .getByRole("button", { name: "Synthetic browser fixture · 1 menu record", exact: true })
    .click();
  await expect(page.locator("#detail-view h2")).toHaveText("Synthetic browser fixture");
  await expect(page.getByRole("cell", { name: "$0.00", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "Price unrecorded", exact: true })).toBeVisible();
  await expect(page.locator("#detail-view img")).toHaveCount(0);
  await page.getByText("Fixture page · existing transcript", { exact: true }).click();
  await expect(page.locator(".transcript")).toContainText("<script>not markup</script>");
  await expect(page.locator("#detail-view script")).toHaveCount(0);
});
