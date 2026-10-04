import { test, expect } from "@playwright/test";
test("landing, keyboard tabs, clipboard and local connection instructions", async ({
  page,
  context,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Philly");
  await expect(
    page.getByRole("img", { name: "Philly", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(
    page.getByRole("button", { name: "Copy terminal install command" }),
  ).toBeInViewport();
  await expect(
    page.getByRole("link", { name: "Connect an MCP client" }),
  ).toBeInViewport();
  expect(await page.locator("main [class]").count()).toBe(0);
  await page.getByRole("tab", { name: "Terminal" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Python" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.keyboard.press("End");
  await expect(
    page.getByRole("tab", { name: "MCP", exact: true }),
  ).toBeFocused();
  await expect(page.getByRole("tabpanel")).toContainText("npm ci");
  await page.getByRole("button", { name: "Copy local MCP setup" }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(
    "npm run stdio",
  );
  await page.getByRole("tab", { name: "Terminal" }).click();
  await page.screenshot({
    path: `${info.project.outputDir}/philly-${info.project.name}.png`,
    fullPage: true,
  });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.screenshot({
    path: `${info.project.outputDir}/philly-${info.project.name}-dark.png`,
    fullPage: true,
  });
  await page.getByRole("link", { name: "Connect", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Connect Philly" }),
  ).toBeVisible();
  await page.getByText("Local HTTP", { exact: true }).click();
  await expect(
    page.getByText("http://127.0.0.1:8787/mcp", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});
test("clipboard denial gives selectable fallback; atlas and gallery routes survive", async ({
  page,
}) => {
  await page.goto("/");
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: async () => {
          throw new Error("denied");
        },
      },
    });
  });
  await page
    .getByRole("button", { name: "Copy terminal install command" })
    .click();
  await expect(page.getByRole("status")).toContainText("Clipboard unavailable");
  expect(
    await page.evaluate(() => window.getSelection()?.toString()),
  ).toContain("uv tool install philly");
  await page.getByRole("link", { name: "Explore", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Explorations", exact: true }),
  ).toBeVisible();
  // Atlas relies on external maps; inspect its route without fetching those services in offline tests.
  const atlas = await page.request.get("/explorations/city-atlas/");
  expect(atlas.ok()).toBe(true);
  expect(await atlas.text()).toContain("City Atlas | Philly");
});
test("City Atlas service worker controls the preserved page route", async ({
  page,
}) => {
  await page.goto("/");
  const scope = await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.register(
      "/explorations/city-atlas/sw.js",
    );
    const scope = registration.scope;
    await registration.unregister();
    return scope;
  });
  expect(scope).toBe("http://127.0.0.1:4321/explorations/city-atlas/");
});
