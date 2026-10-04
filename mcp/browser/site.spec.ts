import { test, expect } from "@playwright/test";
test("all setup options are visible with compact keyboard-accessible copy controls", async ({
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
  expect(
    await page.locator("main [class], [role=tab], details, [hidden]").count(),
  ).toBe(0);
  for (const text of [
    "uv tool install philly",
    "uv add philly",
    "npm run stdio",
    "https://your-worker.example/mcp",
  ]) {
    await expect(page.locator("pre").filter({ hasText: text })).toBeVisible();
  }
  for (const button of await page.locator("button[data-copy]").all()) {
    const box = await button.boundingBox();
    expect(box!.height).toBeGreaterThanOrEqual(24);
    expect(box!.height).toBeLessThanOrEqual(28);
    expect(box!.width).toBeLessThanOrEqual(64);
  }
  await expect(page.locator('a[href*="docs.astral.sh"]')).toHaveCount(0);
  await expect(page.locator("body")).not.toContainText(
    "Hosted MCP is not live",
  );
  await page.screenshot({
    path: `${info.project.outputDir}/philly-${info.project.name}.png`,
    fullPage: true,
  });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.screenshot({
    path: `${info.project.outputDir}/philly-${info.project.name}-dark.png`,
    fullPage: true,
  });
  await page.getByRole("link", { name: "Client setup", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Connect Philly" }),
  ).toBeVisible();
  expect(await page.locator("details, [hidden], [role=tab]").count()).toBe(0);
  for (const label of [
    "Copy server installation",
    "Copy Claude Code setup",
    "Copy stdio client configuration",
    "Copy VS Code configuration",
    "Copy local HTTP setup",
    "Copy local MCP URL",
    "Copy remote MCP configuration",
  ]) {
    const button = page.getByRole("button", { name: label, exact: true });
    await expect(button).toBeVisible();
    await expect(button.locator("..").locator("pre")).toBeVisible();
    await button.focus();
    await expect(button).toBeFocused();
    await page.keyboard.press("Enter");
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
      await button.locator("..").locator("code").textContent(),
    );
    const box = await button.boundingBox();
    expect(box!.height).toBeLessThanOrEqual(28);
    expect(box!.width).toBeLessThanOrEqual(64);
  }
  await expect(
    page.getByText("http://127.0.0.1:8787/mcp", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(page.locator('a[href*="docs.astral.sh"]')).toHaveCount(0);
  await expect(page.locator("body")).not.toContainText(
    "Hosted MCP is not live",
  );
  await page.screenshot({
    path: `${info.project.outputDir}/connect-${info.project.name}.png`,
    fullPage: true,
  });
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
