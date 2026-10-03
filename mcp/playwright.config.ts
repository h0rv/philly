import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./browser",
  outputDir: "./test-results",
  use: {
    baseURL: "http://127.0.0.1:4321",
    colorScheme: "dark",
    launchOptions: process.env.CHROMIUM_PATH
      ? { executablePath: process.env.CHROMIUM_PATH }
      : {},
    screenshot: "only-on-failure",
  },
  webServer: {
    command:
      "node ../website/node_modules/astro/bin/astro.mjs preview --root ../website --host 127.0.0.1",
    url: "http://127.0.0.1:4321",
    reuseExistingServer: !process.env.CI,
    env: { ASTRO_TELEMETRY_DISABLED: "1" },
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1280, height: 1000 } } },
    { name: "mobile", use: { viewport: { width: 375, height: 812 } } },
  ],
});
