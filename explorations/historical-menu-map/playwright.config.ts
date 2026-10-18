import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./browser",
  outputDir: "./test-results",
  fullyParallel: true,
  use: {
    baseURL: "http://127.0.0.1:4173",
    colorScheme: "light",
    launchOptions:
      process.env["CHROMIUM_PATH"] === undefined
        ? {}
        : { executablePath: process.env["CHROMIUM_PATH"] },
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: process.env["CI"] === undefined,
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1100 } } },
    {
      name: "mobile",
      use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true },
    },
  ],
});
