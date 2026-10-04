import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  workers: 1,
  timeout: 45000,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:3111", reducedMotion: "reduce", trace: "retain-on-failure" },
  webServer: { command: "npm run start -- --hostname 127.0.0.1 --port 3111", url: "http://127.0.0.1:3111", reuseExistingServer: process.env.PLAYWRIGHT_REUSE_SERVER === "1" },
});
