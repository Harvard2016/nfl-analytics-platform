import { defineConfig } from "@playwright/test";

// A separate development build enables the upload UI. Every service request is mocked;
// production's unconfigured/no-upload behavior is covered by playwright.config.ts.
export default defineConfig({
  testDir: "./tests/inference",
  fullyParallel: false,
  workers: 1,
  timeout: 45000,
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:3112", reducedMotion: "reduce", trace: "retain-on-failure" },
  webServer: {
    command: "npm run dev -- --hostname 127.0.0.1 --port 3112",
    url: "http://127.0.0.1:3112",
    env: { NEXT_PUBLIC_GRIDIRON_API: "http://127.0.0.1:8765" },
    reuseExistingServer: false,
  },
});
