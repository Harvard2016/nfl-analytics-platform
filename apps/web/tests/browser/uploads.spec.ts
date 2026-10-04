import { expect, test } from "@playwright/test";

// The public build has no inference service configured. It must explain local operation, accept no file,
// and contact nothing on the visitor's machine. With a local service configured these checks do not apply.
for (const path of ["/coverage/analyze", "/highlights/analyze", "/coverage/review", "/predictions/forecasts"]) {
  test(`${path} renders without page overflow or errors`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(() => { try { for (const m of ["coverage-upload", "highlights-upload"]) localStorage.setItem(`gridiron-tour:${m}:v1`, "seen"); } catch { /* storage blocked */ } });
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), path).toBe(true);
    expect(errors).toEqual([]);
  });
}

for (const path of ["/coverage/analyze", "/highlights/analyze"]) {
  test(`${path} without a configured service accepts no file and probes nothing local`, async ({ page }) => {
    const local: string[] = [];
    page.on("request", (r) => { if (/\/\/(127\.0\.0\.1|localhost):8765/.test(r.url())) local.push(r.url()); });
    await page.goto(path);
    const notice = page.getByTestId("backend-notice");
    test.skip(!(await notice.isVisible().catch(() => false)) || !(await notice.textContent())?.includes("your own machine"), "a local inference service is configured for this build");
    await expect(notice).toContainText("does not accept files");
    await expect(page.locator('input[type="file"]')).toHaveCount(0);
    expect(local).toEqual([]);
  });
}
