import { test, expect } from "@playwright/test";
import fs from "node:fs";

test("one real source reports playback or a specific block", async ({page}) => {
  test.skip(process.env.YOUTUBE_SOURCE_PROBE !== "1", "Opt-in network diagnosis; not a deterministic regression test.");
  test.setTimeout(60000);
  await page.addInitScript(() => localStorage.setItem("gridiron-tour:highlights:v1","seen"));
  await page.goto("/highlights");
  await page.locator(".candidate-button").first().click();
  const status=page.locator(".player-status");
  try {
    await expect(status).toContainText(/Playing source video|disabled|unavailable|blocked|Could not|did not finish|could not identify|could not open/, {timeout:40000});
  } finally {
    fs.mkdirSync("../../docs/screenshots/cinematic",{recursive:true});
    const result={environment:"GitHub Actions Chromium / localhost",status:await status.textContent(),source:await page.locator(".player-titlebar a").getAttribute("href"),error:await page.getByTestId("youtube-error").allTextContents()};
    fs.writeFileSync("../../docs/screenshots/cinematic/youtube-source-check.json",JSON.stringify(result,null,2));
    console.log("YOUTUBE_SOURCE_CHECK",JSON.stringify(result));
    // No game frames or audio are captured; only our own status text is saved.
    await status.screenshot({path:"../../docs/screenshots/cinematic/youtube-status.png"});
  }
});
