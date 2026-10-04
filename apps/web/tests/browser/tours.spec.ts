import { test, expect } from "@playwright/test";
import fs from "node:fs";

for (const width of [1440, 390]) for (const moduleName of ["coverage", "predictions", "highlights"]) {
  test(`${moduleName} tour opens once, highlights five steps, finishes and replays at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
    await page.goto(`/${moduleName}`);
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    fs.mkdirSync("../../docs/screenshots/cinematic", {recursive:true});
    for (let step=1;step<=5;step++) {
      await expect(dialog.locator(".tour-count")).toHaveText(`${step} / 5`);
      await expect(dialog.locator(".tour-spotlight")).toBeVisible();
      expect(await dialog.evaluate(el => el.scrollWidth <= window.innerWidth+1)).toBe(true);
      const card = await dialog.locator(".tour-card").boundingBox();
      expect(card).not.toBeNull();
      expect(card!.x).toBeGreaterThanOrEqual(0);
      expect(card!.x+card!.width).toBeLessThanOrEqual(width+1);
      // Native modal focus stays within the tour when Tab wraps.
      await dialog.getByRole("button", {name: step===5 ? "Finish tour" : "Next →",exact:true}).focus();
      await page.keyboard.press("Tab");
      expect(await dialog.evaluate(el => el.contains(document.activeElement))).toBe(true);
      if(step===3) await page.screenshot({path:`../../docs/screenshots/cinematic/tour-${moduleName}-${width}.jpg`,type:"jpeg",quality:85});
      await dialog.getByRole("button", {name: step===5 ? "Finish tour" : "Next →",exact:true}).click();
    }
    await expect(dialog).not.toBeVisible();
    expect(await page.evaluate(moduleName => localStorage.getItem(`gridiron-tour:${moduleName}:v1`),moduleName)).toBe("seen");
    await page.reload();
    await expect(page.getByRole("button",{name:/Page tour/})).toBeEnabled();
    await expect(dialog).not.toBeVisible();
    await page.getByRole("button",{name:/Page tour/}).click();
    await expect(dialog).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible();
    await expect(page.getByRole("button",{name:/Page tour/})).toBeFocused();
  });
}

test("skipping one moduleName does not skip the others",async({page})=>{
  await page.goto("/predictions");
  await page.getByRole("button",{name:"Skip tour ×"}).click();
  await page.goto("/highlights");
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.locator(".tour-count")).toHaveText("1 / 5");
});
