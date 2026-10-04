import { test, expect } from "@playwright/test";
import fs from "node:fs";

const routes = [
  ["home", "/", "Read the field."],
  ["coverage", "/coverage?model=v2_temporal", "Read the defense."],
  ["predictions", "/predictions", "Before kickoff."],
  ["highlights", "/highlights", "Find the moment."],
  ["research", "/research", "The playbook."],
  ["engineering", "/engineering", "Built to be inspected."],
];

for (const width of [1440, 390]) {
  test(`six pages render at ${width}px without page overflow`, async ({ page }) => {
    test.setTimeout(180000);
    await page.setViewportSize({ width, height: width===390 ? 844 : 1000 });
    const errors: string[] = [];
    page.on("pageerror", e => errors.push(e.message));
    fs.mkdirSync("../../docs/screenshots/cinematic", { recursive: true });
    for (const [name, path, heading] of routes) {
      await page.goto(path);
      await expect(page.getByRole("heading", {level:1})).toHaveText(new RegExp(heading.replace(/ /g,"\\s*").replace(/\./g,"\\."),"i"));
      if (name === "home") await expect(page.getByText("Loading the saved benchmark record…")).toHaveCount(0);
      if (name === "coverage") await expect(page.getByTestId("probs")).toBeVisible();
      if (name === "predictions") {
        await expect(page.locator(".team-helmet img")).toHaveCount(2);
        await expect(page.getByTestId("waterfall")).toBeVisible();
      }
      if (name === "highlights") await expect(page.locator(".clip-evidence")).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth+1), path).toBe(true);
      await page.screenshot({path:`../../docs/screenshots/cinematic/${name}-${width}.jpg`,type:"jpeg",quality:85});
    }
    expect(errors).toEqual([]);
  });
}

test("matchup art changes identity and retains the saved prediction", async ({page}) => {
  await page.goto("/predictions?game=2025_01_HOU_LA");
  await expect(page.locator(".team-name")).toHaveText(["Houston Texans","Los Angeles Rams"]);
  await expect(page.locator(".matchup-probabilities")).toContainText("62%");
  await page.getByRole("button",{name:/DAL at PHI/}).click();
  await expect(page.locator(".team-name")).toHaveText(["Dallas Cowboys","Philadelphia Eagles"]);
  await expect(page.locator(".matchup-probabilities")).toContainText("81%");
});

test("candidate click maps time, reuses player, pauses at end, and handles embed failure", async ({page}) => {
  // Controlled API double verifies our state contract; this does not verify a real video permits embedding.
  await page.route("https://www.youtube.com/iframe_api",route => route.fulfill({contentType:"application/javascript",body:`
    window.__ytInstances=[];
    window.YT={Player:class {
      constructor(el,config){this.config=config;this.time=0;this.seeks=[];window.__ytInstances.push(this);queueMicrotask(()=>config.events.onReady({target:this}));}
      cueVideoById(o){this.time=o.startSeconds;}
      seekTo(t){this.time=t;this.seeks.push(t);}
      playVideo(){this.config.events.onStateChange({target:this,data:1});}
      pauseVideo(){this.config.events.onStateChange({target:this,data:2});}
      getCurrentTime(){return this.time;}
      destroy(){this.destroyed=true;}
    }};window.onYouTubeIframeAPIReady();
  `}));
  await page.goto("/highlights");
  const candidates = page.locator(".candidate-button");
  await expect(candidates.first()).toBeVisible();
  await candidates.first().click();
  await expect(page.locator(".player-status")).toHaveText("Playing source video");
  const state = await page.evaluate(() => {
    const p=(window as unknown as {__ytInstances:{time:number;seeks:number[]}[]}).__ytInstances[0];
    return {time:p.time,seeks:p.seeks};
  });
  expect(state.time).toBeCloseTo(435.123448,5);
  await candidates.nth(1).click();
  await expect.poll(()=>page.evaluate(()=>(window as unknown as {__ytInstances:{time:number}[]}).__ytInstances[0].time)).toBeCloseTo(1249.123448,5);
  expect(await page.evaluate(()=>(window as unknown as {__ytInstances:unknown[]}).__ytInstances.length)).toBe(1);
  await page.evaluate(() => { (window as unknown as {__ytInstances:{time:number}[]}).__ytInstances[0].time=1260; });
  await expect(page.locator(".player-status")).toContainText("Interval finished");
  await page.getByRole("button",{name:"Replay ↺"}).click();
  await expect(page.locator(".player-status")).toHaveText("Playing source video");
  await page.evaluate(() => {
    const p=(window as unknown as {__ytInstances:{config:{events:{onError(e:{data:number}):void}}}[]}).__ytInstances[0];
    p.config.events.onError({data:150});
  });
  await expect(page.getByRole("link",{name:"Open source video ↗"})).toBeVisible();
  await expect(page.locator(".player-status")).toContainText("cannot play here");
});
