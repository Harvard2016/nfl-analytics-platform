import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import type { Capabilities, Job } from "../../lib/inference";

// Synthetic fixtures check UI mechanics only: no trained inference or football media is represented.
const caps: Capabilities = {
  privacy: "Synthetic test service", retention_hours: 24, hosting: "local",
  modules: {
    coverage: { ready: true, model: "synthetic", modes: { "release-compatible": { ready: true } }, limits: { max_bytes: 2097152 },
      input: { required: {}, optional: {}, motion: "synthetic", orientation: "synthetic", field: "synthetic", limits: {} } },
    highlights: { modes: { loudness_baseline: { ready: true } }, limits: { max_bytes: 600 * 2 ** 20, min_duration_s: 10, max_duration_s: 1200, full_games: "synthetic" } },
    video_coverage: { ready: false, reason: "not available" },
  },
};

function clipResult(id: string) {
  return {
    mode: `synthetic ${id}`, mode_note: "Synthetic mechanics fixture", score_meaning: "Ranks, not probabilities", domain_shift: "Synthetic", time_origin: "file seconds", clip_seconds: 2,
    media: { duration_s: 30, has_video: true }, event_types: "none: synthetic fixture", evidence_streams_used: ["synthetic audio"],
    timeline: { audio_start_s: 0, audio_end_s: 20.6, bin_start_s: Array.from({ length: 11 }, (_, i) => 2 * i), bin_end_s: Array.from({ length: 11 }, (_, i) => Math.min(2 * i + 2, 20.6)),
      loudness_db: Array(11).fill(-20), loudness_above_background_db: Array(11).fill(2), rank_within_file: Array(11).fill(50) },
    candidates: [{ asset: "a.mp4", rank_in_reel: 1, candidate_moment_s: 10, clip_start_s: 8, clip_end_s: 12, ffprobe_duration_s: 4, loudness_above_background_db: 2 },
      { asset: "b.mp4", rank_in_reel: 2, candidate_moment_s: 18, clip_start_s: 16, clip_end_s: 20, ffprobe_duration_s: 4, loudness_above_background_db: 2 }],
    reel: null, measured: { seconds: 1, peak_memory_mb: 1 },
  };
}

function playResult(id: string) {
  return { play_id: id, mode: "synthetic", scope: "Synthetic fixture", explanation_note: "No real model", horizons: Object.fromEntries(["snap", "post_0_5s", "post_1_0s", "post_1_5s"].map(h => [h, { available: false, reason: "synthetic" }])),
    model: { version: "synthetic", bundle_sha256: "none", input_schema: "synthetic", calibration: "none" },
    input_quality: { defenders: 0, route_runners: 0, frames_supplied: 1, frames_usable: 1, observed_orientation: false, warnings: [] }, comparison_label: null,
    playback: { frames: [0], los_x: 50, entities: [], note: "Synthetic" } };
}

async function setup(page: Page, module: "coverage" | "highlights", delayFirst = false) {
  await page.addInitScript(() => {
    for (const m of ["coverage-upload", "highlights-upload"]) localStorage.setItem(`gridiron-tour:${m}:v1`, "seen");
    // Only the media clock is stubbed; real ffmpeg stream timing is tested by the Python suite.
    Object.defineProperty(HTMLMediaElement.prototype, "duration", { configurable: true, get: () => 30 });
    Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
      configurable: true, get() { return Number(this.dataset.syntheticTime ?? 0); },
      set(v: number) { this.dataset.syntheticTime = String(v); this.dispatchEvent(new Event("timeupdate")); },
    });
  });
  let count = 0;
  let firstRequested!: () => void, release!: () => void;
  const requested = new Promise<void>(r => { firstRequested = r; });
  const gate = new Promise<void>(r => { release = r; });
  await page.route("http://127.0.0.1:8765/**", async route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    const headers = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Authorization, Content-Type", "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS" };
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers });
    let json: unknown;
    if (path.endsWith("capabilities")) json = caps;
    else if (path === "/v1/jobs") json = { id: `job${++count}`, access_token: `token${count}` };
    else if (path.endsWith("/cancel")) json = { state: "cancelled" };
    else if (path.endsWith("/result")) {
      const id = path.split("/")[3];
      if (id === "job1" && delayFirst) { firstRequested(); await gate; }
      json = module === "coverage" ? playResult(id) : clipResult(id);
    } else {
      const job: Job = { id: path.split("/")[3], state: "complete", stage: null, progress: null, warnings: [], error: null, result_available: true, assets: [], expires_at: "2026-10-07T00:00:00+00:00" };
      json = job;
    }
    await route.fulfill({ status: 200, headers, json });
  });
  await page.goto(`/${module}/analyze`);
  await expect(page.locator('input[type="file"]')).toBeVisible();
  return { requested, release };
}

async function select(page: Page, module: "coverage" | "highlights", name: string) {
  await page.locator('input[type="file"]').setInputFiles({ name: `${name}.${module === "coverage" ? "csv" : "mp4"}`, mimeType: module === "coverage" ? "text/csv" : "video/mp4", buffer: Buffer.from("synthetic fixture") });
  if (module === "highlights") await page.getByRole("checkbox", { name: "I have permission to process this file." }).check();
}

async function run(page: Page, module: "coverage" | "highlights") {
  await page.getByRole("button", { name: module === "coverage" ? "Run the model" : "Analyze the clip", exact: true }).click();
}

test("shorter audio uses the video clock for chart, seeking and playhead", async ({ page }) => {
  await setup(page, "highlights");
  await select(page, "highlights", "a"); await run(page, "highlights");
  const graph = page.getByTestId("clip-timeline");
  await expect(graph).toHaveAttribute("viewBox", "0 0 30 82");
  await expect(page.getByTestId("missing-audio")).toHaveAttribute("x", "20.6");
  const box = (await graph.boundingBox())!;
  await page.mouse.click(box.x + box.width / 3, box.y + box.height / 2);
  await expect.poll(() => page.getByTestId("clip-player").evaluate((v: HTMLVideoElement) => v.currentTime)).toBeCloseTo(10, 1);
  await expect.poll(async () => Number(await page.getByTestId("clip-playhead").getAttribute("x1"))).toBeCloseTo(10, 1);
});

test("JSON review flags require explicit per-clip confirmation and reset after edits", async ({ page }) => {
  await setup(page, "highlights"); await select(page, "highlights", "a"); await run(page, "highlights");
  const checks = page.getByRole("checkbox", { name: "I watched this clip and reviewed these settings" });
  await expect(checks).toHaveCount(2);
  async function flags() {
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: /Export.*JSON/i }).click();
    const path = (await (await download).path())!;
    return JSON.parse(await readFile(path, "utf8")).candidates.map((c: { reviewed_by_a_person: boolean }) => c.reviewed_by_a_person);
  }
  expect(await flags()).toEqual([false, false]);
  await checks.first().check();
  expect(await flags()).toEqual([true, false]);
  await page.getByRole("spinbutton", { name: "Start", exact: true }).first().fill("7.5");
  await expect(checks.first()).not.toBeChecked();
  expect(await flags()).toEqual([false, false]);
});

for (const moduleName of ["coverage", "highlights"] as const) {
  test(`${moduleName}: delayed previous result cannot replace the new file's result`, async ({ page }) => {
    const { requested, release } = await setup(page, moduleName, true);
    await select(page, moduleName, "a"); await run(page, moduleName); await requested;
    await select(page, moduleName, "b"); await run(page, moduleName);
    const output = moduleName === "coverage" ? page.getByRole("heading", { name: "Play job2", exact: true }) : page.getByTestId("clip-mode");
    await expect(output).toBeVisible();
    if (moduleName === "highlights") await expect(output).toHaveText("synthetic job2");
    const response = page.waitForResponse(r => r.url().endsWith("/jobs/job1/result"));
    release(); await response;
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    if (moduleName === "coverage") await expect(page.getByRole("heading", { name: "Play job1", exact: true })).toHaveCount(0);
    else await expect(output).toHaveText("synthetic job2");
  });
}
