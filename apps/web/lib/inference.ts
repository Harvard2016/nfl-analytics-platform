// Client for the local inference service. The service address comes only from build-time configuration:
// with none set (the public deployment) no request is made and nothing on the visitor's machine is probed.
export const API_BASE = (process.env.NEXT_PUBLIC_GRIDIRON_API ?? "").replace(/\/$/, "");
export const configured = API_BASE !== "";

export type ModeInfo = { ready: boolean; reason?: string | null; experimental?: boolean; model?: string | null };
export type Capabilities = {
  retention_hours: number; privacy: string; hosting: string;
  modules: {
    coverage: { ready: boolean; model: string; modes: Record<string, ModeInfo>; limits: { max_bytes: number }; input: { required: Record<string, string>; motion: string; orientation: string; field: string; optional: Record<string, string>; limits: Record<string, string> } };
    highlights: { modes: Record<string, ModeInfo>; limits: { max_bytes: number; max_duration_s: number; min_duration_s: number; full_games: string } };
    video_coverage: { ready: boolean; reason: string };
  };
};
export type JobState = "uploading" | "queued" | "validating" | "extracting" | "awaiting_review" | "inferring" | "rendering" | "complete" | "failed" | "cancelled";
export type Job = { id: string; state: JobState; stage: string | null; progress: number | null; warnings: string[]; error: string | null; result_available: boolean; expires_at: string; assets: string[] };
export type Ticket = { id: string; access_token: string };
export const TERMINAL: JobState[] = ["complete", "failed", "cancelled"];
export const STATE_TEXT: Record<JobState, string> = {
  uploading: "Receiving the file", queued: "Waiting for the worker", validating: "Checking the file", extracting: "Extracting signals", awaiting_review: "Waiting for your review",
  inferring: "Running the model", rendering: "Cutting clips", complete: "Done", failed: "Failed", cancelled: "Cancelled",
};

const auth = (t: Ticket) => ({ Authorization: `Bearer ${t.access_token}` });
async function fail(r: Response): Promise<never> {
  let detail = `${r.status}`;
  try { const b = await r.json(); detail = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail ?? b); } catch { /* keep the status code */ }
  throw new Error(detail);
}

export async function capabilities(signal?: AbortSignal): Promise<Capabilities> {
  const r = await fetch(`${API_BASE}/v1/capabilities`, { signal, cache: "no-store" });
  return r.ok ? r.json() : fail(r);
}
export async function submit(module: "coverage" | "highlights", mode: string, options: Record<string, unknown>, file: File): Promise<Ticket> {
  const body = new FormData();
  body.set("module", module); body.set("mode", mode); body.set("options", JSON.stringify(options)); body.set("file", file);
  const r = await fetch(`${API_BASE}/v1/jobs`, { method: "POST", body });
  return r.ok ? r.json() : fail(r);
}
export async function status(t: Ticket): Promise<Job> {
  const r = await fetch(`${API_BASE}/v1/jobs/${t.id}`, { headers: auth(t), cache: "no-store" });
  return r.ok ? r.json() : fail(r);
}
export async function result<T>(t: Ticket): Promise<T> {
  const r = await fetch(`${API_BASE}/v1/jobs/${t.id}/result`, { headers: auth(t), cache: "no-store" });
  return r.ok ? r.json() : fail(r);
}
export async function cancel(t: Ticket): Promise<void> { await fetch(`${API_BASE}/v1/jobs/${t.id}/cancel`, { method: "POST", headers: auth(t) }); }
export async function remove(t: Ticket): Promise<boolean> {
  const r = await fetch(`${API_BASE}/v1/jobs/${t.id}`, { method: "DELETE", headers: auth(t) });
  return r.ok;
}
// Media elements cannot send headers, so the per-job token travels in the query string to the loopback service only.
export const mediaUrl = (t: Ticket, asset: string) => `${API_BASE}/v1/jobs/${t.id}/media/${encodeURIComponent(asset)}?token=${encodeURIComponent(t.access_token)}`;
export const templateUrl = `${API_BASE}/v1/templates/coverage.csv`;
