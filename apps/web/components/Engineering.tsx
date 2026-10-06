"use client";

import { useEffect, useState } from "react";

type Rights = { checked: string; note: string; sources: { id: string; module: string; name: string; url: string; attribution: string; licence_stated: string; processing: string; display: string; independently_reviewed: boolean }[];
  not_used: { id: string; reason: string }[] };
type Baseline = { frozen_at: string; git_commit: string; note: string; coverage: { counts: Record<string, unknown>; models: Record<string, string> } };
const TH = "py-1.5 pr-3 text-left font-normal text-muted align-top";

export default function Engineering() {
  const [rights, setRights] = useState<Rights | null>(null);
  const [base, setBase] = useState<Baseline | null>(null);
  useEffect(() => {
    fetch("/demo/research/rights.json").then((r) => (r.ok ? r.json() : null)).then(setRights).catch(() => null);
    fetch("/demo/research/baseline_v1.json").then((r) => (r.ok ? r.json() : null)).then(setBase).catch(() => null);
  }, []);
  return (
    <>
      <h2 className="display mt-14 text-4xl sm:text-5xl">Data sources and what may be shown</h2>
      {!rights ? <p className="mt-3 text-muted">Rights manifest not exported yet.</p> : (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead><tr><th className={TH}>Source</th><th className={TH}>Module</th><th className={TH}>Licence as stated</th><th className={TH}>What the site shows</th></tr></thead>
            <tbody>
              {rights.sources.map((s) => (
                <tr key={s.id} className="border-t border-line align-top">
                  <td className="py-2 pr-3">{s.url ? <a className="text-teal underline" href={s.url}>{s.name}</a> : s.name}<span className="block text-xs text-muted">{s.attribution}</span></td>
                  <td className="py-2 pr-3">{s.module}</td><td className="py-2 pr-3 text-muted">{s.licence_stated}</td>
                  <td className="py-2 text-muted">{s.display.replace(/\.?$/, ".")}{s.independently_reviewed ? "" : " Terms not independently reviewed."}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-muted">Checked {rights.checked}. Not used: {rights.not_used.map((n) => `${n.id} (${n.reason})`).join("; ")}.</p>
        </div>
      )}
      <h2 className="display mt-14 text-4xl sm:text-5xl">Preserved v1 benchmark</h2>
      {!base ? <p className="mt-3 text-muted">Baseline manifest not exported yet.</p> : (
        <p className="num mt-3 max-w-[70ch] break-all text-sm text-muted">Frozen {base.frozen_at.slice(0, 10)} at commit {base.git_commit.slice(0, 12)}. {base.note} The manifest records data checksums, the split ids, feature lists, {Object.keys(base.coverage.models).length} model file hashes, export hashes, metrics and interface defaults.</p>
      )}
    </>
  );
}
