"use client";

import { useEffect, useId, useRef, useState } from "react";
import { MODULE_TOURS, type TourModule } from "@/lib/moduleTours";

const seenThisSession = new Set<string>();
export default function ModuleTour({ module, ready = true }: { module: TourModule; ready?: boolean }) {
  const steps = MODULE_TOURS[module];
  const key = `gridiron-tour:${module}:v1`;
  const [step, setStep] = useState<number | null>(null);
  const [rect, setRect] = useState<{ top: number; left: number; width: number; height: number } | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const titleId = useId(), bodyId = useId();

  useEffect(() => {
    if (!ready) return;
    const timer = window.setTimeout(() => {
      let seen = seenThisSession.has(key);
      try { seen ||= localStorage.getItem(key) === "seen"; } catch { /* Tour still works when storage is blocked. */ }
      if (!seen) setStep(0);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [ready, key]);

  useEffect(() => {
    if (step === null) return;
    const modal = dialog.current;
    if (modal && !modal.open) modal.showModal();
    const target = document.querySelector<HTMLElement>(`[data-tour="${steps[step].target}"]`);
    target?.scrollIntoView({ block: "center", behavior: "instant" });
    const measure = () => {
      const r = target?.getBoundingClientRect();
      if (!r) { setRect(null); return; }
      const top = Math.max(66, r.top - 6), left = Math.max(8, r.left - 6);
      setRect({ top, left, width: Math.max(0, Math.min(window.innerWidth - left - 8, r.right + 6 - left)), height: Math.max(0, Math.min(window.innerHeight - top - 12, r.bottom + 6 - top)) });
    };
    measure(); heading.current?.focus({ preventScroll: true });
    window.addEventListener("resize", measure); window.addEventListener("scroll", measure, true);
    return () => { window.removeEventListener("resize", measure); window.removeEventListener("scroll", measure, true); };
  }, [step, steps]);

  const close = () => {
    seenThisSession.add(key);
    try { localStorage.setItem(key, "seen"); } catch { /* Optional browser preference only. */ }
    dialog.current?.close(); setStep(null); setRect(null);
    trigger.current?.focus({ preventScroll: true });
  };
  const current = step === null ? null : steps[step];
  return <>
    <button ref={trigger} type="button" className="tour-trigger" onClick={() => setStep(0)} disabled={!ready}>Page tour <span aria-hidden="true">↗</span></button>
    <dialog ref={dialog} className="tour-dialog" aria-labelledby={titleId} aria-describedby={bodyId} onKeyDown={e => {
      e.stopPropagation();
      if (e.key !== "Tab") return;
      const buttons = Array.from(e.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"));
      const first = buttons[0], last = buttons.at(-1);
      if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
      if (e.shiftKey && (document.activeElement === first || document.activeElement === heading.current)) { e.preventDefault(); last?.focus(); }
    }} onCancel={e => { e.preventDefault(); close(); }}>
      {current && <>
        {rect && rect.height > 0 && <div className="tour-spotlight" aria-hidden="true" style={rect} />}
        <section className="tour-card">
          <div className="flex items-center justify-between gap-4"><p className="kicker">{module} / page tour</p><button onClick={close} className="tour-skip">Skip tour ×</button></div>
          <p className="tour-count mono">{(step ?? 0) + 1} / {steps.length}</p>
          <h2 id={titleId} ref={heading} tabIndex={-1} className="narrow text-3xl">{current.title}</h2>
          <p id={bodyId} className="tour-body">{current.body}</p>
          {current.tip && <p className="tour-tip">{current.tip}</p>}
          <div className="tour-actions"><button disabled={step === 0} onClick={() => setStep(s => Math.max(0, (s ?? 0) - 1))}>Back</button><button className="tour-next" onClick={() => step === steps.length - 1 ? close() : setStep(s => (s ?? 0) + 1)}>{step === steps.length - 1 ? "Finish tour" : "Next →"}</button></div>
        </section>
      </>}
    </dialog>
  </>;
}
