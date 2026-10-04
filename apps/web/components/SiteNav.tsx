"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

const NAV = [
  ["/coverage", "Coverage"],
  ["/predictions", "Predictions"],
  ["/highlights", "Highlights"],
  ["/research", "Research"],
  ["/engineering", "Engineering"],
] as const;

export default function SiteNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [compact, setCompact] = useState(false);
  const button = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const onScroll = () => setCompact(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpen(false); button.current?.focus(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className={`sticky top-0 z-40 border-b border-line bg-bg/95 backdrop-blur transition-[padding] duration-200 ${compact ? "py-0" : "py-1.5"}`}>
      <nav aria-label="Main" className="mx-auto flex max-w-[1500px] items-center gap-6 px-4 lg:px-8">
        <Link href="/" className="display py-3 text-2xl tracking-wide" onClick={() => setOpen(false)}>Gridiron <span className="text-teal">Lens</span></Link>
        <ul className="ml-auto hidden items-center gap-7 md:flex">
          {NAV.map(([href, label]) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <li key={href}>
                <Link href={href} aria-current={active ? "page" : undefined}
                  className={`narrow block border-b-2 py-3 text-[17px] transition-colors duration-150 ${active ? "border-teal text-teal" : "border-transparent text-muted hover:text-ink"}`}>{label}</Link>
              </li>
            );
          })}
        </ul>
        <a href="https://github.com/Harvard2016/nfl-analytics-platform" target="_blank" rel="noreferrer" className="nav-source hidden lg:inline-flex" aria-label="View source on GitHub"><svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M12 .8a11.2 11.2 0 0 0-3.54 21.83c.56.1.77-.24.77-.54v-2.1c-3.13.68-3.79-1.33-3.79-1.33-.51-1.3-1.25-1.65-1.25-1.65-1.02-.7.08-.68.08-.68 1.13.08 1.73 1.16 1.73 1.16 1 1.73 2.62 1.23 3.26.94.1-.73.4-1.23.71-1.51-2.5-.29-5.13-1.25-5.13-5.54 0-1.23.44-2.23 1.16-3.01-.12-.28-.51-1.43.11-2.98 0 0 .95-.3 3.08 1.15A10.7 10.7 0 0 1 12 6.16c.95 0 1.91.13 2.81.38 2.13-1.45 3.08-1.15 3.08-1.15.62 1.55.23 2.7.11 2.98.72.78 1.16 1.78 1.16 3.01 0 4.3-2.64 5.25-5.15 5.53.41.36.77 1.04.77 2.1v3.08c0 .3.2.65.77.54A11.2 11.2 0 0 0 12 .8Z" /></svg>Source ↗</a>
        <button ref={button} className="narrow ml-auto border border-line px-3 text-[15px] md:hidden" aria-expanded={open} aria-controls="mobile-menu" onClick={() => setOpen((o) => !o)}>
          {open ? "Close" : "Menu"}
        </button>
      </nav>
      {open && (
        <ul id="mobile-menu" className="fade-swap border-t border-line px-4 pb-4 md:hidden">
          {NAV.map(([href, label]) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <li key={href} className="border-b border-line">
                <Link href={href} aria-current={active ? "page" : undefined} onClick={() => setOpen(false)} className={`display block py-3 text-3xl ${active ? "text-teal" : ""}`}>{label}</Link>
              </li>
            );
          })}
        </ul>
      )}
    </header>
  );
}
