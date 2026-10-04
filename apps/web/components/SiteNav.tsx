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

// No GitHub link: the repository has no remote configured, and the brief forbids a placeholder URL.
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
