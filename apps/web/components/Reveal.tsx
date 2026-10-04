"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";

// Reveals its children once when they scroll into view. Content is in the DOM from the start; with reduced motion
// (or no IntersectionObserver) it is simply visible. Only opacity and transform change.
export default function Reveal({ children, className = "", as: Tag = "div" }: { children: ReactNode; className?: string; as?: "div" | "section" | "li" }) {
  const ref = useRef<HTMLElement>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") { queueMicrotask(() => setSeen(true)); return; }
    const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { setSeen(true); io.disconnect(); } }, { rootMargin: "0px 0px -12% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const T = Tag as any;
  return <T ref={ref} className={`reveal ${seen ? "is-in" : ""} ${className}`}>{children}</T>;
}
