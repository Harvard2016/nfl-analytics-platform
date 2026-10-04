"use client";

import { type ReactNode, useEffect, useRef } from "react";

// Gentle background drift on scroll for the home hero. Off under reduced motion and on small screens.
export default function Parallax({ children, className = "", amount = 0.18 }: { children: ReactNode; className?: string; amount?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches || window.innerWidth < 768) return;
    let raf = 0;
    const onScroll = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { el.style.transform = `translate3d(0, ${Math.min(window.scrollY, 900) * amount}px, 0)`; }); };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => { window.removeEventListener("scroll", onScroll); cancelAnimationFrame(raf); };
  }, [amount]);
  return <div ref={ref} className={`parallax ${className}`}>{children}</div>;
}
