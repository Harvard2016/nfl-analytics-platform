import type { Metadata } from "next";
import { Suspense } from "react";
import Highlights from "@/components/Highlights";

export const metadata: Metadata = { title: "Highlights" };

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading highlight benchmark…</p>}>
      <Highlights />
    </Suspense>
  );
}
