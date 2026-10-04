import type { Metadata } from "next";
import { Suspense } from "react";
import HighlightReview from "@/components/HighlightReview";

export const metadata: Metadata = { title: "Highlight review" };

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading candidates…</p>}>
      <HighlightReview />
    </Suspense>
  );
}
