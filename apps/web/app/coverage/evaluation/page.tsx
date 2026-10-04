import type { Metadata } from "next";
import { Suspense } from "react";
import CoverageEvaluation from "@/components/CoverageEvaluation";

export const metadata: Metadata = { title: "Coverage evaluation" };

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading evaluation…</p>}>
      <CoverageEvaluation />
    </Suspense>
  );
}
