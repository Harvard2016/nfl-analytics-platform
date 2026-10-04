import type { Metadata } from "next";
import { Suspense } from "react";
import CoverageExplorer from "@/components/CoverageExplorer";

export const metadata: Metadata = { title: "Coverage error analysis" };

export default function CoverageErrorsPage() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading plays…</p>}>
      <CoverageExplorer view="errors" />
    </Suspense>
  );
}
