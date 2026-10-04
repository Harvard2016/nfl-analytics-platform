import type { Metadata } from "next";
import { Suspense } from "react";
import CoverageExplorer from "@/components/CoverageExplorer";

export const metadata: Metadata = { title: "Coverage quick throws" };

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading plays…</p>}>
      <CoverageExplorer view="quick" />
    </Suspense>
  );
}
