import type { Metadata } from "next";
import { Suspense } from "react";
import Tendencies from "@/components/Tendencies";

export const metadata: Metadata = { title: "Coverage tendencies" };

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading tendencies…</p>}>
      <Tendencies />
    </Suspense>
  );
}
