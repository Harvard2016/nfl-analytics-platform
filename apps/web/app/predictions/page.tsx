import type { Metadata } from "next";
import { Suspense } from "react";
import Predictions from "@/components/Predictions";

export const metadata: Metadata = { title: "Game predictor" };

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6 text-muted">Loading forecasts…</p>}>
      <Predictions />
    </Suspense>
  );
}
