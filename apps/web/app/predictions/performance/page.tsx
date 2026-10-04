import type { Metadata } from "next";
import PredictionPerformance from "@/components/PredictionPerformance";

export const metadata: Metadata = { title: "Game predictor performance" };

export default function Page() {
  return <PredictionPerformance />;
}
