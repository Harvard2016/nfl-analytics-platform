import type { Metadata } from "next";
import CoverageReview from "@/components/CoverageReview";

export const metadata: Metadata = { title: "Coverage error review" };

export default function Page() {
  return <CoverageReview />;
}
