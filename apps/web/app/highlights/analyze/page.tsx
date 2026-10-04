import type { Metadata } from "next";
import AnalyzeClip from "@/components/AnalyzeClip";

export const metadata: Metadata = { title: "Analyze your clip" };

export default function Page() {
  return <AnalyzeClip />;
}
