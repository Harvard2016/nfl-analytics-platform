import type { Metadata } from "next";
import AnalyzePlay from "@/components/AnalyzePlay";

export const metadata: Metadata = { title: "Analyze your play" };

export default function Page() {
  return <AnalyzePlay />;
}
