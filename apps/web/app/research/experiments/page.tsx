import type { Metadata } from "next";
import Runs from "@/components/Runs";

export const metadata: Metadata = { title: "Experiment records" };

export default function Page() {
  return <Runs />;
}
