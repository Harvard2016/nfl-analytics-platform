import type { Metadata } from "next";
import Forecasts from "@/components/Forecasts";

export const metadata: Metadata = { title: "Live forecast log" };

export default function Page() {
  return <Forecasts />;
}
