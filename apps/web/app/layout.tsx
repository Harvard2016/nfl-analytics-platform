import type { Metadata } from "next";
import { Barlow_Condensed, IBM_Plex_Mono, Inter } from "next/font/google";
import Link from "next/link";
import SiteNav from "@/components/SiteNav";
import "./globals.css";

const barlow = Barlow_Condensed({ variable: "--font-barlow", subsets: ["latin"], weight: ["600", "700", "800"] });
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const plex = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: { default: "Gridiron Lens", template: "%s | Gridiron Lens" },
  description: "Three independent NFL analytics systems in one place: defensive coverage from player tracking, pregame win probabilities, and highlight ranking, each with inspectable evidence.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${barlow.variable} ${inter.variable} ${plex.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:bg-teal focus:px-3 focus:py-2 focus:text-bg">Skip to content</a>
        <SiteNav />
        <main id="main" className="flex-1">{children}</main>
        <footer className="border-t border-line">
          <div className="mx-auto grid max-w-[1500px] gap-6 px-4 py-10 text-sm text-muted md:grid-cols-[1.4fr_1fr_1fr] lg:px-8">
            <div>
              <p className="display text-2xl text-ink">Gridiron <span className="text-teal">Lens</span></p>
              <p className="mt-2 max-w-[46ch]">A research and portfolio project. Three independent systems, trained and evaluated locally, shown through frozen exports. Not affiliated with the NFL. Not betting advice.</p>
            </div>
            <ul className="space-y-1.5">
              <li className="kicker mb-2">Systems</li>
              <li><Link className="hover:text-ink" href="/coverage">Coverage Intelligence</Link></li>
              <li><Link className="hover:text-ink" href="/highlights">Highlight Intelligence</Link></li>
              <li><Link className="hover:text-ink" href="/predictions">Game Prediction</Link></li>
            </ul>
            <ul className="space-y-1.5">
              <li className="kicker mb-2">Evidence</li>
              <li><Link className="hover:text-ink" href="/research">Research</Link></li>
              <li><Link className="hover:text-ink" href="/research/experiments">Run records</Link></li>
              <li><Link className="hover:text-ink" href="/engineering">Engineering and data sources</Link></li>
            </ul>
          </div>
        </footer>
      </body>
    </html>
  );
}
