// Re-mounts on navigation, so each page gets one short fade and rise. CSS only; disabled under reduced motion.
export default function Template({ children }: { children: React.ReactNode }) {
  return <div className="page-in">{children}</div>;
}
