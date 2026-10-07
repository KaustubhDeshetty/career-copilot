import type { Metadata } from "next";
import Link from "next/link";
import { BackendStatus } from "@/components/BackendStatus";
import "./globals.css";

export const metadata: Metadata = {
  title: "Career Copilot",
  description: "Multi-agent assistant for job applications, interview prep and revision",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <div className="container row spread">
            <Link href="/" className="brand">
              <span className="logo" aria-hidden>
                ◆
              </span>
              Career Copilot
            </Link>
            <nav className="row">
              <Link href="/">Assistant</Link>
              <Link href="/tracker">Tracker</Link>
              <BackendStatus />
            </nav>
          </div>
        </header>
        <main className="container">{children}</main>
        <footer className="container muted small footer">
          Router ∥ Planner → Selector → Researcher → Writer ⇄ Critic · Python agents behind a Next.js +
          TypeScript front end
        </footer>
      </body>
    </html>
  );
}
