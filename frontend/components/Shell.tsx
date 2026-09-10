"use client";

/** Application chrome: sidebar navigation, provider banner, sign-out. */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { useAuth, useRequireAuth } from "@/lib/auth";
import { api } from "@/lib/api";
import { Spinner, cx } from "@/components/ui";
import type { Health } from "@/types/api";

const NAV = [
  { href: "/", label: "Dashboard", icon: "M3 12h6v9H3zM10 3h4v18h-4zM15 8h6v13h-6z" },
  { href: "/trending", label: "Trending", icon: "M3 17l6-6 4 4 8-8M21 7v5h-5" },
  { href: "/videos", label: "Videos", icon: "M4 5h16v14H4zM10 9l5 3-5 3z" },
  { href: "/processing", label: "Processing", icon: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l3 3M15 15l3 3M6 18l3-3M15 9l3-3" },
  { href: "/clips", label: "Clips", icon: "M8 4h8v16H8zM8 8h8M8 16h8" },
  { href: "/analytics", label: "Analytics", icon: "M4 20V10M10 20V4M16 20v-7M22 20H2" },
  { href: "/settings", label: "Settings", icon: "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12a7 7 0 00-.1-1l2-1.6-2-3.4-2.4 1a7 7 0 00-1.7-1L14.5 3h-4l-.3 2.6a7 7 0 00-1.7 1l-2.4-1-2 3.4 2 1.6a7 7 0 000 2l-2 1.6 2 3.4 2.4-1a7 7 0 001.7 1l.3 2.6h4l.3-2.6a7 7 0 001.7-1l2.4 1 2-3.4-2-1.6a7 7 0 00.1-1z" },
  { href: "/integrations", label: "Integrations", icon: "M9 3v6M15 3v6M5 9h14v5a7 7 0 01-14 0z" },
];

function Icon({ path }: { path: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-4 w-4 shrink-0"
      aria-hidden
    >
      <path d={path} />
    </svg>
  );
}

/**
 * Tells the user, plainly, when the deployment is running without a model or a
 * transcription provider. Hiding that would make deterministic fallbacks look
 * like model output.
 */
function ProviderBanner({ health }: { health: Health | null }) {
  if (!health) return null;
  const degraded = health.components.filter(
    (c) => c.status !== "ok" && ["llm", "transcription", "task_queue"].includes(c.name),
  );
  if (degraded.length === 0) return null;

  return (
    <div className="border-b border-warn-500/25 bg-warn-500/10 px-6 py-2 text-xs text-warn-500">
      <span className="font-medium">Running in degraded mode. </span>
      {degraded.map((c) => c.detail ?? c.name).join(" ")}
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { user, loading } = useRequireAuth();
  const { logout } = useAuth();
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    api
      .health()
      .then((value) => {
        if (!cancelled) setHealth(value);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [user]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center text-ink-400">
        <Spinner size={20} />
      </div>
    );
  }
  if (!user) return null;

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="flex w-56 shrink-0 flex-col border-r border-ink-800 bg-ink-900">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-600 text-sm font-bold text-white">
            V
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-ink-50">
              Viral Clip Agent
            </div>
            <div className="truncate text-[11px] text-ink-500">
              {health?.environment ?? ""}
            </div>
          </div>
        </div>

        <nav className="flex-1 space-y-0.5 px-3">
          {NAV.map((item) => {
            const active =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cx(
                  "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors",
                  active
                    ? "bg-ink-800 font-medium text-ink-50"
                    : "text-ink-400 hover:bg-ink-850 hover:text-ink-100",
                )}
              >
                <Icon path={item.icon} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-ink-800 px-3 py-3">
          <div className="truncate px-2.5 text-xs text-ink-400" title={user.email}>
            {user.email}
          </div>
          <button
            onClick={logout}
            className="mt-1.5 w-full rounded-lg px-2.5 py-1.5 text-left text-xs text-ink-500 transition-colors hover:bg-ink-850 hover:text-ink-200"
          >
            Sign out
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <ProviderBanner health={health} />
        <main className="flex-1 overflow-y-auto px-6 py-6 lg:px-8">
          <div className="mx-auto max-w-7xl">{children}</div>
        </main>
      </div>
    </div>
  );
}
