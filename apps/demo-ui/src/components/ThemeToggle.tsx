"use client";

import { useSyncExternalStore } from "react";

type Theme = "light" | "dark";

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function currentTheme(): Theme {
  const explicit = document.documentElement.dataset.theme;
  return explicit === "light" || explicit === "dark" ? explicit : systemTheme();
}

const listeners = new Set<() => void>();
const subscribe = (cb: () => void) => {
  listeners.add(cb);
  return () => listeners.delete(cb);
};

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, currentTheme, () => "dark" as Theme);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("pv-theme", next);
    } catch {}
    listeners.forEach((l) => l());
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      className="h-7 rounded border border-line px-2 text-[length:var(--text-micro)] font-medium uppercase tracking-wider text-dim hover:bg-surface-2 hover:text-text"
    >
      {theme === "dark" ? "Dark" : "Light"}
    </button>
  );
}
