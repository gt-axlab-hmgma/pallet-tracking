export type Mode = "select" | "draw";

const TOOLS: { mode: Mode; label: string; key: string }[] = [
  { mode: "select", label: "Select", key: "V" },
  { mode: "draw", label: "Draw zone", key: "Z" },
];

export function Toolbar({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <div role="radiogroup" aria-label="Tool" className="grid grid-cols-2 gap-px rounded border border-line bg-line">
      {TOOLS.map((t) => {
        const active = t.mode === mode;
        return (
          <button
            key={t.mode}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(t.mode)}
            className={`flex h-8 items-center justify-between px-2.5 font-medium first:rounded-l last:rounded-r ${
              active ? "bg-accent text-accent-ink" : "bg-surface text-dim hover:bg-surface-2 hover:text-text"
            }`}
          >
            {t.label}
            <kbd className={`text-[length:var(--text-micro)] ${active ? "opacity-80" : "text-faint"}`}>{t.key}</kbd>
          </button>
        );
      })}
    </div>
  );
}
