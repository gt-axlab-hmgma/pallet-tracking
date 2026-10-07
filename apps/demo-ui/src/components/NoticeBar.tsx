import type { Notice } from "@/hooks/useDemo";

export function NoticeBar({ notice, onDismiss }: { notice: Notice | null; onDismiss: () => void }) {
  if (!notice) return null;
  return (
    <div
      role="alert"
      className="absolute left-1/2 top-3 z-10 flex max-w-[min(560px,90%)] -translate-x-1/2 items-start gap-3 rounded border border-line bg-surface px-3 py-2 shadow-lg"
    >
      <span
        className={`mt-px text-[length:var(--text-micro)] font-semibold uppercase tracking-wider ${
          notice.kind === "error" ? "text-danger" : "text-accent"
        }`}
      >
        {notice.kind === "error" ? "Not saved" : "Heads up"}
      </span>
      <span className="text-text">{notice.text}</span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="ml-1 text-faint hover:text-text"
      >
        ×
      </button>
    </div>
  );
}
