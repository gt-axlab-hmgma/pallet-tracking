import type { StreamStatus } from "@/hooks/useEventStream";

const COPY: Record<StreamStatus, { label: string; dot: string }> = {
  connecting: { label: "Connecting", dot: "bg-faint" },
  live: { label: "Live", dot: "bg-join" },
  reconnecting: { label: "Reconnecting", dot: "bg-danger" },
};

export function StatusPill({ status }: { status: StreamStatus }) {
  const { label, dot } = COPY[status];
  return (
    <span
      className="inline-flex items-center gap-1.5 text-dim"
      role="status"
      aria-label={`Event stream: ${label}`}
    >
      <span className={`size-1.5 rounded-full ${dot}`} aria-hidden />
      {label}
    </span>
  );
}
