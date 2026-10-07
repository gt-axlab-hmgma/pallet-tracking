import type { ZoneEvent } from "@/lib/types";

interface Props {
  events: ZoneEvent[]; // newest first
  sink: string | undefined;
  sinkError: string | null;
  configError: string | null;
}

const clock = (iso: string) => {
  const d = new Date(iso);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
};

const STORED_COPY: Record<string, string> = {
  jsonl: "appended to runs/demo/events.jsonl",
  postgres: "written to Postgres",
  outbox: "queued in the outbox for pv-sync",
  stdout: "printed by the API server",
};

export function EventFeed({ events, sink, sinkError, configError }: Props) {
  return (
    <section aria-labelledby="feed-h" className="flex h-full min-h-0 flex-col">
      <header className="border-b border-line p-3">
        <h2
          id="feed-h"
          className="flex items-baseline justify-between font-display text-[length:var(--text-micro)] font-semibold uppercase tracking-[0.08em] text-faint"
        >
          Event log
          <span className="font-sans tracking-normal">{events.length}</span>
        </h2>
        {sink && (
          <p className="mt-1 text-faint">Each event is {STORED_COPY[sink] ?? `sent to ${sink}`}.</p>
        )}
      </header>
      {configError && (
        <p role="alert" className="border-b border-line bg-accent-soft px-3 py-2 text-text">
          <strong className="font-semibold">Zones not saved.</strong> They apply for this session,
          but the database could not store them. Start Postgres, then save the zones again or move a
          box. {configError}
        </p>
      )}
      {sinkError && (
        <p role="alert" className="border-b border-line bg-accent-soft px-3 py-2 text-text">
          <strong className="font-semibold">Shown here but not stored.</strong> {sinkError}
        </p>
      )}
      {events.length === 0 ? (
        <ol className="list-decimal space-y-1.5 p-3 pl-7 text-dim">
          <li>
            Draw a zone with <kbd className="rounded border border-line px-1">Z</kbd>.
          </li>
          <li>Add a mock object.</li>
          <li>Drag it across the zone edge.</li>
          <li className="list-none text-faint">Joins and leaves appear here as they happen.</li>
        </ol>
      ) : (
        <ol
          role="log"
          aria-label="Zone events"
          className="scroll-thin min-h-0 flex-1 overflow-y-auto"
        >
          {events.map((e) => (
            <li key={e.event_id} className="feed-in border-b border-line px-3 py-1.5">
              <div className="flex items-baseline gap-2 text-faint">
                <span
                  aria-hidden
                  className={`w-3 text-center font-semibold ${e.event_type === "enter" ? "text-join" : "text-leave"}`}
                >
                  {e.event_type === "enter" ? "+" : "−"}
                </span>
                <time dateTime={e.device_ts}>{clock(e.device_ts)}</time>
                <span className="ml-auto">
                  #{e.track_id} · seq {e.seq}
                </span>
              </div>
              <p className="pl-5 text-text">{e.message}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
