"use client";

interface Props {
  /** How many zones are proposed and waiting; 0 when there is nothing to accept. */
  proposed: number;
  busy: boolean;
  /** True while waiting for a click on a painted line. */
  picking: boolean;
  onFind: () => void;
  onPick: () => void;
  onAdd: () => void;
  onDiscard: () => void;
}

/** One button that proposes zones from the painted floor lines, then add or discard. */
export function AutoZonePanel({
  proposed,
  busy,
  picking,
  onFind,
  onPick,
  onAdd,
  onDiscard,
}: Props) {
  const button =
    "h-8 rounded border border-line px-3 font-medium hover:bg-surface-2 disabled:opacity-50";
  return (
    <section aria-labelledby="auto-h">
      <h2
        id="auto-h"
        className="mb-1.5 font-display text-[length:var(--text-micro)] font-semibold uppercase tracking-[0.08em] text-faint"
      >
        Auto zones
      </h2>
      {proposed === 0 ? (
        <>
          <button type="button" onClick={onFind} disabled={busy} className={`${button} w-full`}>
            {busy ? "Looking for floor lines…" : "Auto-detect zones"}
          </button>
          <button
            type="button"
            onClick={onPick}
            disabled={busy}
            aria-pressed={picking}
            className={`${button} mt-1.5 w-full ${picking ? "border-accent bg-accent-soft" : ""}`}
          >
            {picking ? "Click a painted line… (Esc)" : "Pick line color"}
          </button>
          <p className="mt-1.5 text-dim">
            Finds the painted lines on this frame and proposes zones. If it finds none, pick the
            color by clicking on a painted line. You can adjust the zones after.
          </p>
        </>
      ) : (
        <>
          <p className="mb-1.5">
            {proposed} {proposed === 1 ? "zone" : "zones"} proposed, shown dashed on the frame.
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onAdd}
              className={`${button} flex-1 border-accent bg-accent-soft`}
            >
              Add {proposed === 1 ? "zone" : `${proposed} zones`}
            </button>
            <button type="button" onClick={onDiscard} className={button}>
              Discard
            </button>
          </div>
        </>
      )}
    </section>
  );
}
