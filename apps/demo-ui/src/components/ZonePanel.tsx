"use client";

import { useState } from "react";
import { zoneColor } from "@/lib/geometry";
import type { MockObject, Zone } from "@/lib/types";

interface Props {
  zones: Zone[];
  objects: MockObject[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRename: (id: string, next: string) => string | null; // returns an error message, or null
  onDelete: (id: string) => void;
}

export function ZonePanel({ zones, objects, selectedId, onSelect, onRename, onDelete }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);

  function startEdit(id: string) {
    setEditing(id);
    setDraft(id);
    setError(null);
  }

  function commit(id: string) {
    const next = draft.trim();
    if (next === id) return setEditing(null);
    const err = onRename(id, next);
    if (err) setError(err);
    else setEditing(null);
  }

  return (
    <section aria-labelledby="zones-h" className="min-h-0">
      <h2
        id="zones-h"
        className="mb-1.5 flex items-baseline justify-between font-display text-[length:var(--text-micro)] font-semibold uppercase tracking-[0.08em] text-faint"
      >
        Zones
        <span className="font-sans tracking-normal">{zones.length}</span>
      </h2>
      {zones.length === 0 ? (
        <p className="text-dim">
          No zones yet. Press <kbd className="rounded border border-line px-1">Z</kbd> and click the
          corners of a floor area. Use as many corners as its outline needs.
        </p>
      ) : (
        <ul className="scroll-thin -mx-1 max-h-56 overflow-y-auto">
          {zones.map((zone, i) => {
            const active = zone.id === selectedId;
            const inside = objects.filter((o) => o.zones.includes(zone.id)).length;
            return (
              <li key={zone.id} className="group">
                <div
                  className={`flex h-8 items-center gap-2 rounded px-1 ${
                    active ? "bg-accent-soft" : "hover:bg-surface-2"
                  }`}
                >
                  <span
                    className="size-2.5 shrink-0 rounded-sm"
                    style={{
                      background: zoneColor(i).stroke,
                    }}
                    aria-hidden
                  />
                  {editing === zone.id ? (
                    <input
                      autoFocus
                      value={draft}
                      onChange={(e) => {
                        setDraft(e.target.value);
                        setError(null);
                      }}
                      onBlur={() => commit(zone.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commit(zone.id);
                        if (e.key === "Escape") setEditing(null);
                      }}
                      aria-label={`Rename ${zone.id}`}
                      className="h-6 min-w-0 flex-1 rounded border border-accent bg-surface px-1.5 text-text"
                    />
                  ) : (
                    <button
                      type="button"
                      onClick={() => onSelect(zone.id)}
                      onDoubleClick={() => startEdit(zone.id)}
                      className="min-w-0 flex-1 truncate text-left font-medium"
                      title="Click to select, double-click to rename"
                    >
                      {zone.id}
                    </button>
                  )}
                  <span className="shrink-0 text-faint">
                    {zone.polygon.length} pts
                    {inside > 0 && ` · ${inside} in`}
                  </span>
                  <button
                    type="button"
                    onClick={() => onDelete(zone.id)}
                    aria-label={`Delete ${zone.id}`}
                    className="shrink-0 px-1 text-faint opacity-0 hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    ×
                  </button>
                </div>
                {editing === zone.id && error && (
                  <p className="px-1 pb-1 text-danger" role="alert">
                    {error}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
