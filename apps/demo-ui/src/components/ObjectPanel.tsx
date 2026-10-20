"use client";

import { useState } from "react";
import type { MockObject } from "@/lib/types";

interface Props {
  objects: MockObject[];
  types: string[];
  /** Tracked from a video: the list is read-only. */
  live: boolean;
  selectedId: number | null;
  liveDrag: boolean;
  onLiveDrag: (v: boolean) => void;
  onSelect: (id: number) => void;
  onAdd: (type: string) => void;
  onChangeType: (id: number, type: string) => void;
  onDelete: (id: number) => void;
}

const selectClass =
  "h-7 min-w-0 rounded border border-line bg-surface px-1.5 text-text hover:bg-surface-2";

export function ObjectPanel({
  objects,
  types,
  live,
  selectedId,
  liveDrag,
  onLiveDrag,
  onSelect,
  onAdd,
  onChangeType,
  onDelete,
}: Props) {
  const [newType, setNewType] = useState(types[0] ?? "");
  const type = types.includes(newType) ? newType : (types[0] ?? "");

  return (
    <section aria-labelledby="objects-h" className="flex min-h-0 flex-1 flex-col">
      <h2
        id="objects-h"
        className="mb-1.5 flex items-baseline justify-between font-display text-[length:var(--text-micro)] font-semibold uppercase tracking-[0.08em] text-faint"
      >
        {live ? "Tracked objects" : "Mock objects"}
        <span className="font-sans tracking-normal">{objects.length}</span>
      </h2>
      {!live && (
        <div className="mb-2 flex gap-1.5">
          <select
            value={type}
            onChange={(e) => setNewType(e.target.value)}
            aria-label="Type of the new object"
            className={`${selectClass} flex-1`}
          >
            {types.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => onAdd(type)}
            className="h-7 rounded bg-accent px-3 font-medium text-accent-ink hover:brightness-110"
          >
            Add
          </button>
        </div>
      )}
      {objects.length === 0 ? (
        <p className="text-dim">
          {live
            ? "No pallet is being tracked right now. Boxes appear here as the tracker confirms them."
            : "Add an object, then drag it across a zone edge. Every crossing is logged as an event."}
        </p>
      ) : (
        <ul className="scroll-thin -mx-1 min-h-0 flex-1 overflow-y-auto">
          {objects.map((o) => (
            <li key={o.id} className="group">
              <div
                className={`flex min-h-9 items-center gap-1.5 rounded px-1 py-1 ${
                  o.id === selectedId ? "bg-accent-soft" : "hover:bg-surface-2"
                }`}
              >
                <button
                  type="button"
                  onClick={() => onSelect(o.id)}
                  className="w-7 shrink-0 text-left font-medium"
                  aria-label={`Select object ${o.id}`}
                >
                  #{o.id}
                </button>
                {live ? (
                  <span className="min-w-0 flex-1 truncate text-dim">{o.object_type}</span>
                ) : (
                  <select
                    value={o.object_type}
                    onChange={(e) => onChangeType(o.id, e.target.value)}
                    aria-label={`Type of object ${o.id}`}
                    className={`${selectClass} flex-1`}
                  >
                    {types.map((t) => (
                      <option key={t}>{t}</option>
                    ))}
                  </select>
                )}
                <span className="w-16 shrink-0 truncate text-right text-faint">
                  {o.zones.length > 0 ? o.zones.join(", ") : "outside"}
                </span>
                {!live && (
                  <button
                    type="button"
                    onClick={() => onDelete(o.id)}
                    aria-label={`Delete object ${o.id}`}
                    className="shrink-0 px-1 text-faint opacity-0 hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    ×
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {!live && (
        <label className="mt-2 flex cursor-pointer items-center gap-2 border-t border-line pt-2 text-dim">
          <input
            type="checkbox"
            checked={liveDrag}
            onChange={(e) => onLiveDrag(e.target.checked)}
            className="accent-[var(--accent)]"
          />
          Log events while dragging
        </label>
      )}
    </section>
  );
}
