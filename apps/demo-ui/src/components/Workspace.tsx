"use client";

import { useCallback, useEffect, useState } from "react";
import { useDemo } from "@/hooks/useDemo";
import { useEventStream } from "@/hooks/useEventStream";
import { nextZoneId } from "@/lib/geometry";
import type { Point } from "@/lib/types";
import { NoticeBar } from "./NoticeBar";
import { Stage } from "./Stage";
import { type Mode, Toolbar } from "./Toolbar";
import { TopBar } from "./TopBar";

export function Workspace() {
  const demo = useDemo();
  const status = useEventStream(demo.pushEvent, demo.backfillEvents);
  const [mode, setMode] = useState<Mode>("select");
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  const { state, saveZones } = demo;

  const createZone = useCallback(
    (polygon: Point[]) => {
      if (!state) return;
      const id = nextZoneId(state.zones);
      saveZones([...state.zones, { id, polygon, min_dwell_s: 0 }]);
      setSelectedZone(id);
      setMode("select");
    },
    [state, saveZones],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLElement && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "v" || e.key === "V") setMode("select");
      else if (e.key === "z" || e.key === "Z") setMode("draw");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="grid h-full grid-rows-[40px_1fr]">
      <TopBar state={state} status={status} />
      <div className="grid min-h-0 grid-cols-[280px_1fr_320px]">
        <aside
          className="flex flex-col gap-4 border-r border-line bg-surface p-3"
          aria-label="Tools"
        >
          <Toolbar mode={mode} onChange={setMode} />
        </aside>
        <main className="relative min-h-0 min-w-0 bg-stage" aria-label="Camera view">
          <NoticeBar notice={demo.notice} onDismiss={demo.dismissNotice} />
          {state ? (
            <Stage
              state={state}
              mode={mode}
              selectedZone={selectedZone}
              onSelectZone={setSelectedZone}
              onCreateZone={createZone}
            />
          ) : (
            <p className="p-6 text-dim">{demo.notice?.text ?? "Loading…"}</p>
          )}
        </main>
        <aside className="border-l border-line bg-surface" aria-label="Event feed" />
      </div>
    </div>
  );
}
